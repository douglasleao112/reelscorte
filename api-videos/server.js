const express = require('express');
const multer = require('multer');
const ffmpeg = require('fluent-ffmpeg');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { OpenAI } = require('openai');
const { exec } = require('child_process');
const util = require('util');

const execPromise = util.promisify(exec);

const app = express();
const port = process.env.PORT || 3000;

// Configuração do OpenAI
// IMPORTANTE: Você precisa configurar a variável OPENAI_API_KEY no EasyPanel
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Configuração de pastas
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const OUTPUTS_DIR = path.join(__dirname, 'outputs');

if (!fs.existsSync(UPLOADS_DIR))
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(OUTPUTS_DIR))
  fs.mkdirSync(OUTPUTS_DIR, { recursive: true });

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.options('*', cors()); 

app.use(express.json());

app.use('/videos', express.static(OUTPUTS_DIR));

app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok', message: 'Servidor rodando perfeitamente!' });
});

// Configuração do Multer para receber o vídeo
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  },
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 500 * 1024 * 1024 }, // Limite de 500MB
});

// --- FUNÇÕES AUXILIARES ---

// Extrai o áudio do vídeo para enviar ao Whisper (arquivos menores = menos RAM e banda)
const extractAudio = (videoPath, audioPath) => {
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .output(audioPath)
      .noVideo()
      .audioCodec('libmp3lame')
      .audioBitrate('64k') // Bitrate baixo para arquivo pequeno
      .on('end', () => resolve(audioPath))
      .on('error', (err) => reject(err))
      .run();
  });
};

// Transcreve o áudio usando OpenAI Whisper
const transcribeAudio = async (audioPath) => {
  console.log('Iniciando transcrição com Whisper...');
  const transcription = await openai.audio.transcriptions.create({
    file: fs.createReadStream(audioPath),
    model: 'whisper-1',
    response_format: 'verbose_json', // Retorna timestamps
    timestamp_granularities: ['segment'],
  });

  // Formata para o GPT-4 (apenas o necessário para economizar tokens)
  const segments = transcription.segments.map((s) => ({
    start: s.start,
    end: s.end,
    text: s.text.trim(),
  }));

  return segments;
};

// Analisa a transcrição com GPT-4 para encontrar os melhores cortes
const analyzeContext = async (segments, prompt, theme, duration, clipCount) => {
  console.log('Analisando contexto com GPT-4...');

  const systemPrompt = `Você é um editor de vídeo viral especialista e obcecado em retenção e viralização de conteúdo.
Sua tarefa é analisar a transcrição de um vídeo e identificar os melhores trechos contínuos que formam cortes perfeitos para Reels/TikTok aonde as pessoas assistem até o fim. 

IMPORTANTE: 
- Remover os silêncios entre as falas, ajustar o enquadramento para acompanhar o movimento da cabeça do usuário e editar os cortes de forma fluida, unindo os trechos para manter a fala contínua e dinâmica.

DIRETRIZES:
- O corte DEVE ter sentido completo (início, meio e fim). Não corte no meio de uma frase.
- Duração alvo: aproximadamente ${duration}.
 - O primeiro 1-2s deve ter gancho (curiosidade, promessa, contradição, tensão, pergunta).
- Tema desejado: ${theme || 'Qualquer tema interessante'}.
- Instruções específicas do usuário: ${prompt || 'Nenhuma'}.
- Se houver tema, priorize o tema. Se não houver, pegue os trechos mais fortes do vídeo.
- Evitar partes burocráticas: cumprimentos longos, “galera…”, “deixa eu te falar”, enrolação.
- Priorize trechos com alta emoção, dicas valiosas, histórias curtas ou ganchos fortes.
- Retorne EXATAMENTE ${clipCount} cortes.

FORMATO DE RESPOSTA (JSON estrito):
{
  "clips": [
    {
      "id": "1",
      "title": "Título chamativo (max 5 palavras)",
      "description": "Por que este corte é bom com 90 a 110 caractéres",
      "start": 12.5, // tempo em segundos (número)
      "end": 45.2, // tempo em segundos (número)
      "score": 95 // nota de 0 a 100 de quão viral isso pode ser
    }
  ]
}`;

  const userMessage = `Aqui está a transcrição do vídeo com timestamps (em segundos):\n${JSON.stringify(segments, null, 2)}`;

  const response = await openai.chat.completions.create({
    model: 'gpt-4o', // Modelo mais inteligente para contexto
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMessage },
    ],
    response_format: { type: 'json_object' },
    temperature: 0.7,
  });

  const content = response.choices[0].message.content;
  const parsed = JSON.parse(content);

if (!parsed.clips || !Array.isArray(parsed.clips)) {
  throw new Error('Resposta da IA não retornou "clips" no formato esperado.');
}

return parsed.clips.slice(0, clipCount);
};

// Corta o vídeo usando FFmpeg com base nos timestamps
const cutVideo = (inputPath, outputPath, start, end) => {
  return new Promise((resolve, reject) => {
    const duration = end - start;
    ffmpeg(inputPath)
      .setStartTime(start)
      .setDuration(duration)
      .output(outputPath)
      .videoCodec('libx264')
      .audioCodec('aac')
      // Otimizações para VPS fraca: preset ultrafast, crf alto (menor qualidade, mais rápido)
      .outputOptions(['-preset', 'ultrafast', '-crf', '28'])
      .on('end', () => resolve(outputPath))
      .on('error', (err) => reject(err))
      .run();
  });
};

// --- ROTA PRINCIPAL DA API ---

app.post('/api/process-video', upload.single('video'), async (req, res) => {
  req.setTimeout(600000);
  res.setTimeout(600000);

  if (!req.file) {
    return res.status(400).json({ error: 'Nenhum vídeo enviado.' });
  }

  const videoPath = req.file.path;
 const { duration = '30s', prompt = '', theme = '', clipCount = '3' } = req.body;

const clipsN = Math.min(Math.max(parseInt(clipCount, 10) || 3, 1), 10); 
// limita de 1 a 10 pra evitar estouro de custo/tempo


  const baseUrl = `${req.protocol}://${req.get('host')}`;
  const tempAudioPath = path.join(UPLOADS_DIR, `temp_audio_${Date.now()}.mp3`);

  try {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY não configurada no servidor.');
    }

    console.log(`Processando vídeo: ${req.file.filename}`);

    // PASSO 1: Extrair áudio e Transcrever com Whisper
    console.log('Passo 1: Extraindo áudio...');
    await extractAudio(videoPath, tempAudioPath);

    console.log('Passo 1: Transcrevendo...');
    const segments = await transcribeAudio(tempAudioPath);

    if (segments.length === 0) {
      throw new Error('Não foi possível detectar fala no vídeo.');
    }

    // PASSO 2: Analisar com GPT-4
    console.log('Passo 2: Analisando com IA...');
    const aiClips = await analyzeContext(segments, prompt, theme, duration, clipsN);

    // PASSO 3: Cortar o vídeo com FFmpeg
    console.log('Passo 3: Gerando cortes reais...');
    const finalClips = [];

    for (let i = 0; i < aiClips.length; i++) {
      const clip = aiClips[i];
      const outputFilename = `corte_${Date.now()}_${i}.mp4`;
      const outputPath = path.join(OUTPUTS_DIR, outputFilename);

      console.log(`Cortando trecho ${i + 1}: ${clip.start}s até ${clip.end}s`);
      await cutVideo(videoPath, outputPath, clip.start, clip.end);

      // Calcula a duração real em formato mm:ss
      const durationSecs = Math.round(clip.end - clip.start);
      const mins = Math.floor(durationSecs / 60);
      const secs = durationSecs % 60;
      const formattedDuration = `${mins}:${secs.toString().padStart(2, '0')}`;

      finalClips.push({
        id: clip.id || String(i + 1),
        title: clip.title,
        description: clip.description,
        start: clip.start,
        end: clip.end,
        duration: formattedDuration,
        score: clip.score,
        url: `${baseUrl}/videos/${outputFilename}`,
      });
    }

    console.log('Processamento concluído com sucesso!');

    // Limpa o arquivo de áudio temporário
    if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);

    // Envia a resposta final
    res.json({ success: true, clips: finalClips });
  } catch (error) {
    console.error('Erro no processamento:', error);
    // Limpa o arquivo de áudio temporário em caso de erro
    if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);
    res
      .status(500)
      .json({ error: 'Erro ao processar vídeo', details: error.message });
  } 
});

// Limpeza automática a cada 10 minutos (arquivos mais velhos que 30 min)
setInterval(() => {
  const now = Date.now();
  const maxAge = 30 * 60 * 1000; // 30 minutos

  [UPLOADS_DIR, OUTPUTS_DIR].forEach((dir) => {
    fs.readdir(dir, (err, files) => {
      if (err) return console.error(`Erro ao ler diretório ${dir}:`, err);

      files.forEach((file) => {
        const filePath = path.join(dir, file);
        fs.stat(filePath, (err, stats) => {
          if (err) return;
          if (now - stats.mtimeMs > maxAge) {
            fs.unlink(filePath, (err) => {
              if (err) console.error(`Erro ao deletar ${filePath}:`, err);
              else
                console.log(
                  `Arquivo deletado por limpeza automática: ${filePath}`
                );
            });
          }
        });
      });
    });
  });
}, 10 * 60 * 1000);

app.listen(port, () => {
  console.log(
    `Servidor de processamento de vídeo com IA rodando na porta ${port}`
  );
});