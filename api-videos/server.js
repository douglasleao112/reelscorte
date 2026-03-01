const express = require('express');
const multer = require('multer');
const ffmpeg = require('fluent-ffmpeg');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { OpenAI } = require('openai');

const app = express();
const port = process.env.PORT || 3000;

// OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Pastas
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const OUTPUTS_DIR = path.join(__dirname, 'outputs');

if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(OUTPUTS_DIR)) fs.mkdirSync(OUTPUTS_DIR, { recursive: true });

// CORS + JSON
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));
app.use(express.json());
app.use('/videos', express.static(OUTPUTS_DIR));

// Health check
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok', message: 'Servidor rodando perfeitamente!' });
});

// Multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 500 * 1024 * 1024 }, // 500MB
});

// Prompt fixo do sistema (UMA VEZ)
const BASE_SYSTEM_PROMPT = `
Você é um editor de vídeos curtos (Reels/TikTok) obcecado por retenção.

OBJETIVO:
Selecionar trechos contínuos do vídeo que virem 3 cortes que as pessoas assistem até o fim.

REGRAS OBRIGATÓRIAS:
- Nunca cortar no meio de frase.
- O corte precisa ter começo, meio e fim (ideia completa).
- O primeiro 1-2s deve ter gancho (curiosidade, promessa, contradição, tensão, pergunta).
- Evitar partes burocráticas: cumprimentos longos, “galera…”, “deixa eu te falar”, enrolação.
- Priorizar: emoção, polêmica leve, história curta, dica prática, erro comum, antes/depois, “verdade que ninguém fala”.
- Se houver tema, priorize o tema. Se não houver, pegue os trechos mais fortes do vídeo.
- Retorne EXATAMENTE 3 cortes.

FORMATO DE RESPOSTA:
Responda SOMENTE com JSON válido no formato:
{
  "clips": [
    { "id":"1", "title":"...", "description":"...", "start": 0.0, "end": 0.0, "score": 0 }
  ]
}
`;

// Extrai áudio
const extractAudio = (videoPath, audioPath) => {
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .output(audioPath)
      .noVideo()
      .audioCodec('libmp3lame')
      .audioBitrate('64k')
      .on('end', () => resolve(audioPath))
      .on('error', reject)
      .run();
  });
};

// Transcreve com Whisper
const transcribeAudio = async (audioPath) => {
  const transcription = await openai.audio.transcriptions.create({
    file: fs.createReadStream(audioPath),
    model: 'whisper-1',
    response_format: 'verbose_json',
    timestamp_granularities: ['segment'],
  });

  const segments = (transcription.segments || []).map((s) => ({
    start: s.start,
    end: s.end,
    text: (s.text || '').trim(),
  }));

  return segments;
};

// GPT escolhe os cortes
const analyzeContext = async (segments, prompt, theme, duration) => {
  const systemPrompt = `
${BASE_SYSTEM_PROMPT}

CONFIGURAÇÕES DO USUÁRIO:
- Duração alvo: ${duration}
- Tema: ${theme ? theme : 'Nenhum (escolha os melhores trechos)'}
- Instrução extra do usuário (opcional): ${prompt ? prompt : 'Nenhuma'}
`;

  const userMessage =
    `Aqui está a transcrição do vídeo com timestamps (segundos):\n` +
    `${JSON.stringify(segments, null, 2)}`;

  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
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

  return parsed.clips;
};

// Corta vídeo
const cutVideo = (inputPath, outputPath, start, end) => {
  return new Promise((resolve, reject) => {
    const clipDuration = end - start;
    ffmpeg(inputPath)
      .setStartTime(start)
      .setDuration(clipDuration)
      .output(outputPath)
      .videoCodec('libx264')
      .audioCodec('aac')
      .outputOptions(['-preset', 'ultrafast', '-crf', '28'])
      .on('end', () => resolve(outputPath))
      .on('error', reject)
      .run();
  });
};

// Rota principal
app.post('/api/process-video', upload.single('video'), async (req, res) => {
  req.setTimeout(600000);
  res.setTimeout(600000);

  if (!req.file) return res.status(400).json({ error: 'Nenhum vídeo enviado.' });
  if (!process.env.OPENAI_API_KEY) {
    return res.status(500).json({ error: 'OPENAI_API_KEY não configurada no servidor.' });
  }

  const videoPath = req.file.path;
  const { duration = '30s', prompt = '', theme = '' } = req.body;

  const baseUrl = `${req.protocol}://${req.get('host')}`;
  const tempAudioPath = path.join(UPLOADS_DIR, `temp_audio_${Date.now()}.mp3`);

  try {
    // 1) áudio
    await extractAudio(videoPath, tempAudioPath);

    // 2) transcrição
    const segments = await transcribeAudio(tempAudioPath);
    if (!segments.length) throw new Error('Não foi possível detectar fala no vídeo.');

    // 3) GPT escolhe cortes
    const aiClips = await analyzeContext(segments, prompt, theme, duration);

    // 4) corta
    const finalClips = [];
    for (let i = 0; i < aiClips.length; i++) {
      const clip = aiClips[i];

      const outputFilename = `corte_${Date.now()}_${i}.mp4`;
      const outputPath = path.join(OUTPUTS_DIR, outputFilename);

      await cutVideo(videoPath, outputPath, Number(clip.start), Number(clip.end));

      const durationSecs = Math.max(0, Math.round(Number(clip.end) - Number(clip.start)));
      const mins = Math.floor(durationSecs / 60);
      const secs = durationSecs % 60;

      finalClips.push({
        id: clip.id || String(i + 1),
        title: clip.title,
        description: clip.description,
        start: clip.start,
        end: clip.end,
        duration: `${mins}:${String(secs).padStart(2, '0')}`,
        score: clip.score,
        url: `${baseUrl}/videos/${outputFilename}`,
      });
    }

    // limpa áudio temp
    if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);

    // opcional: apagar o vídeo original logo após gerar os cortes
    // (se quiser manter até o job de limpeza, pode comentar essa linha)
    if (fs.existsSync(videoPath)) fs.unlinkSync(videoPath);

    res.json({ success: true, clips: finalClips });
  } catch (error) {
    if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);
    res.status(500).json({ error: 'Erro ao processar vídeo', details: error.message });
  }
});

// Limpeza automática a cada 10 minutos
setInterval(() => {
  const now = Date.now();
  const maxAge = 30 * 60 * 1000;

  [UPLOADS_DIR, OUTPUTS_DIR].forEach((dir) => {
    fs.readdir(dir, (err, files) => {
      if (err) return;
      files.forEach((file) => {
        const filePath = path.join(dir, file);
        fs.stat(filePath, (err, stats) => {
          if (err) return;
          if (now - stats.mtimeMs > maxAge) {
            fs.unlink(filePath, () => { });
          }
        });
      });
    });
  });
}, 10 * 60 * 1000);

app.listen(port, () => {
  console.log(`Servidor rodando na porta ${port}`);
});