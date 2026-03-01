const express = require('express');
const multer = require('multer');
const ffmpeg = require('fluent-ffmpeg');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { OpenAI } = require('openai');
const util = require('util');
const { exec } = require('child_process');

const execPromise = util.promisify(exec);

const app = express();
const port = process.env.PORT || 3000;

// CORS
app.use(
  cors({
    origin: '*',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  })
);
app.options('*', cors());
app.use(express.json());

// OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Pastas
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const OUTPUTS_DIR = path.join(__dirname, 'outputs');

if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(OUTPUTS_DIR)) fs.mkdirSync(OUTPUTS_DIR, { recursive: true });

app.use('/videos', express.static(OUTPUTS_DIR));

// Health
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

// =======================
// FUNÇÕES AUXILIARES
// =======================

// Extrai áudio leve para Whisper
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

// Transcreve com Whisper (segmentos com timestamps)
const transcribeAudio = async (audioPath) => {
  console.log('Iniciando transcrição com Whisper...');
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

// IA escolhe os melhores cortes
const analyzeContext = async (segments, prompt, theme, duration, clipCount) => {
  console.log('Analisando contexto com GPT-4...');

  const systemPrompt = `Você é um editor de vídeo viral especialista e obcecado em retenção.
Sua tarefa é analisar a transcrição e escolher trechos contínuos que virem cortes perfeitos para Reels/TikTok.

DIRETRIZES:
- O corte DEVE ter sentido completo (início, meio e fim). Nunca cortar no meio de frase.
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
      "description": "Por que este corte é bom (90 a 110 caracteres)",
      "start": 12.5,
      "end": 45.2,
      "score": 95
    }
  ]
}`;

  const userMessage = `Aqui está a transcrição com timestamps (segundos):\n${JSON.stringify(
    segments,
    null,
    2
  )}`;

  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMessage },
    ],
    response_format: { type: 'json_object' },
    temperature: 0.7,
  });

  const content = response.choices?.[0]?.message?.content || '{}';
  const parsed = JSON.parse(content);

  if (!parsed.clips || !Array.isArray(parsed.clips)) {
    throw new Error('Resposta da IA não retornou "clips" no formato esperado.');
  }

  // garante quantidade
  return parsed.clips.slice(0, clipCount);
};

// Corta trecho bruto
const cutVideo = (inputPath, outputPath, start, end, speed = 1.0) => {
  return new Promise((resolve, reject) => {
    const dur = Math.max(end - start, 0.01);

    const s = Number.isFinite(speed) && speed > 0 ? speed : 1.0;
    const videoFilter = `setpts=${(1 / s).toFixed(6)}*PTS`;
    const audioFilter = `atempo=${s}`;

    ffmpeg(inputPath)
      .setStartTime(start)
      .setDuration(dur)
      .output(outputPath)
      .videoCodec('libx264')
      .audioCodec('aac')
      .videoFilters(videoFilter)
      .audioFilters(audioFilter)
      .outputOptions(['-preset', 'ultrafast', '-crf', '28'])
      .on('end', () => resolve(outputPath))
      .on('error', reject)
      .run();
  });
};

// -----------------------------------------
// REMOVER SILÊNCIOS (implementação real)
// Estratégia:
// 1) Rodar silencedetect no CLIP já cortado
// 2) Criar vários pedacinhos com fala
// 3) Concatenar tudo num MP4 final
// -----------------------------------------

// Detecta intervalos com fala dentro de um arquivo
const detectSpeechRanges = async (inputPath, noiseDb = -30, minSilence = 0.35) => {
  const cmd = `ffmpeg -i "${inputPath}" -af silencedetect=noise=${noiseDb}dB:d=${minSilence} -f null -`;
  const { stderr } = await execPromise(cmd);

  const silenceStarts = [];
  const silenceEnds = [];

  for (const line of stderr.split('\n')) {
    const s = line.match(/silence_start:\s*([0-9.]+)/);
    if (s) silenceStarts.push(parseFloat(s[1]));

    const e = line.match(/silence_end:\s*([0-9.]+)/);
    if (e) silenceEnds.push(parseFloat(e[1]));
  }

  // se não detectou silencios, retorna null (não mexe)
  if (silenceStarts.length === 0 && silenceEnds.length === 0) return null;

  // pega duração do arquivo
  const probeCmd = `ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "${inputPath}"`;
  const { stdout } = await execPromise(probeCmd);
  const totalDuration = Math.max(parseFloat(stdout.trim()) || 0, 0);

  const pad = 0.12; // evita cortar sílabas
  const minChunk = 0.35; // ignora trechos minúsculos

  const ranges = [];
  let cursor = 0;

  for (let i = 0; i < silenceStarts.length; i++) {
    const startSil = silenceStarts[i];
    const endSil = silenceEnds[i] ?? startSil;

    const a = Math.max(cursor, 0);
    const b = Math.max(startSil - pad, a);

    if (b - a >= minChunk) ranges.push([a, b]);
    cursor = Math.max(endSil + pad, cursor);
  }

  // trecho final
  if (totalDuration > 0) {
    const a = Math.min(cursor, totalDuration);
    const b = totalDuration;
    if (b - a >= minChunk) ranges.push([a, b]);
  }

  if (ranges.length === 0) return null;
  return ranges;
};


// Remove silêncios e garante um "respiro" entre falas (gap fixo)
const removeSilences = async (inputPath, outputPath, gapMs = 850) => {
  const ranges = await detectSpeechRanges(inputPath);

  // se não há ranges, copia o arquivo
  if (!ranges) {
    fs.copyFileSync(inputPath, outputPath);
    return outputPath;
  }

  // monta comando ffmpeg com concat filter (vídeo + áudio) e gap entre trechos
  // Ideia:
  // - Para cada trecho: trim + setpts / atrim + asetpts
  // - Antes de concatenar, "empurra" o áudio com adelay e o vídeo com tpad (preenche com último frame)
  //   para gerar o gap desejado entre os blocos.
  //
  // Observação:
  // - No vídeo: tpad=stop_mode=clone:stop_duration=0.85
  // - No áudio: apad + atrim mantendo duração e adelay de 850ms (por trecho, exceto o primeiro)
  //
  // Isso cria um espacinho natural entre falas.

  const gapSec = (gapMs / 1000).toFixed(3);

  // cria entrada repetida do mesmo arquivo para cada trecho
  const inputArgs = [];
  for (let i = 0; i < ranges.length; i++) {
    inputArgs.push(`-i "${inputPath}"`);
  }

  // filter_complex
  const vLabels = [];
  const aLabels = [];
  const filters = [];

  for (let i = 0; i < ranges.length; i++) {
    const [start, end] = ranges[i];
    const dur = Math.max(end - start, 0.01);

    // vídeo do trecho
    // trim no intervalo + reset pts
    let vChain = `[${i}:v]trim=start=${start}:duration=${dur},setpts=PTS-STARTPTS`;

    // adiciona gap no vídeo (exceto no último, mas pode manter também)
    // tpad clone mantém o último frame por gapSec segundos
    vChain += `,tpad=stop_mode=clone:stop_duration=${gapSec}[v${i}]`;
    filters.push(vChain);
    vLabels.push(`[v${i}]`);

    // áudio do trecho
    let aChain = `[${i}:a]atrim=start=${start}:duration=${dur},asetpts=PTS-STARTPTS`;

    // adiciona gap no áudio: apad cria “silêncio”, depois atrim limita ao dur + gap
    // e a gente mantém um espacinho entre os blocos.
    aChain += `,apad=pad_dur=${gapSec},atrim=duration=${(dur + parseFloat(gapSec)).toFixed(3)}[a${i}]`;

    filters.push(aChain);
    aLabels.push(`[a${i}]`);
  }

  // concatena todos os blocos (n = quantidade de trechos)
  // v=1 a=1 para concatenar vídeo e áudio juntos
  filters.push(`${vLabels.join('')}${aLabels.join('')}concat=n=${ranges.length}:v=1:a=1[outv][outa]`);

  const filterComplex = filters.join(';');

  const cmd =
    `ffmpeg -y ${inputArgs.join(' ')} ` +
    `-filter_complex "${filterComplex}" ` +
    `-map "[outv]" -map "[outa]" ` +
    `-c:v libx264 -preset ultrafast -crf 28 ` +
    `-c:a aac -b:a 128k ` +
    `"${outputPath}"`;

  await execPromise(cmd);

  return outputPath;
};





// =======================
// ROTA PRINCIPAL
// =======================
app.post('/api/process-video', upload.single('video'), async (req, res) => {
  req.setTimeout(600000);
  res.setTimeout(600000);

  if (!req.file) return res.status(400).json({ error: 'Nenhum vídeo enviado.' });

  const videoPath = req.file.path;

  const {
  duration = '30s',
  prompt = '',
  theme = '',
  clipCount = '3',
  videoSpeed = '1'
} = req.body;

// 🔒 aceita somente essas velocidades
const allowed = new Set(['1', '1.2', '1.5', '1.7', '2']);
const speedStr = String(videoSpeed).trim();
const safeSpeed = allowed.has(speedStr) ? parseFloat(speedStr) : 1.0;

  const clipsN = Math.min(Math.max(parseInt(clipCount, 10) || 3, 1), 10);
  const baseUrl = `${req.protocol}://${req.get('host')}`;
  const tempAudioPath = path.join(UPLOADS_DIR, `temp_audio_${Date.now()}.mp3`);

  try {
    if (!process.env.OPENAI_API_KEY) throw new Error('OPENAI_API_KEY não configurada no servidor.');

    console.log(`Processando vídeo: ${req.file.filename}`);

    // 1) áudio + transcrição
    console.log('Passo 1: Extraindo áudio...');
    await extractAudio(videoPath, tempAudioPath);

    console.log('Passo 1: Transcrevendo...');
    const segments = await transcribeAudio(tempAudioPath);
    if (!segments.length) throw new Error('Não foi possível detectar fala no vídeo.');

    // 2) IA escolhe cortes
    console.log('Passo 2: Analisando com IA...');
    const aiClips = await analyzeContext(segments, prompt, theme, duration, clipsN);

    // 3) gera cortes + remove silêncios
    console.log('Passo 3: Gerando cortes reais (com remoção de silêncios)...');
    const finalClips = [];

    for (let i = 0; i < aiClips.length; i++) {
      const clip = aiClips[i];

      const rawFilename = `corte_raw_${Date.now()}_${i}.mp4`;
      const rawPath = path.join(OUTPUTS_DIR, rawFilename);

      // corte bruto
     console.log(`Cortando bruto ${i + 1}: ${clip.start}s até ${clip.end}s (speed=${safeSpeed})`);
    await cutVideo(videoPath, rawPath, clip.start, clip.end, safeSpeed);

      // remove silêncios
      const finalFilename = `corte_${Date.now()}_${i}.mp4`;
      const finalPath = path.join(OUTPUTS_DIR, finalFilename);

      console.log(`Removendo silêncios do corte ${i + 1}...`);
      await removeSilences(rawPath, finalPath, 650);   // tempo do silencio entre o corte

      // apaga bruto para economizar espaço
      if (fs.existsSync(rawPath)) fs.unlinkSync(rawPath);

      // duração aproximada (antes de remover silêncios)
      const durationSecs = Math.max(Math.round(clip.end - clip.start), 0);
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
        url: `${baseUrl}/videos/${finalFilename}`,
      });
    }

    // limpa áudio temporário
    if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);

    // opcional: apagar o vídeo original logo após gerar tudo (economiza MUITO espaço)
    // se quiser isso, descomenta:
    // if (fs.existsSync(videoPath)) fs.unlinkSync(videoPath);

    console.log('Processamento concluído com sucesso!');
    return res.json({ success: true, clips: finalClips });
  } catch (error) {
    console.error('Erro no processamento:', error);
    if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);
    return res.status(500).json({ error: 'Erro ao processar vídeo', details: error.message });
  }
});

// Limpeza automática a cada 10 minutos (arquivos > 30 min)
setInterval(() => {
  const now = Date.now();
  const maxAge = 30 * 60 * 1000;

  [UPLOADS_DIR, OUTPUTS_DIR].forEach((dir) => {
    fs.readdir(dir, (err, files) => {
      if (err) return;
      files.forEach((file) => {
        const filePath = path.join(dir, file);
        fs.stat(filePath, (err2, stats) => {
          if (err2) return;
          if (now - stats.mtimeMs > maxAge) {
            fs.unlink(filePath, () => {});
          }
        });
      });
    });
  });
}, 10 * 60 * 1000);

app.listen(port, () => {
  console.log(`Servidor de processamento de vídeo com IA rodando na porta ${port}`);
});