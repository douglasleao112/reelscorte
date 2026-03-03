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
const analyzeContext = async (segments, prompt, duration, clipCount) => {
  console.log('Analisando contexto com GPT-4...');

  const systemPrompt = `Você é um editor de vídeo viral especialista e obcecado em retenção.
Sua tarefa é analisar a transcrição e escolher trechos contínuos que virem cortes perfeitos para Reels/TikTok.

DIRETRIZES:
- O corte DEVE ter sentido completo (início, meio e fim). Não corte no meio de uma frase.
- Duração alvo: aproximadamente ${duration}.
- O primeiro 3-5s deve ter gancho (curiosidade, promessa, contradição, tensão, pergunta).
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

  // Se não achou silêncio, assume que é tudo fala
  const probeCmd = `ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "${inputPath}"`;
  const { stdout } = await execPromise(probeCmd);
  const totalDuration = Math.max(parseFloat(stdout.trim()) || 0, 0);

  if (totalDuration <= 0) return null;

  if (silenceStarts.length === 0 && silenceEnds.length === 0) {
    return [[0, totalDuration]];
  }

  // ranges de "fala" são os espaços ENTRE silêncios
  const ranges = [];
  let cursor = 0;

  for (let i = 0; i < silenceStarts.length; i++) {
    const startSil = silenceStarts[i];
    const endSil = silenceEnds[i] ?? startSil;

    const a = Math.max(cursor, 0);
    const b = Math.min(startSil, totalDuration);

    if (b > a) ranges.push([a, b]);

    cursor = Math.min(Math.max(endSil, cursor), totalDuration);
  }

  if (cursor < totalDuration) {
    ranges.push([cursor, totalDuration]);
  }

  // remove pedaços muito curtos (evita micro-cortes)
  const minChunk = 0.18; // ajuste fino
  const filtered = ranges.filter(([a, b]) => (b - a) >= minChunk);

  return filtered.length ? filtered : null;
};


// Remove silêncios mantendo "tail/respiro" (ex: +650ms de vídeo real após cada fala)
// Sem inserir silêncio, sem clonar frame.
const removeSilences = async (inputPath, outputPath, tailMs = 650) => {
  const ranges = await detectSpeechRanges(inputPath);

  if (!ranges || ranges.length === 0) {
    fs.copyFileSync(inputPath, outputPath);
    return outputPath;
  }

  // duração total (pra clamp)
  const probeCmd = `ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "${inputPath}"`;
  const { stdout } = await execPromise(probeCmd);
  const totalDuration = Math.max(parseFloat(stdout.trim()) || 0, 0);

  const tailSec = tailMs / 1000;

  // 1) estende o final de cada fala em +tailSec
  // 2) evita overlap: se encostar no próximo trecho, a gente "cola" e depois mescla
  const extended = ranges.map(([start, end]) => {
    const newEnd = Math.min(end + tailSec, totalDuration);
    return [Math.max(0, start), Math.max(start, newEnd)];
  });

  // Mescla ranges que se sobrepõem/encostam (pra não dar micro-corte desnecessário)
  extended.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const [s, e] of extended) {
    if (!merged.length) {
      merged.push([s, e]);
      continue;
    }
    const last = merged[merged.length - 1];
    // se o próximo começa antes do último acabar (ou muito perto), mescla
    if (s <= last[1] + 0.02) {
      last[1] = Math.max(last[1], e);
    } else {
      merged.push([s, e]);
    }
  }

  // Se sobrou só 1 range, só recorta e pronto
  if (merged.length === 1) {
    const [s, e] = merged[0];
    const dur = Math.max(e - s, 0.01);
    const cmd =
      `ffmpeg -y -i "${inputPath}" ` +
      `-ss ${s.toFixed(3)} -t ${dur.toFixed(3)} ` +
      `-c:v libx264 -preset ultrafast -crf 28 ` +
      `-c:a aac -b:a 128k -movflags +faststart ` +
      `"${outputPath}"`;
    await execPromise(cmd);
    return outputPath;
  }

  // Monta filter_complex com 1 input só
  const filters = [];
  let concatInputs = '';

  merged.forEach(([start, end], i) => {
    const dur = Math.max(end - start, 0.01);

    // vídeo
    filters.push(
      `[0:v]trim=start=${start.toFixed(3)}:duration=${dur.toFixed(3)},setpts=PTS-STARTPTS[v${i}]`
    );

    // áudio
    filters.push(
      `[0:a]atrim=start=${start.toFixed(3)}:duration=${dur.toFixed(3)},asetpts=PTS-STARTPTS[a${i}]`
    );

    concatInputs += `[v${i}][a${i}]`;
  });

  filters.push(`${concatInputs}concat=n=${merged.length}:v=1:a=1[outv][outa]`);
  const filterComplex = filters.join(';');

  const cmd =
    `ffmpeg -y -i "${inputPath}" ` +
    `-filter_complex "${filterComplex}" ` +
    `-map "[outv]" -map "[outa]" ` +
    `-c:v libx264 -preset ultrafast -crf 28 ` +
    `-c:a aac -b:a 128k -movflags +faststart ` +
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
    clipCount = '3',
    videoSpeed = '1'
  } = req.body;

  // Permite qualquer velocidade entre 0.5 e 2.0
  const parsedSpeed = parseFloat(videoSpeed);
  const safeSpeed = (parsedSpeed >= 0.5 && parsedSpeed <= 2.0) ? parsedSpeed : 1.0;

  // Se o usuário clicou em "Max", definimos um limite alto (ex: 15). Senão, usamos o número escolhido.
  let clipsN = 15; 
  if (clipCount !== 'max') {
    clipsN = Math.min(Math.max(parseInt(clipCount, 10) || 3, 1), 15);
  }

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
    const aiClips = await analyzeContext(segments, prompt, duration, clipsN);

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