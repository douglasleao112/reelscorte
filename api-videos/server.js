const express = require("express");
const multer = require("multer");
const ffmpeg = require("fluent-ffmpeg");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const { OpenAI } = require("openai");
const { exec } = require("child_process");
const util = require("util");

const execPromise = util.promisify(exec);

const app = express();
const port = process.env.PORT || 80; // EasyPanel geralmente usa porta 80 ou 3000

// Configuração do OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Configuração de pastas
const UPLOADS_DIR = path.join(__dirname, "uploads");
const OUTPUTS_DIR = path.join(__dirname, "outputs");

if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(OUTPUTS_DIR)) fs.mkdirSync(OUTPUTS_DIR, { recursive: true });

// Middlewares
app.use(cors());
app.use(express.json());

// Servir vídeos com cabeçalhos CORS para permitir download direto no frontend
app.use("/videos", express.static(OUTPUTS_DIR, {
  setHeaders: (res) => {
    res.set('Access-Control-Allow-Origin', '*');
    res.set('Cross-Origin-Resource-Policy', 'cross-origin');
  }
}));

// Configuração do Multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  },
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 500 * 1024 * 1024 }, // 500MB
});

// --- FILA DE PROCESSAMENTO (Essencial para VPS 2GB) ---
let isProcessing = false;
const processingQueue = [];

const processNextInQueue = async () => {
  if (isProcessing || processingQueue.length === 0) return;
  isProcessing = true;
  const { task, res } = processingQueue.shift();
  try {
    await task();
  } catch (error) {
    console.error("Erro na tarefa de processamento:", error);
    if (!res.headersSent) {
      res.status(500).json({ error: "Erro interno no processamento", details: error.message });
    }
  } finally {
    isProcessing = false;
    processNextInQueue();
  }
};

// --- FUNÇÕES AUXILIARES ---

const extractAudio = (videoPath, audioPath) => {
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .output(audioPath)
      .noVideo()
      .audioCodec("libmp3lame")
      .audioBitrate("64k")
      .on("end", () => resolve(audioPath))
      .on("error", reject)
      .run();
  });
};

const transcribeAudio = async (audioPath) => {
  console.log("Iniciando transcrição com Whisper...");
  const transcription = await openai.audio.transcriptions.create({
    file: fs.createReadStream(audioPath),
    model: "whisper-1",
    response_format: "verbose_json",
    timestamp_granularities: ["segment"],
  });
  return transcription.segments.map((s) => ({
    start: s.start,
    end: s.end,
    text: s.text.trim(),
  }));
};

const analyzeContext = async (segments, prompt, duration, clipCount) => {
  console.log("Analisando contexto com GPT-4o...");
  const systemPrompt = `Você é um editor de vídeo viral especialista em retenção.
Analise a transcrição e identifique os melhores trechos contínuos para Reels/TikTok.

DIRETRIZES:
1. O corte DEVE ter sentido completo (início, meio e fim).
2. Duração alvo: ${duration}.
3. Instruções do usuário: ${prompt || "Nenhuma"}.
4. Priorize ganchos fortes nos primeiros 3 segundos.
5. Retorne EXATAMENTE ${clipCount} cortes.

FORMATO DE RESPOSTA (JSON):
{
  "clips": [
    {
      "id": "1",
      "title": "Título chamativo",
      "description": "Por que é viral",
      "start": 12.5,
      "end": 45.2,
      "score": 95
    }
  ]
}`;

  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: `Transcrição:\n${JSON.stringify(segments)}` },
    ],
    response_format: { type: "json_object" },
  });

  return JSON.parse(response.choices[0].message.content).clips;
};

const getCropFilter = (aspectRatio) => {
  let wRatio, hRatio;
  switch (aspectRatio) {
    case '9:16': wRatio = 9; hRatio = 16; break;
    case '1:1': wRatio = 1; hRatio = 1; break;
    case '4:5': wRatio = 4; hRatio = 5; break;
    case '16:9': wRatio = 16; hRatio = 9; break;
    default: return null;
  }
  return `crop=w='min(iw,ih*(${wRatio}/${hRatio}))':h='min(ih,iw*(${hRatio}/${wRatio}))':x='(iw-w)/2':y='(ih-h)/2'`;
};

const cutVideo = (inputPath, outputPath, start, end, speed = 1.0, aspectRatio = null) => {
  return new Promise((resolve, reject) => {
    const duration = end - start;
    const s = parseFloat(speed) || 1.0;
    
    const vFilters = [`setpts=${(1/s).toFixed(6)}*PTS`];
    const cropFilter = getCropFilter(aspectRatio);
    if (cropFilter) vFilters.push(cropFilter);

    ffmpeg(inputPath)
      .setStartTime(start)
      .setDuration(duration)
      .output(outputPath)
      .videoCodec("libx264")
      .audioCodec("aac")
      .videoFilters(vFilters)
      .audioFilters(`atempo=${s}`)
      .outputOptions(["-preset ultrafast", "-crf 28"])
      .on("end", () => resolve(outputPath))
      .on("error", reject)
      .run();
  });
};

const detectSpeechRanges = async (inputPath, noiseDb = -30, minSilence = 0.35) => {
  const cmd = `ffmpeg -i "${inputPath}" -af silencedetect=noise=${noiseDb}dB:d=${minSilence} -f null -`;
  const { stderr } = await execPromise(cmd);
  const silenceStarts = [];
  const silenceEnds = [];

  stderr.split('\n').forEach(line => {
    const s = line.match(/silence_start:\s*([0-9.]+)/);
    if (s) silenceStarts.push(parseFloat(s[1]));
    const e = line.match(/silence_end:\s*([0-9.]+)/);
    if (e) silenceEnds.push(parseFloat(e[1]));
  });

  const { stdout } = await execPromise(`ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "${inputPath}"`);
  const totalDuration = parseFloat(stdout.trim()) || 0;

  if (silenceStarts.length === 0) return [[0, totalDuration]];

  const ranges = [];
  let cursor = 0;
  for (let i = 0; i < silenceStarts.length; i++) {
    if (silenceStarts[i] > cursor) ranges.push([cursor, silenceStarts[i]]);
    cursor = silenceEnds[i] || silenceStarts[i];
  }
  if (cursor < totalDuration) ranges.push([cursor, totalDuration]);
  
  return ranges.filter(([a, b]) => (b - a) >= 0.2);
};

const removeSilences = async (inputPath, outputPath, tailMs = 500) => {
  const ranges = await detectSpeechRanges(inputPath);
  if (!ranges || ranges.length <= 1) {
    fs.copyFileSync(inputPath, outputPath);
    return outputPath;
  }

  const { stdout } = await execPromise(`ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "${inputPath}"`);
  const totalDuration = parseFloat(stdout.trim()) || 0;
  const tailSec = tailMs / 1000;

  const merged = [];
  ranges.forEach(([s, e]) => {
    const newEnd = Math.min(e + tailSec, totalDuration);
    if (!merged.length) { merged.push([s, newEnd]); return; }
    const last = merged[merged.length - 1];
    if (s <= last[1] + 0.05) last[1] = Math.max(last[1], newEnd);
    else merged.push([s, newEnd]);
  });

  const filters = [];
  let concatInputs = '';
  merged.forEach(([s, e], i) => {
    const d = e - s;
    filters.push(`[0:v]trim=start=${s.toFixed(3)}:duration=${d.toFixed(3)},setpts=PTS-STARTPTS[v${i}]`);
    filters.push(`[0:a]atrim=start=${s.toFixed(3)}:duration=${d.toFixed(3)},asetpts=PTS-STARTPTS[a${i}]`);
    concatInputs += `[v${i}][a${i}]`;
  });

  filters.push(`${concatInputs}concat=n=${merged.length}:v=1:a=1[outv][outa]`);
  
  const cmd = `ffmpeg -y -i "${inputPath}" -filter_complex "${filters.join(';')}" -map "[outv]" -map "[outa]" -c:v libx264 -preset ultrafast -crf 28 -c:a aac -b:a 128k "${outputPath}"`;
  await execPromise(cmd);
  return outputPath;
};

// --- ROTA PRINCIPAL ---

app.post("/api/process-video", upload.single("video"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "Nenhum vídeo enviado." });

  const task = async () => {
    const videoPath = req.file.path;
    const { duration = "30s", prompt = "", videoCount = "3", videoSpeed = "1", aspectRatio = "9:16" } = req.body;
    const baseUrl = `${req.protocol}://${req.get("host")}`;
    const tempAudioPath = path.join(UPLOADS_DIR, `temp_audio_${Date.now()}.mp3`);

    try {
      console.log(`Processando: ${req.file.filename}`);
      await extractAudio(videoPath, tempAudioPath);
      const segments = await transcribeAudio(tempAudioPath);
      
      const clipsN = videoCount === 'max' ? 10 : Math.min(parseInt(videoCount) || 3, 10);
      const aiClips = await analyzeContext(segments, prompt, duration, clipsN);

      const finalClips = [];
      for (let i = 0; i < aiClips.length; i++) {
        const clip = aiClips[i];
        const rawPath = path.join(OUTPUTS_DIR, `raw_${Date.now()}_${i}.mp4`);
        const finalFilename = `corte_${Date.now()}_${i}.mp4`;
        const finalPath = path.join(OUTPUTS_DIR, finalFilename);

        // 1. Corte e Crop
        await cutVideo(videoPath, rawPath, clip.start, clip.end, videoSpeed, aspectRatio);
        
        // 2. Remoção de Silêncio (o "respiro")
        await removeSilences(rawPath, finalPath, 500);
        
        if (fs.existsSync(rawPath)) fs.unlinkSync(rawPath);

        finalClips.push({
          ...clip,
          url: `${baseUrl}/videos/${finalFilename}`,
          duration: `${Math.floor((clip.end - clip.start)/60)}:${Math.round((clip.end - clip.start)%60).toString().padStart(2, '0')}`
        });
      }

      if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);
      res.json({ success: true, clips: finalClips });
    } catch (error) {
      console.error(error);
      if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);
      res.status(500).json({ error: error.message });
    }
  };

  processingQueue.push({ task, res });
  processNextInQueue();
});

// Limpeza automática (30 min)
setInterval(() => {
  const now = Date.now();
  [UPLOADS_DIR, OUTPUTS_DIR].forEach(dir => {
    fs.readdir(dir, (err, files) => {
      if (err) return;
      files.forEach(file => {
        const p = path.join(dir, file);
        fs.stat(p, (err, stats) => {
          if (!err && (now - stats.mtimeMs > 30 * 60 * 1000)) fs.unlink(p, () => {});
        });
      });
    });
  });
}, 10 * 60 * 1000);

app.listen(port, () => console.log(`Servidor rodando na porta ${port}`));