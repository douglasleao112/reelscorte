import express from "express";
import { createServer as createViteServer } from "vite";
import OpenAI from "openai";
import dotenv from "dotenv";
import multer from "multer";
import fs from "fs";
import path from "path";
import ffmpeg from "fluent-ffmpeg";
import { exec } from "child_process";
import util from "util";

dotenv.config();

const execPromise = util.promisify(exec);

// Configuração de pastas
const __dirname = path.resolve();
const UPLOADS_DIR = path.join(__dirname, "uploads");
const OUTPUTS_DIR = path.join(__dirname, "outputs");

if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(OUTPUTS_DIR)) fs.mkdirSync(OUTPUTS_DIR, { recursive: true });

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

// --- FUNÇÕES AUXILIARES ---

const extractAudio = (videoPath: string, audioPath: string) => {
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .output(audioPath)
      .noVideo()
      .audioCodec("libmp3lame")
      .audioBitrate("64k")
      .on("end", () => resolve(audioPath))
      .on("error", (err) => reject(err))
      .run();
  });
};

const transcribeAudio = async (openai: OpenAI, audioPath: string) => {
  console.log("Iniciando transcrição com Whisper...");
  const transcription = await openai.audio.transcriptions.create({
    file: fs.createReadStream(audioPath),
    model: "whisper-1",
    response_format: "verbose_json",
    timestamp_granularities: ["segment"],
  });

  const segments = (transcription as any).segments.map((s: any) => ({
    start: s.start,
    end: s.end,
    text: s.text.trim(),
  }));

  return segments;
};

const analyzeContext = async (openai: OpenAI, segments: any[], prompt: string, duration: string, clipCount: number) => {
  console.log("Analisando contexto com GPT-4...");

  const systemPrompt = `Você é um editor de vídeo viral especialista em retenção.
Sua tarefa é analisar a transcrição de um vídeo e identificar os melhores trechos contínuos que formam cortes perfeitos para Reels/TikTok.

DIRETRIZES:
1. O corte DEVE ter sentido completo (início, meio e fim). Não corte no meio de uma frase.
2. Duração alvo: aproximadamente ${duration}.
3. Instruções específicas do usuário: ${prompt || "Nenhuma"}.
4. Priorize trechos com alta emoção, dicas valiosas, histórias curtas ou ganchos fortes.
5. Retorne EXATAMENTE ${clipCount} cortes.

FORMATO DE RESPOSTA (JSON estrito):
{
  "clips": [
    {
      "id": "1",
      "title": "Título chamativo (max 5 palavras)",
      "description": "Por que este corte é bom",
      "start": 12.5,
      "end": 45.2,
      "score": 95
    }
  ]
}`;

  const userMessage = `Aqui está a transcrição do vídeo com timestamps (em segundos):\n${JSON.stringify(segments, null, 2)}`;

  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userMessage },
    ],
    response_format: { type: "json_object" },
    temperature: 0.7,
  });

  const content = response.choices[0].message.content;
  if (!content) throw new Error("Sem resposta do OpenAI");
  return JSON.parse(content).clips;
};

const getCropFilter = (aspectRatio: string) => {
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

const cutVideo = (inputPath: string, outputPath: string, start: number, end: number, speed: number = 1.0, aspectRatio: string | null = null) => {
  return new Promise((resolve, reject) => {
    const dur = Math.max(end - start, 0.01);
    const s = Number.isFinite(speed) && speed > 0 ? speed : 1.0;
    const speedFilter = `setpts=${(1 / s).toFixed(6)}*PTS`;
    const audioFilter = `atempo=${s}`;

    const vFilters = [speedFilter];
    const cropFilter = getCropFilter(aspectRatio || '');
    if (cropFilter) {
      vFilters.push(cropFilter);
    }

    ffmpeg(inputPath)
      .setStartTime(start)
      .setDuration(dur)
      .output(outputPath)
      .videoCodec("libx264")
      .audioCodec("aac")
      .videoFilters(vFilters)
      .audioFilters(audioFilter)
      .outputOptions(["-preset ultrafast", "-crf 28"])
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(err))
      .run();
  });
};

const detectSpeechRanges = async (inputPath: string, noiseDb = -35, minSilence = 0.3) => {
  const cmd = `ffmpeg -i "${inputPath}" -af silencedetect=noise=${noiseDb}dB:d=${minSilence} -f null -`;
  try {
    const { stderr } = await execPromise(cmd);

    const silenceStarts: number[] = [];
    const silenceEnds: number[] = [];

    for (const line of stderr.split('\n')) {
      const s = line.match(/silence_start:\s*([0-9.]+)/);
      if (s) silenceStarts.push(parseFloat(s[1]));

      const e = line.match(/silence_end:\s*([0-9.]+)/);
      if (e) silenceEnds.push(parseFloat(e[1]));
    }

    const probeCmd = `ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "${inputPath}"`;
    const { stdout } = await execPromise(probeCmd);
    const totalDuration = Math.max(parseFloat(stdout.trim()) || 0, 0);

    if (totalDuration <= 0) return null;

    if (silenceStarts.length === 0 && silenceEnds.length === 0) {
      return [[0, totalDuration]];
    }

    const ranges: [number, number][] = [];
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

    const minChunk = 0.1;
    const filtered = ranges.filter(([a, b]) => (b - a) >= minChunk);

    return filtered.length ? filtered : null;
  } catch (err) {
    console.error("Erro ao detectar fala:", err);
    return null;
  }
};

const removeSilences = async (inputPath: string, outputPath: string, tailMs = 650, headMs = 200) => {
  console.log(`Detectando falas em: ${inputPath}`);
  const ranges = await detectSpeechRanges(inputPath);

  if (!ranges || ranges.length === 0) {
    console.log("Nenhuma fala detectada. Copiando original.");
    fs.copyFileSync(inputPath, outputPath);
    return outputPath;
  }

  console.log(`Falas detectadas: ${ranges.length} trechos.`);

  const probeCmd = `ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "${inputPath}"`;
  const { stdout } = await execPromise(probeCmd);
  const totalDuration = Math.max(parseFloat(stdout.trim()) || 0, 0);

  const tailSec = tailMs / 1000;
  const headSec = headMs / 1000;

  const extended = ranges.map(([start, end]) => {
    const newStart = Math.max(0, start - headSec); // Corta um pouco antes
    const newEnd = Math.min(end + tailSec, totalDuration); // Respiro depois
    return [newStart, newEnd];
  });

  extended.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const [s, e] of extended) {
    if (!merged.length) {
      merged.push([s, e]);
      continue;
    }
    const last = merged[merged.length - 1];
    if (s <= last[1] + 0.05) { // Margem de segurança para colar trechos próximos
      last[1] = Math.max(last[1], e);
    } else {
      merged.push([s, e]);
    }
  }

  if (merged.length === 1) {
    const [s, e] = merged[0];
    const dur = Math.max(e - s, 0.01);
    const cmd = `ffmpeg -y -i "${inputPath}" -ss ${s.toFixed(3)} -t ${dur.toFixed(3)} -c:v libx264 -preset ultrafast -crf 28 -c:a aac -b:a 128k -movflags +faststart "${outputPath}"`;
    await execPromise(cmd);
    return outputPath;
  }

  const filters: string[] = [];
  let concatInputs = '';

  merged.forEach(([start, end], i) => {
    const dur = Math.max(end - start, 0.01);
    filters.push(`[0:v]trim=start=${start.toFixed(3)}:duration=${dur.toFixed(3)},setpts=PTS-STARTPTS[v${i}]`);
    filters.push(`[0:a]atrim=start=${start.toFixed(3)}:duration=${dur.toFixed(3)},asetpts=PTS-STARTPTS[a${i}]`);
    concatInputs += `[v${i}][a${i}]`;
  });

  filters.push(`${concatInputs}concat=n=${merged.length}:v=1:a=1[outv][outa]`);
  const filterComplex = filters.join(';');

  const cmd = `ffmpeg -y -i "${inputPath}" -filter_complex "${filterComplex}" -map "[outv]" -map "[outa]" -c:v libx264 -preset ultrafast -crf 28 -c:a aac -b:a 128k -movflags +faststart "${outputPath}"`;
  await execPromise(cmd);
  return outputPath;
};

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // OpenAI Proxy Route
  app.post("/api/chat", async (req, res) => {
    const { messages } = req.body;
    
    const apiKey = process.env.OPENAI_API_KEY;
    
    if (!apiKey) {
      return res.status(500).json({ error: "OPENAI_API_KEY not configured on server." });
    }

    try {
      const openai = new OpenAI({ apiKey });
      const response = await openai.chat.completions.create({
        model: "gpt-4o", // or gpt-4-turbo, etc.
        messages: messages,
      });

      res.json({ text: response.choices[0].message.content });
    } catch (error: any) {
      console.error("OpenAI Error:", error);
      res.status(500).json({ error: error.message });
    }
  });

  // Generate Reels Route
  app.post("/api/generate-reels", async (req, res) => {
    try {
      const { duration, prompt, theme, interactiveSubtitles } = req.body;

      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) {
        return res.status(500).json({ error: "OPENAI_API_KEY not configured on server." });
      }

      const openai = new OpenAI({ apiKey });

      const systemPrompt = `Você é um especialista em edição de vídeos curtos (Reels/TikTok) e curadoria de conteúdo viral.
O usuário enviou um vídeo (simulado) com as seguintes preferências:
- Duração desejada por corte: ${duration}
- Tema: ${theme || 'Geral'}
- Instruções adicionais (prompt): ${prompt || 'Nenhuma'}
- Legendas dinâmicas: ${interactiveSubtitles === 'true' ? 'Sim' : 'Não'}

Gere 3 sugestões de cortes virais baseadas no tema e nas instruções do usuário.
Retorne APENAS um JSON válido no seguinte formato exato:
{
  "clips": [
    {
      "id": "1",
      "title": "Título chamativo do corte",
      "duration": "0:30",
      "score": 95,
      "description": "Descrição do que acontece no corte"
    }
  ]
}`;

      const response = await openai.chat.completions.create({
        model: "gpt-4o-mini", // Use mini for faster/cheaper JSON generation
        messages: [{ role: "system", content: systemPrompt }],
        response_format: { type: "json_object" },
      });

      const content = response.choices[0].message.content;
      if (!content) throw new Error("Sem resposta do OpenAI");

      const parsed = JSON.parse(content);
      res.json(parsed);
    } catch (error: any) {
      console.error("Erro na API do OpenAI:", error);
      res.status(500).json({ error: error.message || "Erro ao processar o vídeo" });
    }
  });

  // Process Video Route (Actual logic)
  app.post("/api/process-video", upload.single("video"), async (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: "Nenhum vídeo enviado." });
    }

    const videoPath = req.file.path;
    const { duration = "30s", prompt = "", videoCount = "3", videoSpeed = "1", aspectRatio = "9:16" } = req.body;
    const baseUrl = `${req.protocol}://${req.get("host")}`;
    const tempAudioPath = path.join(UPLOADS_DIR, `temp_audio_${Date.now()}.mp3`);

    try {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error("OPENAI_API_KEY não configurada no servidor.");
      }

      console.log(`Processando vídeo: ${req.file.filename}`);

      const openai = new OpenAI({ apiKey });

      // PASSO 1: Extrair áudio e Transcrever com Whisper
      console.log("Passo 1: Extraindo áudio...");
      await extractAudio(videoPath, tempAudioPath);

      console.log("Passo 1: Transcrevendo...");
      const segments = await transcribeAudio(openai, tempAudioPath);

      if (segments.length === 0) {
        throw new Error("Não foi possível detectar fala no vídeo.");
      }

      // PASSO 2: Analisar com GPT-4
      console.log("Passo 2: Analisando com IA...");
      const clipsN = videoCount === 'max' ? 10 : Math.min(Math.max(parseInt(videoCount, 10) || 3, 1), 15);
      const aiClips = await analyzeContext(openai, segments, prompt, duration, clipsN);

      // PASSO 3: Cortar o vídeo com FFmpeg
      console.log("Passo 3: Gerando cortes reais...");
      const finalClips = [];
      const safeSpeed = parseFloat(videoSpeed) || 1.0;

      for (let i = 0; i < aiClips.length; i++) {
        const clip = aiClips[i];
        const rawFilename = `corte_raw_${Date.now()}_${i}.mp4`;
        const rawPath = path.join(OUTPUTS_DIR, rawFilename);

        console.log(`Cortando trecho bruto ${i + 1}: ${clip.start}s até ${clip.end}s`);
        await cutVideo(videoPath, rawPath, clip.start, clip.end, safeSpeed, aspectRatio);

        const finalFilename = `corte_${Date.now()}_${i}.mp4`;
        const finalPath = path.join(OUTPUTS_DIR, finalFilename);

        console.log(`Removendo silêncios do corte ${i + 1} (head 200ms, tail 650ms)...`);
        await removeSilences(rawPath, finalPath, 650, 200);

        if (fs.existsSync(rawPath)) fs.unlinkSync(rawPath);

        const durationSecs = Math.round(clip.end - clip.start);
        const mins = Math.floor(durationSecs / 60);
        const secs = durationSecs % 60;
        const formattedDuration = `${mins}:${secs.toString().padStart(2, "0")}`;

        finalClips.push({
          id: clip.id || String(i + 1),
          title: clip.title,
          description: clip.description,
          start: clip.start,
          end: clip.end,
          duration: formattedDuration,
          score: clip.score,
          url: `/videos/${finalFilename}`,
        });
      }

      console.log("Processamento concluído com sucesso!");
      if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);

      res.json({ success: true, clips: finalClips });
    } catch (error: any) {
      console.error("Erro no processamento:", error);
      if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);
      res.status(500).json({ error: "Erro ao processar vídeo", details: error.message });
    }
  });

  // Servir vídeos estáticos
  app.use("/videos", express.static(OUTPUTS_DIR));

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static("dist"));
    app.get("*", (req, res) => {
      res.sendFile("index.html", { root: "dist" });
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });

  // Limpeza automática a cada 10 minutos (arquivos mais velhos que 30 min)
  setInterval(() => {
    const now = Date.now();
    const maxAge = 30 * 60 * 1000; // 30 minutos

    [UPLOADS_DIR, OUTPUTS_DIR].forEach((dir) => {
      fs.readdir(dir, (err, files) => {
        if (err) return;
        files.forEach((file) => {
          const filePath = path.join(dir, file);
          fs.stat(filePath, (err, stats) => {
            if (err) return;
            if (now - stats.mtimeMs > maxAge) {
              fs.unlink(filePath, () => {});
            }
          });
        });
      });
    });
  }, 10 * 60 * 1000);
}

startServer();
