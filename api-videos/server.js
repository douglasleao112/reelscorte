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
const port = 80; 

// Configuração do OpenAI
// Certifique-se de configurar a variável de ambiente OPENAI_API_KEY na sua VPS
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Configuração de pastas
const UPLOADS_DIR = path.join(__dirname, "uploads");
const OUTPUTS_DIR = path.join(__dirname, "outputs");

if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(OUTPUTS_DIR)) fs.mkdirSync(OUTPUTS_DIR, { recursive: true });

app.use(cors());
app.use(express.json());
app.use("/videos", express.static(OUTPUTS_DIR));

// Configuração do Multer para receber o vídeo
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  },
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 500 * 1024 * 1024 }, // Limite de 500MB
});

// Fila de processamento simples para evitar overload na VPS (2GB RAM)
let isProcessing = false;
const processingQueue = [];

const processNextInQueue = async () => {
  if (isProcessing || processingQueue.length === 0) return;

  isProcessing = true;
  const task = processingQueue.shift();

  try {
    await task();
  } catch (error) {
    console.error("Erro na tarefa de processamento:", error);
  } finally {
    isProcessing = false;
    processNextInQueue();
  }
};

// --- FUNÇÕES AUXILIARES ---

// Extrai o áudio do vídeo para enviar ao Whisper (arquivos menores)
const extractAudio = (videoPath, audioPath) => {
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .output(audioPath)
      .noVideo()
      .audioCodec("libmp3lame")
      .audioBitrate("64k") // Bitrate baixo para arquivo pequeno
      .on("end", () => resolve(audioPath))
      .on("error", (err) => reject(err))
      .run();
  });
};

// Transcreve o áudio usando OpenAI Whisper
const transcribeAudio = async (audioPath) => {
  console.log("Iniciando transcrição com Whisper...");
  const transcription = await openai.audio.transcriptions.create({
    file: fs.createReadStream(audioPath),
    model: "whisper-1",
    response_format: "verbose_json", // Retorna timestamps
    timestamp_granularities: ["segment"],
  });

  // Formata para o GPT-4
  const segments = transcription.segments.map((s) => ({
    start: s.start,
    end: s.end,
    text: s.text.trim(),
  }));

  return segments;
};

// Analisa a transcrição com GPT-4 para encontrar os melhores cortes
const analyzeContext = async (segments, prompt, theme, duration) => {
  console.log("Analisando contexto com GPT-4...");

  const systemPrompt = `Você é um editor de vídeo viral especialista em retenção.
Sua tarefa é analisar a transcrição de um vídeo e identificar os melhores trechos contínuos que formam cortes perfeitos para Reels/TikTok.

DIRETRIZES:
1. O corte DEVE ter sentido completo (início, meio e fim). Não corte no meio de uma frase.
2. Duração alvo: aproximadamente ${duration}.
3. Tema desejado: ${theme || "Qualquer tema interessante"}.
4. Instruções específicas do usuário: ${prompt || "Nenhuma"}.
5. Priorize trechos com alta emoção, dicas valiosas, histórias curtas ou ganchos fortes.
6. Retorne EXATAMENTE 3 cortes.

FORMATO DE RESPOSTA (JSON estrito):
{
  "clips": [
    {
      "id": "1",
      "title": "Título chamativo (max 5 palavras)",
      "description": "Por que este corte é bom",
      "start": 12.5, // tempo em segundos (número)
      "end": 45.2, // tempo em segundos (número)
      "score": 95 // nota de 0 a 100 de quão viral isso pode ser
    }
  ]
}`;

  const userMessage = `Aqui está a transcrição do vídeo com timestamps (em segundos):\n${JSON.stringify(segments, null, 2)}`;

  const response = await openai.chat.completions.create({
    model: "gpt-4-turbo", // gpt-4o    gpt-4-turbo
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userMessage },
    ],
    response_format: { type: "json_object" },
    temperature: 0.7,
  });

  const content = response.choices[0].message.content;
  return JSON.parse(content).clips;
};

// Corta o vídeo usando FFmpeg com base nos timestamps
const cutVideo = (inputPath, outputPath, start, end) => {
  return new Promise((resolve, reject) => {
    const duration = end - start;
    ffmpeg(inputPath)
      .setStartTime(start)
      .setDuration(duration)
      .output(outputPath)
      .videoCodec("libx264")
      .audioCodec("aac")
      // Otimizações para VPS fraca: preset ultrafast, crf alto (menor qualidade, mais rápido)
      .outputOptions(["-preset ultrafast", "-crf 28"])
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(err))
      .run();
  });
};



// Detecta intervalos com fala dentro de um arquivo (com base nos silêncios)
const detectSpeechRanges = async (inputPath, noiseDb = -35, minSilence = 0.25) => {
  const cmd = `ffmpeg -i "${inputPath}" -af silencedetect=noise=${noiseDb}dB:d=${minSilence} -f null -`;
  const { stderr } = await execPromise(cmd);

  const silenceStarts = [];
  const silenceEnds = [];

  for (const line of stderr.split("\n")) {
    const s = line.match(/silence_start:\s*([0-9.]+)/);
    if (s) silenceStarts.push(parseFloat(s[1]));

    const e = line.match(/silence_end:\s*([0-9.]+)/);
    if (e) silenceEnds.push(parseFloat(e[1]));
  }

  // duração total
  const probeCmd = `ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "${inputPath}"`;
  const { stdout } = await execPromise(probeCmd);
  const totalDuration = Math.max(parseFloat(stdout.trim()) || 0, 0);

  if (totalDuration <= 0) return null;

  // Se não achou silêncio, considera tudo como "fala"
  if (silenceStarts.length === 0 && silenceEnds.length === 0) {
    return [[0, totalDuration]];
  }

  // ranges de fala são os espaços ENTRE silêncios
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

  if (cursor < totalDuration) ranges.push([cursor, totalDuration]);

  // remove micro-trechos
  const minChunk = 0.18;
  const filtered = ranges.filter(([a, b]) => (b - a) >= minChunk);

  return filtered.length ? filtered : null;
};

// Remove silêncios mantendo "pre" e "tail" (sem silêncio artificial)
// preRollMs: pega um pouco ANTES da fala (evita cortar em cima)
// tailMs: mantém um pouco DEPOIS da fala (respiro real)
const removeSilences = async (inputPath, outputPath, tailMs = 650, preRollMs = 120) => {
  const ranges = await detectSpeechRanges(inputPath);

  if (!ranges || ranges.length === 0) {
    fs.copyFileSync(inputPath, outputPath);
    return outputPath;
  }

  // duração total
  const probeCmd = `ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "${inputPath}"`;
  const { stdout } = await execPromise(probeCmd);
  const totalDuration = Math.max(parseFloat(stdout.trim()) || 0, 0);

  const tailSec = tailMs / 1000;
  const preSec = preRollMs / 1000;

  // estende cada range: começa um pouco antes e termina um pouco depois
  const extended = ranges.map(([start, end]) => {
    const newStart = Math.max(0, start - preSec);
    const newEnd = Math.min(end + tailSec, totalDuration);
    return [newStart, Math.max(newStart, newEnd)];
  });

  // mescla overlaps/encostes
  extended.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const [s, e] of extended) {
    if (!merged.length) {
      merged.push([s, e]);
      continue;
    }
    const last = merged[merged.length - 1];
    if (s <= last[1] + 0.02) last[1] = Math.max(last[1], e);
    else merged.push([s, e]);
  }

  // Se só 1 range, recorta simples
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

  // filter_complex (1 input só)
  const filters = [];
  let concatInputs = "";

  merged.forEach(([start, end], i) => {
    const dur = Math.max(end - start, 0.01);

    filters.push(
      `[0:v]trim=start=${start.toFixed(3)}:duration=${dur.toFixed(3)},setpts=PTS-STARTPTS[v${i}]`
    );
    filters.push(
      `[0:a]atrim=start=${start.toFixed(3)}:duration=${dur.toFixed(3)},asetpts=PTS-STARTPTS[a${i}]`
    );

    concatInputs += `[v${i}][a${i}]`;
  });

  filters.push(`${concatInputs}concat=n=${merged.length}:v=1:a=1[outv][outa]`);
  const filterComplex = filters.join(";");

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



// --- ROTA PRINCIPAL DA API ---

app.post("/api/process-video", upload.single("video"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "Nenhum vídeo enviado." });
  }

  const videoPath = req.file.path;
  const { duration = "30s", prompt = "", theme = "" } = req.body;
  const baseUrl = `${req.protocol}://${req.get("host")}`;
  const tempAudioPath = path.join(UPLOADS_DIR, `temp_audio_${Date.now()}.mp3`);

  try {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY não configurada no servidor.");
    }

    console.log(`Processando vídeo: ${req.file.filename}`);

    // PASSO 1: Extrair áudio e Transcrever com Whisper
    console.log("Passo 1: Extraindo áudio...");
    await extractAudio(videoPath, tempAudioPath);

    console.log("Passo 1: Transcrevendo...");
    const segments = await transcribeAudio(tempAudioPath);

    if (segments.length === 0) {
      throw new Error("Não foi possível detectar fala no vídeo.");
    }

    // PASSO 2: Analisar com GPT-4
    console.log("Passo 2: Analisando com IA...");
    const aiClips = await analyzeContext(segments, prompt, theme, duration);

    // PASSO 3: Cortar o vídeo com FFmpeg
    console.log("Passo 3: Gerando cortes reais...");
    const finalClips = [];

    for (let i = 0; i < aiClips.length; i++) {
      const clip = aiClips[i];
     // 1) corta bruto
const rawFilename = `corte_raw_${Date.now()}_${i}.mp4`;
const rawPath = path.join(OUTPUTS_DIR, rawFilename);

console.log(`Cortando bruto ${i + 1}: ${clip.start}s até ${clip.end}s`);
await cutVideo(videoPath, rawPath, clip.start, clip.end);

// 2) remove silêncios com respiro real (650ms)
const outputFilename = `corte_${Date.now()}_${i}.mp4`;
const outputPath = path.join(OUTPUTS_DIR, outputFilename);

console.log(`Removendo silêncios do corte ${i + 1} (tail 650ms)...`);
await removeSilences(rawPath, outputPath, 650, 120);    // respiro depois da fala,  pega antes da fala começar

// 3) apaga bruto pra economizar espaço
if (fs.existsSync(rawPath)) fs.unlinkSync(rawPath);

      // Calcula a duração real em formato mm:ss
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
        url: `${baseUrl}/videos/${outputFilename}`,
      });
    }

    console.log("Processamento concluído com sucesso!");

    // Limpa o arquivo de áudio temporário
    if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);

    // Envia a resposta final
    res.json({ success: true, clips: finalClips });
  } catch (error) {
    console.error("Erro no processamento:", error);
    // Limpa o arquivo de áudio temporário em caso de erro
    if (fs.existsSync(tempAudioPath)) fs.unlinkSync(tempAudioPath);
    res
      .status(500)
      .json({ error: "Erro ao processar vídeo", details: error.message });
  }
});

// Limpeza automática a cada 10 minutos (arquivos mais velhos que 30 min)
setInterval(
  () => {
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
                    `Arquivo deletado por limpeza automática: ${filePath}`,
                  );
              });
            }
          });
        });
      });
    });
  },
  10 * 60 * 1000,
);

app.listen(port, () => {
  console.log(
    `Servidor de processamento de vídeo com IA rodando na porta ${port}`,
  );
});
