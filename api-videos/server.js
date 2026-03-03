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
const port = process.env.PORT || 80; // EasyPanel geralmente usa porta 80 ou mapeia a 3000

// Configuração de CORS para permitir que seu frontend acesse o backend
app.use(cors());
app.use(express.json());

// OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Pastas de trabalho
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const OUTPUTS_DIR = path.join(__dirname, 'outputs');

if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(OUTPUTS_DIR)) fs.mkdirSync(OUTPUTS_DIR, { recursive: true });

// Servir os vídeos processados
app.use('/videos', express.static(OUTPUTS_DIR));

// --- FUNÇÕES DE PROCESSAMENTO ---

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

const transcribeAudio = async (audioPath) => {
  const transcription = await openai.audio.transcriptions.create({
    file: fs.createReadStream(audioPath),
    model: 'whisper-1',
    response_format: 'verbose_json',
    timestamp_granularities: ['segment'],
  });
  return transcription.segments.map(s => ({
    start: s.start,
    end: s.end,
    text: s.text.trim(),
  }));
};

const analyzeContext = async (segments, prompt, duration, clipCount) => {
  const systemPrompt = `Você é um editor de vídeo viral. Analise a transcrição e escolha ${clipCount} cortes de aprox. ${duration}. 
  Retorne APENAS JSON: { "clips": [ { "title": "...", "start": 0, "end": 10, "score": 90, "description": "..." } ] }`;
  
  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: JSON.stringify(segments) }
    ],
    response_format: { type: 'json_object' }
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
    const dur = Math.max(end - start, 0.01);
    const vFilters = [`setpts=${(1/speed)}*PTS`];
    const crop = getCropFilter(aspectRatio);
    if (crop) vFilters.push(crop);

    ffmpeg(inputPath)
      .setStartTime(start)
      .setDuration(dur)
      .output(outputPath)
      .videoFilters(vFilters)
      .audioFilters(`atempo=${speed}`)
      .outputOptions(['-preset ultrafast', '-crf 28'])
      .on('end', () => resolve(outputPath))
      .on('error', reject)
      .run();
  });
};

const removeSilences = async (inputPath, outputPath, tailMs = 650, headMs = 200) => {
  const noiseDb = -35;
  const minSilence = 0.3;
  const cmd = `ffmpeg -i "${inputPath}" -af silencedetect=noise=${noiseDb}dB:d=${minSilence} -f null -`;
  const { stderr } = await execPromise(cmd);

  const silenceStarts = (stderr.match(/silence_start: [0-9.]+/g) || []).map(s => parseFloat(s.split(': ')[1]));
  const silenceEnds = (stderr.match(/silence_end: [0-9.]+/g) || []).map(s => parseFloat(s.split(': ')[1]));

  const { stdout: durOut } = await execPromise(`ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "${inputPath}"`);
  const totalDuration = parseFloat(durOut);

  let ranges = [];
  let cursor = 0;
  for (let i = 0; i < silenceStarts.length; i++) {
    if (silenceStarts[i] > cursor) ranges.push([cursor, silenceStarts[i]]);
    cursor = silenceEnds[i] || cursor;
  }
  if (cursor < totalDuration) ranges.push([cursor, totalDuration]);

  const tailSec = tailMs / 1000;
  const headSec = headMs / 1000;
  const merged = [];
  ranges.forEach(([s, e]) => {
    const ns = Math.max(0, s - headSec);
    const ne = Math.min(totalDuration, e + tailSec);
    if (!merged.length || ns > merged[merged.length-1][1] + 0.05) merged.push([ns, ne]);
    else merged[merged.length-1][1] = Math.max(merged[merged.length-1][1], ne);
  });

  const filters = merged.map(([s, e], i) => {
    const d = e - s;
    return `[0:v]trim=start=${s}:duration=${d},setpts=PTS-STARTPTS[v${i}];[0:a]atrim=start=${s}:duration=${d},asetpts=PTS-STARTPTS[a${i}]`;
  }).join(';');
  const concat = merged.map((_, i) => `[v${i}][a${i}]`).join('') + `concat=n=${merged.length}:v=1:a=1[outv][outa]`;

  await execPromise(`ffmpeg -y -i "${inputPath}" -filter_complex "${filters};${concat}" -map "[outv]" -map "[outa]" -c:v libx264 -preset ultrafast -crf 28 "${outputPath}"`);
};

// --- ROTA ---

app.post('/api/process-video', upload.single('video'), async (req, res) => {
  const videoPath = req.file.path;
  const { duration = '30s', videoCount = '3', videoSpeed = '1', aspectRatio = '9:16' } = req.body;
  const tempAudio = path.join(UPLOADS_DIR, `audio_${Date.now()}.mp3`);

  try {
    await extractAudio(videoPath, tempAudio);
    const segments = await transcribeAudio(tempAudio);
    const clips = await analyzeContext(segments, '', duration, parseInt(videoCount) || 3);
    
    const results = [];
    for (let i = 0; i < clips.length; i++) {
      const raw = path.join(OUTPUTS_DIR, `raw_${Date.now()}_${i}.mp4`);
      const final = path.join(OUTPUTS_DIR, `corte_${Date.now()}_${i}.mp4`);
      
      await cutVideo(videoPath, raw, clips[i].start, clips[i].end, parseFloat(videoSpeed), aspectRatio);
      await removeSilences(raw, final, 650, 200);
      fs.unlinkSync(raw);

      results.push({ ...clips[i], url: `/videos/${path.basename(final)}` });
    }
    res.json({ success: true, clips: results });
  } catch (e) {
    res.status(500).json({ error: e.message });
  } finally {
    if (fs.existsSync(tempAudio)) fs.unlinkSync(tempAudio);
  }
});

app.listen(port, () => console.log(`Rodando na porta ${port}`));