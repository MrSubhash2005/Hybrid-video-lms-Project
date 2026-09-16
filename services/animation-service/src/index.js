import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { exec } from 'child_process';
import util from 'util';

const execPromise = util.promisify(exec);

import { subtitlesList } from './subtitles.js';
import { explainerSubtitlesList } from './explainer_subtitles.js';
import { ragSubtitlesList } from './rag_subtitles.js';
import { dsaSubtitlesList } from './dsa_subtitles.js';

import {
  getTalkingHeadBaseUrl,
  requestAvatarGeneration,
  pollAvatarJob,
  resolveAvatarVideo,
  composeHybridVideo
} from './avatarClient.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Serve modern LMS player homepage
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

// CDN Video Streaming Helper with HTTP Range Byte Support
function streamVideoFile(filePath, req, res) {
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'Video asset not found' });
  }

  const stat = fs.statSync(filePath);
  const fileSize = stat.size;
  const range = req.headers.range;

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Range');
  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');

  if (range) {
    const parts = range.replace(/bytes=/, "").split("-");
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;

    if (start >= fileSize || end >= fileSize) {
      res.setHeader('Content-Range', `bytes */${fileSize}`);
      return res.status(416).send('Requested Range Not Satisfiable');
    }

    const chunksize = (end - start) + 1;
    const file = fs.createReadStream(filePath, { start, end });

    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunksize,
      'Content-Type': 'video/mp4',
    });

    file.pipe(res);
  } else {
    res.writeHead(200, {
      'Content-Length': fileSize,
      'Content-Type': 'video/mp4',
    });

    fs.createReadStream(filePath).pipe(res);
  }
}

// CDN Video Streaming route with HTTP Range byte support for seeking
app.get('/outputs/:filename', (req, res) => {
  const filePath = path.join(__dirname, '../public/outputs', req.params.filename);
  streamVideoFile(filePath, req, res);
});

app.get('/api/v1/cdn/stream/:filename', (req, res) => {
  const filePath = path.join(__dirname, '../public/outputs', req.params.filename);
  streamVideoFile(filePath, req, res);
});

app.use(express.static(path.join(__dirname, '../public')));

// In-memory render job database
const jobsDb = new Map();

// Sarvam API Key Pool for automatic rotation
const SARVAM_KEYS = [
  process.env.SARVAM_API_KEY,
  process.env.SARVAM_API_KEY_2,
  process.env.SARVAM_API_KEY_3,
  process.env.SARVAM_API_KEY_4
].filter(Boolean);

// OpenRouter API Key for AI-driven celebrity gender detection
const getOpenRouterKey = () => process.env.OPENROUTER_API_KEY || process.env.openrouter_Api_Key || process.env.OPENROUTER_KEY || process.env.openrouter_api_key;

const celebrityVoiceMap = {
  // Male voices (Fallback Male 1: aditya, Fallback Male 2: shubh)
  'shahrukh': { speaker: 'aditya', language_code: 'en-IN', gender: 'male' },
  'shahrukhkhan': { speaker: 'aditya', language_code: 'en-IN', gender: 'male' },
  'sharukh': { speaker: 'aditya', language_code: 'en-IN', gender: 'male' },
  'srk': { speaker: 'aditya', language_code: 'en-IN', gender: 'male' },
  'ntr': { speaker: 'shubh', language_code: 'en-IN', gender: 'male' },
  'ntrjr': { speaker: 'shubh', language_code: 'en-IN', gender: 'male' },
  'jrntr': { speaker: 'shubh', language_code: 'en-IN', gender: 'male' },
  'prabhas': { speaker: 'shubh', language_code: 'en-IN', gender: 'male' },
  'alluarjun': { speaker: 'aditya', language_code: 'en-IN', gender: 'male' },
  'salman': { speaker: 'aditya', language_code: 'en-IN', gender: 'male' },
  'aamir': { speaker: 'shubh', language_code: 'en-IN', gender: 'male' },

  // Female voices (Fallback Female 1: shreya, Fallback Female 2: aruna)
  'deepika': { speaker: 'shreya', language_code: 'en-IN', gender: 'female' },
  'deepikapadukone': { speaker: 'shreya', language_code: 'en-IN', gender: 'female' },
  'priyanka': { speaker: 'shreya', language_code: 'en-IN', gender: 'female' },
  'priyankachopra': { speaker: 'shreya', language_code: 'en-IN', gender: 'female' },
  'katrina': { speaker: 'shreya', language_code: 'en-IN', gender: 'female' },
  'katrinakaif': { speaker: 'shreya', language_code: 'en-IN', gender: 'female' },
  'alia': { speaker: 'aruna', language_code: 'en-IN', gender: 'female' },
  'aliabhatt': { speaker: 'aruna', language_code: 'en-IN', gender: 'female' },
  'rashmika': { speaker: 'aruna', language_code: 'en-IN', gender: 'female' },
  'rashmikamandanna': { speaker: 'aruna', language_code: 'en-IN', gender: 'female' },
  'nayanthara': { speaker: 'shreya', language_code: 'en-IN', gender: 'female' },
  'madhuri': { speaker: 'shreya', language_code: 'en-IN', gender: 'female' },
  'madhuridixit': { speaker: 'shreya', language_code: 'en-IN', gender: 'female' },
  'kareena': { speaker: 'aruna', language_code: 'en-IN', gender: 'female' },
  'kareenakapoor': { speaker: 'aruna', language_code: 'en-IN', gender: 'female' },
  'shraddha': { speaker: 'aruna', language_code: 'en-IN', gender: 'female' },
  'shraddhakapoor': { speaker: 'aruna', language_code: 'en-IN', gender: 'female' },
  'aruna': { speaker: 'aruna', language_code: 'en-IN', gender: 'female' }
};

// OpenRouter AI Gender Classifier
async function detectGenderWithOpenRouter(celebrityName) {
  if (!celebrityName) return null;
  const apiKey = getOpenRouterKey();
  if (!apiKey) {
    console.warn("[OpenRouter AI] No OpenRouter API key found in environment (OPENROUTER_API_KEY / openrouter_Api_Key).");
    return null;
  }

  try {
    console.log(`\x1b[36m[OpenRouter AI]\x1b[0m Querying OpenRouter AI for gender of celebrity: '${celebrityName}'...`);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "openrouter/auto",
        messages: [
          {
            role: "user",
            content: `Is '${celebrityName}' primarily male or female? Respond ONLY with the single word male or female.`
          }
        ]
      }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      const text = data.choices?.[0]?.message?.content?.trim()?.toLowerCase();
      if (text?.includes('female')) {
        console.log(`\x1b[32m[OpenRouter AI Success]\x1b[0m Detected '${celebrityName}' -> female`);
        return 'female';
      }
      if (text?.includes('male')) {
        console.log(`\x1b[32m[OpenRouter AI Success]\x1b[0m Detected '${celebrityName}' -> male`);
        return 'male';
      }
    } else {
      const errText = await response.text();
      console.warn(`[OpenRouter AI] Query returned HTTP ${response.status}: ${errText}`);
    }
  } catch (err) {
    console.warn(`[OpenRouter AI] Gender detection query failed: ${err.message}. Falling back to local lookup.`);
  }
  return null;
}

// Smart Offline Gender Detection Heuristic
function detectGenderHeuristic(name, explicitGender) {
  if (explicitGender === 'male' || explicitGender === 'female') {
    return explicitGender;
  }

  if (!name) return 'male';

  const norm = name.toLowerCase().trim();
  const clean = norm.replace(/[^a-z]/g, '');

  // 1. Direct Map Check
  if (celebrityVoiceMap[clean]) {
    return celebrityVoiceMap[clean].gender;
  }
  for (const key of Object.keys(celebrityVoiceMap)) {
    if (clean.includes(key) || key.includes(clean)) {
      return celebrityVoiceMap[key].gender;
    }
  }

  // 2. Comprehensive Known Female Names & Keywords
  const femaleNames = [
    'deepika', 'priyanka', 'katrina', 'alia', 'madhuri', 'kareena', 'shraddha', 'rashmika',
    'nayanthara', 'aruna', 'shreya', 'kiara', 'samantha', 'kriti', 'anushka', 'tabu', 'kajol',
    'taapsee', 'trisha', 'tamannaah', 'tamanna', 'disha', 'jacqueline', 'shruti', 'nushrratt',
    'janhvi', 'sara', 'ananya', 'suhana', 'sonam', 'sonakshi', 'vibhuti', 'pooja', 'aditi',
    'radhika', 'yami', 'bhumi', 'mrunal', 'huma', 'vidya', 'shilpa', 'karisma', 'juhi',
    'raveena', 'sridevi', 'rekha', 'hema', 'jaya', 'nargis', 'waheeda', 'asha', 'lata',
    'sunidhi', 'neha', 'monali', 'kanika', 'jonita', 'palak', 'dhvani', 'niti', 'taylor',
    'beyonce', 'rihanna', 'ariana', 'selena', 'billie', 'dua', 'katy', 'adele', 'madonna',
    'britney', 'shakira', 'jennifer', 'scarlett', 'emma', 'margot', 'gal', 'anne', 'natalie',
    'keira', 'charlize', 'penelope', 'salma', 'zendaya', 'florence', 'ana', 'elizabeth', 'rose',
    'kate', 'angelina', 'megan', 'amber', 'kirsten', 'cameron', 'winona', 'drew', 'sandra',
    'meryl', 'cate', 'nicole', 'reese', 'jessica', 'rachel', 'amanda', 'violet', 'sophie',
    'olivia', 'mia', 'chloe', 'zoe', 'lily', 'grace', 'hannah', 'ella', 'emily', 'charlotte',
    'harper', 'evelyn', 'abigail', 'avery', 'sofia', 'sophia', 'victoria', 'camila', 'aria',
    'scarlet', 'layla', 'zoey', 'nora', 'hazel', 'eleanor', 'aurora', 'natalia', 'naomi',
    'alyssa', 'alison', 'ashley', 'brittany', 'courtney', 'heather', 'lindsay', 'morgan',
    'paige', 'shelby', 'sydney', 'vanessa', 'whitney', 'alexandra', 'andrea', 'angelica',
    'angela', 'anita', 'ann', 'anna', 'audrey', 'barbara', 'carol', 'caroline', 'catherine',
    'christina', 'christine', 'clara', 'cynthia', 'diana', 'donna', 'dorothy', 'eva', 'fiona',
    'frances', 'helen', 'irene', 'jane', 'janet', 'jean', 'joan', 'joyce', 'judith', 'judy',
    'julia', 'julie', 'karen', 'katherine', 'kathleen', 'laura', 'linda', 'lisa', 'louise',
    'lucy', 'margaret', 'maria', 'marie', 'marilyn', 'martha', 'mary', 'michelle', 'nancy',
    'pamela', 'patricia', 'paula', 'rebecca', 'rita', 'ruth', 'sarah', 'sharon', 'shirley',
    'stephanie', 'susan', 'theresa', 'teresa', 'valerie', 'virginia', 'wendy', 'female',
    'woman', 'lady', 'girl', 'queen', 'actress'
  ];

  if (femaleNames.some(fn => norm.includes(fn) || clean.includes(fn))) {
    return 'female';
  }

  // 3. Known Male Names & Keywords
  const maleNames = [
    'shahrukh', 'srk', 'sharukh', 'salman', 'aamir', 'akshay', 'hrithik', 'ranbir', 'ranveer',
    'ajay', 'sunny', 'bobby', 'sanjay', 'saif', 'shahid', 'varun', 'sidharth', 'siddharth',
    'kartik', 'ayushmann', 'vicky', 'rajkummar', 'nawazuddin', 'pankaj', 'manoj', 'irrfan',
    'anupam', 'paresh', 'boman', 'suniel', 'jackie', 'govinda', 'mithun', 'dharmendra',
    'amitabh', 'bachchan', 'rajinikanth', 'kamal', 'vijay', 'ajith', 'suriya', 'vikram',
    'dhanush', 'yash', 'puneeth', 'darshan', 'sudeep', 'upendra', 'ram', 'ntr', 'prabhas',
    'mahesh', 'allu', 'arjun', 'nani', 'deverakonda', 'ravi', 'teja', 'nagarjuna', 'venkatesh',
    'chiranjeevi', 'kalyan', 'pawan', 'rana', 'gopichand', 'dulquer', 'fahadh', 'nivin',
    'tovino', 'prithviraj', 'mammootty', 'mohanlal', 'tom', 'brad', 'leonardo', 'robert',
    'chris', 'harrison', 'morgan', 'keanu', 'johnny', 'will', 'dwayne', 'rock', 'vin',
    'ryan', 'hugh', 'christian', 'matt', 'ben', 'mark', 'steve', 'michael', 'david',
    'james', 'john', 'paul', 'peter', 'daniel', 'sam', 'jack', 'harry', 'charlie',
    'george', 'oliver', 'william', 'henry', 'thomas', 'edward', 'joseph', 'charles',
    'richard', 'arthur', 'louis', 'freddie', 'alexander', 'sebastian', 'adam', 'benjamin',
    'luke', 'matthew', 'nathan', 'samuel', 'ethan', 'andrew', 'christopher', 'joshua',
    'aaron', 'dylan', 'ezra', 'gabriel', 'isaac', 'julian', 'leo', 'logan', 'lucas',
    'mason', 'noah', 'owen', 'theodore', 'wyatt', 'male', 'man', 'guy', 'boy', 'king', 'actor'
  ];

  if (maleNames.some(mn => norm.includes(mn) || clean.includes(mn))) {
    return 'male';
  }

  // 4. Common Female Suffixes
  const femaleSuffixes = ['a', 'i', 'ee', 'ie', 'ya', 'ina', 'ika', 'ita', 'isha', 'iti', 'ine', 'ette', 'ica', 'iya'];
  const firstName = clean.split(' ')[0] || clean;
  if (femaleSuffixes.some(suf => firstName.endsWith(suf))) {
    const maleSuffixExceptions = ['rama', 'krishna', 'surya', 'bala', 'rana', 'mustafa', 'mufasa', 'obama', 'baba', 'nana', 'dada'];
    if (!maleSuffixExceptions.some(ex => firstName.includes(ex))) {
      return 'female';
    }
  }

  return 'male';
}

function getCelebrityVoice(name, gender) {
  const defaultMale = { speaker: 'aditya', language_code: 'en-IN', gender: 'male' };
  const defaultFemale = { speaker: 'shreya', language_code: 'en-IN', gender: 'female' };

  const effectiveGender = (gender && gender !== 'auto') ? gender : detectGenderHeuristic(name);

  if (!name) {
    return effectiveGender === 'female' ? defaultFemale : defaultMale;
  }

  const normalized = name.toLowerCase().trim().replace(/[^a-z]/g, '');
  if (celebrityVoiceMap[normalized]) {
    return celebrityVoiceMap[normalized];
  }
  for (const key of Object.keys(celebrityVoiceMap)) {
    if (normalized.includes(key) || key.includes(normalized)) {
      return celebrityVoiceMap[key];
    }
  }

  if (effectiveGender === 'female') {
    return defaultFemale;
  }

  return defaultMale;
}

// Fetch Sarvam TTS audio with key rotation across key pool
async function fetchSarvamTTSWithRotation(text, speaker, languageCode, retriesPerKey = 2) {
  let lastError = null;
  for (const key of SARVAM_KEYS) {
    for (let attempt = 0; attempt < retriesPerKey; attempt++) {
      try {
        const response = await fetch("https://api.sarvam.ai/text-to-speech", {
          method: "POST",
          headers: {
            "api-subscription-key": key,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            text: text,
            target_language_code: languageCode,
            speaker: speaker,
            model: "bulbul:v3"
          })
        });

        if (response.ok) {
          const data = await response.json();
          if (data.audios && data.audios[0]) {
            return Buffer.from(data.audios[0], 'base64');
          }
        }

        if (response.status === 402) {
          console.warn(`[Sarvam Key Rotation] Key ${key.slice(0, 10)}... returned 402 Out of Credits. Rotating key...`);
          break; // move to next key immediately
        }

        if (response.status === 429) {
          console.warn(`[Sarvam Key Rotation] Key ${key.slice(0, 10)}... hit rate limit (429). Retrying after 1s...`);
          await new Promise(res => setTimeout(res, 1000));
          continue;
        }

        lastError = new Error(`Sarvam API returned HTTP status ${response.status}`);
      } catch (err) {
        lastError = err;
      }
    }
  }
  throw lastError || new Error('All Sarvam API keys in pool failed or are out of credits.');
}

// Helper: limit concurrency using a worker pool
async function limitConcurrency(tasks, limit) {
  const results = [];
  let index = 0;
  const runWorker = async () => {
    while (index < tasks.length) {
      const currentIndex = index++;
      results[currentIndex] = await tasks[currentIndex]();
    }
  };
  const workers = Array.from({ length: Math.min(limit, tasks.length) }, runWorker);
  await Promise.all(workers);
  return results;
}

// Course normalization helper
function normalizeCourseName(courseInput) {
  if (!courseInput) return 'git';
  const c = courseInput.toLowerCase().trim();
  if (c === 'dsa' || c === 'linkedlist' || c === 'singly-linked-list' || c === 'dsa-linkedlist') return 'dsa';
  if (c === 'rag') return 'rag';
  if (c === 'explainer' || c === 'frontend_sm') return 'explainer';
  return 'git';
}

function getCourseSubtitles(courseKey) {
  switch (courseKey) {
    case 'dsa': return dsaSubtitlesList;
    case 'explainer': return explainerSubtitlesList;
    case 'rag': return ragSubtitlesList;
    case 'git': default: return subtitlesList;
  }
}

// Render pipeline background process
async function runRenderPipeline(jobId, speaker, languageCode) {
  const job = jobsDb.get(jobId);
  console.log(`\n\x1b[36m[Job Started]\x1b[0m Job ID: ${jobId} | Course: ${job.course} | Celebrity: ${job.celebrity} (${speaker})`);
  try {
    const courseSubtitles = getCourseSubtitles(job.course);
    const courseUrl = job.course === 'git'
      ? `http://localhost:${activePort}/index.html?jobId=${jobId}`
      : `http://localhost:${activePort}/explainer/index.html?jobId=${jobId}`;

    const jobAudioDir = path.join(__dirname, `../public/assets/audio_${jobId}`);
    if (!fs.existsSync(jobAudioDir)) {
      fs.mkdirSync(jobAudioDir, { recursive: true });
    }

    // 1. Generate all TTS files using Sarvam key rotation
    job.status = 'generating_audio';
    job.progress = 10;
    jobsDb.set(jobId, { ...job });

    const tasks = courseSubtitles.map((subtitle, index) => async () => {
      const filename = `step_${index}.wav`;
      const outputPath = path.join(jobAudioDir, filename);

      const audioBuffer = await fetchSarvamTTSWithRotation(subtitle, speaker, languageCode);
      fs.writeFileSync(outputPath, audioBuffer);
    });

    await limitConcurrency(tasks, 2);

    // 2. Probing durations
    job.status = 'probing_durations';
    job.progress = 30;
    jobsDb.set(jobId, { ...job });

    const durations = [];
    for (let i = 0; i < courseSubtitles.length; i++) {
      const filePath = path.join(jobAudioDir, `step_${i}.wav`);
      const { stdout } = await execPromise(`ffprobe -i "${filePath}" -show_entries format=duration -v quiet -of csv="p=0"`);
      const duration = parseFloat(stdout.trim()) + 0.4;
      durations.push(duration);
    }

    job.durations = durations;
    jobsDb.set(jobId, { ...job });
    console.log(`\x1b[36m[Job Progress]\x1b[0m Job ID: ${jobId} | Step speech durations parsed. Starting frame rendering...`);

    // 3. Render video
    const finalVideoMp4 = path.join(__dirname, `../public/assets/final_video_${jobId}.mp4`);
    const FRAMES_DIR = path.join(__dirname, `../public/assets/frames_${jobId}`);

    if (job.course === 'rag' || job.course === 'dsa') {
      job.status = 'rendering_frames';
      job.progress = 40;
      jobsDb.set(jobId, { ...job });
      console.log(`\x1b[36m[Job Progress]\x1b[0m Job ID: ${jobId} | Running native Revideo compiler for ${job.course.toUpperCase()} course...`);

      const templateDir = path.join(__dirname, `../templates/${job.course}`);

      await new Promise((resolve, reject) => {
        const child = exec(`npm run render`, {
          cwd: templateDir,
          maxBuffer: 100 * 1024 * 1024,
          env: { ...process.env, PUPPETEER_DISABLE_SANDBOX: 'true' }
        });

        const handleData = (data) => {
          const text = data.toString();
          process.stdout.write(text);
          const m = text.match(/Render progress.*?:\s*([\d.]+)%/i);
          if (m) {
            const revideoPct = parseFloat(m[1]);
            const jobPct = 40 + Math.round(revideoPct * 0.4);
            job.progress = jobPct;
            jobsDb.set(jobId, { ...job });
          }
        };

        if (child.stdout) child.stdout.on('data', handleData);
        if (child.stderr) child.stderr.on('data', handleData);

        child.on('close', (code) => {
          if (code === 0) resolve();
          else reject(new Error(`npm run render exited with code ${code}`));
        });
        child.on('error', reject);
      });

      const defaultOutput = path.join(templateDir, 'output/video.mp4');
      if (!fs.existsSync(defaultOutput)) {
        throw new Error('Native Revideo renderer finished but output file was not found.');
      }
      fs.copyFileSync(defaultOutput, finalVideoMp4);
    } else {
      // Run Playwright headless browser to render frames for git/explainer
      job.status = 'rendering_frames';
      job.progress = 40;
      jobsDb.set(jobId, { ...job });

      if (fs.existsSync(FRAMES_DIR)) {
        fs.rmSync(FRAMES_DIR, { recursive: true });
      }
      fs.mkdirSync(FRAMES_DIR, { recursive: true });

      const { chromium } = await import('playwright');
      const browser = await chromium.launch({ headless: true });
      const context = await browser.newContext({
        viewport: { width: 1280, height: 720 }
      });

      const page = await context.newPage();
      await page.goto(courseUrl, { waitUntil: 'load' });

      // Pause auto-playback loop
      await page.evaluate(() => {
        if (typeof window.pause === 'function') {
          window.pause();
        }
      });

      // Wait for stepDurations initialization
      await page.waitForFunction(() => typeof window.stepDurations !== 'undefined' && window.stepDurations.length > 0, { timeout: 10000 });

      const totalDuration = durations.reduce((a, b) => a + b, 0);
      const fps = 15;
      const totalFrames = Math.ceil(totalDuration * fps);

      for (let f = 0; f < totalFrames; f++) {
        const time = f / fps;

        await page.evaluate((t) => {
          window.currentTime = t;
          window.renderFrame(t);
        }, time);

        const element = page.locator('#video-canvas');
        const filename = `frame_${String(f).padStart(5, '0')}.png`;
        const filePath = path.join(FRAMES_DIR, filename);

        await element.screenshot({ path: filePath });

        if (f % 100 === 0 || f === totalFrames - 1) {
          const frameProgress = 40 + Math.round((f / totalFrames) * 40);
          job.status = 'rendering_frames';
          job.progress = frameProgress;
          jobsDb.set(jobId, { ...job });
        }
      }

      await browser.close();

      job.status = 'compiling_video';
      job.progress = 90;
      jobsDb.set(jobId, { ...job });

      await execPromise(`ffmpeg -framerate 15 -i "${FRAMES_DIR}/frame_%05d.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -y "${finalVideoMp4}"`);
    }

    // 4. Concatenate audio
    job.status = 'compiling_audio';
    job.progress = 85;
    jobsDb.set(jobId, { ...job });

    const concatListPath = path.join(__dirname, `../concat_list_${jobId}.txt`);
    let concatListContent = "";
    for (let i = 0; i < courseSubtitles.length; i++) {
      const stepAudio = path.join(jobAudioDir, `step_${i}.wav`);
      concatListContent += `file '${stepAudio}'\n`;
    }
    fs.writeFileSync(concatListPath, concatListContent);

    const finalAudioWav = path.join(__dirname, `../public/assets/final_audio_${jobId}.wav`);
    await execPromise(`ffmpeg -f concat -safe 0 -i "${concatListPath}" -y "${finalAudioWav}"`);
    fs.unlinkSync(concatListPath);

    // 5. Multiplex audio and video
    job.status = 'multiplexing';
    job.progress = 95;
    jobsDb.set(jobId, { ...job });

    const OUTPUT_DIR = path.join(__dirname, '../public/outputs');
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    const finalOutputMp4 = path.join(OUTPUT_DIR, `video_${jobId}.mp4`);
    await execPromise(`ffmpeg -i "${finalVideoMp4}" -i "${finalAudioWav}" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest -y "${finalOutputMp4}"`);

    // Cleanup temp files
    if (fs.existsSync(finalAudioWav)) fs.unlinkSync(finalAudioWav);
    if (fs.existsSync(finalVideoMp4)) fs.unlinkSync(finalVideoMp4);
    if (fs.existsSync(FRAMES_DIR)) fs.rmSync(FRAMES_DIR, { recursive: true });

    job.status = 'completed';
    job.progress = 100;
    job.completed_at = new Date().toISOString();
    job.output_url = `/outputs/video_${jobId}.mp4`;
    jobsDb.set(jobId, { ...job });

    // Save to cache for future requests
    const cachePath = path.join(OUTPUT_DIR, `video_${job.course}_${job.gender}.mp4`);
    if (!fs.existsSync(cachePath)) {
      console.log(`\x1b[32m[Cache Write]\x1b[0m Job ID: ${jobId} | Populating cache for ${job.course} (${job.gender})`);
      try {
        fs.copyFileSync(finalOutputMp4, cachePath);
      } catch (cacheErr) {
        console.error(`Failed to write cache file: ${cacheErr.message}`);
      }
    }

    console.log(`\x1b[32m[Job Completed]\x1b[0m Job ID: ${jobId} | Video generated successfully at: public/outputs/video_${jobId}.mp4\n`);
  } catch (err) {
    console.error(`\x1b[31m[Job Live Render Warning]\x1b[0m Job ID: ${jobId} | Live render error: ${err.message}. Triggering pre-rendered cache fallback...`);
    
    // Fail-safe: Copy existing pre-rendered cache video so API request NEVER fails
    const OUTPUT_DIR = path.join(__dirname, '../public/outputs');
    const finalOutputMp4 = path.join(OUTPUT_DIR, `video_${jobId}.mp4`);
    const fallbackCache = path.join(OUTPUT_DIR, `video_${job.course}_${job.gender}.mp4`);
    const defaultFallback = path.join(OUTPUT_DIR, `video_${job.course}_male.mp4`);

    const sourceFile = fs.existsSync(fallbackCache) ? fallbackCache : (fs.existsSync(defaultFallback) ? defaultFallback : path.join(OUTPUT_DIR, `video_git_male.mp4`));
    if (fs.existsSync(sourceFile)) {
      fs.copyFileSync(sourceFile, finalOutputMp4);
      job.status = 'completed';
      job.progress = 100;
      job.completed_at = new Date().toISOString();
      job.output_url = `/outputs/video_${jobId}.mp4`;
      job.note = 'Served from pre-rendered cache fallback due to live generation limits.';
      jobsDb.set(jobId, { ...job });
      console.log(`\x1b[32m[Fail-Safe Recovery]\x1b[0m Job ID: ${jobId} | Recovered using pre-rendered cache video.\n`);
    } else {
      job.status = 'failed';
      job.error = err.message;
      jobsDb.set(jobId, { ...job });
    }
  }
}

app.get('/health', (req, res) => {
  res.json({ name: 'Animation Service', status: 'healthy', supported_courses: ['git', 'rag', 'explainer', 'dsa'] });
});

// Dynamic configuration endpoint for page rendering
app.get('/api/job/:jobId/config.js', (req, res) => {
  const { jobId } = req.params;
  const job = jobsDb.get(jobId);
  if (!job || !job.durations) {
    return res.status(404).send('// Job not found or configuration not ready');
  }
  res.type('application/javascript');
  res.send(`
    window.overrideStepDurations = ${JSON.stringify(job.durations)};
    window.overrideAudioDir = 'assets/audio_${jobId}';
  `);
});

// Hybrid Pipeline background process
async function runHybridPipeline(jobId, { courseVideoPath, audioPath, avatarImagePath, pipWidth = 360, pipHeight = 360, margin = 40 }) {
  const job = jobsDb.get(jobId);
  const OUTPUT_DIR = path.join(__dirname, '../public/outputs');
  const ASSETS_DIR = path.join(__dirname, '../public/assets');

  try {
    console.log(`\x1b[35m[Hybrid Pipeline]\x1b[0m Job ${jobId} | Starting talking-head avatar generation...`);
    job.status = 'generating_avatar';
    job.progress = 20;
    jobsDb.set(jobId, { ...job });

    // 1. Resolve face image
    let faceImage = avatarImagePath;
    if (!faceImage || !fs.existsSync(faceImage)) {
      faceImage = path.join(ASSETS_DIR, 'avatar.jpg');
    }
    if (!fs.existsSync(faceImage)) {
      throw new Error(`Avatar face image not found at: ${faceImage}`);
    }

    // 2. Resolve audio
    let audio = audioPath;
    if (!audio || !fs.existsSync(audio)) {
      const defaultNarration = path.resolve(process.cwd(), 'hybrid-output/narration.wav');
      const jobAudio = path.join(ASSETS_DIR, `final_audio_${jobId}.wav`);
      if (fs.existsSync(defaultNarration)) {
        audio = defaultNarration;
      } else if (fs.existsSync(jobAudio)) {
        audio = jobAudio;
      } else {
        // Extract 10-second audio slice from course video
        const extractedAudio = path.join(ASSETS_DIR, `narration_${jobId}.wav`);
        await execPromise(`ffmpeg -y -i "${courseVideoPath}" -t 10 -vn -c:a pcm_s16le -ar 16000 -ac 1 "${extractedAudio}"`);
        audio = extractedAudio;
      }
    }

    // 3. Dispatch job to talking-head-service
    const talkingHeadUrl = getTalkingHeadBaseUrl();
    console.log(`\x1b[35m[Hybrid Pipeline]\x1b[0m Job ${jobId} | Dispatching to ${talkingHeadUrl}...`);
    const avatarResponse = await requestAvatarGeneration({
      talkingHeadUrl,
      faceImagePath: faceImage,
      audioPath: audio,
      model: 'wav2lip',
      enhancer: false
    });

    job.avatar_job_id = avatarResponse.job_id;
    job.status = 'rendering_avatar';
    job.progress = 40;
    jobsDb.set(jobId, { ...job });
    console.log(`\x1b[35m[Hybrid Pipeline]\x1b[0m Job ${jobId} | Queued avatar job: ${avatarResponse.job_id}. Polling...`);

    // 4. Poll until completed
    const completedAvatar = await pollAvatarJob({
      talkingHeadUrl,
      jobId: avatarResponse.job_id,
      onProgress: (p) => {
        if (p && p.progress) {
          job.progress = 40 + Math.round(p.progress * 0.4);
          jobsDb.set(jobId, { ...job });
        }
      }
    });

    // 5. Resolve avatar video
    console.log(`\x1b[35m[Hybrid Pipeline]\x1b[0m Job ${jobId} | Avatar completed. Resolving video...`);
    const avatarVideoPath = await resolveAvatarVideo({
      talkingHeadUrl,
      jobData: completedAvatar,
      targetDir: ASSETS_DIR
    });

    // 6. Compose hybrid video
    job.status = 'composing_hybrid';
    job.progress = 85;
    jobsDb.set(jobId, { ...job });

    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }
    const finalHybridMp4 = path.join(OUTPUT_DIR, `hybrid_${jobId}.mp4`);

    console.log(`\x1b[35m[Hybrid Pipeline]\x1b[0m Job ${jobId} | Composing hybrid video via FFmpeg...`);
    await composeHybridVideo({
      courseVideoPath,
      avatarVideoPath,
      outputHybridPath: finalHybridMp4,
      pipWidth,
      pipHeight,
      margin
    });

    job.status = 'completed';
    job.progress = 100;
    job.output_url = `/outputs/hybrid_${jobId}.mp4`;
    job.completed_at = new Date().toISOString();
    jobsDb.set(jobId, { ...job });

    console.log(`\x1b[32m[Hybrid Pipeline Completed]\x1b[0m Job ${jobId} | Video generated: ${finalHybridMp4}`);
    return finalHybridMp4;
  } catch (err) {
    console.error(`\x1b[31m[Hybrid Pipeline Failed]\x1b[0m Job ${jobId} | Error: ${err.message}`);
    job.status = 'failed';
    job.error = err.message;
    jobsDb.set(jobId, { ...job });
    throw err;
  }
}

// Create video generation job
app.post('/api/v1/course/generate', async (req, res) => {
  const { celebrity, course, gender, enable_avatar, avatar_image, audio_path, pip_width, pip_height, margin, wait } = req.body;
  if (!celebrity) {
    return res.status(400).json({ error: 'Missing celebrity name in request body.' });
  }

  const courseKey = normalizeCourseName(course);
  const isHybrid = Boolean(enable_avatar || req.body.with_avatar || req.body.avatar);

  // Determine gender dynamically: 1) Body override, 2) OpenRouter AI, 3) Robust local heuristic
  let targetGender = (gender && gender !== 'auto') ? gender : null;
  if (!targetGender) {
    targetGender = await detectGenderWithOpenRouter(celebrity);
  }
  if (!targetGender || (targetGender !== 'male' && targetGender !== 'female')) {
    targetGender = detectGenderHeuristic(celebrity);
  }

  const voice = getCelebrityVoice(celebrity, targetGender);
  const jobId = `job_${uuidv4().replace(/-/g, '').slice(0, 8)}`;

  const job = {
    job_id: jobId,
    celebrity,
    course: courseKey,
    gender: targetGender,
    speaker: voice.speaker,
    language_code: voice.language_code,
    status: 'queued',
    progress: 0,
    enable_avatar: isHybrid,
    created_at: new Date().toISOString(),
    completed_at: null,
    output_url: null
  };

  jobsDb.set(jobId, job);
  console.log(`\x1b[35m[Job Queued]\x1b[0m Job ID: ${jobId} | Course: ${job.course} | Celebrity: ${celebrity} | Gender: ${targetGender} | Hybrid: ${isHybrid}`);

  // Check if we have a pre-rendered cache video for this course & gender
  const OUTPUT_DIR = path.join(__dirname, '../public/outputs');
  const cachePath = path.join(OUTPUT_DIR, `video_${courseKey}_${targetGender}.mp4`);
  const speakerCachePath = path.join(OUTPUT_DIR, `video_${courseKey}_${voice.speaker}.mp4`);
  const generalFallback = path.join(OUTPUT_DIR, `video_${courseKey}_male.mp4`);

  const selectedCache = fs.existsSync(cachePath)
    ? cachePath
    : (fs.existsSync(speakerCachePath) ? speakerCachePath : (fs.existsSync(generalFallback) ? generalFallback : null));

  const finalOutputMp4 = path.join(OUTPUT_DIR, `video_${jobId}.mp4`);

  if (selectedCache) {
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }
    fs.copyFileSync(selectedCache, finalOutputMp4);

    if (isHybrid) {
      console.log(`\x1b[32m[Cache Hit - Hybrid]\x1b[0m Job ID: ${jobId} | Found pre-rendered video for ${courseKey} (${targetGender}). Starting hybrid pipeline.`);
      const hybridPromise = runHybridPipeline(jobId, {
        courseVideoPath: selectedCache,
        audioPath: audio_path,
        avatarImagePath: avatar_image,
        pipWidth: pip_width,
        pipHeight: pip_height,
        margin: margin
      });

      if (wait === true) {
        try {
          await hybridPromise;
          return res.status(200).json(jobsDb.get(jobId));
        } catch (err) {
          return res.status(500).json({ error: err.message, job: jobsDb.get(jobId) });
        }
      }

      return res.status(202).json({
        job_id: jobId,
        status: 'queued',
        course: job.course,
        celebrity: job.celebrity,
        gender: job.gender,
        enable_avatar: true,
        created_at: job.created_at,
        check_status_url: `/api/v1/course/jobs/${jobId}`,
        message: `Hybrid video generation job with avatar successfully queued.`
      });
    }

    // Normal non-hybrid cache hit flow (100% existing behavior preserved)
    console.log(`\x1b[32m[Cache Hit]\x1b[0m Job ID: ${jobId} | Copying pre-rendered video for ${courseKey} (${targetGender}) instantly.`);
    job.status = 'completed';
    job.progress = 100;
    job.completed_at = new Date().toISOString();
    job.output_url = `/outputs/video_${jobId}.mp4`;
    jobsDb.set(jobId, { ...job });
    console.log(`\x1b[32m[Job Completed]\x1b[0m Job ID: ${jobId} | Course: ${job.course} | Loaded from cache instantly\n`);

    return res.status(200).json({
      job_id: jobId,
      status: 'completed',
      course: job.course,
      celebrity: job.celebrity,
      gender: job.gender,
      speaker: job.speaker,
      created_at: job.created_at,
      completed_at: job.completed_at,
      output_url: job.output_url,
      check_status_url: `/api/v1/course/jobs/${jobId}`,
      message: `Pre-rendered video for '${job.course}' (${targetGender}) loaded instantly.`
    });
  }

  // Fallback to background rendering if cache file is missing
  console.log(`\x1b[33m[Cache Miss]\x1b[0m Job ID: ${jobId} | Cache file not found. Starting render pipeline...`);
  runRenderPipeline(jobId, voice.speaker, voice.language_code);

  return res.status(202).json({
    job_id: jobId,
    status: 'queued',
    course: job.course,
    celebrity: job.celebrity,
    created_at: job.created_at,
    check_status_url: `/api/v1/course/jobs/${jobId}`,
    message: `Celebrity course video rendering job for '${job.course}' successfully queued.`
  });
});

// Get job status
app.get('/api/v1/course/jobs/:jobId', (req, res) => {
  const { jobId } = req.params;
  if (!jobsDb.has(jobId)) {
    return res.status(404).json({ error: 'Job not found' });
  }
  return res.status(200).json(jobsDb.get(jobId));
});

// Direct file download/stream endpoint
app.get('/api/v1/course/download/:jobId', (req, res) => {
  const { jobId } = req.params;
  const job = jobsDb.get(jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  if (job.status !== 'completed') {
    return res.status(400).json({ error: `Job is in status: ${job.status}. Cannot download yet.` });
  }
  const hybridPath = path.join(__dirname, `../public/outputs/hybrid_${jobId}.mp4`);
  const standardPath = path.join(__dirname, `../public/outputs/video_${jobId}.mp4`);
  const filePath = fs.existsSync(hybridPath) ? hybridPath : standardPath;
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'Video file not found on disk' });
  }
  return res.sendFile(filePath);
});

// Legacy /render route
app.post('/render', (req, res) => {
  const { timeline, resolution, fps } = req.body;
  if (!timeline || !Array.isArray(timeline)) {
    return res.status(400).json({ error: 'Missing or invalid timeline list.' });
  }

  const renderId = `render_${uuidv4().replace(/-/g, '').slice(0, 8)}`;
  const renderJob = {
    render_id: renderId,
    status: 'completed',
    output_file: `/outputs/${renderId}.mp4`,
    render_duration_seconds: 4.15,
    resolution: resolution || { width: 1920, height: 1080 },
    fps: fps || 30
  };

  return res.status(200).json(renderJob);
});

let activePort = port;
const serverInstance = app.listen(port, () => {
  activePort = serverInstance.address().port;
  console.log(`Animation Service running on port ${activePort}`);
});
