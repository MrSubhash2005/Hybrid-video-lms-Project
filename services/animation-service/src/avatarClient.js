/**
 * Avatar Client & Hybrid Pipeline Integration Helper
 *
 * Connects animation-service to talking-head-service:
 * 1. Dispatches avatar generation jobs to POST /api/v1/avatar/generate
 * 2. Polls GET /api/v1/avatar/jobs/{job_id} until completion
 * 3. Resolves the generated avatar MP4
 * 4. Invokes composer.py to overlay avatar onto course video
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';
import util from 'util';

const execPromise = util.promisify(exec);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export function getTalkingHeadBaseUrl() {
  return (process.env.TALKING_HEAD_SERVICE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
}

/**
 * Request avatar generation from talking-head-service.
 */
export async function requestAvatarGeneration({
  talkingHeadUrl = getTalkingHeadBaseUrl(),
  faceImagePath,
  audioPath,
  model = 'wav2lip',
  enhancer = false
}) {
  if (!fs.existsSync(faceImagePath)) {
    throw new Error(`Avatar face image not found at: ${faceImagePath}`);
  }
  if (!fs.existsSync(audioPath)) {
    throw new Error(`Narration audio not found at: ${audioPath}`);
  }

  const faceBuffer = fs.readFileSync(faceImagePath);
  const audioBuffer = fs.readFileSync(audioPath);

  const formData = new FormData();
  formData.append('face_image', new Blob([faceBuffer]), path.basename(faceImagePath));
  formData.append('audio', new Blob([audioBuffer]), path.basename(audioPath));
  formData.append('model', model);
  formData.append('enhancer', String(enhancer));

  const response = await fetch(`${talkingHeadUrl}/api/v1/avatar/generate`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Talking Head Service returned HTTP ${response.status}: ${errText}`);
  }

  const data = await response.json();
  return data; // { job_id, status: 'queued', ... }
}

/**
 * Poll talking-head job until completion or timeout.
 */
export async function pollAvatarJob({
  talkingHeadUrl = getTalkingHeadBaseUrl(),
  jobId,
  maxWaitMs = 180000,
  intervalMs = 2000,
  onProgress = null
}) {
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitMs) {
    const response = await fetch(`${talkingHeadUrl}/api/v1/avatar/jobs/${jobId}`);
    if (!response.ok) {
      throw new Error(`Failed to query job status (${response.status}) for job ${jobId}`);
    }

    const jobData = await response.json();
    if (onProgress && typeof onProgress === 'function') {
      onProgress(jobData);
    }

    if (jobData.status === 'completed') {
      return jobData;
    }

    if (jobData.status === 'failed') {
      throw new Error(`Talking head job ${jobId} failed: ${jobData.error_message || 'Unknown error'}`);
    }

    await new Promise(resolve => setTimeout(resolve, intervalMs));
  }

  throw new Error(`Talking head job ${jobId} timed out after ${Math.round(maxWaitMs / 1000)}s`);
}

/**
 * Locate or download the avatar MP4 video produced by talking-head-service.
 */
export async function resolveAvatarVideo({
  talkingHeadUrl = getTalkingHeadBaseUrl(),
  jobData,
  targetDir
}) {
  // Check local filesystem first if running on the same host / volume
  const localJobPath = path.resolve(
    __dirname,
    '../../talking-head-service/storage/jobs',
    jobData.job_id,
    'outputs/avatar.mp4'
  );

  if (fs.existsSync(localJobPath) && fs.statSync(localJobPath).size > 0) {
    return localJobPath;
  }

  // Otherwise download via HTTP
  if (!jobData.output_url) {
    throw new Error(`Avatar job ${jobData.job_id} does not have an output_url`);
  }

  const downloadUrl = `${talkingHeadUrl}${jobData.output_url}`;
  if (!fs.existsSync(targetDir)) {
    fs.mkdirSync(targetDir, { recursive: true });
  }

  const downloadedPath = path.join(targetDir, `avatar_${jobData.job_id}.mp4`);
  const res = await fetch(downloadUrl);
  if (!res.ok) {
    throw new Error(`Failed to download avatar video from ${downloadUrl} (HTTP ${res.status})`);
  }

  const arrayBuffer = await res.arrayBuffer();
  fs.writeFileSync(downloadedPath, Buffer.from(arrayBuffer));
  return downloadedPath;
}

/**
 * Execute composer.py to create the final hybrid video.
 */
export async function composeHybridVideo({
  courseVideoPath,
  avatarVideoPath,
  outputHybridPath,
  pipWidth = 360,
  pipHeight = 360,
  margin = 40
}) {
  const composerScript = path.resolve(__dirname, 'composer.py');
  const pyCmd = process.env.PYTHON_PATH || 'python';

  const cmd = `"${pyCmd}" "${composerScript}" --course "${courseVideoPath}" --avatar "${avatarVideoPath}" --output "${outputHybridPath}" --pip-width ${pipWidth} --pip-height ${pipHeight} --margin ${margin}`;

  const { stdout, stderr } = await execPromise(cmd);
  if (!fs.existsSync(outputHybridPath) || fs.statSync(outputHybridPath).size === 0) {
    throw new Error(`Composer failed to generate ${outputHybridPath}: ${stderr || stdout}`);
  }

  return outputHybridPath;
}
