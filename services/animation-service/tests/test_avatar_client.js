import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  getTalkingHeadBaseUrl,
  requestAvatarGeneration,
  pollAvatarJob,
  resolveAvatarVideo
} from '../src/avatarClient.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Avatar Client Integration', () => {
  it('correctly resolves talking head base URL from default and env', () => {
    delete process.env.TALKING_HEAD_SERVICE_URL;
    assert.strictEqual(getTalkingHeadBaseUrl(), 'http://127.0.0.1:8000');

    process.env.TALKING_HEAD_SERVICE_URL = 'http://localhost:9000/';
    assert.strictEqual(getTalkingHeadBaseUrl(), 'http://localhost:9000');
    delete process.env.TALKING_HEAD_SERVICE_URL;
  });

  it('rejects if face image is missing', async () => {
    await assert.rejects(
      async () => {
        await requestAvatarGeneration({
          faceImagePath: '/non/existent/face.jpg',
          audioPath: '/non/existent/audio.wav'
        });
      },
      /Avatar face image not found/
    );
  });

  it('rejects if audio is missing but face image exists', async () => {
    const existingFace = path.resolve(__dirname, '../public/assets/avatar.jpg');
    await assert.rejects(
      async () => {
        await requestAvatarGeneration({
          faceImagePath: existingFace,
          audioPath: '/non/existent/audio.wav'
        });
      },
      /Narration audio not found/
    );
  });

  it('pollAvatarJob throws error on job failure', async () => {
    // Mock global fetch for this test
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url) => {
      return {
        ok: true,
        json: async () => ({
          status: 'failed',
          error_message: 'Model out of memory'
        })
      };
    };

    try {
      await assert.rejects(
        async () => {
          await pollAvatarJob({
            talkingHeadUrl: 'http://mock-url',
            jobId: 'job_test_fail',
            maxWaitMs: 2000,
            intervalMs: 100
          });
        },
        /Talking head job job_test_fail failed: Model out of memory/
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('pollAvatarJob resolves when job status is completed', async () => {
    const originalFetch = globalThis.fetch;
    let calls = 0;
    globalThis.fetch = async (url) => {
      calls++;
      return {
        ok: true,
        json: async () => ({
          job_id: 'job_test_done',
          status: calls >= 2 ? 'completed' : 'processing',
          progress: calls >= 2 ? 100 : 50,
          output_url: '/outputs/job_test_done/outputs/avatar.mp4'
        })
      };
    };

    try {
      const res = await pollAvatarJob({
        talkingHeadUrl: 'http://mock-url',
        jobId: 'job_test_done',
        maxWaitMs: 5000,
        intervalMs: 50
      });
      assert.strictEqual(res.status, 'completed');
      assert.strictEqual(res.output_url, '/outputs/job_test_done/outputs/avatar.mp4');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
