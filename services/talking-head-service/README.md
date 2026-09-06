# AI Talking Head Service

This service provides a REST API for submitting and tracking talking head avatar
generation jobs. It accepts a face image and audio file, validates inputs, stores
files safely, and dispatches a background pipeline task.

> **Note:** The repository is designed around the official `LatentSync`
> backend. The API keeps the existing `image_path` contract for compatibility,
> but the real official pipeline requires a short reference video plus audio as
> input. The service therefore produces a temporary reference video in the
> compatibility layer before handing off to the actual LatentSync pipeline.
>
> The actual model weights and runtime package are not committed to this
> repository, so the pipeline performs a controlled failure until the runtime and
> checkpoint are configured in the environment.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/api/v1/avatar/generate` | Submit a generation job |
| `GET` | `/api/v1/avatar/jobs/{job_id}` | Query job status |

### POST `/api/v1/avatar/generate`

**Form fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `face_image` | file | Yes | — | JPEG or PNG portrait image (max 25 MB) |
| `audio` | file | Yes | — | WAV/MP3/OGG/FLAC/M4A audio file (max 50 MB) |
| `model` | string | No | `latentsync` | Model to use (currently only `latentsync` is configured) |
| `enhancer` | boolean | No | `true` | Whether to apply enhancement post-processing |

**Response (202 Accepted):**
```json
{
  "job_id": "job_abc123def456",
  "status": "queued",
  "created_at": "2024-01-01T00:00:00Z",
  "message": "Avatar rendering job successfully queued."
}
```

### GET `/api/v1/avatar/jobs/{job_id}`

**Response (200 OK):**
```json
{
  "job_id": "job_abc123def456",
  "status": "failed",
  "progress": 0.0,
  "estimated_time_remaining": 0.0,
  "created_at": "2024-01-01T00:00:00Z",
  "completed_at": "2024-01-01T00:00:01Z",
  "output_url": null,
  "error_message": "Talking Head model/inference engine for 'latentsync' is not implemented in this repository."
}
```

---

## Job Lifecycle

```
queued
  ↓
processing
  ↓
failed  (until real inference engine is integrated)
```

Once a real inference engine is integrated:
```
queued → processing → completed
```

A job will **never** be marked `completed` without a validated, real video output.

---

## Error Codes

| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `VALIDATION_ERROR` | 400 | Generic validation failure |
| `EMPTY_FILE` | 400 | Uploaded file is empty |
| `FILE_TOO_LARGE` | 400 | File exceeds size limit |
| `UNSUPPORTED_IMAGE_FORMAT` | 400 | Image extension/MIME not allowed |
| `UNSUPPORTED_AUDIO_FORMAT` | 400 | Audio extension not allowed |
| `CORRUPTED_IMAGE` | 400 | Image cannot be decoded |
| `CORRUPTED_AUDIO` | 400 | Audio magic header check failed |
| `UNSUPPORTED_MODEL` | 400 | Model not configured in this repository |
| `JOB_NOT_FOUND` | 404 | Job ID does not exist |
| `PIPELINE_ERROR` | 500 | Pipeline execution failed |
| `STORAGE_ERROR` | 500 | File persistence failed |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected error |

---

## Storage

Uploaded files are persisted to `storage/jobs/{job_id}/inputs/` before the
background task is started, so `UploadFile` objects are never used after the
request lifecycle ends. The `storage/` directory is gitignored.

---

## Setup Instructions

### Local Development (Python Virtual Environment)

1. Navigate to this directory:
   ```bash
   cd services/talking-head-service
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install the official LatentSync runtime from source:
   ```bash
   pip install "git+https://github.com/bytedance/LatentSync.git"
   ```
5. Download the official model assets to a local directory such as `./models/latentsync`:
   ```bash
   mkdir -p models/latentsync/checkpoints/whisper
   huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir models/latentsync/checkpoints
   huggingface-cli download ByteDance/LatentSync-1.6 tiny.pt --local-dir models/latentsync/checkpoints/whisper
   ```
   The upstream repo also expects the config directory to include `configs/unet/stage2_512.yaml` and `configs/scheduler_config.json`.
6. Configure the runtime via `.env` or environment variables:
   ```bash
   cp .env.example .env
   ```
7. Start the FastAPI server:
   ```bash
   uvicorn src.main:app --reload --port 8000
   ```
8. Open `http://localhost:8000/docs` for the OpenAPI (Swagger) documentation.

### Running with Docker

```bash
docker-compose up -d talking-head-service
```

### Official LatentSync model files

The repository is designed to use the official LatentSync checkpoint set from the
upstream project:

- `checkpoints/latentsync_unet.pt`
- `checkpoints/whisper/tiny.pt` (or small/other Whisper model variant supported by the upstream repo)
- `configs/unet/stage2_512.yaml`
- `configs/scheduler_config.json`

Do not commit these files or multi-GB model weights to the repository. Keep them
under a local `MODEL_CACHE_DIR` or `LATENTSYNC_*` location and point the service to
that directory via environment variables.

---

## Running Tests

```bash
cd services/talking-head-service
python -m pytest tests/ -v
```

---

## Running Lint

```bash
# From repository root
python -m black --check services/talking-head-service/src services/talking-head-service/tests
python -m isort --check-only services/talking-head-service/src services/talking-head-service/tests
python -m flake8 services/talking-head-service/src services/talking-head-service/tests
```

---

## Pipeline Behaviour (Current)

The `run_talking_head_pipeline()` function is the integration boundary where real
Talking Head model inference will be invoked (e.g. LatentSync — see Issue #2).

Currently, no inference engine, model weights, or checkpoints exist in this
repository. Instead of raising a raw `NotImplementedError`, the pipeline performs
a **controlled failure**:

1. Sets job status → `processing`
2. Raises `PipelineError` (domain exception, not an unhandled crash)
3. Catches the error and sets job status → `failed` with a descriptive `error_message`

This is intentional and production-safe. No fake video is generated and no job is
falsely marked `completed`.
