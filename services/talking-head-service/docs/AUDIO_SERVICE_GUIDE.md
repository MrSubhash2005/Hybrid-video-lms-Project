# Audio Service Configuration & Integration Guide

## Purpose
The audio service prepares user-provided audio files for the talking-head inference pipeline. 
It validates, normalizes, and chunks audio to match the requirements of the inference engine.

## Module Location
`services/talking-head-service/src/audio_service.py`

## Dependencies
- Python 3.13+
- FFmpeg (8.1.2+) — must be in system PATH
- ffprobe (included with FFmpeg)
- subprocess, pathlib, math (stdlib)

Install FFmpeg:
- Windows: Download from [www.gyan.dev](https://www.gyan.dev) or `choco install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `apt-get install ffmpeg` or `yum install ffmpeg`

## Configuration Parameters (Locked)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Target sample rate | 16 kHz | Standard for talking-head/lip-sync models |
| Channels | Mono (1) | Single-channel reduces complexity |
| Format | WAV (LPCM) | No encoder delay; perfect chunk concatenation |
| Bit depth | 16-bit | Standard for audio inference |
| Chunk duration | 10 seconds | Balances memory usage and inference speed |
| Short audio threshold | <= 10 seconds | No chunking needed for short audio |

## Functions

### 1. `validate_audio_with_ffprobe(file_path: str) -> dict`
Validates that a file contains a valid audio stream.

**Input:** File path  
**Output:** 
```python
{
    "duration": float,  # seconds
    "metadata": str     # ffprobe output
}
```
**Raises:** FileNotFoundError, ValueError  

**Example:**
```python
info = validate_audio_with_ffprobe("/path/to/audio.mp3")
print(info["duration"])  # 30.5 seconds
```

---

### 2. `get_audio_duration(file_path: str) -> float`
Returns audio duration in seconds.

**Input:** File path  
**Output:** Duration in seconds (float)  
**Raises:** ValueError  

**Example:**
```python
duration = get_audio_duration("/path/to/audio.wav")
print(duration)  # 30.5
```

---

### 3. `normalize_audio(input_path: str, output_path: str) -> dict`
Normalizes audio to 16 kHz, mono, 16-bit WAV.

**Input:**
- input_path: Path to any audio file (MP3, WAV, AAC, etc.)
- output_path: Path where normalized WAV will be written

**Output:**
```python
{
    "success": True,
    "input_path": str,
    "output_path": str,
    "format": "wav",
    "duration": float,  # seconds
    "sample_rate": 16000,
    "channels": 1
}
```
**Raises:** FileNotFoundError, ValueError, RuntimeError  

**Example:**
```python
result = normalize_audio(
    "/user/uploads/speech.mp3",
    "/jobs/job_123/audio/normalized_audio.wav"
)
print(result["sample_rate"])  # 16000
```

---

### 4. `split_audio_into_chunks(audio_path: str, output_dir: str, chunk_duration: int = 10) -> dict`
Splits normalized audio into 10-second chunks.

**Input:**
- audio_path: Path to normalized audio file (from normalize_audio())
- output_dir: Directory where chunks will be written
- chunk_duration: Seconds per chunk (default 10)

**Output:**
```python
{
    "chunks": [
        {
            "index": 0,
            "path": "/path/to/chunk_0000.wav",
            "start_ms": 0,
            "end_ms": 10000,
            "duration_ms": 10000
        },
        ...
    ],
    "total_duration_ms": 30000,
    "chunk_count": 3
}
```
**Raises:** RuntimeError  

**Example:**
```python
result = split_audio_into_chunks(
    "/jobs/job_123/audio/normalized_audio.wav",
    "/jobs/job_123/audio",
    chunk_duration=10
)
for chunk in result["chunks"]:
    print(f"Chunk {chunk['index']}: {chunk['path']}")
```

---

### 5. `prepare_audio(input_audio_path: str, job_id: str, output_base_dir: str, chunk_duration_seconds: int = 10) -> dict`
**Main entry point. Call this from the inference pipeline.**

Validates, normalizes, and chunks audio in one function.

**Input:**
- input_audio_path: Path to user-provided audio file
- job_id: Unique job identifier (used for directory organization)
- output_base_dir: Base directory for all job outputs
- chunk_duration_seconds: Optional, default 10 seconds

**Output:** Matches AUDIO_INTERFACE.md
```python
{
    "is_chunked": bool,
    "normalized_audio_path": str,
    "total_duration_ms": int,
    "chunks": [...]  # empty if not chunked, populated if chunked
}
```
**Raises:** FileNotFoundError, ValueError, RuntimeError  

**Example:**
```python
audio_data = prepare_audio(
    input_audio_path="/user/uploads/speech.wav",
    job_id="job_abc123",
    output_base_dir="/jobs"
)

# Output directory created: /jobs/job_abc123/audio/
# Files: normalized_audio.wav, chunk_0000.wav, chunk_0001.wav, ...

if audio_data["is_chunked"]:
    for chunk in audio_data["chunks"]:
        print(f"Process chunk: {chunk['path']}")
else:
    print(f"Process single audio: {audio_data['normalized_audio_path']}")
```

## Output Directory Structure

All audio outputs are organized by job:

```text
{output_base_dir}/
└── {job_id}/
    └── audio/
        ├── normalized_audio.wav # Full normalized audio, 16 kHz, 16-bit, mono
        ├── chunk_0000.wav       # If chunked: chunk 0 (0-10 seconds)
        ├── chunk_0001.wav       # If chunked: chunk 1 (10-20 seconds)
        └── chunk_0002.wav       # If chunked: chunk 2 (20-30 seconds)
```

## Error Handling Reference

| Exception | Meaning | Recovery |
|-----------|---------|----------|
| FileNotFoundError | Input file does not exist | Verify file path |
| ValueError (empty) | Audio file is 0 bytes | Check file integrity |
| ValueError (no stream) | File has no audio stream | Provide valid audio file |
| RuntimeError (normalization) | FFmpeg encoding failed | Check FFmpeg installation, verify input format |
| RuntimeError (chunking) | FFmpeg chunking failed | Check output directory permissions |

## Integration Notes for Shankar (Inference Owner)

1. Import the function:
```python
   from services.talking_head_service.src.audio_service import prepare_audio
```

2. In your pipeline.py, call before inference:
```python
   audio_data = prepare_audio(
       input_audio_path=user_audio_file,
       job_id=job_id,
       output_base_dir="/path/to/jobs"
   )
```

3. Process based on chunking:
```python
   if audio_data["is_chunked"]:
       # Long audio: process each chunk in order
       for chunk in audio_data["chunks"]:
           video_chunk = run_inference(
               image_path=face_image,
               audio_path=chunk["path"]
               # Returns video_chunk_path for chunk at index chunk["index"]
           )
   else:
       # Short audio: single inference
       video_path = run_inference(
           image_path=face_image,
           audio_path=audio_data["normalized_audio_path"]
       )
```

4. Pass video chunks in order to Subrat's stitching service

## Testing

Run tests:
```bash
pytest tests/test_audio_service.py -v
```

Expected output:
- 15+ tests passing
- All functions validated
- Short and long audio tested
- Error cases covered

## Known Limitations

- Maximum recommended audio: 5 minutes (to avoid excessive chunks)
- Requires system FFmpeg (not bundled)
- Windows requires FFmpeg in PATH

## Support / Blockers

Document any issues:
- Missing FFmpeg: install from [www.gyan.dev](https://www.gyan.dev)
- Permission errors: check output directory write permissions
- Invalid audio: verify file is valid MP3/WAV/AAC
- Chunking gaps: all output is WAV (no encoder delay)
