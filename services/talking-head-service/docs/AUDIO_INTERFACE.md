# AUDIO INTERFACE CONTRACT

This document defines the exact interface contract between the audio preparation module and the talking-head inference engine.

## Input Contract
- Accepted formats: WAV, MP3, AAC (up to 5 minutes recommended)
- No restrictions on sample rate (will be resampled)
- Mono or stereo (will be converted to mono)

## Normalization Output (Target)
- **Sample rate: 16 kHz** (standard for lip-sync models like Wav2Lip, TalkingHead)
- **Channels: 1 (mono)**
- **Format: WAV (LPCM)** (NOT MP3 — WAV allows perfect chunk concatenation without encoder delay gaps)
- **Bit depth: 16-bit**

## Short Audio Path (No Chunking)
- If audio duration <= 10 seconds:
  - Return single normalized audio file
  - Return metadata:
```json
    {
      "is_chunked": false,
      "normalized_audio_path": "/path/to/normalized_audio.wav",
      "total_duration_ms": 5000,
      "chunks": []
    }
```

## Long Audio Path (With Chunking)
- If audio duration > 10 seconds:
  - Split into 10-second chunks
  - Each chunk is a separate 16 kHz, 16-bit WAV file
  - Chunks named: `chunk_0000.wav`, `chunk_0001.wav`, `chunk_0002.wav` (zero-padded, ordered)
  - Return metadata:
```json
    {
      "is_chunked": true,
      "normalized_audio_path": "/path/to/normalized_audio.wav",
      "total_duration_ms": 30000,
      "chunks": [
        {
          "index": 0,
          "path": "/path/to/chunk_0000.wav",
          "start_ms": 0,
          "end_ms": 10000,
          "duration_ms": 10000
        },
        {
          "index": 1,
          "path": "/path/to/chunk_0001.wav",
          "start_ms": 10000,
          "end_ms": 20000,
          "duration_ms": 10000
        },
        {
          "index": 2,
          "path": "/path/to/chunk_0002.wav",
          "start_ms": 20000,
          "end_ms": 30000,
          "duration_ms": 10000
        }
      ]
    }
```

## Chunk Duration Rationale
- 10 seconds: standard for most inference models (balances memory vs. inference speed)
- Can be overridden in `prepare_audio()` if needed

## Output Directory Structure
All audio processing outputs for a job:
```
{output_base_dir}/{job_id}/audio/
├── normalized_audio.wav (full normalized audio, 16 kHz, 16-bit, mono)
├── chunk_0000.wav (if chunked)
├── chunk_0001.wav (if chunked)
└── chunk_0002.wav (if chunked)
```

## Error Handling
- **FileNotFoundError:** "Audio file not found: {path}"
- **ValueError (empty file):** "Audio file is empty or zero bytes"
- **ValueError (no audio stream):** "Input file does not contain a valid audio stream"
- **ValueError (invalid audio):** "Invalid or corrupted audio file"
- **RuntimeError (normalization failed):** "Audio normalization failed: {ffmpeg stderr}"
- **RuntimeError (chunking failed):** "Failed to create audio chunk {index}: {ffmpeg stderr}"

## Assumptions (Locked)
- Sample rate: 16 kHz (talking-head models standard)
- Channels: Mono (1 channel)
- Format: WAV (no encoder delay)
- Bit depth: 16-bit
- Chunk duration: 10 seconds (overridable)
- Short audio threshold: <= 10 seconds

## Integration Notes for Shankar (Inference Owner)
1. Call `prepare_audio(input_audio_path, job_id, output_base_dir)`
2. Receive metadata dict with `normalized_audio_path` and `chunks` list
3. If `is_chunked=False`: process single normalized file
4. If `is_chunked=True`: process each chunk in order (index 0, 1, 2, ...)
5. Return video chunks in the same order for Subrat to stitch
