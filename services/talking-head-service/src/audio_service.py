import math
import subprocess
from pathlib import Path


def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    duration = float(result.stdout.strip())

    if duration <= 0:
        raise ValueError("Audio duration is invalid or zero.")

    return duration


def validate_audio_with_ffprobe(file_path: str) -> dict:
    """Validate that the file contains a valid audio stream."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if path.stat().st_size == 0:
        raise ValueError("Audio file is empty.")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels",
                "-of",
                "default=noprint_wrappers=1",
                file_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("Invalid or corrupted audio file.") from exc

    if not result.stdout.strip():
        raise ValueError("Input file does not contain a valid audio stream.")

    duration = get_audio_duration(file_path)

    return {
        "duration": duration,
        "metadata": result.stdout.strip(),
    }


def normalize_audio(
    input_path: str,
    output_path: str,
) -> dict:
    """
    Normalize audio to:

    - WAV
    - 16 kHz
    - Mono
    - 16-bit
    """

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    if input_file.stat().st_size == 0:
        raise ValueError("Audio file is empty.")

    # Validate input first.
    validate_audio_with_ffprobe(str(input_file))

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_file),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Audio normalization failed: {exc.stderr}") from exc

    if not output_file.exists():
        raise RuntimeError(
            "Audio normalization completed but output file was not created."
        )

    if output_file.stat().st_size == 0:
        raise RuntimeError("Normalized audio output is empty.")

    # Validate normalized output.
    normalized_info = validate_audio_with_ffprobe(str(output_file))

    return {
        "success": True,
        "input_path": str(input_file),
        "output_path": str(output_file),
        "format": "wav",
        "duration": normalized_info["duration"],
        "sample_rate": 16000,
        "channels": 1,
    }


def split_audio_into_chunks(
    audio_path: str,
    output_dir: str,
    chunk_duration: int = 10,
) -> dict:
    """Split audio into approximately 10-second WAV chunks."""

    total_duration = get_audio_duration(audio_path)
    total_duration_ms = int(total_duration * 1000)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunks_meta = []
    number_of_chunks = math.ceil(total_duration / chunk_duration)

    for index in range(number_of_chunks):
        start_time = index * chunk_duration
        chunk_name = f"chunk_{index:04d}.wav"
        chunk_path = output_path / chunk_name

        # Calculate exact duration for this chunk
        remaining_duration = total_duration - start_time
        actual_chunk_duration = min(chunk_duration, remaining_duration)

        start_ms = int(start_time * 1000)
        duration_ms = int(actual_chunk_duration * 1000)
        end_ms = start_ms + duration_ms

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    audio_path,
                    "-ss",
                    str(start_time),
                    "-t",
                    str(actual_chunk_duration),
                    "-acodec",
                    "pcm_s16le",
                    str(chunk_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to create audio chunk {index}: {exc.stderr}"
            ) from exc

        chunks_meta.append(
            {
                "index": index,
                "path": str(chunk_path),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration_ms,
            }
        )

    return {
        "chunks": chunks_meta,
        "total_duration_ms": total_duration_ms,
        "chunk_count": number_of_chunks,
    }


def prepare_audio(
    input_audio_path: str,
    job_id: str,
    output_base_dir: str,
    chunk_duration_seconds: int = 10,
) -> dict:
    """
    Prepare input audio for the talking-head inference pipeline.

    Validates, normalizes to 16 kHz WAV, and chunks if needed.

    Args:
        input_audio_path: Path to user-provided audio file
        job_id: Unique job identifier
        output_base_dir: Base directory for output (job-specific subdirs will be created)
        chunk_duration_seconds: Duration of each chunk in seconds (default 10)

    Returns:
        dict matching AUDIO_INTERFACE.md:
        {
            "is_chunked": bool,
            "normalized_audio_path": str,
            "total_duration_ms": int,
            "chunks": [
                {
                    "index": int,
                    "path": str,
                    "start_ms": int,
                    "end_ms": int,
                    "duration_ms": int
                },
                ...
            ]
        }

    Raises:
        FileNotFoundError: Input file not found
        ValueError: Invalid or empty audio file
        RuntimeError: Normalization or chunking failed
    """

    # Step 1: Validate input
    validate_audio_with_ffprobe(input_audio_path)

    # Step 2: Create job-specific output directory
    job_audio_dir = Path(output_base_dir) / job_id / "audio"
    job_audio_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Normalize audio
    normalized_output_path = str(job_audio_dir / "normalized_audio.wav")
    normalize_result = normalize_audio(input_audio_path, normalized_output_path)

    # Step 4: Get total duration
    total_duration_ms = int(normalize_result["duration"] * 1000)

    # Step 5: Chunk if necessary
    if total_duration_ms <= chunk_duration_seconds * 1000:
        # Short audio: no chunking
        return {
            "is_chunked": False,
            "normalized_audio_path": normalized_output_path,
            "total_duration_ms": total_duration_ms,
            "chunks": [],
        }
    else:
        # Long audio: chunk it
        chunk_result = split_audio_into_chunks(
            audio_path=normalized_output_path,
            output_dir=str(job_audio_dir),
            chunk_duration=chunk_duration_seconds,
        )

        return {
            "is_chunked": True,
            "normalized_audio_path": normalized_output_path,
            "total_duration_ms": total_duration_ms,
            "chunks": chunk_result["chunks"],
        }
