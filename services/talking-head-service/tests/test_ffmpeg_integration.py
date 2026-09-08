import json
import subprocess
from pathlib import Path

from src.ffmpeg_service import (
    FFmpegAssemblyError,
    concatenate_videos,
    validate_final_mp4,
)


def run_ffmpeg(*args):
    subprocess.run(
        ["ffmpeg", "-y", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def probe_media(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return json.loads(result.stdout)


def get_stream_duration(path, codec_type):
    metadata = probe_media(path)

    stream = next(
        stream for stream in metadata["streams"] if stream["codec_type"] == codec_type
    )

    return float(stream["duration"])


def get_pixel(path, timestamp):
    result = subprocess.run(
        [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            "scale=1:1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    )

    return tuple(result.stdout[:3])


def test_single_chunk_creates_valid_mp4(tmp_path):
    video = tmp_path / "chunk_0.mp4"
    audio = tmp_path / "audio.mp3"
    output = tmp_path / "final.mp4"

    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:r=25",
        "-t",
        "2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video),
    )

    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2",
        "-c:a",
        "libmp3lame",
        str(audio),
    )

    result = concatenate_videos(
        [str(video)],
        str(audio),
        str(output),
    )

    assert result == str(output)
    assert output.exists()
    assert output.stat().st_size > 0

    metadata = probe_media(output)

    assert any(stream["codec_type"] == "video" for stream in metadata["streams"])
    assert any(stream["codec_type"] == "audio" for stream in metadata["streams"])


def test_multiple_chunks_preserve_order(tmp_path):
    chunk_0 = tmp_path / "chunk_0.mp4"
    chunk_1 = tmp_path / "chunk_1.mp4"
    audio = tmp_path / "audio.mp3"
    output = tmp_path / "final.mp4"

    # First chunk = black
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:r=25",
        "-t",
        "1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(chunk_0),
    )

    # Second chunk = white
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=320x240:r=25",
        "-t",
        "1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(chunk_1),
    )

    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2",
        "-c:a",
        "libmp3lame",
        str(audio),
    )

    concatenate_videos(
        [str(chunk_0), str(chunk_1)],
        str(audio),
        str(output),
    )

    assert output.exists()
    assert output.stat().st_size > 0

    # First second should come from black chunk.
    first_pixel = get_pixel(output, 0.5)

    # Second second should come from white chunk.
    second_pixel = get_pixel(output, 1.5)

    assert sum(first_pixel) < 100
    assert sum(second_pixel) > 650


def test_final_audio_and_video_are_synchronized(tmp_path):
    chunk_0 = tmp_path / "chunk_0.mp4"
    chunk_1 = tmp_path / "chunk_1.mp4"
    audio = tmp_path / "audio.mp3"
    output = tmp_path / "final.mp4"

    for chunk in [chunk_0, chunk_1]:
        run_ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=25",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(chunk),
        )

    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2",
        "-c:a",
        "libmp3lame",
        str(audio),
    )

    concatenate_videos(
        [str(chunk_0), str(chunk_1)],
        str(audio),
        str(output),
    )

    video_duration = get_stream_duration(output, "video")
    audio_duration = get_stream_duration(output, "audio")

    assert abs(video_duration - audio_duration) < 0.15


def test_invalid_video_input_returns_ffmpeg_error(tmp_path):
    invalid_video = tmp_path / "invalid.mp4"
    audio = tmp_path / "audio.mp3"
    output = tmp_path / "final.mp4"

    invalid_video.write_text("this is not a valid video")

    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=1",
        "-c:a",
        "libmp3lame",
        str(audio),
    )

    try:
        concatenate_videos(
            [str(invalid_video)],
            str(audio),
            str(output),
        )
        assert False, "Expected FFmpegAssemblyError"
    except FFmpegAssemblyError as exc:
        assert "FFmpeg video assembly failed" in str(exc)
        assert "Invalid data found when processing input" in str(exc)


def test_final_mp4_validation_rejects_invalid_output(tmp_path):
    invalid_output = tmp_path / "invalid.mp4"

    invalid_output.write_text("not a real mp4")

    try:
        validate_final_mp4(str(invalid_output))
        assert False, "Expected FFmpegAssemblyError"
    except FFmpegAssemblyError as exc:
        assert "Final MP4 validation failed" in str(exc)


def test_temporary_concat_file_is_cleaned_up(tmp_path):
    chunk_0 = tmp_path / "chunk_0.mp4"
    chunk_1 = tmp_path / "chunk_1.mp4"
    audio = tmp_path / "audio.mp3"
    output = tmp_path / "final.mp4"

    for chunk in [chunk_0, chunk_1]:
        run_ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=25",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(chunk),
        )

    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2",
        "-c:a",
        "libmp3lame",
        str(audio),
    )

    concatenate_videos(
        [str(chunk_0), str(chunk_1)],
        str(audio),
        str(output),
    )

    assert output.exists()

    concat_files = list(tmp_path.glob("concat_list_*.txt"))
    assert concat_files == []
