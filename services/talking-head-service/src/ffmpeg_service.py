import json
import os
import subprocess
import time
from typing import Dict, List


class FFmpegAssemblyError(RuntimeError):
    """Raised when video assembly or final MP4 validation fails."""


def probe_video_stream(file_path: str) -> Dict:
    """Probe video stream metadata to check for matching properties."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        file_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    metadata = json.loads(result.stdout)

    video_stream = next(
        (s for s in metadata.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )

    if not video_stream:
        raise ValueError(f"No video stream found in {file_path}")

    fps = 0.0

    if video_stream.get("r_frame_rate"):
        num, den = video_stream["r_frame_rate"].split("/")

        if num and den and int(den) != 0:
            fps = int(num) / int(den)

    return {
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "fps": round(fps, 2),
        "codec_name": video_stream.get("codec_name"),
        "pix_fmt": video_stream.get("pix_fmt"),
    }


def validate_final_mp4(output_path: str) -> bool:
    """
    Validate that the final output is a readable MP4 containing
    both video and audio streams.
    """

    if not os.path.exists(output_path):
        raise FFmpegAssemblyError(
            f"Final MP4 validation failed: output file does not exist: "
            f"{output_path}"
        )

    if os.path.getsize(output_path) == 0:
        raise FFmpegAssemblyError(
            f"Final MP4 validation failed: output file is empty: " f"{output_path}"
        )

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        metadata = json.loads(result.stdout)

    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            error = exc.stderr.strip() or "Unknown ffprobe error"
        else:
            error = "Invalid ffprobe output"

        raise FFmpegAssemblyError(f"Final MP4 validation failed: {error}") from exc

    streams = metadata.get("streams", [])

    has_video = any(stream.get("codec_type") == "video" for stream in streams)

    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)

    if not has_video:
        raise FFmpegAssemblyError("Final MP4 validation failed: no video stream found")

    if not has_audio:
        raise FFmpegAssemblyError("Final MP4 validation failed: no audio stream found")

    return True


def _run_ffmpeg(cmd: List[str]) -> None:
    """Run FFmpeg and expose a useful error message on failure."""

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

    except subprocess.CalledProcessError as exc:
        error = exc.stderr.strip() or "Unknown FFmpeg error"

        raise FFmpegAssemblyError(f"FFmpeg video assembly failed: {error}") from exc


def concatenate_videos(
    video_paths: List[str],
    master_audio_path: str,
    output_path: str,
):
    """
    Concatenate ordered video chunks and merge the master audio
    into one final MP4.
    """

    if not video_paths:
        raise ValueError("No video paths provided for concatenation")

    # Fast path for single video clip
    if len(video_paths) == 1:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_paths[0],
            "-i",
            master_audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-shortest",
            output_path,
        ]

        _run_ffmpeg(cmd)
        validate_final_mp4(output_path)

        return output_path

    concat_list_path = None

    try:
        # Probe all input videos to check if they match perfectly
        metadatas = [probe_video_stream(path) for path in video_paths]

        first_meta = metadatas[0]

        is_mismatch = any(
            metadata["width"] != first_meta["width"]
            or metadata["height"] != first_meta["height"]
            or metadata["fps"] != first_meta["fps"]
            or metadata["codec_name"] != first_meta["codec_name"]
            or metadata["pix_fmt"] != first_meta["pix_fmt"]
            for metadata in metadatas
        )

        cmd = ["ffmpeg", "-y"]

        if not is_mismatch:
            # Mode A: concat demuxer / stream copy
            concat_list_path = os.path.join(
                os.path.dirname(output_path),
                f"concat_list_{int(time.time())}.txt",
            )

            with open(
                concat_list_path,
                "w",
                encoding="utf-8",
            ) as file:
                for path in video_paths:
                    safe_path = os.path.abspath(path).replace("'", "'\\''")

                    file.write(f"file '{safe_path}'\n")

            cmd.extend(
                [
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_list_path,
                ]
            )

            cmd.extend(
                [
                    "-i",
                    master_audio_path,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    "-movflags",
                    "+faststart",
                    "-shortest",
                    output_path,
                ]
            )

        else:
            # Mode B: filter complex / safe re-encoding
            for path in video_paths:
                cmd.extend(["-i", path])

            cmd.extend(["-i", master_audio_path])

            filter_parts = []
            concat_v_inputs = ""

            for index in range(len(video_paths)):
                filter_parts.append(
                    f"[{index}:v]"
                    "fps=25,"
                    "scale=1920:1080:"
                    "force_original_aspect_ratio=decrease,"
                    "pad=1920:1080:"
                    "(ow-iw)/2:(oh-ih)/2,"
                    "setsar=1,"
                    "format=yuv420p"
                    f"[v{index}]"
                )

                concat_v_inputs += f"[v{index}]"

            filter_parts.append(
                f"{concat_v_inputs}" f"concat=n={len(video_paths)}:v=1:a=0" "[vout]"
            )

            cmd.extend(
                [
                    "-filter_complex",
                    "; ".join(filter_parts),
                    "-map",
                    "[vout]",
                    "-map",
                    f"{len(video_paths)}:a:0",
                    "-c:v",
                    "libx264",
                    "-profile:v",
                    "high",
                    "-level",
                    "4.1",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    "-movflags",
                    "+faststart",
                    "-shortest",
                    output_path,
                ]
            )

        _run_ffmpeg(cmd)

        # Explicitly validate the final assembled MP4.
        validate_final_mp4(output_path)

        return output_path

    finally:
        # Always clean up temporary concat list.
        if concat_list_path and os.path.exists(concat_list_path):
            try:
                os.remove(concat_list_path)
            except OSError:
                pass
