"""
Hybrid Video LMS - FFmpeg Video Composer Module.

Provides minimal, reusable composition to overlay a talking-head avatar video
as a Picture-in-Picture (PiP) window onto a base course video.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)


class ComposerError(RuntimeError):
    """Raised when video composition or validation fails."""
    pass


def has_audio_stream(file_path: Path) -> bool:
    """Check whether a media file contains an audio stream."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(file_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return "audio" in res.stdout
    except Exception:
        return False


def build_ffmpeg_command(
    course_path: Path,
    avatar_path: Path,
    output_path: Path,
    pip_width: int = 360,
    pip_height: int = 360,
    margin: int = 40,
    audio_source: str = "avatar",
    overwrite: bool = True,
) -> List[str]:
    """Build the exact FFmpeg command array for PiP composition."""
    filter_complex = (
        f"[1:v]scale={pip_width}:{pip_height}:force_original_aspect_ratio=decrease,"
        f"pad={pip_width}:{pip_height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p[avatar];"
        f"[0:v][avatar]overlay=W-w-{margin}:H-h-{margin}:shortest=1[vout]"
    )

    cmd = ["ffmpeg"]
    if overwrite:
        cmd.append("-y")

    cmd.extend([
        "-i", str(course_path),
        "-i", str(avatar_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
    ])

    # Audio mapping logic:
    avatar_has_audio = has_audio_stream(avatar_path)
    course_has_audio = has_audio_stream(course_path)

    if audio_source == "avatar" and avatar_has_audio:
        cmd.extend(["-map", "1:a:0"])
    elif audio_source == "course" and course_has_audio:
        cmd.extend(["-map", "0:a:0"])
    elif avatar_has_audio:
        cmd.extend(["-map", "1:a:0"])
    elif course_has_audio:
        cmd.extend(["-map", "0:a:0"])

    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ])

    return cmd


def compose_hybrid_video(
    course_video: str | Path,
    avatar_video: str | Path,
    output_video: str | Path,
    pip_width: int = 360,
    pip_height: int = 360,
    margin: int = 40,
    audio_source: str = "avatar",
    overwrite: bool = True,
) -> Path:
    """
    Compose a final hybrid LMS MP4 by overlaying an avatar video onto a course video.

    Parameters:
    - course_video: Path to base 1920x1080 course MP4.
    - avatar_video: Path to talking-head avatar MP4.
    - output_video: Output MP4 destination.
    - pip_width: Width of PiP window (default: 360).
    - pip_height: Height of PiP window (default: 360).
    - margin: Margin in px from bottom-right (default: 40).
    - audio_source: Audio track priority ('avatar' or 'course', default: 'avatar').
    - overwrite: Overwrite destination file if it exists (default: True).

    Returns:
    - Path to created MP4.
    """
    course_path = Path(course_video).resolve()
    avatar_path = Path(avatar_video).resolve()
    output_path = Path(output_video).resolve()

    if not course_path.exists():
        raise FileNotFoundError(f"Course video not found: {course_path}")
    if not avatar_path.exists():
        raise FileNotFoundError(f"Avatar video not found: {avatar_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_ffmpeg_command(
        course_path=course_path,
        avatar_path=avatar_path,
        output_path=output_path,
        pip_width=pip_width,
        pip_height=pip_height,
        margin=margin,
        audio_source=audio_source,
        overwrite=overwrite,
    )

    logger.info("Executing FFmpeg composition: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.strip() if exc.stderr else "Unknown FFmpeg execution error"
        raise ComposerError(f"FFmpeg composition failed: {err}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ComposerError(f"Composition output file is missing or empty: {output_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose a hybrid LMS course video with a talking-head avatar PiP overlay."
    )
    parser.add_argument(
        "--course",
        required=True,
        help="Path to the course MP4 video.",
    )
    parser.add_argument(
        "--avatar",
        required=True,
        help="Path to the talking-head avatar MP4 video.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output hybrid MP4 video.",
    )
    parser.add_argument(
        "--pip-width",
        type=int,
        default=360,
        help="Width of the PiP overlay (default: 360).",
    )
    parser.add_argument(
        "--pip-height",
        type=int,
        default=360,
        help="Height of the PiP overlay (default: 360).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=40,
        help="Bottom-right margin in pixels (default: 40).",
    )
    parser.add_argument(
        "--audio-source",
        choices=["avatar", "course"],
        default="avatar",
        help="Which audio track to use (default: avatar).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    try:
        out = compose_hybrid_video(
            course_video=args.course,
            avatar_video=args.avatar,
            output_video=args.output,
            pip_width=args.pip_width,
            pip_height=args.pip_height,
            margin=args.margin,
            audio_source=args.audio_source,
        )
        print(f"SUCCESS: Hybrid video created at {out}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
