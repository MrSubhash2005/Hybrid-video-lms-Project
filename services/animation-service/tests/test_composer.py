"""
Tests for FFmpeg video composer module.
"""

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Add services/animation-service to sys.path to support hyphenated directory
ANIM_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(ANIM_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ANIM_SERVICE_DIR))

from src.composer import (
    ComposerError,
    build_ffmpeg_command,
    compose_hybrid_video,
    has_audio_stream,
)


class TestComposer(unittest.TestCase):
    def setUp(self):
        self.course_path = Path("/mock/path/course.mp4")
        self.avatar_path = Path("/mock/path/avatar.mp4")
        self.output_path = Path("/mock/path/output.mp4")

    @patch("src.composer.has_audio_stream")
    def test_build_ffmpeg_command_structure(self, mock_has_audio):
        mock_has_audio.return_value = True

        cmd = build_ffmpeg_command(
            course_path=self.course_path,
            avatar_path=self.avatar_path,
            output_path=self.output_path,
            pip_width=360,
            pip_height=360,
            margin=40,
            audio_source="avatar",
            overwrite=True,
        )

        cmd_str = " ".join(cmd)
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert str(self.course_path) in cmd
        assert str(self.avatar_path) in cmd
        assert "scale=360:360" in cmd_str
        assert "overlay=W-w-40:H-h-40:shortest=1" in cmd_str
        assert "-shortest" in cmd
        assert "-c:v" in cmd and "libx264" in cmd
        assert "-c:a" in cmd and "aac" in cmd
        assert str(self.output_path) == cmd[-1]

    @patch("src.composer.has_audio_stream")
    def test_build_ffmpeg_command_custom_geometry(self, mock_has_audio):
        mock_has_audio.return_value = True

        cmd = build_ffmpeg_command(
            course_path=self.course_path,
            avatar_path=self.avatar_path,
            output_path=self.output_path,
            pip_width=400,
            pip_height=400,
            margin=50,
        )

        cmd_str = " ".join(cmd)
        assert "scale=400:400" in cmd_str
        assert "overlay=W-w-50:H-h-50:shortest=1" in cmd_str

    def test_missing_course_video_raises(self):
        with pytest.raises(FileNotFoundError, match="Course video not found"):
            compose_hybrid_video(
                course_video="/non/existent/course.mp4",
                avatar_video=self.avatar_path,
                output_video=self.output_path,
            )

    def test_missing_avatar_video_raises(self):
        with patch.object(Path, "exists", autospec=True) as mock_exists:
            # Make course exist, avatar not exist
            def side_effect(p):
                return "course" in str(p)

            mock_exists.side_effect = side_effect
            with pytest.raises(FileNotFoundError, match="Avatar video not found"):
                compose_hybrid_video(
                    course_video="course.mp4",
                    avatar_video="avatar.mp4",
                    output_video="output.mp4",
                )

    @patch("src.composer.has_audio_stream", return_value=True)
    @patch("src.composer.subprocess.run")
    def test_ffmpeg_failure_raises_composer_error(self, mock_run, mock_has_audio):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="Invalid input or corrupted stream",
        )

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "mkdir"):
                with pytest.raises(ComposerError, match="FFmpeg composition failed.*Invalid input"):
                    compose_hybrid_video(
                        course_video="course.mp4",
                        avatar_video="avatar.mp4",
                        output_video="output.mp4",
                    )

    @patch("src.composer.has_audio_stream", return_value=True)
    @patch("src.composer.subprocess.run")
    def test_successful_composition(self, mock_run, mock_has_audio):
        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "mkdir"):
                with patch.object(Path, "stat") as mock_stat:
                    mock_stat.return_value.st_size = 1048576
                    out = compose_hybrid_video(
                        course_video="course.mp4",
                        avatar_video="avatar.mp4",
                        output_video="output.mp4",
                    )
                    assert out == Path("output.mp4").resolve()
                    mock_run.assert_called_once()
