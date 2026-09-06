import unittest
from unittest.mock import patch, MagicMock
from src.ffmpeg_service import concatenate_videos, probe_video_stream


class TestFFmpegService(unittest.TestCase):

    @patch("src.ffmpeg_service.subprocess.run")
    def test_probe_video_stream(self, mock_run):
        # Mock ffprobe output
        mock_result = MagicMock()
        mock_result.stdout = '{"streams": [{"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "25/1", "codec_name": "h264", "pix_fmt": "yuv420p"}]}'
        mock_run.return_value = mock_result

        result = probe_video_stream("dummy.mp4")
        self.assertEqual(result["width"], 1920)
        self.assertEqual(result["height"], 1080)
        self.assertEqual(result["fps"], 25.0)
        self.assertEqual(result["codec_name"], "h264")

    @patch("src.ffmpeg_service.validate_final_mp4")
    @patch("src.ffmpeg_service.probe_video_stream")
    @patch("src.ffmpeg_service.subprocess.run")
    def test_concatenate_videos_mode_a(self, mock_run, mock_probe, mock_validate):
        # Mock probe to return identical streams (triggers Mode A)
        mock_probe.return_value = {
            "width": 1920,
            "height": 1080,
            "fps": 25.0,
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
        }

        concatenate_videos(["clip1.mp4", "clip2.mp4"], "audio.mp3", "out.mp4")

        # Verify ffmpeg was called with concat demuxer (-c:v copy)
        cmd_args = mock_run.call_args[0][0]
        self.assertIn("-f", cmd_args)
        self.assertIn("concat", cmd_args)
        self.assertIn("-c:v", cmd_args)
        self.assertIn("copy", cmd_args)
        self.assertIn("+faststart", cmd_args)

    @patch("src.ffmpeg_service.validate_final_mp4")
    @patch("src.ffmpeg_service.probe_video_stream")
    @patch("src.ffmpeg_service.subprocess.run")
    def test_concatenate_videos_mode_b(self, mock_run, mock_probe, mock_validate):
        # Mock probe to return mismatched streams (triggers Mode B)
        mock_probe.side_effect = [
            {
                "width": 1920,
                "height": 1080,
                "fps": 25.0,
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
            },
            {
                "width": 1280,
                "height": 720,
                "fps": 30.0,
                "codec_name": "hevc",
                "pix_fmt": "yuv420p",
            },
        ]

        concatenate_videos(["clip1.mp4", "clip2.mp4"], "audio.mp3", "out.mp4")

        # Verify ffmpeg was called with filter_complex
        cmd_args = mock_run.call_args[0][0]
        self.assertIn("-filter_complex", cmd_args)
        self.assertIn("libx264", cmd_args)
        self.assertIn("+faststart", cmd_args)


if __name__ == "__main__":
    unittest.main()
