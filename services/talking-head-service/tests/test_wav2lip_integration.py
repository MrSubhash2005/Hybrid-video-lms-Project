"""
Lightweight integration tests for the Wav2Lip backend wiring.

These tests verify:
  - config imports cleanly
  - "wav2lip" is in SUPPORTED_MODELS
  - DEFAULT_MODEL is "wav2lip"
  - get_inference_backend("wav2lip") returns _Wav2LipBackend
  - get_inference_backend("latentsync") still works (does not crash at import)
  - missing Wav2Lip directory / checkpoint raises PipelineError
  - command construction uses sys.executable
  - unsupported model names raise PipelineError

No real inference is executed.
"""

import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers — make the latentsync import in get_inference_backend survivable
# even though the latentsync package is not installed in the test environment.
# ---------------------------------------------------------------------------

def _inject_latentsync_stub():
    """Inject a minimal latentsync stub into sys.modules so that
    importlib.import_module('latentsync') doesn't raise ModuleNotFoundError
    when we test the LatentSync path."""
    pkg = types.ModuleType("latentsync")
    pipeline_mod = types.ModuleType("latentsync.pipelines.lipsync_pipeline")
    pipeline_mod.LipsyncPipeline = MagicMock()
    sys.modules.setdefault("latentsync", pkg)
    sys.modules.setdefault("latentsync.pipelines", types.ModuleType("latentsync.pipelines"))
    sys.modules.setdefault("latentsync.pipelines.lipsync_pipeline", pipeline_mod)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_model_is_wav2lip(self):
        from src.config import DEFAULT_MODEL
        assert DEFAULT_MODEL == "wav2lip"

    def test_wav2lip_in_supported_models(self):
        from src.config import SUPPORTED_MODELS
        assert "wav2lip" in SUPPORTED_MODELS

    def test_latentsync_still_in_supported_models(self):
        from src.config import SUPPORTED_MODELS
        assert "latentsync" in SUPPORTED_MODELS

    def test_wav2lip_dir_is_string(self):
        from src.config import WAV2LIP_DIR
        assert isinstance(WAV2LIP_DIR, str)
        assert len(WAV2LIP_DIR) > 0

    def test_wav2lip_checkpoint_path_is_string(self):
        from src.config import WAV2LIP_CHECKPOINT_PATH
        assert isinstance(WAV2LIP_CHECKPOINT_PATH, str)
        assert WAV2LIP_CHECKPOINT_PATH.endswith("wav2lip_gan.pth")

    def test_wav2lip_checkpoint_inside_wav2lip_dir(self):
        from src.config import WAV2LIP_DIR, WAV2LIP_CHECKPOINT_PATH
        assert WAV2LIP_CHECKPOINT_PATH.startswith(WAV2LIP_DIR)


# ---------------------------------------------------------------------------
# _Wav2LipBackend unit tests
# ---------------------------------------------------------------------------

class TestWav2LipBackend:

    def _make_backend(self, wav2lip_dir: str, checkpoint_path: str):
        from src.pipeline import _Wav2LipBackend
        return _Wav2LipBackend(wav2lip_dir=wav2lip_dir, checkpoint_path=checkpoint_path)

    def test_missing_wav2lip_dir_raises(self, tmp_path):
        from src.exceptions import PipelineError
        backend = self._make_backend(
            wav2lip_dir=str(tmp_path / "nonexistent_wav2lip"),
            checkpoint_path=str(tmp_path / "nonexistent.pth"),
        )
        with pytest.raises(PipelineError, match="Wav2Lip directory not found"):
            backend.run(
                image_path=str(tmp_path / "img.jpg"),
                audio_path=str(tmp_path / "audio.wav"),
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_missing_inference_script_raises(self, tmp_path):
        from src.exceptions import PipelineError
        wav2lip_dir = tmp_path / "Wav2Lip"
        wav2lip_dir.mkdir()
        # No inference.py inside
        backend = self._make_backend(
            wav2lip_dir=str(wav2lip_dir),
            checkpoint_path=str(wav2lip_dir / "checkpoints" / "wav2lip_gan.pth"),
        )
        with pytest.raises(PipelineError, match="inference.py not found"):
            backend.run(
                image_path=str(tmp_path / "img.jpg"),
                audio_path=str(tmp_path / "audio.wav"),
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_missing_checkpoint_raises(self, tmp_path):
        from src.exceptions import PipelineError
        wav2lip_dir = tmp_path / "Wav2Lip"
        wav2lip_dir.mkdir()
        (wav2lip_dir / "inference.py").write_text("# stub")
        # No checkpoint file
        backend = self._make_backend(
            wav2lip_dir=str(wav2lip_dir),
            checkpoint_path=str(wav2lip_dir / "checkpoints" / "wav2lip_gan.pth"),
        )
        with pytest.raises(PipelineError, match="checkpoint not found"):
            backend.run(
                image_path=str(tmp_path / "img.jpg"),
                audio_path=str(tmp_path / "audio.wav"),
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_missing_image_raises(self, tmp_path):
        from src.exceptions import PipelineError
        wav2lip_dir = tmp_path / "Wav2Lip"
        wav2lip_dir.mkdir()
        (wav2lip_dir / "inference.py").write_text("# stub")
        ckpt = wav2lip_dir / "checkpoints"
        ckpt.mkdir()
        (ckpt / "wav2lip_gan.pth").write_bytes(b"fake_weight")
        backend = self._make_backend(
            wav2lip_dir=str(wav2lip_dir),
            checkpoint_path=str(ckpt / "wav2lip_gan.pth"),
        )
        with pytest.raises(PipelineError, match="Face image not found"):
            backend.run(
                image_path=str(tmp_path / "missing_image.jpg"),
                audio_path=str(tmp_path / "audio.wav"),
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_missing_audio_raises(self, tmp_path):
        from src.exceptions import PipelineError
        wav2lip_dir = tmp_path / "Wav2Lip"
        wav2lip_dir.mkdir()
        (wav2lip_dir / "inference.py").write_text("# stub")
        ckpt = wav2lip_dir / "checkpoints"
        ckpt.mkdir()
        (ckpt / "wav2lip_gan.pth").write_bytes(b"fake_weight")
        img = tmp_path / "face.jpg"
        img.write_bytes(b"fake_image")
        backend = self._make_backend(
            wav2lip_dir=str(wav2lip_dir),
            checkpoint_path=str(ckpt / "wav2lip_gan.pth"),
        )
        with pytest.raises(PipelineError, match="Audio file not found"):
            backend.run(
                image_path=str(img),
                audio_path=str(tmp_path / "missing_audio.wav"),
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_command_uses_sys_executable(self, tmp_path):
        """Verify subprocess is called with sys.executable as the Python interpreter."""
        from src.pipeline import _Wav2LipBackend

        wav2lip_dir = tmp_path / "Wav2Lip"
        wav2lip_dir.mkdir()
        (wav2lip_dir / "inference.py").write_text("# stub")
        ckpt = wav2lip_dir / "checkpoints"
        ckpt.mkdir()
        (ckpt / "wav2lip_gan.pth").write_bytes(b"fake_weight")
        img = tmp_path / "face.jpg"
        img.write_bytes(b"fake_image")
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"fake_audio")
        out = tmp_path / "out.mp4"
        out.write_bytes(b"fake_video_output")  # simulate successful write

        backend = _Wav2LipBackend(
            wav2lip_dir=str(wav2lip_dir),
            checkpoint_path=str(ckpt / "wav2lip_gan.pth"),
        )

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = ""
        fake_result.stderr = ""

        with patch("subprocess.run", return_value=fake_result) as mock_run:
            backend.run(
                image_path=str(img),
                audio_path=str(audio_file),
                output_path=str(out),
            )
            called_cmd = mock_run.call_args[0][0]
            assert called_cmd[0] == sys.executable, (
                f"Expected sys.executable ({sys.executable!r}) "
                f"but got {called_cmd[0]!r}"
            )
            assert "--static" in called_cmd
            assert "True" in called_cmd
            assert "--fps" in called_cmd
            assert "25" in called_cmd

    def test_nonzero_returncode_raises_pipeline_error(self, tmp_path):
        from src.exceptions import PipelineError
        from src.pipeline import _Wav2LipBackend

        wav2lip_dir = tmp_path / "Wav2Lip"
        wav2lip_dir.mkdir()
        (wav2lip_dir / "inference.py").write_text("# stub")
        ckpt = wav2lip_dir / "checkpoints"
        ckpt.mkdir()
        (ckpt / "wav2lip_gan.pth").write_bytes(b"fake_weight")
        img = tmp_path / "face.jpg"
        img.write_bytes(b"fake_image")
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"fake_audio")

        backend = _Wav2LipBackend(
            wav2lip_dir=str(wav2lip_dir),
            checkpoint_path=str(ckpt / "wav2lip_gan.pth"),
        )

        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = ""
        fake_result.stderr = "Some error from inference.py"

        with patch("subprocess.run", return_value=fake_result):
            with pytest.raises(PipelineError, match="exit code 1"):
                backend.run(
                    image_path=str(img),
                    audio_path=str(audio_file),
                    output_path=str(tmp_path / "out.mp4"),
                )

    def test_empty_output_file_raises(self, tmp_path):
        from src.exceptions import PipelineError
        from src.pipeline import _Wav2LipBackend

        wav2lip_dir = tmp_path / "Wav2Lip"
        wav2lip_dir.mkdir()
        (wav2lip_dir / "inference.py").write_text("# stub")
        ckpt = wav2lip_dir / "checkpoints"
        ckpt.mkdir()
        (ckpt / "wav2lip_gan.pth").write_bytes(b"fake_weight")
        img = tmp_path / "face.jpg"
        img.write_bytes(b"fake_image")
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"fake_audio")
        out = tmp_path / "out.mp4"
        out.write_bytes(b"")  # empty

        backend = _Wav2LipBackend(
            wav2lip_dir=str(wav2lip_dir),
            checkpoint_path=str(ckpt / "wav2lip_gan.pth"),
        )

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = ""
        fake_result.stderr = ""

        with patch("subprocess.run", return_value=fake_result):
            with pytest.raises(PipelineError, match="empty output file"):
                backend.run(
                    image_path=str(img),
                    audio_path=str(audio_file),
                    output_path=str(out),
                )


# ---------------------------------------------------------------------------
# get_inference_backend tests
# ---------------------------------------------------------------------------

class TestGetInferenceBackend:

    def setup_method(self):
        # Clear lru_cache between tests
        from src.pipeline import get_inference_backend
        get_inference_backend.cache_clear()

    def test_wav2lip_returns_wav2lip_backend(self):
        from src.pipeline import get_inference_backend, _Wav2LipBackend
        backend = get_inference_backend("wav2lip")
        assert isinstance(backend, _Wav2LipBackend)

    def test_wav2lip_case_insensitive(self):
        from src.pipeline import get_inference_backend, _Wav2LipBackend
        get_inference_backend.cache_clear()
        backend = get_inference_backend("Wav2Lip")
        assert isinstance(backend, _Wav2LipBackend)

    def test_unsupported_model_raises(self):
        from src.exceptions import PipelineError
        from src.pipeline import get_inference_backend
        get_inference_backend.cache_clear()
        with pytest.raises(PipelineError, match="Unsupported model"):
            get_inference_backend("unknown_model_xyz")

    def test_latentsync_attempt_raises_on_missing_package(self):
        """LatentSync backend raises PipelineError when the package isn't installed
        (which is the expected state in this environment)."""
        from src.exceptions import PipelineError
        from src.pipeline import get_inference_backend
        get_inference_backend.cache_clear()
        # Patch _resolve_checkpoint_path to avoid 'no checkpoint' error before
        # we reach the importlib check for 'latentsync'.
        with patch("src.pipeline._resolve_checkpoint_path", return_value="/fake/ckpt.pt"):
            with patch("src.pipeline._resolve_whisper_path", return_value="/fake/tiny.pt"):
                with patch("src.pipeline._resolve_config_path", return_value="/fake/cfg.yaml"):
                    with patch("src.pipeline._resolve_scheduler_config_path", return_value="/fake/sched"):
                        # latentsync is not installed — PipelineError is expected
                        with pytest.raises(PipelineError):
                            get_inference_backend("latentsync")
