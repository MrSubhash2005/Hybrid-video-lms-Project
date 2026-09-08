import importlib
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2

from src.config import (
    DEFAULT_MODEL,
    LATENTSYNC_CHECKPOINT_PATH,
    LATENTSYNC_CONFIG_DIR,
    LATENTSYNC_CONFIG_PATH,
    LATENTSYNC_MODEL_REPO,
    LATENTSYNC_SCHEDULER_CONFIG_PATH,
    LATENTSYNC_WHISPER_PATH,
    MODEL_CACHE_DIR,
    STORAGE_DIR,
    TALKING_HEAD_DEVICE,
)
from src.exceptions import PipelineError
from src.logging_config import get_logger
from src.job_store import update_job, get_job

logger = get_logger("pipeline")


def validate_video_output(output_path: str) -> bool:
    """Validates that a generated video file exists and can be decoded."""
    path = Path(output_path)
    if not path.is_file():
        logger.error(f"Output validation failed: file '{output_path}' does not exist.")
        return False

    if path.stat().st_size == 0:
        logger.error(f"Output validation failed: file '{output_path}' is 0 bytes.")
        return False

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error(
            f"Output validation failed: cv2 cannot open video '{output_path}'."
        )
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.error(
            f"Output validation failed: cv2 failed to read frame from '{output_path}'."
        )
        return False

    logger.info(f"Output validation passed for video '{output_path}'.")
    return True


def _resolve_service_path(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        return None
    if isinstance(path_value, (list, tuple)):
        return [
            _resolve_service_path(str(item)) for item in path_value if str(item).strip()
        ]

    candidate = Path(str(path_value))
    if candidate.exists():
        return str(candidate)

    repo_root = Path(__file__).resolve().parent.parent
    repo_relative = repo_root / candidate
    if repo_relative.exists():
        return str(repo_relative)

    if not candidate.is_absolute():
        cwd_relative = Path.cwd() / candidate
        if cwd_relative.exists():
            return str(cwd_relative)

    return str(candidate)


def _resolve_default_output_path(
    job_id: str,
    output_path: Optional[str] = None,
) -> str:
    if output_path:
        final_path = Path(output_path)
        if not final_path.is_absolute():
            final_path = Path(__file__).resolve().parent.parent / final_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        return str(final_path)

    job_output_dir = STORAGE_DIR / job_id / "outputs"
    job_output_dir.mkdir(parents=True, exist_ok=True)
    return str(job_output_dir / "avatar.mp4")


def _resolve_checkpoint_path(model_name: str) -> str:
    normalized = (model_name or DEFAULT_MODEL).strip().lower()
    if normalized != "latentsync":
        raise PipelineError(
            f"Unsupported model '{model_name}' in the current talking-head pipeline. "
            "The repository only targets 'latentsync'."
        )

    candidates: list[str] = []
    if LATENTSYNC_CHECKPOINT_PATH:
        candidates.append(LATENTSYNC_CHECKPOINT_PATH)
    if MODEL_CACHE_DIR:
        candidates.extend(
            [
                str(MODEL_CACHE_DIR / "latentsync" / "latentsync_unet.pt"),
                str(MODEL_CACHE_DIR / "latentsync" / "latentsync.pth"),
                str(
                    MODEL_CACHE_DIR
                    / "latentsync"
                    / "checkpoints"
                    / "latentsync_unet.pt"
                ),
                str(MODEL_CACHE_DIR / "latentsync" / "checkpoints" / "latest.pth"),
                str(MODEL_CACHE_DIR / "latentsync" / "model.pth"),
                str(MODEL_CACHE_DIR / "latentsync" / "checkpoint.pt"),
                str(MODEL_CACHE_DIR / "latentsync" / "weights.pt"),
            ]
        )

    seen = set()
    for candidate in candidates:
        normalized_candidate = str(candidate).strip()
        if not normalized_candidate or normalized_candidate in seen:
            continue
        seen.add(normalized_candidate)
        checkpoint = Path(normalized_candidate)
        if checkpoint.exists() and checkpoint.is_file():
            return str(checkpoint)

    raise PipelineError(
        "LatentSync inference checkpoint is not available. The repository expects the "
        "official LatentSync runtime and checkpoint file named `latentsync_unet.pt` "
        "plus Whisper model assets, configured through LATENTSYNC_CHECKPOINT_PATH, "
        "LATENTSYNC_WHISPER_PATH, and the official Hugging Face model repo."
    )


def _resolve_whisper_path() -> str:
    candidates: list[str] = []
    if LATENTSYNC_WHISPER_PATH:
        candidates.append(LATENTSYNC_WHISPER_PATH)
    if MODEL_CACHE_DIR:
        candidates.extend(
            [
                str(
                    MODEL_CACHE_DIR
                    / "latentsync"
                    / "checkpoints"
                    / "whisper"
                    / "tiny.pt"
                ),
                str(MODEL_CACHE_DIR / "latentsync" / "whisper" / "tiny.pt"),
                str(
                    MODEL_CACHE_DIR
                    / "latentsync"
                    / "checkpoints"
                    / "whisper"
                    / "small.pt"
                ),
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return str(path)
    raise PipelineError(
        "LatentSync Whisper audio encoder checkpoint is missing. Configure "
        "LATENTSYNC_WHISPER_PATH or place the Whisper model under "
        "MODEL_CACHE_DIR/latentsync/checkpoints/whisper/."
    )


def _resolve_config_path() -> str:
    candidates: list[str] = []
    if LATENTSYNC_CONFIG_PATH:
        candidates.append(LATENTSYNC_CONFIG_PATH)
    if LATENTSYNC_CONFIG_DIR:
        candidates.extend(
            [
                str(Path(LATENTSYNC_CONFIG_DIR) / "unet" / "stage2_512.yaml"),
                str(Path(LATENTSYNC_CONFIG_DIR) / "stage2_512.yaml"),
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return str(path)
    raise PipelineError(
        "LatentSync UNet config is missing. Configure LATENTSYNC_CONFIG_PATH or "
        "LATENTSYNC_CONFIG_DIR to point at the official repo config directory."
    )


def _resolve_scheduler_config_path() -> str:
    candidates: list[str] = []
    if LATENTSYNC_SCHEDULER_CONFIG_PATH:
        candidates.append(LATENTSYNC_SCHEDULER_CONFIG_PATH)
    if LATENTSYNC_CONFIG_PATH:
        config_path = Path(LATENTSYNC_CONFIG_PATH)
        if config_path.is_file() and config_path.suffix.lower() == ".yaml":
            candidates.extend(
                [
                    str(config_path.parent / "scheduler_config.json"),
                    str(config_path.parent.parent / "scheduler_config.json"),
                ]
            )
    if LATENTSYNC_CONFIG_DIR:
        config_dir = Path(LATENTSYNC_CONFIG_DIR)
        candidates.extend(
            [
                str(config_dir / "scheduler_config.json"),
                str(config_dir.parent / "scheduler_config.json"),
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and (path.is_file() or path.is_dir()):
            return str(path)
    raise PipelineError(
        "LatentSync scheduler config is missing. Configure LATENTSYNC_SCHEDULER_CONFIG_PATH "
        "or LATENTSYNC_CONFIG_DIR to the official LatentSync config directory."
    )


def _merge_audio_chunks(audio_path: str) -> str:
    """Accept a single audio file or chunk directory/list and return a valid audio file."""
    raw = audio_path
    if isinstance(raw, (list, tuple)):
        chunk_paths = [str(p) for p in raw if str(p).strip()]
        if not chunk_paths:
            raise PipelineError("No audio chunk paths were provided for inference.")
        if len(chunk_paths) == 1:
            return chunk_paths[0]
        merged = (
            Path(tempfile.mkdtemp(prefix="talking_head_audio_")) / "merged_input.wav"
        )
        list_path = (
            Path(tempfile.mkdtemp(prefix="talking_head_audio_list_")) / "chunks.txt"
        )
        with list_path.open("w", encoding="utf-8") as handle:
            for chunk in chunk_paths:
                handle.write(f"file '{Path(chunk).resolve().as_posix()}'\n")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(merged),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return str(merged)

    path = Path(raw)
    if path.is_dir():
        chunk_files = sorted(path.glob("*"))
        audio_candidates = [
            p
            for p in chunk_files
            if p.is_file()
            and p.suffix.lower() in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        ]
        if not audio_candidates:
            raise PipelineError(
                f"Audio directory '{raw}' does not contain supported audio files."
            )
        return _merge_audio_chunks([str(p) for p in audio_candidates])

    if not path.exists():
        raise PipelineError(f"Audio input file not found: '{audio_path}'")
    return str(path)


def _make_reference_video(image_path: str, output_video_path: str) -> str:
    """Convert the repo's still face image into a minimal video input accepted by LatentSync."""
    source = Path(image_path)
    if not source.exists() or not source.is_file():
        raise PipelineError(f"Face input image not found: '{image_path}'")

    frame = cv2.imread(str(source))
    if frame is None:
        raise PipelineError(f"Face input image could not be decoded: '{image_path}'")

    height, width = frame.shape[:2]
    target_width = max(64, width)
    target_height = max(64, height)
    reference_fd, reference_path = tempfile.mkstemp(
        prefix="latentsync_ref_", suffix=".mp4"
    )
    os.close(reference_fd)
    Path(reference_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(reference_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (target_width, target_height),
    )
    if not writer.isOpened():
        raise PipelineError(
            "OpenCV could not initialize a temporary MP4 reference video for LatentSync."
        )

    for _ in range(25):
        writer.write(frame)
    writer.release()
    return str(reference_path)


class _OfficialLatentSyncBackend:
    def __init__(
        self,
        checkpoint_path: str,
        whisper_path: str,
        config_path: str,
        scheduler_config_path: str,
        device: str,
    ):
        self.checkpoint_path = checkpoint_path
        self.whisper_path = whisper_path
        self.config_path = config_path
        self.scheduler_config_path = scheduler_config_path
        self.device = device
        self._pipeline = None

    def _resolve_device(self) -> str:
        normalized = (self.device or TALKING_HEAD_DEVICE or "cpu").strip().lower()
        if normalized in {"cuda", "gpu", "cuda:0", "gpu:0"}:
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from diffusers.models import AutoencoderKL
            from diffusers.schedulers import DDIMScheduler
            from latentsync.models.unet import UNet3DConditionModel
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
            from latentsync.whisper.audio2feature import Audio2Feature
        except ModuleNotFoundError as exc:
            raise PipelineError(
                "The official LatentSync runtime is not installed. Install the package from "
                f"{LATENTSYNC_MODEL_REPO} with its required dependencies before inference."
            ) from exc

        if not Path(self.checkpoint_path).exists():
            raise PipelineError(
                f"LatentSync checkpoint not found at '{self.checkpoint_path}'. Check "
                "LATENTSYNC_CHECKPOINT_PATH or MODEL_CACHE_DIR/latentsync."
            )
        if not Path(self.whisper_path).exists():
            raise PipelineError(
                f"LatentSync Whisper checkpoint not found at '{self.whisper_path}'. Check "
                "LATENTSYNC_WHISPER_PATH or MODEL_CACHE_DIR/latentsync/checkpoints/whisper/."
            )
        if not Path(self.config_path).exists():
            raise PipelineError(
                f"LatentSync UNet config not found at '{self.config_path}'. Check "
                "LATENTSYNC_CONFIG_PATH or LATENTSYNC_CONFIG_DIR."
            )
        if not Path(self.scheduler_config_path).exists():
            raise PipelineError(
                f"LatentSync scheduler config not found at '{self.scheduler_config_path}'. Check "
                "LATENTSYNC_SCHEDULER_CONFIG_PATH or LATENTSYNC_CONFIG_DIR."
            )

        device = self._resolve_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        vae = AutoencoderKL.frompretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=dtype
        )
        audio_encoder = Audio2Feature(model_path=self.whisper_path, device=device)
        unet = UNet3DConditionModel.from_pretrained(
            self.checkpoint_path, torch_dtype=dtype
        )

        scheduler_source = Path(self.scheduler_config_path)
        if scheduler_source.is_file() and scheduler_source.suffix.lower() == ".json":
            scheduler_root = scheduler_source.parent
        else:
            scheduler_root = scheduler_source
        scheduler = DDIMScheduler.from_pretrained(str(scheduler_root))

        self._pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to(device)
        return self._pipeline

    def run(
        self,
        *,
        image_path: str,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        if progress_callback is not None:
            progress_callback(15.0)

        reference_video = _make_reference_video(image_path, output_path)
        try:
            if progress_callback is not None:
                progress_callback(35.0)

            pipeline = self._load_pipeline()
            if progress_callback is not None:
                progress_callback(60.0)

            pipeline(
                video_path=reference_video,
                audio_path=audio_path,
                video_out_path=output_path,
            )

            if progress_callback is not None:
                progress_callback(95.0)

            resolved = Path(output_path)
            if resolved.exists() and resolved.is_file():
                return str(resolved)

            raise PipelineError(
                "LatentSync backend executed without producing a usable MP4 output at the "
                f"requested path '{output_path}'."
            )
        finally:
            try:
                if Path(reference_video).exists():
                    Path(reference_video).unlink()
            except Exception:
                pass


@lru_cache(maxsize=4)
def get_inference_backend(model_name: str):
    """Lazy-load the configured model backend when the runtime is installed."""
    normalized = (model_name or DEFAULT_MODEL).strip().lower()
    if normalized != "latentsync":
        raise PipelineError(
            f"Unsupported model '{model_name}'. "
            "The repository is configured for 'latentsync'."
        )

    checkpoint_path = _resolve_checkpoint_path(normalized)
    whisper_path = _resolve_whisper_path()
    config_path = _resolve_config_path()
    scheduler_config_path = _resolve_scheduler_config_path()

    try:
        importlib.import_module("latentsync")
    except ModuleNotFoundError as exc:
        raise PipelineError(
            "LatentSync inference engine is not installed in this environment. Install the "
            "official LatentSync package and configure LATENTSYNC_CHECKPOINT_PATH, "
            "LATENTSYNC_WHISPER_PATH, and LATENTSYNC_CONFIG_PATH before running the pipeline."
        ) from exc

    try:
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    except ModuleNotFoundError as exc:
        raise PipelineError(
            "The official LatentSync pipeline package is not installed or is incomplete. "
            "Install the repository from the official source and configure its model assets."
        ) from exc

    if not callable(LipsyncPipeline):
        raise PipelineError(
            "Official LatentSync pipeline is available but does not expose LipsyncPipeline."
        )

    return _OfficialLatentSyncBackend(
        checkpoint_path=checkpoint_path,
        whisper_path=whisper_path,
        config_path=config_path,
        scheduler_config_path=scheduler_config_path,
        device=TALKING_HEAD_DEVICE,
    )


def run_talking_head_pipeline(
<<<<<<< HEAD
    job_id: str, image_path: str, audio_path: str, model: str, enhancer: bool
=======
    job_id: str,
    image_path: str,
    audio_path: str,
    model: str,
    enhancer: bool,
    output_path: Optional[str] = None,
    job_context: Optional[Dict[str, Any]] = None,
>>>>>>> 1569b7e62f371dd03fb033169e9fae728c93123d
):
    """Execute a talking-head inference job with lazy model initialization."""
    logger.info(
        f"Pipeline started for job {job_id} using model '{model}' "
        f"(enhancer={enhancer})."
    )

    # Fetch from SQLite instead of jobs_db dictionary
    job = get_job(job_id)
    if not job:
        logger.error(f"Pipeline error: job {job_id} not found in state store.")
        return

    output_target = _resolve_default_output_path(job_id, output_path)
    
    # Prepare updates for the database
    updates = {
        "status": "processing",
        "progress": 10.0,
        "output_url": output_target
    }
    if job_context:
        updates.update(job_context)

    # Apply all updates to SQLite
    update_job(job_id, **updates)

    def update_progress(value: float) -> None:
        clamped_value = max(0.0, min(100.0, float(value)))
        update_job(job_id, progress=clamped_value)

    try:
        resolved_image_path = _resolve_service_path(image_path)
        if not resolved_image_path or not Path(resolved_image_path).exists():
            raise PipelineError(f"Input face image does not exist: '{image_path}'")

        resolved_audio_path = _resolve_service_path(audio_path)
        audio_input = _merge_audio_chunks(resolved_audio_path)
        update_progress(15.0)

        backend = get_inference_backend(model)
        backend_output = backend.run(
            image_path=resolved_image_path,
            audio_path=audio_input,
            output_path=output_target,
            progress_callback=lambda value: update_progress(
                15.0 + (value / 100.0) * 70.0
            ),
        )

        if not Path(backend_output).exists():
            raise PipelineError(
                f"Inference completed but no MP4 was generated at '{backend_output}'."
            )

        if not validate_video_output(backend_output):
            raise PipelineError(
                f"Generated output '{backend_output}' is not a valid MP4 or "
                "failed validation."
            )

        # Success! Update the database
        update_job(
            job_id,
            status="completed",
            progress=100.0,
            output_url=str(Path(backend_output).resolve()),
            completed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            error_message=None
        )
        logger.info(
            f"Pipeline completed successfully for job {job_id}: {backend_output}"
        )
        return backend_output

    except PipelineError as exc:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        update_job(
            job_id,
            status="failed",
            progress=0.0,
            completed_at=timestamp,
            error_message=str(exc),
            output_url=None
        )
        logger.error(f"Pipeline failed for job {job_id}: {exc}")
        return None

    except Exception as exc:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
<<<<<<< HEAD

        # Update failed status in SQLite
=======
>>>>>>> 1569b7e62f371dd03fb033169e9fae728c93123d
        update_job(
            job_id,
            status="failed",
            progress=0.0,
            completed_at=timestamp,
            error_message=str(exc),
<<<<<<< HEAD
        )

        logger.error(f"Pipeline failed for job {job_id}: {exc}")
=======
            output_url=None
        )
        logger.exception(f"Unexpected pipeline failure for job {job_id}: {exc}")
        return None
>>>>>>> 1569b7e62f371dd03fb033169e9fae728c93123d
