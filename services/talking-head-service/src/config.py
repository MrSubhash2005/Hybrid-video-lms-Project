import os
from pathlib import Path

# Base directory for the talking head service
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage configuration
STORAGE_DIR = BASE_DIR / "storage" / "jobs"

# Runtime configuration for optional model backends.
MODEL_CACHE_DIR = Path(
    os.getenv("MODEL_CACHE_DIR", str(BASE_DIR / "models"))
).expanduser()

LATENTSYNC_CHECKPOINT_PATH = os.getenv(
    "LATENTSYNC_CHECKPOINT_PATH",
    os.getenv("TALKING_HEAD_CHECKPOINT_PATH"),
)
LATENTSYNC_WHISPER_PATH = os.getenv(
    "LATENTSYNC_WHISPER_PATH",
    os.getenv("WHISPER_MODEL_PATH"),
)
LATENTSYNC_CONFIG_PATH = os.getenv(
    "LATENTSYNC_CONFIG_PATH",
    os.getenv("LATENTSYNC_UNET_CONFIG_PATH"),
)
LATENTSYNC_SCHEDULER_CONFIG_PATH = os.getenv(
    "LATENTSYNC_SCHEDULER_CONFIG_PATH",
    os.getenv("LATENTSYNC_SCHEDULER_PATH"),
)
LATENTSYNC_CONFIG_DIR = os.getenv("LATENTSYNC_CONFIG_DIR", str(BASE_DIR / "configs"))
LATENTSYNC_MODEL_REPO = os.getenv("LATENTSYNC_MODEL_REPO", "ByteDance/LatentSync-1.6")
TALKING_HEAD_DEVICE = os.getenv("TALKING_HEAD_DEVICE", "cpu")

# Wav2Lip backend configuration.
# WAV2LIP_DIR defaults to a sibling "Wav2Lip" directory next to the repo root
# (i.e. <repo_root>/Wav2Lip).  Override via environment variable for any other layout.
WAV2LIP_DIR = os.getenv(
    "WAV2LIP_DIR",
    str(Path(__file__).resolve().parent.parent.parent.parent / "Wav2Lip"),
)
WAV2LIP_CHECKPOINT_PATH = os.getenv(
    "WAV2LIP_CHECKPOINT_PATH",
    str(Path(WAV2LIP_DIR) / "checkpoints" / "wav2lip_gan.pth"),
)

# File validation constraints
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
ALLOWED_AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/x-m4a",
    "audio/m4a",
}

# Models recognized by the service.
# "wav2lip" is the default because the local Wav2Lip installation has been
# verified to produce correct lip-sync output.  "latentsync" remains available
# as a legacy/fallback backend when the official LatentSync runtime is present.
DEFAULT_MODEL = "wav2lip"
SUPPORTED_MODELS = {"wav2lip", "latentsync"}
