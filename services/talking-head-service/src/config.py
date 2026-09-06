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

# Models recognized by the service configuration in this repository.
# The repository's intended backend is the official LatentSync implementation
# using the Hugging Face checkpoint `ByteDance/LatentSync-1.6` and the
# `latentsync_unet.pt` model file.
DEFAULT_MODEL = "latentsync"
SUPPORTED_MODELS = {DEFAULT_MODEL}
