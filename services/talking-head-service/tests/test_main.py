import io
import sys
import uuid
import wave
from pathlib import Path

# Add talking-head-service root to sys.path for repo-root execution.
SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.config import STORAGE_DIR  # noqa: E402
from src.main import (  # noqa: E402
    FEMALE_AVATAR,
    MALE_AVATAR,
    VOICE_GENDER,
    app,
    get_avatar_for_voice,
)
from src.pipeline import (  # noqa: E402
    run_talking_head_pipeline,
    validate_video_output,
)
from src.storage import create_job, get_job  # noqa: E402


client = TestClient(app)


def unique_job_id(prefix: str) -> str:
    """Create a unique job ID for SQLite-backed tests."""
    return f"{prefix}_{uuid.uuid4().hex}"


def get_valid_image_bytes() -> bytes:
    """Generate a small valid PNG image in memory."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    cv2.putText(
        img,
        "Test Face",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )

    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


def get_valid_wav_bytes() -> bytes:
    """Generate a small valid WAV audio file in memory."""
    buf = io.BytesIO()

    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 3200)

    return buf.getvalue()


def make_uploads():
    """Create valid multipart uploads for API tests."""
    return [
        (
            "face_image",
            ("face.png", get_valid_image_bytes(), "image/png"),
        ),
        (
            "audio",
            ("audio.wav", get_valid_wav_bytes(), "audio/wav"),
        ),
    ]


def post_generation(voice, avatar=None):
    """Submit an avatar generation request."""
    data = {
        "voice": voice,
        "model": "latentsync",
        "enhancer": "true",
    }

    if avatar is not None:
        data["avatar"] = avatar

    return client.post(
        "/api/v1/avatar/generate",
        files=make_uploads(),
        data=data,
    )


def create_test_job(job_id: str, **overrides):
    """Create a persistent SQLite job for direct pipeline tests."""
    job = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "estimated_time_remaining": 30.0,
        "created_at": "2024-01-01T00:00:00Z",
        "completed_at": None,
        "output_path": None,
        "output_url": None,
        "error_message": None,
        "voice": "en-US-ChristopherNeural",
        "avatar": "male",
        "gender": "male",
    }

    job.update(overrides)

    create_job(job)
    return job


def create_valid_mp4(output_path: Path):
    """Create a small valid MP4 fixture using OpenCV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 160
    height = 120
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    assert writer.isOpened()

    frame = np.zeros((height, width, 3), dtype=np.uint8)

    for _ in range(10):
        writer.write(frame)

    writer.release()

    assert output_path.exists()
    assert output_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Voice / avatar mapping tests
# ---------------------------------------------------------------------------


def test_male_voices_select_male_avatar():
    male_voices = (
        "en-US-ChristopherNeural",
        "en-GB-RyanNeural",
        "en-IN-PrabhatNeural",
    )

    for voice in male_voices:
        assert VOICE_GENDER[voice] == "male"
        assert get_avatar_for_voice(voice) == MALE_AVATAR


def test_female_voices_select_female_avatar():
    female_voices = (
        "en-US-JennyNeural",
        "en-GB-SoniaNeural",
        "en-IN-NeerjaNeural",
    )

    for voice in female_voices:
        assert VOICE_GENDER[voice] == "female"
        assert get_avatar_for_voice(voice) == FEMALE_AVATAR


def test_male_voice_matches_male_avatar():
    voice = "en-US-ChristopherNeural"

    assert VOICE_GENDER[voice] == "male"
    assert get_avatar_for_voice(voice) == MALE_AVATAR


def test_default_avatar_matches_male_voice_when_avatar_missing():
    voice = "en-US-ChristopherNeural"

    assert VOICE_GENDER[voice] == "male"
    assert get_avatar_for_voice(voice) == MALE_AVATAR


def test_female_voice_matches_female_avatar():
    voice = "en-US-JennyNeural"

    assert VOICE_GENDER[voice] == "female"
    assert get_avatar_for_voice(voice) == FEMALE_AVATAR


def test_default_avatar_matches_female_voice_when_avatar_missing():
    voice = "en-US-JennyNeural"

    assert VOICE_GENDER[voice] == "female"
    assert get_avatar_for_voice(voice) == FEMALE_AVATAR


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


def test_generate_avatar_returns_accepted():
    response = post_generation("en-US-ChristopherNeural")

    assert response.status_code == 202

    body = response.json()

    assert "job_id" in body
    assert body["status"] == "queued"
    assert "created_at" in body
    assert "message" in body


def test_generate_avatar_with_explicit_male_avatar():
    response = post_generation(
        "en-US-ChristopherNeural",
        MALE_AVATAR,
    )

    assert response.status_code == 202

    body = response.json()

    assert "job_id" in body
    assert body["status"] == "queued"


def test_generate_avatar_with_explicit_female_avatar():
    response = post_generation(
        "en-US-JennyNeural",
        FEMALE_AVATAR,
    )

    assert response.status_code == 202

    body = response.json()

    assert "job_id" in body
    assert body["status"] == "queued"


def test_job_status_returns_sqlite_job():
    job_id = unique_job_id("job_status")

    create_test_job(job_id)

    response = client.get(
        f"/api/v1/avatar/jobs/{job_id}"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["job_id"] == job_id
    assert body["status"] == "queued"
    assert body["progress"] == 0.0


def test_missing_job_returns_404():
    job_id = unique_job_id("missing")

    response = client.get(
        f"/api/v1/avatar/jobs/{job_id}"
    )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# SQLite persistence / output URL tests
# ---------------------------------------------------------------------------


def test_completed_job_survives_status_lookup():
    """Verify a completed SQLite job remains available to the API."""
    job_id = unique_job_id("job_persistence")

    output_path = (
        STORAGE_DIR
        / job_id
        / "outputs"
        / "avatar.mp4"
    )

    create_valid_mp4(output_path)

    create_test_job(
        job_id,
        status="completed",
        progress=100.0,
        completed_at="2024-01-01T00:01:00Z",
        output_path=str(
            output_path.resolve()
        ),
        output_url=(
            f"/api/v1/outputs/"
            f"{job_id}/outputs/avatar.mp4"
        ),
    )

    stored_job = get_job(job_id)

    assert stored_job is not None
    assert stored_job["status"] == "completed"
    assert stored_job["progress"] == 100.0
    assert (
        stored_job["completed_at"]
        == "2024-01-01T00:01:00Z"
    )
    assert stored_job["output_path"] == str(
        output_path.resolve()
    )
    assert (
        stored_job["output_url"]
        == f"/api/v1/outputs/{job_id}/outputs/avatar.mp4"
    )

    response = client.get(
        f"/api/v1/avatar/jobs/{job_id}"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["job_id"] == job_id
    assert body["status"] == "completed"
    assert body["progress"] == 100.0
    assert (
        body["completed_at"]
        == "2024-01-01T00:01:00Z"
    )
    assert body["output_path"] == str(
        output_path.resolve()
    )
    assert (
        body["output_url"]
        == f"/api/v1/outputs/{job_id}/outputs/avatar.mp4"
    )

    output_response = client.get(
        body["output_url"]
    )

    assert output_response.status_code == 200
    assert len(output_response.content) > 0


# ---------------------------------------------------------------------------
# Pipeline validation tests
# ---------------------------------------------------------------------------


def test_validate_video_output_accepts_valid_mp4():
    job_id = unique_job_id("valid_video")

    output_path = (
        STORAGE_DIR
        / job_id
        / "outputs"
        / "avatar.mp4"
    )

    create_valid_mp4(output_path)

    assert validate_video_output(str(output_path))


def test_validate_video_output_rejects_missing_file():
    job_id = unique_job_id("missing_video")

    output_path = (
        STORAGE_DIR
        / job_id
        / "outputs"
        / "avatar.mp4"
    )

    assert not validate_video_output(str(output_path))


# ---------------------------------------------------------------------------
# Direct pipeline persistence tests
# ---------------------------------------------------------------------------


def test_pipeline_marks_missing_model_as_failed():
    job_id = unique_job_id("pipeline_missing_model")

    image_path = (
        STORAGE_DIR
        / job_id
        / "inputs"
        / "face.png"
    )

    audio_path = (
        STORAGE_DIR
        / job_id
        / "inputs"
        / "audio.wav"
    )

    image_path.parent.mkdir(parents=True, exist_ok=True)

    image_path.write_bytes(get_valid_image_bytes())
    audio_path.write_bytes(get_valid_wav_bytes())

    create_test_job(job_id)

    run_talking_head_pipeline(
        job_id,
        str(image_path),
        str(audio_path),
        "latentsync",
        False,
    )

    job = get_job(job_id)

    assert job is not None
    assert job["status"] == "failed"
    assert job["error_message"] is not None
