import io
import sys
import wave
from pathlib import Path

# Add talking-head-service root to sys.path for repo-root execution
SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.main import (  # noqa: E402
    DEFAULT_AVATAR,
    FEMALE_AVATAR,
    MALE_AVATAR,
    VOICE_GENDER,
    app,
    get_avatar_for_voice,
    jobs_db,
)
from src.pipeline import (  # noqa: E402
    run_talking_head_pipeline,
    validate_video_output,
)

client = TestClient(app)


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


def test_read_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": "AI Talking Head Service",
        "status": "healthy",
    }


def test_generate_avatar_success_job_creation():
    files = {
        "face_image": (
            "portrait.png",
            get_valid_image_bytes(),
            "image/png",
        ),
        "audio": (
            "speech.wav",
            get_valid_wav_bytes(),
            "audio/wav",
        ),
    }

    data = {
        "model": "latentsync",
        "enhancer": "true",
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
        data=data,
    )

    assert response.status_code == 202

    body = response.json()

    assert "job_id" in body
    assert body["status"] == "queued"
    assert "created_at" in body
    assert body["message"] == "Avatar rendering job successfully queued."

    job_id = body["job_id"]

    status_response = client.get(f"/api/v1/avatar/jobs/{job_id}")

    assert status_response.status_code == 200

    status_body = status_response.json()

    assert status_body["job_id"] == job_id
    assert status_body["status"] in [
        "queued",
        "processing",
        "rendering",
        "failed",
    ]


def test_generate_missing_image():
    files = {
        "audio": (
            "speech.wav",
            get_valid_wav_bytes(),
            "audio/wav",
        )
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
    )

    assert response.status_code == 422
    assert "error_code" in response.json()


def test_generate_missing_audio():
    files = {
        "face_image": (
            "portrait.png",
            get_valid_image_bytes(),
            "image/png",
        )
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
    )

    assert response.status_code == 422
    assert "error_code" in response.json()


def test_generate_empty_image():
    files = {
        "face_image": (
            "portrait.png",
            b"",
            "image/png",
        ),
        "audio": (
            "speech.wav",
            get_valid_wav_bytes(),
            "audio/wav",
        ),
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "EMPTY_FILE"


def test_generate_empty_audio():
    files = {
        "face_image": (
            "portrait.png",
            get_valid_image_bytes(),
            "image/png",
        ),
        "audio": (
            "speech.wav",
            b"",
            "audio/wav",
        ),
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "EMPTY_FILE"


def test_generate_corrupted_image():
    files = {
        "face_image": (
            "portrait.png",
            b"not an image data",
            "image/png",
        ),
        "audio": (
            "speech.wav",
            get_valid_wav_bytes(),
            "audio/wav",
        ),
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "CORRUPTED_IMAGE"


def test_generate_corrupted_audio():
    files = {
        "face_image": (
            "portrait.png",
            get_valid_image_bytes(),
            "image/png",
        ),
        "audio": (
            "speech.wav",
            b"invalid audio bytes data",
            "audio/wav",
        ),
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "CORRUPTED_AUDIO"


def test_generate_unsupported_image_format():
    files = {
        "face_image": (
            "portrait.txt",
            b"hello world",
            "text/plain",
        ),
        "audio": (
            "speech.wav",
            get_valid_wav_bytes(),
            "audio/wav",
        ),
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "UNSUPPORTED_IMAGE_FORMAT"


def test_generate_unsupported_audio_format():
    files = {
        "face_image": (
            "portrait.png",
            get_valid_image_bytes(),
            "image/png",
        ),
        "audio": (
            "speech.txt",
            b"hello audio",
            "text/plain",
        ),
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "UNSUPPORTED_AUDIO_FORMAT"


def test_generate_unsupported_model():
    files = {
        "face_image": (
            "portrait.png",
            get_valid_image_bytes(),
            "image/png",
        ),
        "audio": (
            "speech.wav",
            get_valid_wav_bytes(),
            "audio/wav",
        ),
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
        data={"model": "sad_talker"},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "UNSUPPORTED_MODEL"


def test_generate_oversized_image():
    oversized_bytes = b"0" * (26 * 1024 * 1024)

    files = {
        "face_image": (
            "huge.png",
            oversized_bytes,
            "image/png",
        ),
        "audio": (
            "speech.wav",
            get_valid_wav_bytes(),
            "audio/wav",
        ),
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "FILE_TOO_LARGE"


def test_get_job_status_nonexistent():
    response = client.get("/api/v1/avatar/jobs/nonexistent_job_12345")

    assert response.status_code == 404
    assert response.json()["error_code"] == "JOB_NOT_FOUND"


def test_pipeline_controlled_failure_state():
    """Verify pipeline transitions to failed without real inference."""

    files = {
        "face_image": (
            "portrait.png",
            get_valid_image_bytes(),
            "image/png",
        ),
        "audio": (
            "speech.wav",
            get_valid_wav_bytes(),
            "audio/wav",
        ),
    }

    response = client.post(
        "/api/v1/avatar/generate",
        files=files,
    )

    assert response.status_code == 202

    job_id = response.json()["job_id"]
    assert job_id in jobs_db

    job_dir = f"storage/jobs/{job_id}/inputs"

    run_talking_head_pipeline(
        job_id,
        f"{job_dir}/image_portrait.png",
        f"{job_dir}/audio_speech.wav",
        "latentsync",
        True,
        jobs_db,
    )

    assert jobs_db[job_id]["status"] == "failed"

    assert jobs_db[job_id]["error_message"] is not None

    assert len(jobs_db[job_id]["error_message"]) > 0

    error_msg = jobs_db[job_id]["error_message"].lower()

    assert "not implemented" in error_msg or "inference" in error_msg

    assert jobs_db[job_id]["status"] != "completed"

    assert jobs_db[job_id]["output_url"] is None

    assert jobs_db[job_id]["completed_at"] is not None


def test_validate_video_output_nonexistent():
    assert validate_video_output("nonexistent_file_xyz.mp4") is False


def test_validate_video_output_empty_file(tmp_path):
    empty_file = tmp_path / "empty.mp4"

    empty_file.write_bytes(b"")

    assert validate_video_output(str(empty_file)) is False


# ---------------------------------------------------------
# Avatar / Voice Gender Matching Tests
# ---------------------------------------------------------


def test_male_voices_select_male_avatar():
    male_voices = (
        "en-US-ChristopherNeural",
        "en-US-GuyNeural",
        "en-GB-RyanNeural",
        "en-IN-PrabhatNeural",
    )

    for voice in male_voices:
        response = post_generation(voice)

        assert response.status_code == 202

        body = response.json()

        assert body["gender"] == "male"
        assert body["avatar"] == MALE_AVATAR


def test_female_voices_select_female_avatar():
    female_voices = (
        "en-US-JennyNeural",
        "en-GB-SoniaNeural",
        "en-IN-NeerjaNeural",
    )

    for voice in female_voices:
        response = post_generation(voice)

        assert response.status_code == 202

        body = response.json()

        assert body["gender"] == "female"
        assert body["avatar"] == FEMALE_AVATAR


def test_male_voice_matches_male_avatar():
    response = client.post(
        "/api/v1/avatar/generate",
        files=make_uploads(),
        data={
            "voice": "en-US-ChristopherNeural",
            "avatar": MALE_AVATAR,
        },
    )

    assert response.status_code == 202

    body = response.json()

    assert body["voice"] == "en-US-ChristopherNeural"
    assert body["gender"] == "male"
    assert body["avatar"] == MALE_AVATAR


def test_default_avatar_matches_male_voice_when_avatar_missing():
    response = post_generation("en-US-ChristopherNeural")

    assert response.status_code == 202

    body = response.json()

    assert body["avatar"] == MALE_AVATAR

    job_response = client.get(f"/api/v1/avatar/jobs/{body['job_id']}")

    assert job_response.status_code == 200
    assert job_response.json()["avatar"] == MALE_AVATAR


def test_female_voice_matches_female_avatar():
    response = client.post(
        "/api/v1/avatar/generate",
        files=make_uploads(),
        data={
            "voice": "en-US-JennyNeural",
            "avatar": FEMALE_AVATAR,
        },
    )

    assert response.status_code == 202

    body = response.json()

    assert body["voice"] == "en-US-JennyNeural"
    assert body["gender"] == "female"
    assert body["avatar"] == FEMALE_AVATAR


def test_default_avatar_matches_female_voice_when_avatar_missing():
    response = post_generation("en-US-JennyNeural")

    assert response.status_code == 202

    assert response.json()["avatar"] == FEMALE_AVATAR


def test_unsupported_voice_returns_400():
    response = client.post(
        "/api/v1/avatar/generate",
        files=make_uploads(),
        data={"voice": "en-US-UnknownNeural"},
    )

    assert response.status_code == 400
    assert "Unsupported voice" in response.json()["detail"]


def test_unsupported_avatar_returns_400():
    response = client.post(
        "/api/v1/avatar/generate",
        files=make_uploads(),
        data={
            "voice": "en-US-ChristopherNeural",
            "avatar": "robot",
        },
    )

    assert response.status_code == 400
    assert "Unsupported avatar" in response.json()["detail"]


def test_male_voice_rejects_female_avatar():
    response = client.post(
        "/api/v1/avatar/generate",
        files=make_uploads(),
        data={
            "voice": "en-US-ChristopherNeural",
            "avatar": FEMALE_AVATAR,
        },
    )

    assert response.status_code == 400

    detail = response.json()["detail"].lower()

    assert "male voice" in detail
    assert "female avatar" in detail


def test_female_voice_rejects_male_avatar():
    response = client.post(
        "/api/v1/avatar/generate",
        files=make_uploads(),
        data={
            "voice": "en-US-JennyNeural",
            "avatar": MALE_AVATAR,
        },
    )

    assert response.status_code == 400

    detail = response.json()["detail"].lower()

    assert "female voice" in detail
    assert "male avatar" in detail


def test_unknown_gender_falls_back_to_default_avatar():
    assert get_avatar_for_voice("unknown-voice") == DEFAULT_AVATAR

    assert (
        get_avatar_for_voice(
            "unknown-voice",
            DEFAULT_AVATAR,
        )
        == DEFAULT_AVATAR
    )


def test_voice_gender_mapping_contains_all_supported_voices():
    assert set(VOICE_GENDER) == {
        "en-US-ChristopherNeural",
        "en-US-GuyNeural",
        "en-GB-RyanNeural",
        "en-IN-PrabhatNeural",
        "en-US-JennyNeural",
        "en-GB-SoniaNeural",
        "en-IN-NeerjaNeural",
    }


def test_pipeline_successful_completion_with_backend(monkeypatch, tmp_path):
    job_id = "job_success_001"
    output_path = tmp_path / "result.mp4"
    image_path = tmp_path / "face.png"
    image_path.write_bytes(get_valid_image_bytes())

    audio_path = tmp_path / "voice.wav"
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 64))
    writer.write(frame)
    writer.release()

    class FakeBackend:
        def run(self, *, image_path, audio_path, output_path, progress_callback=None):
            if progress_callback is not None:
                progress_callback(60.0)
            assert Path(image_path).exists()
            assert Path(audio_path).exists()
            return str(output_path)

    monkeypatch.setattr("src.pipeline.get_inference_backend", lambda model: FakeBackend())

    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "estimated_time_remaining": 30.0,
        "created_at": "2024-01-01T00:00:00Z",
        "completed_at": None,
        "output_url": None,
        "error_message": None,
        "voice": "en-US-ChristopherNeural",
        "avatar": "male",
        "gender": "male",
    }

    result = run_talking_head_pipeline(
        job_id=job_id,
        image_path=str(image_path),
        audio_path=str(audio_path),
        model="latentsync",
        enhancer=True,
        jobs_db=jobs_db,
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert jobs_db[job_id]["status"] == "completed"
    assert jobs_db[job_id]["progress"] == 100.0
    assert jobs_db[job_id]["output_url"] == str(output_path.resolve())
    assert jobs_db[job_id]["error_message"] is None


def test_pipeline_accepts_audio_chunk_list(monkeypatch, tmp_path):
    job_id = "job_chunks_001"
    image_path = tmp_path / "face.png"
    image_path.write_bytes(get_valid_image_bytes())

    audio_dir = tmp_path / "chunks"
    audio_dir.mkdir()
    chunk_one = audio_dir / "chunk_1.wav"
    chunk_two = audio_dir / "chunk_2.wav"

    for path in (chunk_one, chunk_two):
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 1600)

    output_path = tmp_path / "chunk_output.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 64))
    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    class FakeBackend:
        def run(self, *, image_path, audio_path, output_path, progress_callback=None):
            assert Path(audio_path).exists()
            return str(output_path)

    monkeypatch.setattr("src.pipeline.get_inference_backend", lambda model: FakeBackend())
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "estimated_time_remaining": 30.0,
        "created_at": "2024-01-01T00:00:00Z",
        "completed_at": None,
        "output_url": None,
        "error_message": None,
        "voice": "en-US-ChristopherNeural",
        "avatar": "male",
        "gender": "male",
    }

    run_talking_head_pipeline(
        job_id=job_id,
        image_path=str(image_path),
        audio_path=[str(chunk_one), str(chunk_two)],
        model="latentsync",
        enhancer=True,
        jobs_db=jobs_db,
        output_path=str(output_path),
    )

    assert jobs_db[job_id]["status"] == "completed"
    assert jobs_db[job_id]["error_message"] is None
    assert jobs_db[job_id]["output_url"] == str(output_path.resolve())


def test_pipeline_missing_backend_stays_controlled_failure(monkeypatch, tmp_path):
    job_id = "job_missing_backend"
    image_path = tmp_path / "face.png"
    image_path.write_bytes(get_valid_image_bytes())

    audio_path = tmp_path / "voice.wav"
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)

    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "estimated_time_remaining": 30.0,
        "created_at": "2024-01-01T00:00:00Z",
        "completed_at": None,
        "output_url": None,
        "error_message": None,
        "voice": "en-US-ChristopherNeural",
        "avatar": "male",
        "gender": "male",
    }

    def missing_backend(model):
        raise Exception("LatentSync inference engine is not installed in this environment.")

    monkeypatch.setattr("src.pipeline.get_inference_backend", missing_backend)

    run_talking_head_pipeline(
        job_id=job_id,
        image_path=str(image_path),
        audio_path=str(audio_path),
        model="latentsync",
        enhancer=True,
        jobs_db=jobs_db,
    )

    assert jobs_db[job_id]["status"] == "failed"
    assert jobs_db[job_id]["progress"] == 0.0
    assert jobs_db[job_id]["output_url"] is None
    assert "inference" in (jobs_db[job_id]["error_message"] or "").lower()
