import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.staticfiles import StaticFiles

from src.job_store import init_db, create_job, get_job
from src.exceptions import JobNotFoundError, register_exception_handlers
from src.logging_config import get_logger
from src.pipeline import run_talking_head_pipeline
from src.schemas import GenerateAvatarResponse, JobStatusResponse
from src.storage import save_job_inputs
from src.validation import (
    validate_audio_file,
    validate_image_file,
    validate_model,
)

logger = get_logger("main")


# ============================================================================
# FastAPI application
# ============================================================================

app = FastAPI(
    title="AI Talking Head Service",
    description="REST API for generating lip-synced talking head avatars",
    version="1.0.0",
)
app.mount(
    "/outputs",
    StaticFiles(directory="storage/jobs"),
    name="outputs",
)


@app.on_event("startup")
def startup_event():
    init_db()


register_exception_handlers(app)


# ============================================================================
# Avatar / voice gender configuration
# ============================================================================

MALE_AVATAR = "male"
FEMALE_AVATAR = "female"

DEFAULT_AVATAR = MALE_AVATAR


VOICE_GENDER = {
    # Male voices
    "en-US-ChristopherNeural": "male",
    "en-US-GuyNeural": "male",
    "en-GB-RyanNeural": "male",
    "en-IN-PrabhatNeural": "male",
    # Female voices
    "en-US-JennyNeural": "female",
    "en-GB-SoniaNeural": "female",
    "en-IN-NeerjaNeural": "female",
}


# ============================================================================
# Voice / avatar helper functions
# ============================================================================


def get_voice_gender(voice: str) -> str:
    """
    Return the gender associated with a voice.
    Unknown voices return 'neutral'.
    """
    return VOICE_GENDER.get(voice, "neutral")


def get_avatar_for_voice(
    voice: str,
    avatar: Optional[str] = None,
) -> str:
    """
    Select an avatar based on voice gender.
    """
    gender = get_voice_gender(voice)

    if gender == "neutral":
        if avatar is None:
            return DEFAULT_AVATAR

        if avatar not in {MALE_AVATAR, FEMALE_AVATAR}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported avatar: {avatar}",
            )
        return avatar

    if avatar is None:
        if gender == "male":
            return MALE_AVATAR
        return FEMALE_AVATAR

    if avatar not in {MALE_AVATAR, FEMALE_AVATAR}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported avatar: {avatar}",
        )

    if gender == "male" and avatar == FEMALE_AVATAR:
        raise HTTPException(
            status_code=400,
            detail=f"Male voice '{voice}' is incompatible with female avatar '{avatar}'.",
        )

    if gender == "female" and avatar == MALE_AVATAR:
        raise HTTPException(
            status_code=400,
            detail=f"Female voice '{voice}' is incompatible with male avatar '{avatar}'.",
        )

    return avatar


# ============================================================================
# Health check
# ============================================================================


@app.get("/")
def read_root():
    logger.info("Health check endpoint 'GET /' called.")
    return {
        "name": "AI Talking Head Service",
        "status": "healthy",
    }


# ============================================================================
# Generate avatar
# ============================================================================


@app.post(
    "/api/v1/avatar/generate",
    response_model=GenerateAvatarResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_avatar(
    background_tasks: BackgroundTasks,
    face_image: UploadFile = File(...),
    audio: UploadFile = File(...),
    model: str = Form("latentsync"),
    enhancer: bool = Form(True),
    voice: str = Form("en-US-ChristopherNeural"),
    avatar: Optional[str] = Form(None),
):
    logger.info(
        "Received avatar generation request: "
        f"image='{face_image.filename if face_image else None}', "
        f"audio='{audio.filename if audio else None}', "
        f"model='{model}', "
        f"enhancer={enhancer}, "
        f"voice='{voice}', "
        f"avatar='{avatar}'"
    )

    if voice not in VOICE_GENDER:
        logger.warning(f"Unsupported voice requested: '{voice}'")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported voice: {voice}",
        )

    gender = get_voice_gender(voice)

    selected_avatar = get_avatar_for_voice(
        voice=voice,
        avatar=avatar,
    )

    logger.info(
        f"Voice '{voice}' resolved to gender='{gender}', " f"avatar='{selected_avatar}'"
    )

    validated_model = validate_model(model)
    img_bytes = await validate_image_file(face_image)
    audio_bytes = await validate_audio_file(audio)

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    saved_paths = save_job_inputs(
        job_id=job_id,
        image_filename=face_image.filename or "face.jpg",
        image_bytes=img_bytes,
        audio_filename=audio.filename or "voice.wav",
        audio_bytes=audio_bytes,
    )

    # ------------------------------------------------------------------
    # Create job state in SQLite DB (Replacing jobs_db dictionary)
    # ------------------------------------------------------------------
    create_job(
        job_id=job_id,
        created_at=created_at,
        voice=voice,
        avatar=selected_avatar,
        gender=gender,
    )

    # ------------------------------------------------------------------
    # Start background pipeline (Removed jobs_db argument)
    # ------------------------------------------------------------------
    background_tasks.add_task(
        run_talking_head_pipeline,
        job_id,
        saved_paths["image_path"],
        saved_paths["audio_path"],
        validated_model,
        enhancer,
    )

    logger.info(f"Successfully queued job {job_id}")

    return GenerateAvatarResponse(
        job_id=job_id,
        status="queued",
        created_at=created_at,
        message="Avatar rendering job successfully queued.",
        voice=voice,
        avatar=selected_avatar,
        gender=gender,
    )


# ============================================================================
# Job status
# ============================================================================


@app.get(
    "/api/v1/avatar/jobs/{job_id}",
    response_model=JobStatusResponse,
)
def get_job_status(job_id: str):
    logger.info(f"Querying job status for job_id='{job_id}'")

    # Fetch job from SQLite DB instead of jobs_db dictionary
    job = get_job(job_id)

    if not job:
        logger.warning(f"Job status query failed: job '{job_id}' not found.")
        raise JobNotFoundError(job_id)

    return JobStatusResponse(**job)
