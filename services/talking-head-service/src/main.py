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

register_exception_handlers(app)


# ============================================================================
# In-memory job store
# ============================================================================

jobs_db: Dict[str, Dict[str, Any]] = {}


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

    Rules:

    1. Known male voice + no avatar
       -> male avatar

    2. Known female voice + no avatar
       -> female avatar

    3. Unknown voice + no avatar
       -> DEFAULT_AVATAR

    4. Explicit avatar
       -> must be male or female

    5. Known voice + incompatible avatar
       -> HTTP 400
    """

    gender = get_voice_gender(voice)

    # ---------------------------------------------------------
    # Unknown / neutral voice
    # ---------------------------------------------------------

    if gender == "neutral":
        if avatar is None:
            return DEFAULT_AVATAR

        if avatar not in {
            MALE_AVATAR,
            FEMALE_AVATAR,
        }:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported avatar: {avatar}",
            )

        return avatar

    # ---------------------------------------------------------
    # No avatar supplied -> automatically select one
    # ---------------------------------------------------------

    if avatar is None:
        if gender == "male":
            return MALE_AVATAR

        return FEMALE_AVATAR

    # ---------------------------------------------------------
    # Validate explicit avatar
    # ---------------------------------------------------------

    if avatar not in {
        MALE_AVATAR,
        FEMALE_AVATAR,
    }:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported avatar: {avatar}",
        )

    # ---------------------------------------------------------
    # Male voice + female avatar
    # ---------------------------------------------------------

    if gender == "male" and avatar == FEMALE_AVATAR:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Male voice '{voice}' is incompatible with "
                f"female avatar '{avatar}'."
            ),
        )

    # ---------------------------------------------------------
    # Female voice + male avatar
    # ---------------------------------------------------------

    if gender == "female" and avatar == MALE_AVATAR:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Female voice '{voice}' is incompatible with "
                f"male avatar '{avatar}'."
            ),
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

    # ------------------------------------------------------------------
    # Validate voice
    # ------------------------------------------------------------------

    if voice not in VOICE_GENDER:
        logger.warning(f"Unsupported voice requested: '{voice}'")

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported voice: {voice}",
        )

    # ------------------------------------------------------------------
    # Determine gender
    # ------------------------------------------------------------------

    gender = get_voice_gender(voice)

    # ------------------------------------------------------------------
    # Determine / validate avatar
    # ------------------------------------------------------------------

    selected_avatar = get_avatar_for_voice(
        voice=voice,
        avatar=avatar,
    )

    logger.info(
        f"Voice '{voice}' resolved to gender='{gender}', " f"avatar='{selected_avatar}'"
    )

    # ------------------------------------------------------------------
    # Validate model
    # ------------------------------------------------------------------

    validated_model = validate_model(model)

    # ------------------------------------------------------------------
    # Validate uploaded files
    # ------------------------------------------------------------------

    img_bytes = await validate_image_file(face_image)

    audio_bytes = await validate_audio_file(audio)

    # ------------------------------------------------------------------
    # Create job ID
    # ------------------------------------------------------------------

    job_id = f"job_{uuid.uuid4().hex[:12]}"

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # ------------------------------------------------------------------
    # Save uploaded files
    # ------------------------------------------------------------------

    saved_paths = save_job_inputs(
        job_id=job_id,
        image_filename=face_image.filename or "face.jpg",
        image_bytes=img_bytes,
        audio_filename=audio.filename or "voice.wav",
        audio_bytes=audio_bytes,
    )

    # ------------------------------------------------------------------
    # Create job state
    # ------------------------------------------------------------------

    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "estimated_time_remaining": 30.0,
        "created_at": created_at,
        "completed_at": None,
        "output_url": None,
        "error_message": None,
        # Voice / avatar metadata
        "voice": voice,
        "avatar": selected_avatar,
        "gender": gender,
    }

    # ------------------------------------------------------------------
    # Start background pipeline
    # ------------------------------------------------------------------

    background_tasks.add_task(
        run_talking_head_pipeline,
        job_id,
        saved_paths["image_path"],
        saved_paths["audio_path"],
        validated_model,
        enhancer,
        jobs_db,
    )

    logger.info(f"Successfully queued job {job_id}")

    # ------------------------------------------------------------------
    # API response
    # ------------------------------------------------------------------

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

    if job_id not in jobs_db:
        logger.warning(f"Job status query failed: " f"job '{job_id}' not found.")

        raise JobNotFoundError(job_id)

    return JobStatusResponse(**jobs_db[job_id])
