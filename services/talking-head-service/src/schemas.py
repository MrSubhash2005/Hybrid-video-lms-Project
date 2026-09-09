
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class GenerateAvatarResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    estimated_time_remaining: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    output_url: Optional[str] = None
    error_message: Optional[str] = None
