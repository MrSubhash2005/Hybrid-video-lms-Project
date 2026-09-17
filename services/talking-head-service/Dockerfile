FROM python:3.10-slim

WORKDIR /app

# Wav2Lip backend environment paths
ENV WAV2LIP_DIR=/app/Wav2Lip
ENV WAV2LIP_CHECKPOINT_PATH=/app/Wav2Lip/checkpoints/wav2lip_gan.pth

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY services/talking-head-service/requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy source code and other artifacts
COPY services/talking-head-service/. .

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
