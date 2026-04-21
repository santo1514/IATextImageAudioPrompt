FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY src/ ./src/

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    openai-whisper \
    transformers \
    torch \
    torchvision \
    torchaudio \
    Pillow \
    cohere

EXPOSE 9015

CMD ["uvicorn", "src.com.ssline.app.main:app", "--host", "0.0.0.0", "--port", "9015"]
