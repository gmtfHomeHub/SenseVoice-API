# CPU-only build (fork of hsiang-han/SenseVoice-API)
# 剔除 CUDA：基础镜像改用 python-slim，torch 走 CPU index，模型运行时下载
FROM python:3.11-slim

# funasr/soundfile 需 libsndfile1；ffmpeg 处理多格式音频；curl 用于 healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU 版 torch（必须早于 funasr 安装；从 CPU index，无 CUDA，约 200MB）
RUN pip install --no-cache-dir torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY docker/scripts/entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV DEVICE=cpu \
    MODEL_ID=FunAudioLLM/SenseVoiceSmall \
    BATCH_SIZE=1 \
    PORT=10095 \
    HF_HOME=/root/.cache/huggingface \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1

EXPOSE 10095
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["./entrypoint.sh"]
