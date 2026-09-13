# ASR 后端镜像：ghcr.io/gmtfhomehub/sensevoice-api-asr
#
# CPU-only fork of hsiang-han/SenseVoice-API。上游只有 docker/gpu/Dockerfile
# （基于 nvidia/cuda，压缩后 6.86GB），这里剔除全部 CUDA：
#   - 基础镜像 python:3.11-slim
#   - torch 走 https://download.pytorch.org/whl/cpu（无 CUDA，约 200MB）
#   - 模型不在镜像里，运行时从 HF 下载到挂载的 HF_HOME
#
# build context 是 api/ 目录本身，所以 COPY 直接写文件名。
FROM python:3.11-slim

# funasr / soundfile 需要 libsndfile1；ffmpeg 处理多格式音频；curl 供 healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU 版 torch 必须早于 funasr 安装，否则 pip 会从默认源拉带 CUDA 的 wheel
RUN pip install --no-cache-dir torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV DEVICE=cpu \
    MODEL_ID=FunAudioLLM/SenseVoiceSmall \
    BATCH_SIZE=1 \
    PORT=10095 \
    HF_HOME=/root/.cache/huggingface \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 \
    # funasr 首次加载某个模型时可能内部跑 `pip install -r <model>/requirements.txt`
    # （无超时、不走镜像源），预设源避免在受限网络下把容器启动卡死
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DEFAULT_TIMEOUT=60 \
    # 启动时预推理一段音频，预热 torch/oneDNN 计算图；0 = 不预热
    PREWARM_SECONDS=2 \
    PYTHONUNBUFFERED=1

EXPOSE 10095

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["./entrypoint.sh"]
