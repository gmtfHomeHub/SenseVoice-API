# WebUI 镜像：ghcr.io/gmtfhomehub/sensevoice-api-webui
#
# Gradio 前端，只做一件事：把上传的音频转发给 sensevoice-api-asr 后端，
# 把结果渲染成纯文本 / 元信息时间轴 / SRT / 原始 JSON。
# 自身不含任何模型，镜像很小。
#
# build context 是 webui/ 目录本身，所以 COPY 直接写文件名。
FROM python:3.11-slim

# curl 仅供 healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖装完再拷代码，充分利用构建缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY app.py .

ENV WEBUI_PORT=7860 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${WEBUI_PORT}/ || exit 1

CMD ["python", "app.py"]
