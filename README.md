# SenseVoice-API (CPU-only fork)

Fork of [hsiang-han/SenseVoice-API](https://github.com/hsiang-han/SenseVoice-API), rebuilt **without CUDA** for CPU-only hosts (NAS / NUC without NVIDIA GPU).

## 改动
- 新增 `docker/cpu/Dockerfile`：基础镜像 `python:3.11-slim`，torch 走 CPU index (`https://download.pytorch.org/whl/cpu`)，无 CUDA。
- 上游 `docker/gpu/` 不再使用（保留作参考未纳入本 fork）。
- 业务代码 `api/main.py`、`entrypoint.sh`、`requirements.txt` 与上游一致，零改动。
- `.github/workflows/docker-publish.yml` 改为构建 CPU 版并推送 GHCR（tag `latest` / `cpu-latest`）。

## 体积对比
| 镜像 | 体积 |
|---|---|
| 上游 GPU (`ghcr.io/hsiang-han/sensevoice-api:latest`) | 6.86 GB |
| 本 fork CPU (`ghcr.io/gmtfhomehub/sensevoice-api:cpu-latest`) | ~1.3 GB |

## 运行
```yaml
services:
  sensevoice-asr:
    image: ghcr.io/gmtfhomehub/sensevoice-api:cpu-latest
    ports: ["10095:10095"]
    environment:
      - DEVICE=cpu
      - ENABLE_SPK=true
      - HF_ENDPOINT=https://hf-mirror.com
    volumes:
      - ./models:/root/.cache/huggingface
    restart: unless-stopped
```
模型在首启时从 HuggingFace 下载到挂载卷（约 900MB，一次性）。

## 接口
- `GET  /health`
- `GET  /v1/models`
- `POST /v1/audio/transcriptions`
- `WS   /v1/audio/transcriptions/stream`

端口 `10095`，与上游完全兼容。

## CPU 变体补充

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `DEVICE` | `cpu` | 推理设备，本 fork 固定 `cpu`（上游 GPU 镜像仍用 `docker/gpu/Dockerfile`） |
| `MAX_AUDIO_SECONDS` | `0` | 音频时长护栏。`0` = 不限制；`>0` 时长超出的请求立即 `413`，避免长时间占住单例模型 |

### `POST /v1/cancel`

取消当前正在进行的转写任务，返回 `{"cancelled": true, "was_active": <bool>}`。

被取消的任务在后端返回 `499` + `{"cancelled": true}` 并丢弃结果。

> funasr 的 `AutoModel.generate()` 是单次同步调用，内部没有可抢占的边界，
> 因此「取消」不能打断途中的计算，语义是「断开连接 + 丢弃结果」。
> 为了让该端点在推理期间仍然可响应，`/v1/audio/transcriptions` 的推理已通过
> `starlette.concurrency.run_in_threadpool` 推入线程池，不再阻塞事件循环
> （上游写法下，转写期间 `/health` 同样无法响应）。
