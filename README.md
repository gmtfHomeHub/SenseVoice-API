# SenseVoice-API（CPU-only fork，含 WebUI）

Fork of [hsiang-han/SenseVoice-API](https://github.com/hsiang-han/SenseVoice-API)，
**剔除全部 CUDA**，面向无 NVIDIA GPU 的主机（NAS / NUC 等）。

本仓库构建并推送 **两个镜像**：

| 镜像 | 内容 | 端口 |
| --- | --- | --- |
| `ghcr.io/gmtfhomehub/sensevoice-api-asr` | FastAPI + funasr 推理后端（CPU-only） | 10095 |
| `ghcr.io/gmtfhomehub/sensevoice-api-webui` | Gradio 前端（无模型，只转发并渲染） | 7860 |

## 项目结构

```
.
├── .github/workflows/docker-publish.yml   # 两个 job 并行构建两个镜像
├── api/                                   # ASR 后端
│   ├── Dockerfile
│   ├── main.py                            # FastAPI 应用（业务逻辑）
│   ├── entrypoint.sh                      # uvicorn 启动
│   ├── requirements.txt
│   └── .dockerignore
├── webui/                                 # Gradio 前端
│   ├── Dockerfile
│   ├── app.py
│   ├── requirements.txt
│   └── .dockerignore
└── README.md
```

每个服务一个目录，自带 Dockerfile / requirements / `.dockerignore`；
build context 就是该目录本身，所以 Dockerfile 里的 `COPY` 直接写文件名。
上游那个 `docker/cpu/` 与 `docker/scripts/` 的中间层已去掉。

## 与上游的差异

- 基础镜像 `python:3.11-slim`，torch 走 CPU index（`https://download.pytorch.org/whl/cpu`），**无 CUDA**。
- 新增 `webui/`：Gradio 前端，可上传音频、实时看进度、随时取消。
- 业务代码有实质改动，不再是「零改动」：
  - `run_in_threadpool` 把推理推入线程池，转写期间 `/health`、`/v1/cancel` 仍可响应。
  - 新增 `POST /v1/cancel`。
  - 新增 `MAX_AUDIO_SECONDS` 时长护栏、`PREWARM_SECONDS` 启动预热。
  - `trust_remote_code=False`（必须，见文末）。
  - 默认开启 FSMN-VAD 分段（必须，见下文）。
  - 过滤 funasr 的 punc_model 误报日志。
- 镜像 tag：两个镜像都用 `latest` / `main` / `sha-*` / `vX.Y.Z`。

## 体积对比

| 镜像 | 体积 |
| --- | --- |
| 上游 GPU（`ghcr.io/hsiang-han/sensevoice-api:latest`） | 6.86 GB |
| 本仓库 ASR（`sensevoice-api-asr:latest`） | ~1.3 GB |
| 本仓库 WebUI（`sensevoice-api-webui:latest`） | ~300 MB |

两个镜像都**不含模型**，模型在首启时从 HuggingFace 下载到挂载卷。

## 快速开始

```yaml
services:
  sensevoice-asr:
    image: ghcr.io/gmtfhomehub/sensevoice-api-asr:latest
    ports: ["10095:10095"]
    environment:
      - DEVICE=cpu
      - ENABLE_SPK=true
      - HF_ENDPOINT=https://hf-mirror.com
      - HF_HUB_DISABLE_XET=1
      - VAD_MODEL=fsmn-vad
      - VAD_MAX_SEGMENT_MS=20000
      - VAD_MERGE_LENGTH_S=15
      - SPK_MODE=vad_segment
    volumes:
      - ./models:/root/.cache/huggingface   # 首启约 900MB，一次性
    mem_limit: 8g
    restart: unless-stopped

  sensevoice-webui:
    image: ghcr.io/gmtfhomehub/sensevoice-api-webui:latest
    ports: ["17860:7860"]
    environment:
      - ASR_API_URL=http://sensevoice-asr:10095   # compose 内部服务名
      - ASR_TIMEOUT=3600
    mem_limit: 2g
    depends_on: [sensevoice-asr]
    restart: unless-stopped
```

`docker compose up -d` 后：

- WebUI：<http://localhost:17860>
- API：<http://localhost:10095>/health

## 接口

- `GET  /health`
- `GET  /v1/models`
- `POST /v1/audio/transcriptions`（multipart，字段名 `file`）
- `POST /v1/cancel`
- `WS   /v1/audio/transcriptions/stream`

端口 `10095`，请求/响应格式与上游兼容。

`response_format=verbose_json` 额外返回 `emotion` / `event` / `language` /
`timestamp` / `words` / `segments`。被取消的任务返回 `499` + `{"cancelled": true}`。

### `POST /v1/cancel`

取消当前正在进行的转写任务，返回 `{"cancelled": true, "was_active": <bool>}`。

> funasr 的 `AutoModel.generate()` 是单次同步调用，内部没有可抢占的边界，
> 因此「取消」不能打断途中的计算，语义是「断开连接 + 丢弃结果」。
> 为了让该端点在推理期间仍然可响应，`/v1/audio/transcriptions` 的推理已通过
> `starlette.concurrency.run_in_threadpool` 推入线程池，不再阻塞事件循环
> （上游写法下，转写期间 `/health` 同样无法响应）。

## 后端环境变量

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `DEVICE` | `cpu` | 推理设备。有 GPU 的镜像用 `cuda:0` |
| `MODEL_ID` | `FunAudioLLM/SenseVoiceSmall` | ASR 模型 |
| `ENABLE_SPK` | `false` | 说话人识别（cam++，首次下载 ~7MB） |
| `NUM_THREADS` | – | torch 线程数，按 CPU 物理核心数设置 |
| `PORT` | `10095` | 监听端口 |
| `MAX_AUDIO_SECONDS` | `0` | 时长护栏。`0` = 不限制；`>0` 超长的请求立即 `413`，避免长时间占住单例模型 |
| `PREWARM_SECONDS` | `2` | 启动时预推理的音频秒数，预热 torch/oneDNN。`0` 关闭 |
| `VAD_MODEL` | `fsmn-vad` | VAD 模型（映射到 HF 的 `funasr/fsmn-vad`，~25MB）。置空字符串关闭，仅用于调试 |
| `VAD_MAX_SEGMENT_MS` | `20000` | 单个 VAD 段的**硬上限**（毫秒）。连续语音超过该值会被强制切开 |
| `VAD_MERGE_LENGTH_S` | `15` | 相邻 VAD 短段合并到的上限（秒） |
| `SPK_MODE` | `vad_segment` | 分段依据，仅 `ENABLE_SPK=true` 时生效。见下文 |
| `HF_ENDPOINT` | – | 国内网络设 `https://hf-mirror.com` |
| `HF_HUB_DISABLE_XET` | `1` | 必须。hf-mirror 不代理 `cas-server.xethub.hf.co`，xet 后端匿名请求返回 401 会导致权重下载失败 |
| `PIP_INDEX_URL` | 清华源 | 兜底：funasr 可能内部执行 `pip install -r <model>/requirements.txt`（无超时），预设镜像源避免卡死 |
| `PIP_DEFAULT_TIMEOUT` | `60` | 同上 |

## 长音频：VAD 分段（必须开启）

SenseVoiceSmall 只在约 30s 以内的音频上训练（LFR 帧率 16.7 帧/s，
`config.yaml` 的 `max_source_length=2000` 帧）。不加 `vad_model` 时，funasr 的
`generate()` 走 `self.vad_model is None` 分支直接调用 `inference()`，
**把整段音频一次性喂进编码器，不做任何切分**：

- 5 分钟音频 = 5000 帧，是训练域上限的数倍。编码器（SANM 全自注意力 +
  `SinusoidalPositionEncoder` 无长度上限，纯外推）超出训练域后输出退化，
  CTC 大面积输出 blank —— 表现为「开头整段丢失 + 每十几秒蹦出一个词」。
- 实测对比（40s 合成音频，CPU）：不加 VAD → 单次 667 帧 forward，19.3s，
  `text='<|nospeech|>…'`，仅 1 个 token；加 VAD → 切成多段逐段推理，
  字级时间戳正确分布在 0.27s–39.94s 全区间。
- 附带约束：`batch_size_s` / `merge_vad` **只在有 `vad_model` 时才生效**，
  不加 VAD 时它们是死参数。`spk_model="cam++"` 同理（`inference()` 路径不碰
  spk_model），`/health` 上的 `spk:true` 只代表模型加载了。

开启后 funasr 会把每段的字级时间戳加上段起始偏移拼回**全局时间轴**
（`auto_model.py` 的 `t[0] += int(vadsegments[j][0])`），所以 `words` / `segments`
的时间戳直接就是相对整个文件的绝对时间，无需客户端换算。

> ⚠ `merge_vad` **只合并、不切分**：`merge_vad(vad_result, max_length)` 在
> `len(vad_result) <= 1` 时直接原样返回。所以真正限制单段长度的是
> `VAD_MAX_SEGMENT_MS`，不是 `VAD_MERGE_LENGTH_S`。
> SenseVoice 文档上限 30s，默认取 20s 留一档安全裕量。

## 标点：不加 punc_model（SenseVoice 自带）

SenseVoiceSmall 的 sentencepiece 词表（25055 token）里 `，` `。` `？` `、` `！` `；`
`：` **都是单字符 token** —— SenseVoice 的 CTC 本身就输出标点，不需要额外的
标点模型。

**不要**给 SenseVoice 配 `punc_model`，否则输出会全是重复标点。实测：

```
输入（已带标点） 今天天气不错，我们去公园散步。
输出            今天天气不错，，，我们去公园散步。。

输入（无标点）   今天天气不错我们去公园散步
输出            今天天气不错，我们去公园散步。        ← 只有无标点输入才正确
```

原因是 funasr 的 `_punctuate_surface_text()` 是「原文照抄 + 每个 token 后追加
ct-punc 的标点」，不会先剥离 ASR 已有的标点。funasr 自己的文档也这么写
（`auto_model.py:431`、`ct_transformer/model.py:49`）：

> Not needed for Fun-ASR-Nano/SenseVoice/Qwen3-ASR (they output punctuation natively).

ct-punc 模型 1.13GB、推理约 10 字/秒，既慢又会让输出变成「。。，，？？」，所以本
仓库不加载它。

## SPK_MODE

仅在 `ENABLE_SPK=true` 时生效。默认 `vad_segment`。

- `vad_segment` —— 按 VAD 边界切句，段长稳定（受 `VAD_MAX_SEGMENT_MS` 限制），
  段内 `text` 是 SenseVoice 原始输出（**自带标点**）。
- `punc_segment` —— 需要 `punc_model` 才能产出有意义的句段。没有 punc_model 时，
  funasr 第一次请求打 warning 并把共享实例的 `self.spk_mode` 永久改成
  `vad_segment`（状态泄漏），之后每次请求都走 `elif` 分支刷
  `[ERROR] Missing punc_model, which is required by spk_model.`

字级 `timestamp` / `words` 在两种模式下都是 VAD 校正后的全局时间，不受影响。

`_SuppressFunasrPuncNoise` 日志过滤器会精确屏蔽上述三条 punc 误报消息
（误报原因：`raw_text` 只在 `self.punc_model is not None` 时才赋值），
OOM、Traceback 等真实错误照常放行。

## WebUI 环境变量

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `ASR_API_URL` | `http://sensevoice-asr:10095` | 后端地址。compose 内用服务名，不要用 `localhost` |
| `ASR_TIMEOUT` | `3600` | 单次转写的 HTTP 超时（秒），界面里也可改（30–86400） |
| `WEBUI_PORT` | `7860` | 容器内监听端口 |

WebUI 提供：音频上传（wav/mp3/m4a/flac/ogg）、上传时预估时长与耗时、
2 秒一次的任务进度、后端健康状态条（device / spk / vad / model）、
随时取消、四种输出格式（完整文本 / 元信息时间轴 / SRT 字幕 / 原始 JSON）。

关于「取消」的诚实说明：请求会立即断开、界面立刻可用、后端返回 `499`，
但 funasr 是单次同步推理无法中途抢占，**服务端计算会自然跑完**（结果不返回）。

## 启动加速与网络

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `PREWARM_SECONDS` | `2` | 启动时对一段音频做一次推理，预热 torch/oneDNN 计算图。CPU 上首个真实请求否则会因内核图缓存阻塞数分钟，表现为「一直转最后超时」。`0` 关闭 |
| `PIP_INDEX_URL` | 清华源 | 兜底：funasr 可能内部执行 `pip install -r <model>/requirements.txt`（无超时），预设镜像源避免卡死 |
| `PIP_DEFAULT_TIMEOUT` | `60` | 同上 |

> **为什么 `trust_remote_code=False` 是必须的**：funasr 在 `trust_remote_code=True`
> 时会执行 `install_model_requirements`，即对一个**没有超时**的裸
> `pip install -r <model>/requirements.txt` 子进程。SenseVoiceSmall 的该文件要求
> `torch`、`modelscope`、`huggingface`、`gradio`、`numpy<=1.26.4` —— 在受限网络下
> 会让容器启动永久卡住，且每次重启都会尝试把 numpy 降级。SenseVoiceSmall 与
> campplus 都是 funasr 原生注册模型，不需要远程代码，跳过该步骤是安全的。

## 构建镜像

推送 `main` 分支即自动触发，两个 job 并行构建：

- `build-asr`：context `./api` → `ghcr.io/gmtfhomehub/sensevoice-api-asr`
- `build-webui`：context `./webui` → `ghcr.io/gmtfhomehub/sensevoice-api-webui`

tag 规则：`latest`（仅默认分支）、`main`、`sha-*`、`vX.Y.Z`。
两个 job 各自独立的 GHA 构建缓存（`scope=asr` / `scope=webui`）。
