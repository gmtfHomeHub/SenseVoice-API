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

### 长音频：VAD 分段（默认开启，不建议关闭）

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `VAD_MODEL` | `fsmn-vad` | VAD 模型（映射到 HF 的 `funasr/fsmn-vad`，~25MB）。置为空字符串关闭，仅用于调试 |
| `VAD_MAX_SEGMENT_MS` | `20000` | 单个 VAD 段的**硬上限**（毫秒）。连续语音超过该值会被强制切开 |
| `VAD_MERGE_LENGTH_S` | `15` | 相邻 VAD 短段合并到的上限（秒） |

> ⚠ `merge_vad` **只合并、不切分**：`merge_vad(vad_result, max_length)` 在
> `len(vad_result) <= 1` 时直接原样返回。所以真正限制单段长度的是
> `VAD_MAX_SEGMENT_MS`，不是 `VAD_MERGE_LENGTH_S`。
> SenseVoice 文档上限为 30s，默认取 20s 留一档安全裕量。

**为什么要开 VAD**：SenseVoiceSmall 只在约 30s 以内的音频上训练（LFR 帧率 16.7 帧/s，
`config.yaml` 的 `max_source_length=2000` 帧）。不加 `vad_model` 时，funasr 的
`generate()` 走 `self.vad_model is None` 分支直接调用 `inference()`，
**把整段音频一次性喂进编码器，不做任何切分**：

- 5 分钟音频 = 5000 帧，是训练域上限的数倍。编码器（SANM 全自注意力 +
  `SinusoidalPositionEncoder` 无长度上限，纯外推）超出训练域后输出退化，
  CTC 大面积输出 blank —— 表现为「开头整段丢失 + 每十几秒蹦出一个词」。
- 实测对比（40s 合成音频，CPU）：不加 VAD → 单次 667 帧 forward，19.3s，
  `text='<|nospeech|>…'`，仅 1 个 token；加 VAD → 切成多段逐段推理，
  字级时间戳正确分布在 0.27s–39.94s 全区间。
- 附带收益：`batch_size_s` / `merge_vad` 这两个参数**只在有 `vad_model` 时才生效**，
  不加 VAD 时它们是完全无操作的死参数。
- 附带收益：`spk_model="cam++"` 也不加 VAD 就不生效（`inference()` 路径不碰
  spk_model），`/health` 上的 `spk:true` 只代表模型加载了。

开启后 funasr 会把每段的字级时间戳加上段起始偏移拼回**全局时间轴**
（`auto_model.py` 的 `t[0] += int(vadsegments[j][0])`），所以 `/v1/audio/transcriptions`
返回的 `words` / `segments` 时间戳直接就是相对整个文件的绝对时间，无需客户端换算。

### 标点：ct-punc（默认开启）

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `PUNC_MODEL` | `ct-punc` | CT-Transformer 标点模型（映射到 HF 的 `funasr/ct-punc`，**1.13GB**）。置为空字符串关闭 |
| `SPK_MODE` | `punc_segment` | 分段依据。`punc_segment` 按标点切句，`vad_segment` 按 VAD 切句（仅 `ENABLE_SPK=true` 时生效） |

SenseVoice 的 CTC 几乎不出标点（偶尔蹦一个「。」），不加的话 5 分钟转出来是
「我想我想去我想然后回家吃饭」这种。加上 `punc_model` 后 funasr 在 ASR 之后多跑一遍
标点模型，插回 `。` `，` `？` `、`。模型本身效果正常（实测）：

```
输入: 今天天气不错我们去公园散步看了一场电影然后回家吃饭休息明天还要上班所以早点睡
输出: 今天天气不错，我们去公园散步看了一场电影。然后回家吃饭休息，明天还要上班，所以早点睡。
```

**代价（实测，J3455 CPU 4 线程）**：

| 项目 | 数值 |
| --- | --- |
| 模型大小 | 1.13GB（`model.pt` 1.126GB），首启需下载 |
| 标点推理 | **约 10 字/秒**（500 字 5.1s，2000 字 21s，3 次一致） |
| 典型开销 | 5 分钟视频约 2000–3000 字 → **+20–30s**，相对转写耗时约 +12–18% |
| 内存 | 后端约 2.1GiB → 2.85GiB（`mem_limit: 8g` 内） |

### SPK_MODE：punc_segment vs vad_segment

300s 音频、三种配置同一份音频的对照实测：

| 配置 | segments | 段长范围 | 段内文本 |
| --- | --- | --- | --- |
| 无 punc | 12 | 0.6 – 15.4s | 原始 ASR 输出（带 `<\|…\|>` 标签） |
| `punc_segment` | **3** | 16.0 – **43.7s** ⚠ | **带标点** |
| `vad_segment` | 12 | 0.6 – 15.4s | 原始 ASR 输出（无标点插入） |

`punc_segment` 下句段依据标点边界，**标点稀疏时多个段会塔成一个巨句**。
合成音频几乎不出文本，标点极少，所以塔得厉害；真实语音标点密集，句段一般是
3–10s 的合理长度。遇到音乐/噪声/非中文/长停顿单声道这类标点稀疏的输入，
设 `SPK_MODE=vad_segment` 可换回稳定分段，代价是段内文本不再有标点
（顶层 `text` 仍有）。

字级 `timestamp` / `words` 在两种模式下都是 VAD 校正后的全局时间，**不受影响**。
关闭 `punc_model` 后，`spk_model` 仍可用但日志会反复刷
`[ERROR] Missing punc_model, which is required by spk_model.`——这是 funasr 的
**状态泄漏 bug**：第一次请求打 warning 并把共享实例的 `self.spk_mode` 永久改成
`"vad_segment"`，之后每次请求都走 `elif` 分支打 ERROR。

### 启动加速与网络相关

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `PREWARM_SECONDS` | `2` | 启动时对一段静音做一次推理，预热 torch/oneDNN 图。CPU 上首个真实请求否则会因内核图缓存阻塞 2-4 分钟，表现为「一直转最后超时」。`0` 关闭 |
| `PIP_INDEX_URL` | 清华源 | 兜底：funasr 可能内部执行 `pip install -r <model>/requirements.txt`（无超时），预设镜像源避免卡死 |

> **为什么 `trust_remote_code=False` 是必须的**：funasr 在 `trust_remote_code=True` 时会执行
> `install_model_requirements`，即对一个**没有超时**的裸 `pip install -r <model>/requirements.txt`
> 子进程。SenseVoiceSmall 的该文件要求 `torch`、`modelscope`、`huggingface`、`gradio`、
> `numpy<=1.26.4` —— 在受限网络下会让容器启动永久卡住，且每次重启都会尝试把 numpy 降级。
> SenseVoiceSmall 与 campplus 都是 funasr 原生注册模型，不需要远程代码，跳过该步骤是安全的。
