import io
import math
import os
import re
import struct
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

MODEL_ID = os.getenv("MODEL_ID", "FunAudioLLM/SenseVoiceSmall")
DEVICE = os.getenv("DEVICE", "cuda:0")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
ENABLE_SPK = os.getenv("ENABLE_SPK", "false").lower() == "true"
# 0 = 不限制；>0 时长超过该秒数的请求立即 413，避免长时间占住单例模型
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "0"))
# 启动时预推理的音频时长（秒），0 = 不预热
PREWARM_SECONDS = float(os.getenv("PREWARM_SECONDS", "2"))

# ── VAD 分段 ───────────────────────────────────────────────────────────────
# 必须开启。不加 vad_model 时 funasr 走 `vad_model is None` 分支，直接调用
# inference()，把整段音频一次性喂进 SenseVoiceSmall 编码器——没有任何切分。
# SenseVoiceSmall 只在约 30s 以内的音频上训练（LFR 帧率 16.7 帧/s，
# config.yaml max_source_length=2000 帧），5 分钟 = 5000 帧，远超训练域，
# 编码器输出退化为近随机分布、CTC 大面积输出 blank，表现为「开头整段丢失
# + 每十几秒蹦出一个词」。加了 vad_model 后 funasr 会先跑 FSMN-VAD 分段，
# 逐段推理，并把每段的字级时间戳加上段偏移拼回全局时间轴。
# 设为空字符串可关闭（仅用于调试）。
VAD_MODEL = os.getenv("VAD_MODEL", "fsmn-vad")
# 单个 VAD 段的最大时长（毫秒）。连续语音超过该值会被强制切开，
# 保证每段都落在 SenseVoice 的有效输入范围内。
VAD_MAX_SEGMENT_MS = int(os.getenv("VAD_MAX_SEGMENT_MS", "20000"))
# merge_vad 把相邻 VAD 段合并到的上限（秒），让 ASR 有上下文又不超长。
VAD_MERGE_LENGTH_S = int(os.getenv("VAD_MERGE_LENGTH_S", "15"))

_model = None
_model_lock = threading.Lock()
_prewarmed = False


class _ActiveJob:
    """进程内当前任务句柄。

    单例模型 + BATCH_SIZE=1，任一时刻至多一个转写任务，因此用一个全局
    cancel 标志即可表达「取消当前任务」。
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.started_at: Optional[float] = None

    def begin(self) -> "threading.Event":
        with self.lock:
            self.cancel.clear()
            self.started_at = time.time()
            return self.cancel

    def end(self) -> None:
        with self.lock:
            self.started_at = None

    @property
    def is_active(self) -> bool:
        with self.lock:
            return self.started_at is not None


_ACTIVE = _ActiveJob()

# SenseVoice outputs: <|lang|><|emotion|><|event|><|textnorm|>text
_TAG_PATTERN = re.compile(r"<\|([^|]*)\|>")

_EMOTION_TAGS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL"}
_EVENT_TAGS = {"Speech", "Applause", "BGM", "Laughter", "Cry", "Cough", "Sneeze", "Breath", "Music"}
_LANG_TAGS = {"zh", "en", "ja", "ko", "yue", "nospeech"}


def _parse_rich_text(raw_text: str) -> dict:
    """Parse SenseVoice rich transcription tags from raw output.

    Returns dict with keys: text, language, emotion, event
    """
    tags = _TAG_PATTERN.findall(raw_text)
    clean_text = _TAG_PATTERN.sub("", raw_text).strip()

    language = None
    emotion = None
    event = None

    for tag in tags:
        tag_upper = tag.upper()
        tag_orig = tag
        if tag_orig in _LANG_TAGS:
            language = tag_orig
        elif tag_upper in _EMOTION_TAGS:
            emotion = tag_upper.lower()
        elif tag_orig in _EVENT_TAGS:
            event = tag_orig
        # textnorm tags (withitn/woitn) are internal, skip

    return {
        "text": clean_text,
        "language": language,
        "emotion": emotion,
        "event": event,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    from funasr import AutoModel

    kwargs = {
        "model": MODEL_ID,
        "hub": "hf",
        # SenseVoiceSmall 与 campplus 都是 funasr 原生模型，不需要远程代码。
        # 必须设为 False：为 True 时 funasr 会执行 install_model_requirements，
        # 即对一个无超时的裸 `pip install -r <model>/requirements.txt` 子进程，
        # 在受限网络下会卡死容器启动，并尝试把 numpy 降级到 <=1.26.4。
        "trust_remote_code": False,
        "disable_update": True,
        "device": DEVICE,
    }
    if ENABLE_SPK:
        kwargs["spk_model"] = "cam++"
    if VAD_MODEL:
        # merge_length_s 必须放在 AutoModel kwargs 里，而不是 generate()：
        # inference_with_vad() 先读 kwargs.get("merge_length_s", 15) 才做
        # deep_update(kwargs, cfg)，所以 generate() 里传不进去。
        kwargs["vad_model"] = VAD_MODEL
        kwargs["vad_kwargs"] = {"max_single_segment_time": VAD_MAX_SEGMENT_MS}
        kwargs["merge_length_s"] = VAD_MERGE_LENGTH_S

    _model = AutoModel(**kwargs)

    if PREWARM_SECONDS > 0:
        await run_in_threadpool(_prewarm)

    yield
    _model = None


def _prewarm() -> None:
    """推理一次音频，预热 torch/oneDNN 图。

    CPU 上首个真实请求会因 torch 内核编译/图缓存而阻塞 2-4 分钟，
    用户侧表现为「一直转、最后报超时」。预热后首个请求也只需正常耗时。
    PREWARM_SECONDS=0 可关闭。

    注意：预热点必须是「有声」信号，不能用纯静音或低电平噪声。开了 VAD 后
    VAD 会判定静音/噪声为无语音，直接短路返回空结果，ASR 编码器完全不会执行，
    预热就失效了（实测静音 0.06s / 噪声 0.05s / 音调 1.30s，只有音调跑到
    encoder）。这里用 440Hz 音调 + 淡入淡出，确保 VAD 放行。
    """
    import wave as _wave

    global _prewarmed
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
            with _wave.open(tmp.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                n = int(16000 * PREWARM_SECONDS)
                frames = bytearray()
                for i in range(n):
                    t = i / 16000.0
                    fade = min(1.0, t / 0.02, (PREWARM_SECONDS - t) / 0.02)
                    v = 0.3 * math.sin(2 * math.pi * 440.0 * t) * fade
                    frames += struct.pack(
                        "<h", int(max(-1.0, min(1.0, v)) * 32767)
                    )
                w.writeframes(bytes(frames))

        t0 = time.time()
        with _model_lock:
            _model.generate(
                input=path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=300,
                merge_vad=True,
            )
        print(f"[prewarm] done in {time.time() - t0:.1f}s", flush=True)
        _prewarmed = True
    except Exception as e:
        print(f"[prewarm] failed: {e}", flush=True)
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


app = FastAPI(title="SenseVoice-API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok" if _model else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
        "prewarmed": _prewarmed,
        "features": ["emotion", "event", "language_detection", "timestamps", "itn"],
        "spk": ENABLE_SPK,
        "vad": VAD_MODEL or None,
        "vad_max_segment_s": VAD_MAX_SEGMENT_MS / 1000.0,
        "vad_merge_length_s": VAD_MERGE_LENGTH_S,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "sensevoice-small",
                "object": "model",
                "owned_by": "FunAudioLLM",
                "capabilities": {
                    "languages": ["zh", "en", "ja", "ko", "yue"],
                    "emotion_detection": True,
                    "event_detection": True,
                    "timestamps": True,
                },
            }
        ],
    }


@app.post("/v1/cancel")
async def cancel_transcription() -> JSONResponse:
    """取消当前正在进行的转写任务。

    funasr 的 generate() 是单次同步调用，内部没有可抢占的边界，
    因此取消的实际语义是：客户端断开后，后端跳过结果后处理并丢弃结果。
    generate() 本身会自然跑完，无法在途中打断。
    该端点之所以能返回，依赖推理被推入线程池（run_in_threadpool），
    事件循环未阻塞。
    """
    was_active = _ACTIVE.is_active
    _ACTIVE.cancel.set()
    return JSONResponse(
        content={
            "cancelled": True,
            "was_active": was_active,
            "note": (
                "result is discarded and post-processing is skipped; "
                "the in-flight generate() call finishes on its own"
            ),
        }
    )


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    timestamp_granularities: Optional[str] = Form(default=None),
):
    """OpenAI-compatible audio transcription endpoint.

    SenseVoice extensions (returned in verbose_json):
    - emotion: detected emotion (happy, sad, angry, neutral)
    - event: detected audio event (Speech, Music, Applause, Laughter, etc.)
    - language: auto-detected language code

    实现要点：
    - 推理通过 run_in_threadpool 推入线程池。否则同步 CPU 计算会霸占事件
      循环，转写期间 /health 与 /v1/cancel 都无法响应。
    - 每个任务一个 cancel 标志，供 /v1/cancel 设置。
    - 推理完成后若已被取消，返回 499 + {"cancelled": true} 并丢弃结果。
    """
    if not _model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    audio_bytes = await file.read()

    if MAX_AUDIO_SECONDS > 0:
        duration = _get_duration(audio_bytes)
        if duration and duration > MAX_AUDIO_SECONDS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"audio duration {duration:.1f}s exceeds "
                    f"MAX_AUDIO_SECONDS={MAX_AUDIO_SECONDS:.0f}s"
                ),
            )

    cancel_evt = _ACTIVE.begin()
    try:
        return await run_in_threadpool(
            _transcribe_sync,
            audio_bytes,
            _get_suffix(file.filename),
            language,
            response_format,
            timestamp_granularities,
            cancel_evt,
        )
    finally:
        _ACTIVE.end()


def _transcribe_sync(
    audio_bytes: bytes,
    suffix: str,
    language: Optional[str],
    response_format: str,
    timestamp_granularities: Optional[str],
    cancel_evt: "threading.Event",
) -> JSONResponse:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        if cancel_evt.is_set():
            return JSONResponse(content={"cancelled": True}, status_code=499)

        want_timestamps = (
            response_format == "verbose_json"
            or (timestamp_granularities and "word" in timestamp_granularities)
        )

        start_time = time.time()
        with _model_lock:
            result = _model.generate(
                input=tmp_path,
                cache={},
                language=_map_language(language),
                use_itn=True,
                batch_size_s=300,
                merge_vad=True,
                output_timestamp=want_timestamps,
            )
        elapsed = time.time() - start_time
    finally:
        os.unlink(tmp_path)

    # 推理已结束但客户端已取消：丢弃结果，跳过后续后处理
    if cancel_evt.is_set():
        return JSONResponse(content={"cancelled": True}, status_code=499)

    raw_text = _extract_raw(result)
    parsed = _parse_rich_text(raw_text)

    if response_format == "verbose_json":
        response = {
            "task": "transcribe",
            "language": parsed["language"] or language or "auto",
            "duration": _get_duration(audio_bytes),
            "text": parsed["text"],
            "emotion": parsed["emotion"],
            "event": parsed["event"],
            "processing_time": round(elapsed, 3),
        }

        # 说话人分段（cam++ + VAD 的 sentence_info）与字级时间戳都要返回，
        # 两者不互斥：原始 result 同时带 sentence_info / timestamp / words，
        # 写成 if/else 会让客户端在开启 ENABLE_SPK 后拿不到字级时间戳。
        speaker_segments = _extract_speaker_segments(result)
        if speaker_segments:
            response["segments"] = speaker_segments
        word_segments = _extract_timestamps(result)
        if word_segments:
            response["words"] = word_segments

        return JSONResponse(content=response)

    # Standard OpenAI json format
    return JSONResponse(content={"text": parsed["text"]})


@app.websocket("/v1/audio/transcriptions/stream")
async def transcribe_stream(websocket: WebSocket):
    """WebSocket streaming transcription.

    Send raw audio bytes, receive JSON with text, emotion, event, language.
    """
    await websocket.accept()

    if not _model:
        await websocket.send_json({"error": "Model not loaded"})
        await websocket.close(code=1013)
        return

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                break

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
                with _model_lock:
                    result = _model.generate(
                        input=tmp_path,
                        cache={},
                        language="auto",
                        use_itn=True,
                        batch_size_s=300,
                        merge_vad=True,
                    )
            finally:
                os.unlink(tmp_path)

            raw_text = _extract_raw(result)
            parsed = _parse_rich_text(raw_text)

            await websocket.send_json({
                "text": parsed["text"],
                "language": parsed["language"],
                "emotion": parsed["emotion"],
                "event": parsed["event"],
                "is_final": True,
            })

    except WebSocketDisconnect:
        pass


def _extract_raw(result) -> str:
    if not result:
        return ""
    if isinstance(result, list) and len(result) > 0:
        item = result[0]
        if isinstance(item, dict):
            return item.get("text", "")
        if hasattr(item, "text"):
            return item.text
    return str(result)


def _extract_speaker_segments(result) -> list:
    """Extract speaker diarization segments from cam++ sentence_info."""
    if not result or not isinstance(result, list) or len(result) == 0:
        return []
    item = result[0]
    if not isinstance(item, dict):
        return []
    sentence_info = item.get("sentence_info", [])
    if not sentence_info:
        return []
    segments = []
    for info in sentence_info:
        if isinstance(info, dict):
            raw = info.get("text", "")
            parsed = _parse_rich_text(raw)
            segments.append({
                "start": round(info.get("start", 0) / 1000.0, 3),
                "end": round(info.get("end", 0) / 1000.0, 3),
                "text": parsed["text"],
                "speaker": info.get("spk", None),
                "emotion": parsed["emotion"],
            })
    return segments


def _extract_timestamps(result) -> list:
    """Extract word-level timestamps from FunASR result."""
    if not result or not isinstance(result, list) or len(result) == 0:
        return []

    item = result[0]
    if not isinstance(item, dict):
        return []

    timestamps = item.get("timestamp", [])
    words = item.get("words", [])

    if not timestamps or not words:
        return []

    segments = []
    for i, ts in enumerate(timestamps):
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            word = words[i] if i < len(words) else ""
            segments.append({
                "word": word,
                "start": round(ts[0] / 1000.0, 3),
                "end": round(ts[1] / 1000.0, 3),
            })

    return segments


def _map_language(lang: str | None) -> str:
    if not lang:
        return "auto"
    mapping = {
        "zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "yue": "yue",
        "chinese": "zh", "english": "en", "japanese": "ja",
        "korean": "ko", "cantonese": "yue",
    }
    return mapping.get(lang.lower(), "auto")


def _get_suffix(filename: str | None) -> str:
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix
    return ".wav"


def _get_duration(audio_bytes: bytes) -> float:
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        return round(len(data) / sr, 2)
    except Exception:
        return 0.0
