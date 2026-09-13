import http.client
import json
import os
import socket
import threading
import time
import uuid
from urllib.parse import urlsplit

import gradio as gr
import requests

# ===== 配置 =====
ASR_API_URL = os.environ.get("ASR_API_URL", "http://sensevoice-asr:10095").rstrip("/")
# 默认 1 小时：CPU 转写速度取决于机器性能，且首次请求含 torch 预热开销
TIMEOUT = int(os.environ.get("ASR_TIMEOUT", "3600"))
PORT = int(os.environ.get("WEBUI_PORT", "7860"))
HEALTH_INTERVAL = 15.0   # 后端状态轮询间隔（秒）
PROGRESS_STEP = 2.0      # 进度刷新间隔（秒）

TRANSCRIBE_ENDPOINT = f"{ASR_API_URL}/v1/audio/transcriptions"
CANCEL_ENDPOINT = f"{ASR_API_URL}/v1/cancel"
HEALTH_ENDPOINT = f"{ASR_API_URL}/health"

# 当前任务的取消标志（单例：同一时刻只跑一个任务）
_CURRENT = {"cancel": None}


class _Cancelled(Exception):
    """客户端主动取消。"""


# ===== 格式化工具 =====
def format_timestamp(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS.mmm"""
    if seconds is None:
        return "00:00:00.000"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _fmt_dur(seconds):
    if seconds is None:
        return "未知"
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.0f} 秒"
    m = int(s // 60)
    if m < 60:
        return f"{m} 分 {int(s % 60)} 秒"
    return f"{m // 60} 小时 {m % 60} 分"


def _fmt_size(nbytes):
    if nbytes is None:
        return "未知"
    b = max(0.0, float(nbytes))
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b:.1f} {unit}" if unit != "B" else f"{int(b)} B"
        b /= 1024
    return f"{b:.1f} GB"


def _audio_info(path):
    """返回 (时长秒, 字节数)。时长解析失败时时长为 None。"""
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    dur = None
    try:
        from mutagen import File as _MFile

        audio = _MFile(path)
        if audio is not None and audio.info is not None:
            length = getattr(audio.info, "length", None)
            if length:
                dur = float(length)
    except Exception:
        dur = None
    return dur, size


def _seg_meta(seg):
    """取 (说话人, 情感)，统一转成字符串。

    后端返回的 speaker 是 **int**（distribute_spk 写入的 0-based id），
    直接拼进 `" | ".join(...)` 会抛
    `sequence item 0: expected str instance, int found`。
    用 `in (None, "")` 判断而不是真值判断：说话人 0 不能被当作假值丢掉。
    """
    speaker = seg.get("speaker")
    emotion = seg.get("emotion")
    speaker = "" if speaker in (None, "") else str(speaker)
    emotion = "" if emotion in (None, "") else str(emotion)
    return speaker, emotion


def build_srt(segments):
    """将 segments 转换为 SRT 字幕格式"""
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.get("start"))
        end = format_timestamp(seg.get("end"))
        text = str(seg.get("text", "")).strip()
        speaker, emotion = _seg_meta(seg)
        tag = ""
        if speaker:
            tag += f"[{speaker}]"
        if emotion:
            tag += f"({emotion})"
        if tag:
            text = f"{tag} {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def build_readable(segments):
    """将 segments 转换为带时间轴、说话人和情感的可读文本"""
    lines = []
    for seg in segments:
        start = format_timestamp(seg.get("start"))
        end = format_timestamp(seg.get("end"))
        text = str(seg.get("text", "")).strip()
        speaker, emotion = _seg_meta(seg)

        meta_parts = []
        if speaker:
            meta_parts.append(f"说话人{speaker}")
        if emotion:
            meta_parts.append(f"情感:{emotion}")
        meta = f" [{' | '.join(meta_parts)}]" if meta_parts else ""

        lines.append(f"[{start} - {end}]{meta} {text}")
    return "\n".join(lines)


def build_word_level(words):
    """将 words 数组转换为字级时间轴文本"""
    lines = []
    for w in words:
        start = format_timestamp(w.get("start"))
        end = format_timestamp(w.get("end"))
        word = w.get("word", "")
        lines.append(f"[{start} - {end}] {word}")
    return "\n".join(lines)


# ===== 带取消能力的 HTTP 客户端 =====
def _post_transcribe(audio_path, form_fields, timeout_s, cancel_evt: threading.Event):
    """提交转写请求，返回 (status, body_bytes)。

    后端在推理期间不发送任何响应，客户端只能阻塞在「等响应头」。
    这里把 socket 超时设得很短并轮询，让 cancel 标志真正生效：
    命中取消就抛 _Cancelled（finally 中关闭连接，后端随之断连）。
    超过 timeout_s 抛 TimeoutError。
    """
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    u = urlsplit(TRANSCRIBE_ENDPOINT)
    host = u.hostname
    port = u.port or 80
    path = (u.path or "/") + (("?" + u.query) if u.query else "")

    boundary = "----sensevoice" + uuid.uuid4().hex
    parts = bytearray()
    for k, v in form_fields.items():
        parts += (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
        ).encode()
    parts += (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; "
        f"filename=\"{os.path.basename(audio_path)}\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode()
    parts += audio_bytes
    parts += f"\r\n--{boundary}--\r\n".encode()

    deadline = time.time() + timeout_s
    conn = http.client.HTTPConnection(host, port, timeout=PROGRESS_STEP)
    try:
        conn.request(
            "POST",
            path,
            body=bytes(parts),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(parts)),
            },
        )

        # 阻塞点 1：等响应头（后端推理完成前不会有任何字节）
        resp = None
        while resp is None:
            if cancel_evt.is_set():
                raise _Cancelled()
            if time.time() > deadline:
                raise TimeoutError()
            try:
                resp = conn.getresponse()
            except socket.timeout:
                continue

        # 阻塞点 2：读响应体
        body = bytearray()
        while True:
            if cancel_evt.is_set():
                raise _Cancelled()
            if time.time() > deadline:
                raise TimeoutError()
            try:
                chunk = resp.read(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            body += chunk
        return resp.status, bytes(body)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _notify_cancel():
    """通知后端丢弃当前任务结果。失败静默。"""
    try:
        requests.post(CANCEL_ENDPOINT, timeout=5)
    except Exception:
        pass


# ===== 转写主流程（Gradio generator：边跑边推进度） =====
def _out(text, meta, srt, raw, status, running):
    """统一 7 个输出，保证 generator 每次 yield 的元数一致。"""
    return (
        text,
        meta,
        srt,
        raw,
        status,
        gr.update(interactive=not running),
        gr.update(interactive=running),
    )


def run_transcription(audio_file, response_format, timeout_s):
    if not audio_file:
        yield _out("请先上传音频文件", "", "", "", "⏸ 未开始", False)
        return

    try:
        timeout_s = min(86400.0, max(30.0, float(timeout_s)))
    except (TypeError, ValueError):
        timeout_s = float(TIMEOUT)

    dur, size = _audio_info(audio_file)
    pre = f"📁 **{os.path.basename(audio_file)}**\n"
    pre += f"大小：{_fmt_size(size)}\n"
    if dur is not None:
        pre += f"时长：{_fmt_dur(dur)}　·　CPU 估算耗时 ≈ {_fmt_dur(dur / 0.6)}\n"
    pre += f"超时阈值：{timeout_s:.0f}s"

    cancel_evt = threading.Event()
    _CURRENT["cancel"] = cancel_evt
    box = {}

    def _worker():
        try:
            box["resp"] = _post_transcribe(
                audio_file,
                {"response_format": response_format, "language": "auto"},
                timeout_s,
                cancel_evt,
            )
        except BaseException as exc:
            box["err"] = exc

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    t0 = time.time()
    last = ""

    # 运行期间每 PROGRESS_STEP 秒刷新一次，避免用户以为卡死
    while th.is_alive():
        line = f"⏳ **已运行 {time.time() - t0:.0f}s** — 后端 CPU 满载推理中，可点「取消」终止"
        if line != last:
            yield _out(
                "",
                pre + "\n\n" + line,
                "",
                "",
                f"⏳ 已运行 {time.time() - t0:.0f}s",
                True,
            )
            last = line
        time.sleep(PROGRESS_STEP)
    th.join()
    total = time.time() - t0

    if "err" in box:
        err = box["err"]
        if isinstance(err, _Cancelled):
            yield _out(
                "",
                f"⏹ **已取消**\n\n{pre}\n\n已运行 {total:.0f}s，"
                "请求已终止、结果已丢弃。\n\n"
                "⚠️ 注意：后端 funasr 是单次同步推理调用，无法在途中抢占，"
                "服务端计算会自然跑完，但结果不会返回、连接已关闭。",
                "",
                "",
                "⏹ 已取消",
                False,
            )
        elif isinstance(err, TimeoutError):
            yield _out(
                "",
                f"⏰ **请求超时**（>{timeout_s:.0f}s）\n\n{pre}\n\n"
                "可放大「超时时间」或缩短音频后重试。",
                "",
                "",
                "⏰ 超时",
                False,
            )
        else:
            yield _out("", f"❌ 发生错误：`{err}`\n\n{pre}", "", "", "❌ 错误", False)
        return

    status, body = box["resp"]
    if status == 499:
        yield _out(
            "",
            f"⏹ **已取消**\n\n{pre}\n\n后端已收到取消并丢弃结果。",
            "",
            "",
            "⏹ 已取消",
            False,
        )
        return

    if status != 200:
        yield _out(
            "",
            f"❌ 请求失败 HTTP {status}\n\n{pre}\n\n`{body[:400].decode('utf-8', 'replace')}`",
            "",
            "",
            f"❌ HTTP {status}",
            False,
        )
        return

    try:
        result = json.loads(body.decode("utf-8"))
    except Exception as e:
        yield _out("", f"❌ 响应解析失败：`{e}`\n\n{pre}", "", "", "❌ 解析错误", False)
        return

    full_text = result.get("text", "")
    language = result.get("language", "未知")
    emotion = result.get("emotion", "")
    event = result.get("event", "")

    meta_info = f"语种：{language}"
    if emotion:
        meta_info += f" | 情感：{emotion}"
    if event:
        meta_info += f" | 事件：{event}"
    meta_info += f" | 后端处理耗时：{result.get('processing_time', '—')}s"
    meta_info += f" | 端到端耗时：{total:.1f}s"

    segments = result.get("segments", [])
    words = result.get("words", [])
    # 渲染失败不能把已经跑完的转写结果弄丢：降级为展示原始 JSON
    try:
        if segments:
            readable = build_readable(segments)
            srt = build_srt(segments)
        elif words:
            readable = build_word_level(words)
            srt = "（字级模式无分段，无法生成 SRT）"
        else:
            readable = "（当前返回格式不包含时间轴信息，请使用 verbose_json）"
            srt = "（当前返回格式不包含时间轴信息）"
    except Exception as e:
        readable = (
            f"（分段渲染失败，已降级为原始 JSON：`{type(e).__name__}: {e}`）"
        )
        srt = "（渲染失败，见右侧原始 JSON）"

    raw_json = json.dumps(result, ensure_ascii=False, indent=2)
    yield _out(full_text, meta_info + "\n\n" + readable, srt, raw_json, f"✅ 完成 · {total:.1f}s", False)


def do_cancel():
    """取消当前运行中的任务。"""
    evt = _CURRENT.get("cancel")
    if evt is not None:
        evt.set()
    _notify_cancel()
    return "🚫 已发送取消请求"


async def refresh_health():
    """轮询后端健康状态。"""
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        j = r.json()
        spk = "开启" if j.get("spk") else "关闭"
        feats = ", ".join(j.get("features", []))
        return (
            f"🟢 **后端正常**　`{j.get('device')}`　模型 `{j.get('model')}`　"
            f"说话人识别：{spk}　能力：{feats}"
        )
    except Exception:
        return "🔴 **后端不可达** — 请确认 ASR 容器已启动"


# ===== Gradio 界面 =====
with gr.Blocks(title="SenseVoice 语音转文字") as demo:
    gr.Markdown(
        "## 🎙️ SenseVoice 语音转文字\n"
        "上传音频文件，获取带**时间轴**、**说话人标签**和**情感识别**的转写结果。"
    )

    health_status = gr.Markdown("🟡 正在检测后端状态…")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="上传音频",
                type="filepath",
                sources=["upload", "microphone"],
            )
            response_format = gr.Dropdown(
                label="返回格式",
                choices=["verbose_json", "json"],
                value="verbose_json",
                info="verbose_json 返回时间轴和说话人信息",
            )
            timeout_input = gr.Number(
                label="超时时间（秒）",
                value=TIMEOUT,
                minimum=30,
                maximum=86400,
                step=30,
                precision=0,
                info="首次请求含 torch 预热开销，长音频请适当调大",
            )
            with gr.Row():
                submit_btn = gr.Button("开始转写", variant="primary", size="lg", scale=2)
                cancel_btn = gr.Button("取消", variant="stop", size="lg", scale=1, interactive=False)

        with gr.Column(scale=2):
            status_output = gr.Textbox(label="任务状态", lines=1)
            text_output = gr.Textbox(label="完整文本", lines=4)
            meta_output = gr.Textbox(label="元信息与时间轴", lines=12)
            srt_output = gr.Textbox(label="SRT 字幕格式", lines=12)
            json_output = gr.Code(label="原始 JSON", language="json", lines=8)

    cancel_hint = gr.Textbox(label="取消反馈", lines=1, value=" ")

    submit_btn.click(
        fn=run_transcription,
        inputs=[audio_input, response_format, timeout_input],
        outputs=[text_output, meta_output, srt_output, json_output, status_output, submit_btn, cancel_btn],
    )
    cancel_btn.click(fn=do_cancel, outputs=[cancel_hint])

    gr.Timer(HEALTH_INTERVAL).tick(fn=refresh_health, outputs=health_status)
    demo.load(fn=refresh_health, outputs=health_status)

    gr.Markdown(
        "**说明**：\n"
        "- 容器重启后的**第一个请求**含 torch 预热，可能需数分钟，后续请求正常。\n"
        "- 「取消」会立即断开连接并通知后端丢弃结果，界面立刻可用；但 funasr 是单次同步推理、"
        "无法在途中抢占，服务端计算会自然跑完（结果不返回）。\n"
        "- 说话人识别由后端环境变量 `ENABLE_SPK=true` 决定，见上方后端状态。\n"
        "- 后端可选环境变量 `MAX_AUDIO_SECONDS`（默认 0 = 不限制），用于快速拒绝超长音频，"
        "避免长时间占住单例模型。\n"
        "- 说话人识别开启时返回说话人分段；情感检测支持开心/伤心/愤怒/中性；"
        "事件检测支持语音/音乐/掌声/笑声/背景音乐等；语种支持中文/英文/日文/韩文/粤语。"
    )

if __name__ == "__main__":
    # concurrency_limit 按事件监听器计，因此 cancel 能与正在跑的转写并行执行；
    # max_size 是全局队列上限，必须留够余量，否则转写运行时「取消」会被拒
    demo.queue(default_concurrency_limit=1, max_size=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        theme=gr.themes.Soft(),
        show_error=True,
    )
