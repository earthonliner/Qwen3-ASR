# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Qwen3-ASR 课堂转写网页工具（本地运行，适配 Apple Silicon / MPS）。

功能：
  1. 点击按钮开始/停止录音，边录边（准实时）显示识别文字；停止后自动保存录音 wav 和转写 txt。
  2. 上传音频/视频文件（wav / mp3 / m4a / mp4 …），显示并保存识别文本。
  3. 历史记录列表，可在线查看/下载已保存的转写结果。

启动：
    # MLX 后端（推荐，默认模型）
    python scripts/web_app.py
    # transformers 原生后端（-hf 模型）
    python scripts/web_app.py --model Qwen/Qwen3-ASR-0.6B-hf
    # 然后浏览器打开 http://127.0.0.1:8000

说明：
  - 「实时」是通过分段增量转写实现的（浏览器每隔数秒上传音频块，服务端对当前段落
    反复转写刷新结果；段落超过 --segment-seconds 后固化，开始新段落）。
  - 模型推理串行化，多人同时使用会排队。
"""

import argparse
import datetime as _dt
import os
import queue
import re
import sys
import tempfile
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

# 复用命令行脚本里的设备选择 / 模型检查 / 输入构造 / 解码逻辑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import record_and_transcribe as rat  # noqa: E402

from flask import Flask, jsonify, render_template, request, send_from_directory  # noqa: E402

SAMPLE_RATE = rat.SAMPLE_RATE

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui"),
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui"),
    static_url_path="/static",
)

ENGINE: Optional["ASREngine"] = None
OUTPUT_DIR = "./recordings"
SEGMENT_SECONDS = 25.0     # 每段最长时长，超过后固化该段、开启新段
CHUNK_SECONDS_FILE = 30.0  # 文件转写的分段时长


# ---------------------------------------------------------------------------
# 推理引擎（两种后端复用命令行脚本的逻辑）
# ---------------------------------------------------------------------------
class ASREngine:
    def __init__(self, model_path: str, backend: str, device: str, dtype_str: str,
                 aligner: Optional[str] = None, max_new_tokens: int = 1024):
        self.model_path = model_path
        self.backend = rat.resolve_backend(backend, model_path)
        self.max_new_tokens = max_new_tokens
        self.lock = threading.Lock()  # 串行化推理，避免并发挤爆内存

        if self.backend == "mlx":
            # MLX 自行管理设备（Apple Silicon / Metal），无需 torch。
            # MLX 的计算流(Stream)绑定创建线程；Flask 每个请求在不同线程，直接调用会报
            # "There is no Stream(gpu, N) in current thread"。
            # 因此加载与推理全部固定在一个专用工作线程里，通过队列提交任务。
            self.device = "mlx"
            self.dtype = None
            print("[engine] 后端：mlx（Apple Silicon / Metal）")
            self._mlx_jobs: "queue.Queue" = queue.Queue()
            self._mlx_ready = threading.Event()
            self._mlx_load_error: Optional[BaseException] = None
            self._mlx_thread = threading.Thread(
                target=self._mlx_worker, args=(model_path,), daemon=True, name="mlx-worker",
            )
            self._mlx_thread.start()
            self._mlx_ready.wait()
            if self._mlx_load_error is not None:
                raise SystemExit(f"MLX 模型加载失败：{self._mlx_load_error}")
        else:
            from transformers import AutoProcessor

            self.device = rat.pick_device(device)
            self.dtype = rat.pick_dtype(dtype_str, self.device)
            print(f"[engine] 后端：transformers，设备：{self.device}，"
                  f"精度：{str(self.dtype).replace('torch.', '')}")

            rat._ensure_torch_enabled_in_transformers()
            rat._ensure_native_qwen3_asr_support(model_path)
            asr_cls = rat._import_asr_model_class()

            print(f"[engine] 正在加载 transformers 原生模型：{model_path} ……")
            t0 = time.time()
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = asr_cls.from_pretrained(model_path, dtype=self.dtype)
            self.model.to(self.device)
            self.model.eval()
            print(f"[engine] 模型加载完成，用时 {time.time() - t0:.1f} 秒。")

    def transcribe(self, audio_16k: np.ndarray, language: Optional[str]) -> Tuple[str, str]:
        """转写一段 16k float32 音频，返回 (language, text)。"""
        if audio_16k is None or audio_16k.size < int(0.3 * SAMPLE_RATE):
            return language or "", ""

        with self.lock:
            if self.backend == "mlx":
                return self._transcribe_mlx(audio_16k, language)
            return self._transcribe_transformers(audio_16k, language)

    def _mlx_worker(self, model_path: str) -> None:
        """专用 MLX 线程：加载模型 + 顺序执行所有推理任务。"""
        try:
            print(f"[engine] 正在加载 MLX 模型：{model_path} ……")
            t0 = time.time()
            self.model = rat._load_mlx_model(model_path)
            print(f"[engine] 模型加载完成，用时 {time.time() - t0:.1f} 秒。")
        except BaseException as e:  # SystemExit 也要拦截并回传主线程
            self._mlx_load_error = e
            self._mlx_ready.set()
            return
        self._mlx_ready.set()

        while True:
            audio_path, language, holder, done = self._mlx_jobs.get()
            try:
                gen_kwargs = rat._mlx_generate_kwargs(self.model, language, self.max_new_tokens)
                result = self.model.generate(audio_path, **gen_kwargs)
                holder["result"] = rat._parse_raw_asr(getattr(result, "text", "") or "", language)
            except Exception as e:
                holder["error"] = e
            finally:
                done.set()

    def _transcribe_mlx(self, audio_16k: np.ndarray, language: Optional[str]) -> Tuple[str, str]:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            rat.save_wav(audio_16k, tmp_path)
            holder: dict = {}
            done = threading.Event()
            self._mlx_jobs.put((tmp_path, language, holder, done))
            done.wait()
            if "error" in holder:
                raise holder["error"]
            return holder["result"]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _transcribe_transformers(self, audio_16k: np.ndarray, language: Optional[str]) -> Tuple[str, str]:
        import torch

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            rat.save_wav(audio_16k, tmp_path)
            inputs = rat._prepare_asr_inputs(self.processor, tmp_path, language)
            inputs = inputs.to(self.model.device, self.model.dtype)
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            return rat._decode_asr(self.processor, generated_ids, language)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ---------------------------------------------------------------------------
# 录音会话（准实时：分段增量转写）
# ---------------------------------------------------------------------------
class LiveSession:
    def __init__(self, language: Optional[str]):
        self.id = uuid.uuid4().hex[:12]
        self.language = language
        self.created = _dt.datetime.now()
        self.lock = threading.Lock()

        self.done_audio: List[np.ndarray] = []   # 已固化段落的音频
        self.seg_audio = np.zeros((0,), dtype=np.float32)  # 当前段落音频
        self.final_text_parts: List[str] = []    # 已固化段落文本
        self.partial_text = ""                   # 当前段落最新转写
        self.languages: List[str] = []

    def total_seconds(self) -> float:
        n = sum(int(a.shape[0]) for a in self.done_audio) + int(self.seg_audio.shape[0])
        return n / float(SAMPLE_RATE)

    def full_audio(self) -> np.ndarray:
        parts = list(self.done_audio)
        if self.seg_audio.size > 0:
            parts.append(self.seg_audio)
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    def add_language(self, lang: str) -> None:
        lang = (lang or "").strip()
        if lang and (not self.languages or self.languages[-1] != lang):
            self.languages.append(lang)


SESSIONS: Dict[str, LiveSession] = {}
SESSIONS_LOCK = threading.Lock()


def _safe_name(name: str) -> str:
    name = os.path.basename(name or "")
    return re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", name).strip("._") or "audio"


def _save_transcript(prefix: str, text: str, language: str, source: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{stamp}.txt"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Qwen3-ASR 课堂转写\n")
        f.write(f"# 时间：{_dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# 来源：{source}\n")
        f.write(f"# 语言：{language or '(未知)'}\n\n")
        f.write((text or "").strip() + "\n")
    return fname


# ---------------------------------------------------------------------------
# 页面与 API
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/session/start")
def session_start():
    data = request.get_json(silent=True) or {}
    language = (data.get("language") or "").strip() or None
    s = LiveSession(language=language)
    with SESSIONS_LOCK:
        SESSIONS[s.id] = s
    return jsonify({"session_id": s.id})


@app.post("/api/session/<sid>/chunk")
def session_chunk(sid: str):
    s = SESSIONS.get(sid)
    if s is None:
        return jsonify({"error": "会话不存在或已结束"}), 404

    raw = request.get_data()
    if not raw:
        return jsonify({"error": "空音频块"}), 400
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    with s.lock:
        s.seg_audio = np.concatenate([s.seg_audio, pcm], axis=0)
        seg = s.seg_audio.copy()

    # 对当前段落做增量转写（引擎内部有全局锁）
    lang, text = ENGINE.transcribe(seg, s.language)

    finalized = False
    with s.lock:
        s.partial_text = text
        s.add_language(lang)
        # 段落到达上限 => 固化，开启新段落
        if s.seg_audio.shape[0] >= int(SEGMENT_SECONDS * SAMPLE_RATE):
            s.done_audio.append(s.seg_audio)
            s.seg_audio = np.zeros((0,), dtype=np.float32)
            if text.strip():
                s.final_text_parts.append(text.strip())
            s.partial_text = ""
            finalized = True

        return jsonify({
            "final_text": " ".join(s.final_text_parts),
            "partial_text": s.partial_text,
            "language": ",".join(s.languages),
            "seconds": round(s.total_seconds(), 1),
            "finalized": finalized,
        })


@app.post("/api/session/<sid>/stop")
def session_stop(sid: str):
    with SESSIONS_LOCK:
        s = SESSIONS.pop(sid, None)
    if s is None:
        return jsonify({"error": "会话不存在或已结束"}), 404

    # 收尾：转写剩余段落
    with s.lock:
        seg = s.seg_audio.copy()
    if seg.size >= int(0.3 * SAMPLE_RATE):
        lang, text = ENGINE.transcribe(seg, s.language)
        with s.lock:
            s.add_language(lang)
            if text.strip():
                s.final_text_parts.append(text.strip())
            s.done_audio.append(seg)
            s.seg_audio = np.zeros((0,), dtype=np.float32)
            s.partial_text = ""

    full_text = " ".join(s.final_text_parts).strip()
    language = ",".join(s.languages)

    saved = {}
    audio = s.full_audio()
    if audio.size > 0:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_name = f"live_{stamp}.wav"
        rat.save_wav(audio, os.path.join(OUTPUT_DIR, wav_name))
        saved["audio"] = wav_name
    if full_text:
        saved["transcript"] = _save_transcript("live", full_text, language, source="浏览器录音")

    return jsonify({
        "final_text": full_text,
        "language": language,
        "seconds": round(audio.shape[0] / float(SAMPLE_RATE), 1),
        "saved": saved,
    })


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "未选择文件"}), 400
    language = (request.form.get("language") or "").strip() or None

    suffix = os.path.splitext(f.filename)[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        f.save(tmp_path)
        try:
            audio = rat.load_audio_16k(tmp_path)
        except Exception as e:
            return jsonify({"error": f"音频解码失败（确认已安装 ffmpeg，格式受支持）：{e}"}), 400

        chunks = rat.split_fixed(audio, CHUNK_SECONDS_FILE)
        texts: List[str] = []
        langs: List[str] = []
        for chunk, _off in chunks:
            lang, text = ENGINE.transcribe(chunk, language)
            if text.strip():
                texts.append(text.strip())
            if lang and (not langs or langs[-1] != lang):
                langs.append(lang)

        full_text = " ".join(texts).strip()
        merged_lang = ",".join(langs)
        saved = {}
        if full_text:
            saved["transcript"] = _save_transcript(
                f"upload_{_safe_name(os.path.splitext(f.filename)[0])[:40]}",
                full_text, merged_lang, source=f"上传文件：{f.filename}",
            )
        return jsonify({
            "text": full_text,
            "language": merged_lang,
            "seconds": round(audio.shape[0] / float(SAMPLE_RATE), 1),
            "chunks": len(chunks),
            "saved": saved,
        })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/history")
def history():
    items = []
    for name in os.listdir(OUTPUT_DIR):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(OUTPUT_DIR, name)
        st = os.stat(path)
        items.append({
            "name": name,
            "size": st.st_size,
            "mtime": _dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify({"items": items[:100]})


@app.get("/files/<path:name>")
def files(name: str):
    return send_from_directory(os.path.abspath(OUTPUT_DIR), name, as_attachment=False)


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qwen3-ASR 课堂转写网页工具（录音准实时转写 + 文件上传转写）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str, default="mlx-community/Qwen3-ASR-0.6B-8bit",
                   help="ASR 模型名或本地目录。名字含 mlx 用 MLX 后端（推荐），带 -hf 用 transformers 原生后端。")
    p.add_argument("--backend", type=str, default="auto",
                   choices=["auto", "mlx", "transformers"],
                   help="推理后端；auto 按模型名自动选择。")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"],
                   help="transformers 后端的运行设备；auto 优先 mps。MLX 后端自行管理设备。")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"], help="计算精度。")
    p.add_argument("--max-new-tokens", type=int, default=1024, help="每段最多生成的 token 数。")
    p.add_argument("--segment-seconds", type=float, default=25.0,
                   help="录音准实时转写的段落上限（秒），越小固化越快、越大上下文越连贯。")
    p.add_argument("--output-dir", type=str, default="./recordings", help="录音与转写结果保存目录。")
    p.add_argument("--host", type=str, default="127.0.0.1", help="监听地址（0.0.0.0 可局域网访问）。")
    p.add_argument("--port", type=int, default=8000, help="监听端口。")
    return p


def main() -> None:
    global ENGINE, OUTPUT_DIR, SEGMENT_SECONDS

    args = build_arg_parser().parse_args()
    OUTPUT_DIR = args.output_dir
    SEGMENT_SECONDS = float(args.segment_seconds)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ENGINE = ASREngine(
        model_path=args.model,
        backend=args.backend,
        device=args.device,
        dtype_str=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n打开浏览器访问： http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}\n")
    # threaded=True：录音 chunk 与页面请求并行；模型推理由引擎内部锁串行化
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
