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
在 Apple Silicon Mac（如 MacBook Air M2 / 16GB）上，用 Qwen3-ASR-0.6B 录制并转写
课堂上老师与学生的对话。

支持两套模型/后端，脚本会根据模型名自动选择：

1) transformers 原生版（推荐给带 -hf 后缀的仓库，如 Qwen/Qwen3-ASR-0.6B-hf）
   - 用 🤗 transformers 原生的 AutoModelForMultimodalLM + processor.apply_transcription_request
   - 需要较新的 transformers（含 Qwen3-ASR 原生支持）

2) qwen-asr 包版（用于不带 -hf 的仓库，如 Qwen/Qwen3-ASR-0.6B）
   - 用 qwen_asr.Qwen3ASRModel

两种使用方式：

   # 现场录音后转写（讲课过程中运行，结束时按 Ctrl+C 停止录音）
   python scripts/record_and_transcribe.py --record --model Qwen/Qwen3-ASR-0.6B-hf --timestamps

   # 转写一个已有的音频/视频文件（wav / mp3 / m4a / mp4 …，需已安装 ffmpeg）
   python scripts/record_and_transcribe.py --audio ./lesson.m4a --model Qwen/Qwen3-ASR-0.6B-hf --language Chinese

说明：
- macOS 上通过 MPS(Metal) 做 GPU 加速；vLLM 在 macOS 上不可用。
- 0.6B 模型在 16GB 内存的 M2 上可离线运行；长音频会自动分段处理。
- 该模型只做「语音识别 + 时间戳」，不做说话人分离(diarization)，
  因此无法自动区分「谁是老师、谁是学生」；时间戳可帮助你按时间顺序回看对话。
"""

import argparse
import datetime as _dt
import os
import re
import sys
import tempfile
import time
from typing import List, Optional, Tuple

# 让 MPS 上不支持的算子自动回退到 CPU，避免在 Apple Silicon 上直接报错。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

SAMPLE_RATE = 16000  # Qwen3-ASR 使用 16kHz 单声道输入


# ---------------------------------------------------------------------------
# 后端 / 设备 / 精度选择
# ---------------------------------------------------------------------------
def resolve_backend(backend: str, model_name: str) -> str:
    """auto 时：带 -hf 后缀走 transformers 原生后端，否则走 qwen-asr 包后端。"""
    if backend != "auto":
        return backend
    base = os.path.basename(str(model_name).rstrip("/"))
    return "transformers" if base.endswith("-hf") else "package"


def pick_device(requested: str) -> str:
    """选择运行设备：auto 时优先 Apple Silicon 的 mps，否则回退到 cpu。"""
    import torch

    if requested != "auto":
        return requested

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def pick_dtype(requested: str, device: str):
    """选择精度：mps 默认 float16，cpu 默认 float32（更稳）。"""
    import torch

    if requested != "auto":
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[requested]

    if device.startswith("mps"):
        return torch.float16
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


# ---------------------------------------------------------------------------
# 录音 / 音频读取
# ---------------------------------------------------------------------------
def record_from_microphone(max_seconds: Optional[float] = None) -> np.ndarray:
    """
    从默认麦克风录制 16kHz 单声道音频。

    - 若 max_seconds 为 None：一直录音直到用户按 Ctrl+C。
    - 若指定了 max_seconds：录满该秒数后自动停止（也可提前 Ctrl+C）。

    返回 float32、范围约 [-1, 1] 的一维 numpy 数组。
    """
    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "缺少 sounddevice 依赖，无法录音。请先安装：\n"
            "    pip install sounddevice\n"
            "并确保系统已安装 portaudio（macOS：brew install portaudio）。\n"
            f"原始错误：{e}"
        )

    frames = []

    def _callback(indata, _frames, _time, status):
        if status:
            print(f"[录音警告] {status}", file=sys.stderr)
        frames.append(indata.copy())

    print("=" * 60)
    print("开始录音……对着麦克风讲课即可。")
    if max_seconds is None:
        print("按 Ctrl+C 结束录音并开始转写。")
    else:
        print(f"将最多录制 {max_seconds:.0f} 秒（也可随时按 Ctrl+C 提前结束）。")
    print("=" * 60)

    start = time.time()
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=_callback,
        ):
            while True:
                sd.sleep(200)
                elapsed = time.time() - start
                print(f"\r已录制 {elapsed:6.1f} 秒", end="", flush=True)
                if max_seconds is not None and elapsed >= max_seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print()

    if not frames:
        raise SystemExit("没有录到任何音频，请检查麦克风权限（系统设置 → 隐私与安全性 → 麦克风）。")

    audio = np.concatenate(frames, axis=0).reshape(-1).astype(np.float32)
    print(f"录音结束，共 {len(audio) / SAMPLE_RATE:.1f} 秒。")
    return audio


def load_audio_16k(path: str) -> np.ndarray:
    """把本地音频/视频文件读取为 16kHz 单声道 float32 波形。"""
    import librosa

    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(audio, dtype=np.float32)


def save_wav(audio: np.ndarray, path: str) -> None:
    import soundfile as sf

    sf.write(path, audio, SAMPLE_RATE)


def split_fixed(audio: np.ndarray, chunk_seconds: float) -> List[Tuple[np.ndarray, float]]:
    """把长音频按固定时长切成若干段，返回 [(chunk, offset_sec), ...]。"""
    if chunk_seconds is None or chunk_seconds <= 0 or len(audio) <= int(chunk_seconds * SAMPLE_RATE):
        return [(audio, 0.0)]
    step = int(chunk_seconds * SAMPLE_RATE)
    out: List[Tuple[np.ndarray, float]] = []
    for start in range(0, len(audio), step):
        out.append((audio[start : start + step], start / float(SAMPLE_RATE)))
    return out


# ---------------------------------------------------------------------------
# 输出格式化
# ---------------------------------------------------------------------------
def _fmt_ts(seconds: float) -> str:
    total = float(seconds)
    h, rem = divmod(int(total), 3600)
    m, s = divmod(rem, 60)
    ms = int((total - int(total)) * 1000)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def format_timestamps(timestamps: Optional[List[dict]]) -> str:
    """把统一后的时间戳列表 [{text,start_time,end_time}] 汇总为可读文本。"""
    if not timestamps:
        return ""
    lines = []
    for item in timestamps:
        lines.append(f"[{_fmt_ts(item['start_time'])} -> {_fmt_ts(item['end_time'])}] {item['text']}")
    return "\n".join(lines)


def write_outputs(args, text: str, language: str, timestamps: Optional[List[dict]],
                  saved_wav_path: Optional[str], source: Optional[str], stamp: str) -> None:
    print("=" * 60)
    print(f"检测/使用语言：{language or '(未知)'}")
    print("-" * 60)
    print(text or "(未识别到语音内容)")
    print("=" * 60)

    out_path = args.output or os.path.join(args.output_dir, f"transcript_{stamp}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Qwen3-ASR 课堂转写\n")
        f.write(f"# 时间：{_dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# 模型：{args.model}\n")
        if saved_wav_path:
            f.write(f"# 录音文件：{saved_wav_path}\n")
        elif source:
            f.write(f"# 源文件：{source}\n")
        f.write(f"# 语言：{language or '(未知)'}\n\n")
        f.write("## 全文\n")
        f.write((text or "").strip() + "\n")
        if args.timestamps:
            ts_text = format_timestamps(timestamps)
            if ts_text:
                f.write("\n## 时间戳（逐词/逐字）\n")
                f.write(ts_text + "\n")

    print(f"\n转写文本已保存到：{out_path}")
    if args.timestamps and timestamps:
        print("（文件中包含逐词/逐字时间戳，可按时间顺序回看老师与学生的对话。）")


# ---------------------------------------------------------------------------
# 后端 A：transformers 原生（-hf 模型）
# ---------------------------------------------------------------------------
def _ensure_torch_enabled_in_transformers() -> None:
    """
    新版 transformers 需要较新的 PyTorch（>= 2.4）。若 PyTorch 太旧，transformers 会
    「禁用 PyTorch」，导致模型无法加载、processor 也缺少音频相关方法。这里给出精确报错。
    """
    import torch

    try:
        from transformers.utils import is_torch_available
    except Exception:
        is_torch_available = None

    torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    disabled = (is_torch_available is not None) and (not is_torch_available())

    if disabled or torch_ver < (2, 4):
        import platform

        arch_hint = ""
        if platform.system() == "Darwin" and platform.machine() == "x86_64":
            arch_hint = (
                "\n\n⚠️ 检测到当前 Python 是 x86_64（Intel / Rosetta）架构！\n"
                "   PyTorch 在 macOS 上最后的 Intel 版本就是 2.2.2，所以 pip 找不到 >=2.4，\n"
                "   这几乎肯定是因为你的 conda/Python 环境不是 Apple Silicon 原生（arm64）。\n"
                "   需要新建一个原生 arm64 环境（示例）：\n"
                "       CONDA_SUBDIR=osx-arm64 conda create -n qwen3-asr-arm python=3.12 -y\n"
                "       conda activate qwen3-asr-arm\n"
                "       conda config --env --set subdir osx-arm64\n"
                "       python -c \"import platform; print(platform.machine())\"   # 应显示 arm64\n"
                "   然后重新安装依赖：pip install -U 'torch>=2.4' torchaudio transformers ..."
            )
        raise SystemExit(
            f"检测到 PyTorch {torch.__version__}，但当前 transformers 需要 PyTorch >= 2.4，"
            "否则会禁用 PyTorch，导致 -hf 模型无法加载。\n\n"
            "请升级 PyTorch（Apple Silicon 原生 wheel 已内置 MPS）：\n"
            "    pip install -U 'torch>=2.4' torchaudio"
            f"{arch_hint}\n\n升级后重新运行本脚本即可。"
        )


def _ensure_native_qwen3_asr_support(model_path: str) -> None:
    """
    检查当前 transformers 是否原生认识 `qwen3_asr` 架构。很多已发布版本尚未包含，
    需要从源码安装开发版。这里在真正加载前给出清晰指引，避免难懂的 KeyError 堆栈。
    """
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(model_path)
    except Exception as e:  # KeyError / ValueError: model type qwen3_asr not recognized
        msg = str(e)
        if "qwen3_asr" in msg or "does not recognize this architecture" in msg:
            raise SystemExit(
                "当前 transformers 不认识 `qwen3_asr` 架构，说明这个版本还没有内置 Qwen3-ASR 原生支持。\n"
                "请安装含该支持的 transformers 开发版（-hf 模型必需）：\n\n"
                "    pip install -U 'git+https://github.com/huggingface/transformers'\n\n"
                "安装后可用这条命令确认：\n"
                "    python -c \"from transformers import Qwen3ASRForConditionalGeneration; print('ok')\"\n"
            )
        raise


def _import_asr_model_class():
    import transformers

    # 优先用显式类，避免 AutoModelForMultimodalLM 在某些版本上解析成空模型类型。
    for name in ("Qwen3ASRForConditionalGeneration", "AutoModelForMultimodalLM"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise SystemExit(
        "当前 transformers 版本没有 Qwen3-ASR 的原生支持（找不到 Qwen3ASRForConditionalGeneration / "
        "AutoModelForMultimodalLM）。\n"
        "使用 -hf 模型需要较新的 transformers，请升级：\n"
        "    pip install -U transformers\n"
        "或安装开发版：\n"
        "    pip install -U git+https://github.com/huggingface/transformers\n"
    )


def _prepare_asr_inputs(processor, audio_path: str, language: Optional[str]):
    """构造 ASR 模型输入；优先用 apply_transcription_request，缺失时回退到 chat template。"""
    if hasattr(processor, "apply_transcription_request"):
        return processor.apply_transcription_request(audio=audio_path, language=language)

    # 回退：手动拼 chat template（兼容不同 transformers 版本）
    messages = []
    if language:
        messages.append({"role": "system", "content": [{"type": "text", "text": language}]})
    messages.append({"role": "user", "content": [{"type": "audio", "path": audio_path}]})
    return processor.apply_chat_template(
        [messages], tokenize=True, return_dict=True, add_generation_prompt=True,
    )


def _parse_raw_asr(raw: str, forced_language: Optional[str]) -> Tuple[str, str]:
    """从原始解码文本里解析出 (language, text)，兼容 'language X<asr_text>...' 格式。"""
    s = (raw or "").strip()
    lang = forced_language or ""
    text = s
    if "<asr_text>" in s:
        meta, text = s.split("<asr_text>", 1)
        m = re.search(r"language\s+([A-Za-z]+)", meta)
        if m and not lang:
            lang = m.group(1)
    # 去掉可能残留的特殊 token，如 <|im_end|> / <asr_text> 等
    text = re.sub(r"<\|.*?\|>", "", text)
    text = text.replace("<asr_text>", "").strip()
    return lang, text


def _decode_asr(processor, generated_ids, forced_language: Optional[str]) -> Tuple[str, str]:
    """解码单条 ASR 输出，返回 (language, text)；优先用 return_format='parsed'。"""
    try:
        parsed = processor.decode(generated_ids, return_format="parsed")[0]
        return (parsed.get("language") or forced_language or "",
                (parsed.get("transcription") or "").strip())
    except (TypeError, AttributeError, KeyError):
        texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
        return _parse_raw_asr(texts[0] if texts else "", forced_language)


def run_transformers_backend(args, audio_16k: np.ndarray, device: str, dtype,
                             saved_wav_path: Optional[str], source: Optional[str], stamp: str) -> None:
    import torch
    from transformers import AutoProcessor

    _ensure_torch_enabled_in_transformers()
    _ensure_native_qwen3_asr_support(args.model)
    asr_cls = _import_asr_model_class()

    print(f"正在加载 transformers 原生模型：{args.model}（首次运行会自动下载权重，请耐心等待）……")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = asr_cls.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    aligner_model = None
    aligner_processor = None
    if args.timestamps:
        from transformers import AutoModelForTokenClassification

        print(f"正在加载对齐模型：{args.aligner} ……")
        aligner_processor = AutoProcessor.from_pretrained(args.aligner)
        aligner_model = AutoModelForTokenClassification.from_pretrained(args.aligner, dtype=dtype)
        aligner_model.to(device)
        aligner_model.eval()
    print(f"模型加载完成，用时 {time.time() - t0:.1f} 秒。开始转写……")

    chunks = split_fixed(audio_16k, args.chunk_seconds)
    if len(chunks) > 1:
        print(f"音频较长，已按 {args.chunk_seconds:.0f} 秒切成 {len(chunks)} 段处理。")

    all_text: List[str] = []
    all_lang: List[str] = []
    all_ts: List[dict] = []

    t0 = time.time()
    for idx, (chunk, offset) in enumerate(chunks):
        if len(chunks) > 1:
            print(f"\r正在转写第 {idx + 1}/{len(chunks)} 段……", end="", flush=True)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            save_wav(chunk, tmp_path)

            inputs = _prepare_asr_inputs(processor, tmp_path, args.language)
            inputs = inputs.to(model.device, model.dtype)
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            lang, text = _decode_asr(processor, generated_ids, args.language)

            if text:
                all_text.append(text)
            if lang:
                all_lang.append(lang)

            if args.timestamps and text:
                ts = _align_transformers(
                    aligner_model, aligner_processor, tmp_path, text,
                    lang or args.language or "English", offset, device,
                )
                all_ts.extend(ts)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if len(chunks) > 1:
        print()
    print(f"转写完成，用时 {time.time() - t0:.1f} 秒。\n")

    # 合并语言（去掉连续重复）
    merged_lang: List[str] = []
    for l in all_lang:
        if l and (not merged_lang or merged_lang[-1] != l):
            merged_lang.append(l)

    write_outputs(
        args,
        text=" ".join(all_text).strip(),
        language=",".join(merged_lang),
        timestamps=all_ts if args.timestamps else None,
        saved_wav_path=saved_wav_path,
        source=source,
        stamp=stamp,
    )


def _align_transformers(aligner_model, aligner_processor, audio_path: str, transcript: str,
                        language: str, offset_sec: float, device: str) -> List[dict]:
    """用 transformers 原生的 ForcedAligner 计算单段时间戳，并加上时间偏移。"""
    import torch

    try:
        aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
            audio=audio_path, transcript=transcript, language=language,
        )
        aligner_inputs = aligner_inputs.to(aligner_model.device, aligner_model.dtype)
        with torch.inference_mode():
            outputs = aligner_model(**aligner_inputs)
        timestamps = aligner_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=aligner_model.config.timestamp_token_id,
        )[0]
    except Exception as e:  # 对齐失败不应影响转写主流程
        print(f"\n[对齐警告] 本段时间戳计算失败，已跳过：{e}", file=sys.stderr)
        return []

    out: List[dict] = []
    for item in timestamps:
        out.append({
            "text": item["text"],
            "start_time": float(item["start_time"]) + offset_sec,
            "end_time": float(item["end_time"]) + offset_sec,
        })
    return out


# ---------------------------------------------------------------------------
# 后端 B：qwen-asr 包（非 -hf 模型）
# ---------------------------------------------------------------------------
def run_package_backend(args, audio_input, device: str, dtype,
                        saved_wav_path: Optional[str], source: Optional[str], stamp: str) -> None:
    from qwen_asr import Qwen3ASRModel

    load_kwargs = dict(
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    if args.timestamps:
        load_kwargs["forced_aligner"] = args.aligner
        load_kwargs["forced_aligner_kwargs"] = dict(dtype=dtype, device_map=device)

    print(f"正在加载 qwen-asr 模型：{args.model}（首次运行会自动下载权重，请耐心等待）……")
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(args.model, **load_kwargs)
    print(f"模型加载完成，用时 {time.time() - t0:.1f} 秒。开始转写……")

    t0 = time.time()
    results = model.transcribe(
        audio=audio_input,
        language=args.language,
        return_time_stamps=args.timestamps,
    )
    print(f"转写完成，用时 {time.time() - t0:.1f} 秒。\n")

    result = results[0]
    timestamps = None
    if args.timestamps and getattr(result, "time_stamps", None):
        timestamps = [
            {"text": it.text, "start_time": float(it.start_time), "end_time": float(it.end_time)}
            for it in result.time_stamps
        ]

    write_outputs(
        args,
        text=result.text or "",
        language=result.language or "",
        timestamps=timestamps,
        saved_wav_path=saved_wav_path,
        source=source,
        stamp=stamp,
    )


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="在 Apple Silicon Mac 上用 Qwen3-ASR-0.6B 录制并转写课堂对话。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--record", action="store_true", help="从麦克风现场录音后转写。")
    src.add_argument("--audio", type=str, help="转写已有的本地音频/视频文件路径（需 ffmpeg 支持的格式）。")

    p.add_argument("--duration", type=float, default=None,
                   help="录音模式下的最长录制秒数；不填则一直录到 Ctrl+C。")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-0.6B-hf",
                   help="ASR 模型名或本地目录。带 -hf 用 transformers 原生后端，不带 -hf 用 qwen-asr 包后端。")
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "transformers", "package"],
                   help="推理后端；auto 会根据模型名是否带 -hf 自动选择。")
    p.add_argument("--language", type=str, default=None,
                   help="强制识别语言（如 Chinese / English，或 zh / en）；不填则自动检测，适合中英混说的课堂。")
    p.add_argument("--timestamps", action="store_true",
                   help="输出逐词/逐字时间戳（会额外加载 ForcedAligner，占用更多内存/时间）。")
    p.add_argument("--aligner", type=str, default=None,
                   help="时间戳对齐模型（不填则根据后端自动选择 -hf / 非 -hf 版本）。")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "mps", "cpu", "cuda:0"],
                   help="运行设备；auto 会在 Apple Silicon 上自动选择 mps。")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"],
                   help="计算精度；auto 时 mps 用 float16、cpu 用 float32。")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="每段最多生成的 token 数；课堂长音频建议设大一些。")
    p.add_argument("--chunk-seconds", type=float, default=30.0,
                   help="transformers 后端处理长音频时的分段时长（秒）。qwen-asr 后端会自行分段，忽略该值。")
    p.add_argument("--output-dir", type=str, default="./recordings",
                   help="录音与转写结果的保存目录。")
    p.add_argument("--output", type=str, default=None,
                   help="转写文本输出文件路径；不填则自动放到 --output-dir 下。")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    backend = resolve_backend(args.backend, args.model)

    # 自动选择对齐模型版本，与后端保持一致（-hf 配 -hf）。
    if args.aligner is None:
        args.aligner = (
            "Qwen/Qwen3-ForcedAligner-0.6B-hf" if backend == "transformers"
            else "Qwen/Qwen3-ForcedAligner-0.6B"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) 取得音频输入
    saved_wav_path = None
    source = None
    audio_16k = None       # transformers 后端使用的 16k np 数组
    audio_input = None     # qwen-asr 后端使用的输入（(array, sr) 或 路径）

    if args.record:
        audio_16k = record_from_microphone(max_seconds=args.duration)
        saved_wav_path = os.path.join(args.output_dir, f"class_{stamp}.wav")
        save_wav(audio_16k, saved_wav_path)
        print(f"录音已保存到：{saved_wav_path}")
        audio_input = (audio_16k, SAMPLE_RATE)
    else:
        is_url = str(args.audio).startswith(("http://", "https://"))
        if backend == "transformers" and is_url:
            raise SystemExit("transformers 后端仅支持本地文件；请先把音频下载到本地，再用 --audio 指向它。")
        if not is_url and not os.path.exists(args.audio):
            raise SystemExit(f"找不到音频文件：{args.audio}")
        source = args.audio
        audio_input = args.audio  # qwen-asr 后端可直接吃路径 / URL

    # 2) 设备 / 精度
    import torch  # noqa: F401

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    print(f"后端：{backend}，设备：{device}，精度：{str(dtype).replace('torch.', '')}")
    if device == "cpu":
        print("[提示] 未启用 MPS，正在使用 CPU，速度会明显偏慢。")

    # 3) 分后端执行
    if backend == "transformers":
        if audio_16k is None:
            audio_16k = load_audio_16k(args.audio)
        run_transformers_backend(args, audio_16k, device, dtype, saved_wav_path, source, stamp)
    else:
        run_package_backend(args, audio_input, device, dtype, saved_wav_path, source, stamp)


if __name__ == "__main__":
    main()
