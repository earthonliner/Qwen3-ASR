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
在 Apple Silicon Mac（如 MacBook Air M2 / 16GB）上，用 Qwen3-ASR 录制并转写
课堂上老师与学生的对话。

支持两套模型/后端，按模型名自动选择：

1) MLX（推荐，Apple Silicon 上最快；模型名含 mlx，如 mlx-community/Qwen3-ASR-0.6B-8bit）
   - 依赖：pip install mlx-audio

2) transformers 原生（-hf 后缀仓库，如 Qwen/Qwen3-ASR-0.6B-hf，走 MPS/Metal）
   - 依赖：PyTorch >= 2.4 + 含 qwen3_asr 支持的 transformers（源码开发版）

两种使用方式：

   # 现场录音后转写（讲课过程中运行，结束时按 Ctrl+C 停止录音）
   python scripts/record_and_transcribe.py --record --timestamps

   # 转写一个已有的音频/视频文件（wav / mp3 / m4a / mp4 …，需已安装 ffmpeg）
   python scripts/record_and_transcribe.py --audio ./lesson.m4a --language Chinese

说明：
- 模型只做「语音识别 + 时间戳」，不做说话人分离(diarization)，
  无法自动区分「谁是老师、谁是学生」；时间戳可帮助按时间顺序回看对话。
- 长音频会按 --chunk-seconds 自动分段处理。
"""

import argparse
import datetime as _dt
import os
import re
import sys
import tempfile
import time
from typing import List, Optional, Tuple

# 让 MPS 上不支持的算子自动回退到 CPU（transformers 后端用）
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

SAMPLE_RATE = 16000  # Qwen3-ASR 使用 16kHz 单声道输入


# ---------------------------------------------------------------------------
# 后端 / 设备选择
# ---------------------------------------------------------------------------
def resolve_backend(backend: str, model_name: str) -> str:
    """
    auto 时按模型名判断：
      - 名字含 mlx（如 mlx-community/...）→ mlx 后端
      - 带 -hf 后缀 → transformers 原生后端
    """
    if backend != "auto":
        return backend
    name = str(model_name).rstrip("/")
    if "mlx" in name.lower():
        return "mlx"
    if os.path.basename(name).endswith("-hf"):
        return "transformers"
    raise SystemExit(
        f"无法从模型名判断后端：{model_name}\n"
        "本仓库仅支持 Apple Silicon 上的两种模型：\n"
        "  - MLX：mlx-community/Qwen3-ASR-0.6B-8bit（推荐）\n"
        "  - transformers 原生：Qwen/Qwen3-ASR-0.6B-hf\n"
        "也可用 --backend mlx / transformers 手动指定。"
    )


def pick_device(requested: str) -> str:
    """transformers 后端设备：auto 时优先 mps，否则 cpu。"""
    import torch

    if requested != "auto":
        return requested
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested: str, device: str):
    """transformers 后端精度：mps 用 float16，cpu 用 float32。"""
    import torch

    if requested != "auto":
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[requested]
    return torch.float16 if device.startswith("mps") else torch.float32


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
            "并确保系统已安装 portaudio（brew install portaudio）。\n"
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
# 输出格式化 / 解析
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
    """把时间戳列表 [{text,start_time,end_time}] 汇总为可读文本。"""
    if not timestamps:
        return ""
    lines = []
    for item in timestamps:
        lines.append(f"[{_fmt_ts(item['start_time'])} -> {_fmt_ts(item['end_time'])}] {item['text']}")
    return "\n".join(lines)


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
    text = re.sub(r"<\|.*?\|>", "", text)
    text = text.replace("<asr_text>", "").strip()
    return lang, text


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
# 后端 A：MLX（推荐）
# ---------------------------------------------------------------------------
def _patch_mlx_lm_transformers_compat() -> None:
    """
    兼容补丁：mlx-lm 在导入时调用 AutoTokenizer.register("NewlineTokenizer", ...)（传字符串），
    transformers v5 起要求传类对象（内部访问 key.__module__），导致
    `AttributeError: 'str' object has no attribute '__module__'`。
    这里包一层：字符串注册失败时静默跳过（Qwen3-ASR 不使用 NewlineTokenizer，跳过无影响）。
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return

    orig = AutoTokenizer.register.__func__ if hasattr(AutoTokenizer.register, "__func__") \
        else AutoTokenizer.register

    def _safe_register(config_class, *args, **kwargs):
        if isinstance(config_class, str):
            try:
                return orig(config_class, *args, **kwargs)
            except AttributeError:
                return None
        return orig(config_class, *args, **kwargs)

    AutoTokenizer.register = staticmethod(_safe_register)


def _load_mlx_model(model_path: str):
    """加载 mlx-audio 的 STT 模型，兼容不同版本的入口。"""
    _patch_mlx_lm_transformers_compat()
    try:
        try:
            from mlx_audio.stt.utils import load_model as _load
        except ImportError:
            from mlx_audio.stt import load as _load
    except ImportError as e:
        raise SystemExit(
            "缺少 mlx-audio，无法使用 MLX 后端。请安装：\n"
            "    pip install -U mlx-audio\n"
            f"原始错误：{e}"
        )
    return _load(model_path)


def _mlx_generate_kwargs(model, language: Optional[str], max_tokens: int) -> dict:
    """按 model.generate 的实际签名过滤 kwargs，兼容不同 mlx-audio 版本。"""
    import inspect

    candidates = {"language": language, "max_tokens": max_tokens}
    try:
        params = inspect.signature(model.generate).parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        return {k: v for k, v in candidates.items()
                if v is not None and (has_var_kw or k in params)}
    except (TypeError, ValueError):
        return {k: v for k, v in candidates.items() if v is not None}


def run_mlx_backend(args, audio_16k: np.ndarray,
                    saved_wav_path: Optional[str], source: Optional[str], stamp: str) -> None:
    print(f"正在加载 MLX 模型：{args.model}（首次运行会自动下载权重，请耐心等待）……")
    t0 = time.time()
    model = _load_mlx_model(args.model)

    aligner = None
    if args.timestamps:
        print(f"正在加载 MLX 对齐模型：{args.aligner} ……")
        aligner = _load_mlx_model(args.aligner)
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

            gen_kwargs = _mlx_generate_kwargs(model, args.language, args.max_new_tokens)
            result = model.generate(tmp_path, **gen_kwargs)
            lang, text = _parse_raw_asr(getattr(result, "text", "") or "", args.language)

            if text:
                all_text.append(text)
            if lang:
                all_lang.append(lang)

            if args.timestamps and text and aligner is not None:
                try:
                    align_kwargs = {"text": text}
                    if lang or args.language:
                        align_kwargs["language"] = lang or args.language
                    align_result = aligner.generate(tmp_path, **align_kwargs)
                    for item in align_result:
                        all_ts.append({
                            "text": item.text,
                            "start_time": float(item.start_time) + offset,
                            "end_time": float(item.end_time) + offset,
                        })
                except Exception as e:
                    print(f"\n[对齐警告] 本段时间戳计算失败，已跳过：{e}", file=sys.stderr)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if len(chunks) > 1:
        print()
    print(f"转写完成，用时 {time.time() - t0:.1f} 秒。\n")

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


# ---------------------------------------------------------------------------
# 后端 B：transformers 原生（-hf 模型，MPS）
# ---------------------------------------------------------------------------
def _ensure_torch_enabled_in_transformers() -> None:
    """
    新版 transformers 需要 PyTorch >= 2.4，否则会「禁用 PyTorch」，导致模型无法加载、
    processor 缺少音频相关方法。这里给出精确报错。
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
        if platform.machine() == "x86_64":
            arch_hint = (
                "\n\n⚠️ 检测到当前 Python 是 x86_64（Intel / Rosetta）架构！\n"
                "   PyTorch 在 macOS 上最后的 Intel 版本是 2.2.2，需要新建 Apple Silicon 原生（arm64）环境：\n"
                "       CONDA_SUBDIR=osx-arm64 conda create -n qwen3-asr-arm python=3.12 -y\n"
                "       conda activate qwen3-asr-arm\n"
                "       conda config --env --set subdir osx-arm64"
            )
        raise SystemExit(
            f"检测到 PyTorch {torch.__version__}，但当前 transformers 需要 PyTorch >= 2.4。\n"
            "请升级：pip install -U 'torch>=2.4' torchaudio"
            f"{arch_hint}"
        )


def _ensure_native_qwen3_asr_support(model_path: str) -> None:
    """检查当前 transformers 是否原生认识 `qwen3_asr` 架构（很多发布版尚未包含）。"""
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(model_path)
    except Exception as e:
        msg = str(e)
        if "qwen3_asr" in msg or "does not recognize this architecture" in msg:
            raise SystemExit(
                "当前 transformers 不认识 `qwen3_asr` 架构。请安装含该支持的开发版：\n"
                "    pip install -U 'git+https://github.com/huggingface/transformers'\n"
            )
        raise


def _import_asr_model_class():
    import transformers

    for name in ("Qwen3ASRForConditionalGeneration", "AutoModelForMultimodalLM"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise SystemExit(
        "当前 transformers 版本没有 Qwen3-ASR 的原生支持。请安装开发版：\n"
        "    pip install -U 'git+https://github.com/huggingface/transformers'\n"
    )


def _prepare_asr_inputs(processor, audio_path: str, language: Optional[str]):
    """构造 ASR 模型输入；优先用 apply_transcription_request，缺失时回退到 chat template。"""
    if hasattr(processor, "apply_transcription_request"):
        return processor.apply_transcription_request(audio=audio_path, language=language)

    messages = []
    if language:
        messages.append({"role": "system", "content": [{"type": "text", "text": language}]})
    messages.append({"role": "user", "content": [{"type": "audio", "path": audio_path}]})
    return processor.apply_chat_template(
        [messages], tokenize=True, return_dict=True, add_generation_prompt=True,
    )


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
                    lang or args.language or "English", offset,
                )
                all_ts.extend(ts)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if len(chunks) > 1:
        print()
    print(f"转写完成，用时 {time.time() - t0:.1f} 秒。\n")

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
                        language: str, offset_sec: float) -> List[dict]:
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
# 命令行
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="在 Apple Silicon Mac 上用 Qwen3-ASR 录制并转写课堂对话。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--record", action="store_true", help="从麦克风现场录音后转写。")
    src.add_argument("--audio", type=str, help="转写已有的本地音频/视频文件路径（需 ffmpeg 支持的格式）。")

    p.add_argument("--duration", type=float, default=None,
                   help="录音模式下的最长录制秒数；不填则一直录到 Ctrl+C。")
    p.add_argument("--model", type=str, default="mlx-community/Qwen3-ASR-0.6B-8bit",
                   help="ASR 模型名或本地目录。名字含 mlx 用 MLX 后端（推荐），带 -hf 用 transformers 原生后端。")
    p.add_argument("--backend", type=str, default="auto",
                   choices=["auto", "mlx", "transformers"],
                   help="推理后端；auto 按模型名自动选择。")
    p.add_argument("--language", type=str, default=None,
                   help="强制识别语言（如 Chinese / English）；不填则自动检测，适合中英混说的课堂。")
    p.add_argument("--timestamps", action="store_true",
                   help="输出逐词/逐字时间戳（会额外加载 ForcedAligner，占用更多内存/时间）。")
    p.add_argument("--aligner", type=str, default=None,
                   help="时间戳对齐模型（不填则根据后端自动选择）。")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"],
                   help="transformers 后端的运行设备；auto 优先 mps。MLX 后端自行管理设备。")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"],
                   help="transformers 后端的计算精度；auto 时 mps 用 float16。")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="每段最多生成的 token 数；课堂长音频建议设大一些。")
    p.add_argument("--chunk-seconds", type=float, default=30.0,
                   help="长音频的分段时长（秒）。")
    p.add_argument("--output-dir", type=str, default="./recordings",
                   help="录音与转写结果的保存目录。")
    p.add_argument("--output", type=str, default=None,
                   help="转写文本输出文件路径；不填则自动放到 --output-dir 下。")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    backend = resolve_backend(args.backend, args.model)

    # 自动选择对齐模型版本，与后端保持一致
    if args.aligner is None:
        args.aligner = (
            "mlx-community/Qwen3-ForcedAligner-0.6B-8bit" if backend == "mlx"
            else "Qwen/Qwen3-ForcedAligner-0.6B-hf"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) 取得音频输入
    saved_wav_path = None
    source = None

    if args.record:
        audio_16k = record_from_microphone(max_seconds=args.duration)
        saved_wav_path = os.path.join(args.output_dir, f"class_{stamp}.wav")
        save_wav(audio_16k, saved_wav_path)
        print(f"录音已保存到：{saved_wav_path}")
    else:
        if not os.path.exists(args.audio):
            raise SystemExit(f"找不到音频文件：{args.audio}")
        source = args.audio
        audio_16k = load_audio_16k(args.audio)

    # 2) 分后端执行
    if backend == "mlx":
        print("后端：mlx（Apple Silicon / Metal）")
        run_mlx_backend(args, audio_16k, saved_wav_path, source, stamp)
    else:
        device = pick_device(args.device)
        dtype = pick_dtype(args.dtype, device)
        print(f"后端：transformers，设备：{device}，精度：{str(dtype).replace('torch.', '')}")
        if device == "cpu":
            print("[提示] 未启用 MPS，正在使用 CPU，速度会明显偏慢。")
        run_transformers_backend(args, audio_16k, device, dtype, saved_wav_path, source, stamp)


if __name__ == "__main__":
    main()
