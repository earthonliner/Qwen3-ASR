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

两种使用方式：

1) 现场录音后转写（讲课过程中运行，结束时按 Ctrl+C 停止录音）：
       python scripts/record_and_transcribe.py --record --timestamps

2) 转写一个已有的音频/视频文件（wav / mp3 / m4a / mp4 …，需已安装 ffmpeg）：
       python scripts/record_and_transcribe.py --audio ./lesson.m4a --language Chinese

说明：
- 该脚本使用 transformers 后端 + Apple Silicon 的 MPS(Metal) 加速；vLLM 在 macOS 上不可用。
- 0.6B 模型在 16GB 内存的 M2 上可以离线运行；长音频会自动分段处理。
- 模型本身只做「语音识别 + 时间戳」，不做说话人分离(diarization)，
  因此无法自动区分「谁是老师、谁是学生」；时间戳可帮助你按时间顺序回看对话。
"""

import argparse
import datetime as _dt
import os
import sys
import time
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000  # Qwen3-ASR 使用 16kHz 单声道输入


# ---------------------------------------------------------------------------
# 设备 / 精度自动选择
# ---------------------------------------------------------------------------
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
# 录音
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
    except Exception as e:  # pragma: no cover - 依赖缺失时的友好提示
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
        print()  # 换行

    if not frames:
        raise SystemExit("没有录到任何音频，请检查麦克风权限（系统设置 → 隐私与安全性 → 麦克风）。")

    audio = np.concatenate(frames, axis=0).reshape(-1).astype(np.float32)
    print(f"录音结束，共 {len(audio) / SAMPLE_RATE:.1f} 秒。")
    return audio


def save_wav(audio: np.ndarray, path: str) -> None:
    import soundfile as sf

    sf.write(path, audio, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# 输出格式化
# ---------------------------------------------------------------------------
def _fmt_ts(seconds: float) -> str:
    td = _dt.timedelta(seconds=float(seconds))
    total = td.total_seconds()
    h, rem = divmod(int(total), 3600)
    m, s = divmod(rem, 60)
    ms = int((total - int(total)) * 1000)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def format_timestamps(result) -> str:
    """把 forced aligner 的逐词/逐字时间戳汇总为一段可读文本。"""
    ts = getattr(result, "time_stamps", None)
    if not ts:
        return ""
    lines = []
    for item in ts:
        lines.append(f"[{_fmt_ts(item.start_time)} -> {_fmt_ts(item.end_time)}] {item.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="在 Apple Silicon Mac 上用 Qwen3-ASR-0.6B 录制并转写课堂对话。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--record", action="store_true", help="从麦克风现场录音后转写。")
    src.add_argument("--audio", type=str, help="转写已有的音频/视频文件路径（需 ffmpeg 支持的格式）。")

    p.add_argument("--duration", type=float, default=None,
                   help="录音模式下的最长录制秒数；不填则一直录到 Ctrl+C。")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-0.6B",
                   help="ASR 模型名或本地目录（默认 0.6B，适合 16GB 的 M2）。")
    p.add_argument("--language", type=str, default=None,
                   help="强制识别语言（如 Chinese / English）；不填则自动检测，适合中英混说的课堂。")
    p.add_argument("--timestamps", action="store_true",
                   help="输出逐词/逐字时间戳（会额外加载 ForcedAligner，占用更多内存/时间）。")
    p.add_argument("--aligner", type=str, default="Qwen/Qwen3-ForcedAligner-0.6B",
                   help="时间戳对齐模型名或本地目录（配合 --timestamps 使用）。")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "mps", "cpu", "cuda:0"],
                   help="运行设备；auto 会在 Apple Silicon 上自动选择 mps。")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"],
                   help="计算精度；auto 时 mps 用 float16、cpu 用 float32。")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="每段最多生成的 token 数；课堂长音频建议设大一些。")
    p.add_argument("--output-dir", type=str, default="./recordings",
                   help="录音与转写结果的保存目录。")
    p.add_argument("--output", type=str, default=None,
                   help="转写文本输出文件路径；不填则自动放到 --output-dir 下。")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) 取得音频输入（录音或读取文件）
    audio_input = None  # 传给 model.transcribe 的对象
    saved_wav_path = None
    if args.record:
        audio = record_from_microphone(max_seconds=args.duration)
        saved_wav_path = os.path.join(args.output_dir, f"class_{stamp}.wav")
        save_wav(audio, saved_wav_path)
        print(f"录音已保存到：{saved_wav_path}")
        audio_input = (audio, SAMPLE_RATE)
    else:
        if not os.path.exists(args.audio):
            raise SystemExit(f"找不到音频文件：{args.audio}")
        audio_input = args.audio

    # 2) 选择设备/精度并加载模型（延迟 import，加快 --help）
    import torch  # noqa: F401
    from qwen_asr import Qwen3ASRModel

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    print(f"使用设备：{device}，精度：{str(dtype).replace('torch.', '')}")
    if device == "cpu":
        print("[提示] 未启用 MPS，正在使用 CPU，速度会明显偏慢。")

    load_kwargs = dict(
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,  # 单路课堂音频，batch=1 更省内存
        max_new_tokens=args.max_new_tokens,
    )
    if args.timestamps:
        load_kwargs["forced_aligner"] = args.aligner
        load_kwargs["forced_aligner_kwargs"] = dict(dtype=dtype, device_map=device)

    print(f"正在加载模型：{args.model}（首次运行会自动下载权重，请耐心等待）……")
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(args.model, **load_kwargs)
    print(f"模型加载完成，用时 {time.time() - t0:.1f} 秒。开始转写……")

    # 3) 转写
    t0 = time.time()
    results = model.transcribe(
        audio=audio_input,
        language=args.language,
        return_time_stamps=args.timestamps,
    )
    print(f"转写完成，用时 {time.time() - t0:.1f} 秒。\n")

    result = results[0]

    # 4) 打印 + 保存
    print("=" * 60)
    print(f"检测/使用语言：{result.language or '(未知)'}")
    print("-" * 60)
    print(result.text or "(未识别到语音内容)")
    print("=" * 60)

    out_path = args.output or os.path.join(args.output_dir, f"transcript_{stamp}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Qwen3-ASR 课堂转写\n")
        f.write(f"# 时间：{_dt.datetime.now().isoformat(timespec='seconds')}\n")
        if saved_wav_path:
            f.write(f"# 录音文件：{saved_wav_path}\n")
        elif args.audio:
            f.write(f"# 源文件：{args.audio}\n")
        f.write(f"# 语言：{result.language or '(未知)'}\n\n")
        f.write("## 全文\n")
        f.write((result.text or "").strip() + "\n")
        if args.timestamps:
            ts_text = format_timestamps(result)
            if ts_text:
                f.write("\n## 时间戳（逐词/逐字）\n")
                f.write(ts_text + "\n")

    print(f"\n转写文本已保存到：{out_path}")
    if args.timestamps:
        print("（文件中包含逐词/逐字时间戳，可按时间顺序回看老师与学生的对话。）")


if __name__ == "__main__":
    main()
