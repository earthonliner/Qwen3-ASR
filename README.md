# Qwen3-ASR

<br>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/logo.png" width="400"/>
<p>

<p align="center">
&nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen3-asr">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen/Qwen3-ASR">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwen.ai/blog?id=qwen3asr">Blog</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2601.21337">Paper</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen3-ASR">Hugging Face Demo</a>&nbsp&nbsp | &nbsp&nbsp 🖥️ <a href="https://modelscope.cn/studios/Qwen/Qwen3-ASR">ModelScope Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/qwen-speech-recognition">API</a>

</p>

We release **Qwen3-ASR**, a family that includes two powerful all-in-one speech recognition models that support language identification and ASR for 52 languages and dialects, as well as a novel non-autoregressive speech forced-alignment model that can align text–speech pairs in 11 languages.


## News
* 2026.6.26: 🤗🤗🤗 Native Transformers support is now available! It includes `torch.compile` support for faster inference. See the new model cards for example usage:
  - [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
  - [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)
  - [Qwen/Qwen3-ForcedAligner-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
* 2026.1.29: 🎉🎉🎉 We have released the [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) series (0.6B/1.7B) and the Qwen3-ForcedAligner-0.6B model. Please check out our [blog](https://qwen.ai/blog?id=qwen3asr)!


## Contents <!-- omit in toc -->

- [Overview](#overview)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Released Models Description and Download](#released-models-description-and-download)
- [Quickstart](#quickstart)
  - [Environment Setup](#environment-setup)
  - [Python Package Usage](#python-package-usage)
    - [Quick Inference](#quick-inference)
    - [vLLM Backend](#vllm-backend)
    - [Streaming Inference](#streaming-inference)
    - [ForcedAligner Usage](#forcedaligner-usage)
  - [DashScope API Usage](#dashscope-api-usage)
- [macOS (Apple Silicon) 本地运行指南 · 课堂录音转写](#macos-apple-silicon-本地运行指南--课堂录音转写)
- [Launch Local Web UI Demo](#launch-local-web-ui-demo)
  - [Gradio Demo](#gradio-demo)
  - [Streaming Demo](#streaming-demo)
- [Deployment with vLLM](#deployment-with-vllm)
- [Fine Tuning](#fine-tuning)
- [Docker](#docker)
- [Evaluation](#evaluation)
- [Citation](#citation)


## Overview

### Introduction

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/qwen3_asr_introduction.png" width="90%"/>
<p>

The Qwen3-ASR family includes Qwen3-ASR-1.7B and Qwen3-ASR-0.6B, which support language identification and ASR for 52 languages and dialects. Both leverage large-scale speech training data and the strong audio understanding capability of their foundation model, Qwen3-Omni. Experiments show that the 1.7B version achieves state-of-the-art performance among open-source ASR models and is competitive with the strongest proprietary commercial APIs. Here are the main features:

* **All-in-one**: Qwen3-ASR-1.7B and Qwen3-ASR-0.6B support language identification and speech recognition for 30 languages and 22 Chinese dialects, so as to English accents from multiple countries and regions.

* **Excellent and Fast**: The Qwen3-ASR family ASR models maintains high-quality and robust recognition under complex acoustic environments and challenging text patterns. Qwen3-ASR-1.7B achieves strong performance on both open-sourced and internal benchmarks. While the 0.6B version achieves accuracy-efficient trade-off, it reaches 2000 times throughput at a concurrency of 128. They both achieve streaming / offline unified inference with single model and support transcribe long audio.

* **Novel and strong forced alignment Solution**: We introduce Qwen3-ForcedAligner-0.6B, which supports timestamp prediction for arbitrary units within up to 5 minutes of speech in 11 languages. Evaluations show its timestamp accuracy surpasses E2E based forced-alignment models.

* **Comprehensive inference toolkit**: In addition to open-sourcing the architectures and weights of the Qwen3-ASR series, we also release a powerful, full-featured inference framework that supports vLLM-based batch inference, asynchronous serving, streaming inference, timestamp prediction, and more.

### Model Architecture

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/overview.jpg" width="100%"/>
<p>


### Released Models Description and Download

Below is an introduction and download information for the Qwen3-ASR models. Please select and download the model that fits your needs.

| Model | Supported Languages | Supported Dialects | Inference Mode | Audio Types |
|---|---|---|---|---|
| Qwen3-ASR-1.7B & Qwen3-ASR-0.6B | Chinese (zh), English (en), Cantonese (yue), Arabic (ar), German (de), French (fr), Spanish (es), Portuguese (pt), Indonesian (id), Italian (it), Korean (ko), Russian (ru), Thai (th), Vietnamese (vi), Japanese (ja), Turkish (tr), Hindi (hi), Malay (ms), Dutch (nl), Swedish (sv), Danish (da), Finnish (fi), Polish (pl), Czech (cs), Filipino (fil), Persian (fa), Greek (el), Hungarian (hu), Macedonian (mk), Romanian (ro) | Anhui, Dongbei, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei, Hunan, Jiangxi, Ningxia, Shandong, Shaanxi, Shanxi, Sichuan, Tianjin, Yunnan, Zhejiang, Cantonese (Hong Kong accent), Cantonese (Guangdong accent), Wu language, Minnan language. | Offline / Streaming | Speech, Singing Voice, Songs with BGM |
| Qwen3-ForcedAligner-0.6B | Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish | -- | NAR | Speech |

During model loading in the `qwen-asr` package or vLLM, model weights will be downloaded automatically based on the model name. However, if your runtime environment does not allow downloading weights during execution, you can use the following commands to manually download the model weights to a local directory:

```bash
# Download through ModelScope (recommended for users in Mainland China)
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B  --local_dir ./Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen3-ForcedAligner-0.6B
# Download through Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./Qwen3-ForcedAligner-0.6B
```


## Quickstart

### Environment Setup

The easiest way to use Qwen3-ASR is to install the `qwen-asr` Python package from PyPI. This will pull in the required runtime dependencies and allow you to load any released Qwen3-ASR model. If you’d like to simplify environment setup further, you can also use our official [Docker image](#docker). The `qwen-asr` package provides two backends: the transformers backend and the vLLM backend. For usage instructions for different backends, please refer to [Python Package Usage](#python-package-usage). We recommend using a **fresh, isolated environment** to avoid dependency conflicts with existing packages. You can create a clean Python 3.12 environment like this:

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
```

Run the following command to get the minimal installation with transformers-backend support:

```bash
pip install -U qwen-asr
```

To enable the vLLM backend for faster inference and streaming support, run:

```bash
pip install -U qwen-asr[vllm]
```

If you want to develop or modify the code locally, install from source in editable mode:

```bash
git clone https://github.com/QwenLM/Qwen3-ASR.git
cd Qwen3-ASR
pip install -e .
# support vLLM backend
# pip install -e ".[vllm]"
```

Additionally, we recommend using FlashAttention 2 to reduce GPU memory usage and accelerate inference speed, especially for long inputs and large batch sizes.

```bash
pip install -U flash-attn --no-build-isolation
```

If your machine has less than 96GB of RAM and lots of CPU cores, run:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention 2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

### Python Package Usage

#### Quick Inference

The `qwen-asr` package provides two backends: **transformers backend** and **vLLM backend**. You can pass audio inputs as a local path, a URL, base64 data, or a `(np.ndarray, sr)` tuple, and run batch inference. To quickly try Qwen3-ASR, you can use `Qwen3ASRModel.from_pretrained(...)` for the transformers backend with the following code:

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    language=None, # set "English" to force the language
)

print(results[0].language)
print(results[0].text)
```

If you want to return timestamps, pass `forced_aligner` and its init kwargs. Here is an example of batch inference with timestamps output:

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
        # attn_implementation="flash_attention_2",
    ),
)

results = model.transcribe(
    audio=[
      "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
      "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    ],
    language=["Chinese", "English"], # can also be set to None for automatic language detection
    return_time_stamps=True,
)

for r in results:
    print(r.language, r.text, r.time_stamps[0])
```

For more detailed usage examples, please refer to the [example code](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_transformers.py) for the transformers backend.

#### vLLM Backend

If you want the fastest inference speed with Qwen3-ASR, we strongly recommend using the vLLM backend by initializing the model with `Qwen3ASRModel.LLM(...)`. Example code is provided below. Note that you must install it via `pip install -U qwen-asr[vllm]`. If you want the model to output timestamps, it’s best to install FlashAttention via `pip install -U flash-attn --no-build-isolation` to speed up inference for the forced aligner model. Remember to wrap your code under `if __name__ == '__main__':` to avoid the `spawn` error described in [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/#python-multiprocessing).

```python
import torch
from qwen_asr import Qwen3ASRModel

if __name__ == '__main__':
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.7,
        max_inference_batch_size=128, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=4096, # Maximum number of tokens to generate. Set a larger value for long audio input.
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
        ),
    )

    results = model.transcribe(
        audio=[
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
        ],
        language=["Chinese", "English"], # can also be set to None for automatic language detection
        return_time_stamps=True,
    )

    for r in results:
        print(r.language, r.text, r.time_stamps[0])
```

For more detailed usage examples, please refer to the [example code](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm.py) for the vLLM backend. In addition, you can start a vLLM server via the `qwen-asr-serve` command, which is a wrapper around `vllm serve`. You can pass any arguments supported by `vllm serve`, for example:

```bash
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000
```

And send requests to the server via:

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
                    },
                }
            ],
        }
    ]
}

response = requests.post(url, headers=headers, json=data, timeout=300)
response.raise_for_status()
content = response.json()['choices'][0]['message']['content']
print(content)

# parse ASR output if you want
from qwen_asr import parse_asr_output
language, text = parse_asr_output(content)
print(language)
print(text)
```

#### Streaming Inference

Qwen3-ASR fully supports streaming inference. Currently, streaming inference is only available with the vLLM backend. Note that streaming inference does not support batch inference or returning timestamps. Please refer to the [example code](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py) for details. You can also launch a streaming web demo through the [guide](#streaming-demo) to experience Qwen3-ASR’s streaming transcription capabilities. 

#### ForcedAligner Usage

`Qwen3-ForcedAligner-0.6B` can align text–speech pairs and return word or character level timestamps. Here is an example of using the forced aligner directly:

```python
import torch
from qwen_asr import Qwen3ForcedAligner

model = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
)

results = model.align(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    text="甚至出现交易几乎停滞的情况。",
    language="Chinese",
)

print(results[0])
print(results[0][0].text, results[0][0].start_time, results[0][0].end_time)
```

In addition, the forced aligner supports local paths / URLs / base64 data / `(np.ndarray, sr)` inputs and batch inference. Please refer to the [example code](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_forced_aligner.py) for details.

### DashScope API Usage

To further explore Qwen3-ASR, we encourage you to try our DashScope API for a faster and more efficient experience. For detailed API information and documentation, please refer to the following:

| API Description | API Documentation (Mainland China) | API Documentation (International) |
|------------------|-----------------------------------|------------------------------------|
| Real-time API for Qwen3-ASR. | [https://help.aliyun.com/zh/model-studio/qwen-real-time-speech-recognition](https://help.aliyun.com/zh/model-studio/qwen-real-time-speech-recognition) | [https://www.alibabacloud.com/help/en/model-studio/qwen-real-time-speech-recognition](https://www.alibabacloud.com/help/en/model-studio/qwen-real-time-speech-recognition) |
| FileTrans API for Qwen3-ASR. | [https://help.aliyun.com/zh/model-studio/qwen-speech-recognition](https://help.aliyun.com/zh/model-studio/qwen-speech-recognition) | [https://www.alibabacloud.com/help/en/model-studio/qwen-speech-recognition](https://www.alibabacloud.com/help/en/model-studio/qwen-speech-recognition) |


## macOS (Apple Silicon) 本地运行指南 · 课堂录音转写

> 本节面向想在 **MacBook Air / Pro（M1/M2/M3 等 Apple Silicon）** 上离线运行 **Qwen3-ASR-0.6B**，用来**录制并转写老师与学生课堂对话**的用户。以 **MacBook Air M2 / 16GB** 为例。

### 为什么用 0.6B + MPS

- **vLLM 后端在 macOS 上不可用**，Apple Silicon 也无法安装 `flash-attn`。因此在 Mac 上用 CPU/**MPS（Metal）** 做推理即可。
- **0.6B** 模型体量小（bf16/fp16 权重约 1~2GB），在 **16GB 统一内存**的 M2 上可以流畅离线运行；1.7B 也能跑，但更吃内存、更慢，日常课堂转写推荐 **0.6B**。
- 该模型只做 **语音识别 + 时间戳对齐**，**不做说话人分离（diarization）**，所以无法自动区分「哪句是老师、哪句是学生」；开启时间戳后可按时间顺序回看整段对话。

### 三种模型 / 后端

本节脚本 [`scripts/record_and_transcribe.py`](scripts/record_and_transcribe.py) 会**根据模型名自动选择后端**：

| 模型仓库 | 后端 | 说明 |
|---|---|---|
| `mlx-community/Qwen3-ASR-0.6B-8bit`（名字含 `mlx`，**Apple Silicon 推荐**） | [MLX](https://github.com/Blaizzy/mlx-audio)（`mlx-audio`） | Apple 官方 ML 框架，Apple Silicon 上速度最快、量化版更省内存；`pip install mlx-audio` 即可，无需 torch/transformers |
| `Qwen/Qwen3-ASR-0.6B-hf`（带 `-hf`） | 🤗 transformers 原生 | `AutoModelForMultimodalLM` + `apply_transcription_request`，需 PyTorch ≥ 2.4 与含 `qwen3_asr` 支持的 transformers |
| `Qwen/Qwen3-ASR-0.6B`（不带 `-hf`） | `qwen-asr` 包 | `qwen_asr.Qwen3ASRModel` |

MLX 可用模型：`mlx-community/Qwen3-ASR-0.6B-8bit`、`mlx-community/Qwen3-ASR-1.7B-8bit`、`mlx-community/Qwen3-ASR-1.7B-4bit`、对齐模型 `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`。1.7B 的 4bit/8bit 量化版在 16GB 的 M2 上也能流畅运行，识别质量高于 0.6B。

> 三者**不能混用**：`-hf` 模型要用原生 transformers（`qwen-asr` 包会报 `rope_scaling=None` 的 `AttributeError`）；MLX 模型要用 `mlx-audio`。脚本已按模型名自动路由，也可用 `--backend mlx/transformers/package` 手动指定。

### 一键安装

仓库内提供了安装脚本 [`scripts/setup_mac.sh`](scripts/setup_mac.sh)，会自动安装 `ffmpeg`、`portaudio`、PyTorch（含 MPS）、transformers、`sounddevice` 等：

```bash
git clone https://github.com/QwenLM/Qwen3-ASR.git
cd Qwen3-ASR

# 需要先安装 Homebrew（https://brew.sh）
# MLX 后端（Apple Silicon 最快，推荐）：
USE_MLX=1 bash scripts/setup_mac.sh

# 或默认安装 -hf（原生 transformers）所需环境：
# bash scripts/setup_mac.sh

# 或非 -hf 的 qwen-asr 包版本：
# USE_HF=0 bash scripts/setup_mac.sh

# 国内网络推荐用 ModelScope 预下载模型权重（可选）：
# USE_MODELSCOPE=1 bash scripts/setup_mac.sh
```

MLX 后端手动安装只需两行：

```bash
pip install -U mlx-audio librosa soundfile sounddevice flask
# 模型会在首次运行时自动从 HuggingFace 下载（国内可先 export HF_ENDPOINT=https://hf-mirror.com）
```

如果你更喜欢手动安装（`-hf` 原生 transformers 路线）：

```bash
# 系统依赖（录音与音视频解码）
brew install ffmpeg portaudio

# 建议使用独立的 Python 3.12 环境
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr

# PyTorch 的 macOS 官方 wheel 已内置 MPS 支持（需 >= 2.4）
pip install -U "torch>=2.4" torchaudio
# -hf 模型需要含 Qwen3-ASR(`qwen3_asr`) 原生支持的 transformers。很多已发布版本尚未包含，
# 直接装源码开发版最稳：
pip install -U "git+https://github.com/huggingface/transformers" accelerate
pip install -U librosa soundfile sounddevice   # 音频读取 + 录音
pip install -U nagisa soynlp                    # 仅日语/韩语时间戳需要

# 注意：不要在 Mac 上安装 vLLM 或 flash-attn（不支持）
```

> 从 HuggingFace 下载模型较慢时，可先设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

### 录制并转写课堂对话

脚本支持「现场录音」与「转写已有文件」两种模式，默认使用 `Qwen/Qwen3-ASR-0.6B-hf`。

**方式一：上课时现场录音，下课按 `Ctrl+C` 结束并自动转写**

```bash
# MLX 后端（推荐）：自动检测语言，输出逐词/逐字时间戳
python scripts/record_and_transcribe.py --record --model mlx-community/Qwen3-ASR-0.6B-8bit --timestamps

# 也可强制指定语言，识别更稳
python scripts/record_and_transcribe.py --record --model mlx-community/Qwen3-ASR-0.6B-8bit --language Chinese

# 用 transformers 原生后端（-hf 模型）则：
python scripts/record_and_transcribe.py --record --timestamps   # 默认 Qwen/Qwen3-ASR-0.6B-hf
```

第一次运行时，请在 **系统设置 → 隐私与安全性 → 麦克风** 中允许终端（或你的 IDE）访问麦克风。录音文件与转写文本默认保存在 `./recordings/` 目录下。

**方式二：转写课后导出的录音/录像文件（wav / mp3 / m4a / mp4 …）**

```bash
python scripts/record_and_transcribe.py --audio ./lesson.m4a --model mlx-community/Qwen3-ASR-1.7B-4bit --language Chinese --timestamps
```

MLX 后端也可以直接用 `mlx-audio` 的 Python API（不经过本仓库脚本）：

```python
from mlx_audio.stt import load

model = load("mlx-community/Qwen3-ASR-1.7B-4bit")
result = model.generate("path_to_audio.wav", language="Chinese")
print(result.text)
```

**用本地已下载的模型目录（`--model` 指向本地路径即可，脚本按目录名是否带 `-hf` 判断后端）：**

```bash
# 例如用 ModelScope / huggingface-cli 下载到 ./models 后
python scripts/record_and_transcribe.py --record --model ./models/Qwen3-ASR-0.6B-hf
```

常用参数（完整列表见 `python scripts/record_and_transcribe.py --help`）：

| 参数 | 说明 |
|---|---|
| `--record` / `--audio PATH` | 现场录音 / 转写已有文件（二选一） |
| `--duration SECONDS` | 录音模式下的最长录制秒数（不填则录到 `Ctrl+C`） |
| `--model` | ASR 模型名或本地目录（默认 `Qwen/Qwen3-ASR-0.6B-hf`；Apple Silicon 推荐 `mlx-community/Qwen3-ASR-0.6B-8bit`） |
| `--backend` | `auto`/`mlx`/`transformers`/`package`；默认 `auto`，按模型名判断（含 `mlx` → mlx，`-hf` → transformers） |
| `--language` | 强制语言（如 `Chinese`/`English`，或 `zh`/`en`）；不填则自动检测 |
| `--timestamps` | 额外加载 ForcedAligner 输出逐词/逐字时间戳 |
| `--chunk-seconds` | transformers 后端处理长音频的分段时长（默认 30 秒） |
| `--device` | `auto`/`mps`/`cpu`（默认 `auto`，在 Apple Silicon 上自动选 `mps`） |
| `--output-dir` | 录音与转写结果的保存目录（默认 `./recordings`） |

### 网页工具（录音准实时转写 + 文件上传转写）

仓库内提供了一个本地网页工具 [`scripts/web_app.py`](scripts/web_app.py)：

- **录音转写**：点击按钮开始录音，识别文字**边录边刷新**显示；停止后自动保存录音 wav 与转写 txt。
- **文件转写**：上传音频/视频文件（wav / mp3 / m4a / mp4 …），显示并自动保存识别文本。
- **历史记录**：页面下方可查看/下载所有已保存的转写结果。

启动（需已装 `flask`，安装脚本已包含）：

```bash
# MLX 后端（Apple Silicon 最快，推荐）
python scripts/web_app.py --model mlx-community/Qwen3-ASR-0.6B-8bit
# 或 transformers 原生后端（-hf 模型）
python scripts/web_app.py --model ./models/Qwen3-ASR-0.6B-hf
# 打开浏览器访问 http://127.0.0.1:8000
```

常用参数：`--port` 端口（默认 8000）；`--host 0.0.0.0` 允许局域网访问（注意浏览器只在 localhost 或 HTTPS 下允许录音）；`--segment-seconds` 准实时转写的段落上限（默认 25 秒，段落固化后文字不再变动）；`--output-dir` 保存目录（默认 `./recordings`）。

> 实现说明：浏览器每 3 秒把麦克风音频块发给本地服务，服务端对当前段落做**增量转写**刷新结果（紫色为临时文字），段落满 25 秒后固化。macOS 没有 vLLM 无法做真流式解码，这种分段方案在 0.6B + MPS 下延迟约数秒，课堂场景够用。模型推理有全局锁，多人同时使用会排队。

### 常见问题

- **MLX 后端报 `No module named 'mlx_audio'` 或安装失败？** `pip install -U mlx-audio`；注意 MLX 仅支持 Apple Silicon 原生（arm64）Python 环境，Intel/Rosetta 环境装不了（检查：`python -c "import platform; print(platform.machine())"` 应为 `arm64`）。
- **网页工具 + MLX 报 `RuntimeError: There is no Stream(gpu, N) in current thread`？** MLX 的计算流绑定创建线程，而 Flask 每个请求在不同线程。本仓库网页工具已把 MLX 加载与推理固定到专用工作线程（拉取最新代码即可）；自己写多线程服务时也需同样处理。
- **MLX 后端报 `AttributeError: 'str' object has no attribute '__module__'`（`AutoTokenizer.register`）？** 这是 `mlx-lm` 与新版 transformers（v5+/开发版）的兼容问题：`mlx-lm` 导入时用字符串注册 `NewlineTokenizer`，新版 transformers 要求传类对象。本仓库脚本已内置兼容补丁（加载 MLX 模型前自动生效），拉取最新代码即可；也可尝试 `pip install -U mlx-lm mlx-audio` 升级到已修复的版本。
- **报错 `AttributeError: 'NoneType' object has no attribute 'get'`（`rope_scaling`）？** 这是把 `-hf` 模型喂给了 `qwen-asr` 包。用本仓库脚本时它会自动选原生 transformers 后端；若你手写代码，请对 `-hf` 模型用 `AutoModelForMultimodalLM`，不要用 `qwen_asr.Qwen3ASRModel`。
- **报错 `KeyError: 'qwen3_asr'` 或 `Transformers does not recognize this architecture`？** 你的 transformers 还没有内置 Qwen3-ASR 原生支持（很多已发布版本尚未包含）。安装源码开发版：

  ```bash
  pip install -U "git+https://github.com/huggingface/transformers"
  # 确认：
  python -c "from transformers import Qwen3ASRForConditionalGeneration; print('ok')"
  ```

- **报错找不到 `AutoModelForMultimodalLM` / `apply_transcription_request`，或日志出现 `Disabling PyTorch because PyTorch >= 2.4 is required`？** 这是 **PyTorch 版本太旧**（新版 transformers 需要 PyTorch ≥ 2.4，否则会禁用 PyTorch，导致模型无法加载、processor 缺少音频方法）。升级 PyTorch：

  ```bash
  pip install -U "torch>=2.4" torchaudio
  ```

  若升级后仍提示缺少原生支持，再升级 transformers：`pip install -U transformers`（或 `pip install -U git+https://github.com/huggingface/transformers`）。

- **`pip install "torch>=2.4"` 报 `Could not find a version that satisfies the requirement torch>=2.4 (from versions: 2.2.0, 2.2.1, 2.2.2)`？** 说明你的 conda/Python 是 **x86_64（Intel/Rosetta）** 架构——PyTorch 在 macOS 上最后的 Intel 版本就是 2.2.2，Apple Silicon 原生（arm64）才有 2.4+。先确认：

  ```bash
  python -c "import platform; print(platform.machine())"   # 若显示 x86_64 即为此问题
  ```

  新建一个**原生 arm64 环境**再安装（从现有 miniconda 即可，无需重装 conda）：

  ```bash
  CONDA_SUBDIR=osx-arm64 conda create -n qwen3-asr-arm python=3.12 -y
  conda activate qwen3-asr-arm
  conda config --env --set subdir osx-arm64
  python -c "import platform; print(platform.machine())"   # 现在应为 arm64
  pip install -U "torch>=2.4" torchaudio transformers accelerate librosa soundfile sounddevice nagisa soynlp
  ```

  或安装原生 arm64 的 Miniforge：`brew install miniforge`，再 `conda init "$(basename "$SHELL")"` 并重开终端。

- **`qwen-asr 0.0.6 requires transformers==4.57.6 ... incompatible` 警告？** 用 `-hf` 模型走的是原生 transformers，不经过 `qwen-asr` 包，这条 pip 依赖冲突警告可以忽略（不影响 `-hf` 路线）。
- **速度慢？** 确认日志里显示 `设备：mps`。首次运行需下载模型权重与做 MPS 预热，之后会明显变快。长音频会自动按 `--chunk-seconds` 分段处理，请耐心等待。
- **报错 `sounddevice` / PortAudio 找不到设备？** 先 `brew install portaudio` 再 `pip install -U sounddevice`，并检查麦克风权限。
- **MPS 上某个算子不支持报错？** 脚本已默认设置 `PYTORCH_ENABLE_MPS_FALLBACK=1` 让其回退到 CPU；仍报错可加 `--device cpu` 用纯 CPU 运行。
- **内存吃紧？** 使用默认 0.6B 模型；可调小 `--chunk-seconds`（如 20），或关闭 `--timestamps` 省去对齐模型开销。
- **想跑 1.7B？** 加 `--model Qwen/Qwen3-ASR-1.7B-hf` 即可，但在 16GB 机器上更慢、更容易吃满内存，日常转写建议仍用 0.6B。

## Launch Local Web UI Demo

### Gradio Demo

To launch the Qwen3-ASR web UI gradio demo, install the `qwen-asr` package and run `qwen-asr-demo`. Use the command below for help:

```bash
qwen-asr-demo --help
```

To launch the demo, you can use the following commands:

```bash
# Transformers backend
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --ip 0.0.0.0 --port 8000

# Transformers backend + Forced Aligner (enable timestamps)
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"device_map":"cuda:0","dtype":"bfloat16","max_inference_batch_size":8,"max_new_tokens":256}' \
  --aligner-kwargs '{"device_map":"cuda:0","dtype":"bfloat16"}' \
  --ip 0.0.0.0 --port 8000

# vLLM backend + Forced Aligner (enable timestamps)
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B \
  --backend vllm \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"gpu_memory_utilization":0.7,"max_inference_batch_size":8,"max_new_tokens":2048}' \
  --aligner-kwargs '{"device_map":"cuda:0","dtype":"bfloat16"}' \
  --ip 0.0.0.0 --port 8000
```

Then open `http://<your-ip>:8000`, or access it via port forwarding in tools like VS Code.

#### Backend Notes

This demo supports two backends: transformers and vLLM. All backend-specific initialization parameters should be passed via `--backend-kwargs` as a JSON dict. If not provided, the demo will use sensible defaults.

```bash
# Example: override transformers init args with flash attention
--backend-kwargs '{"device_map":"cuda:0","dtype":"bfloat16","attn_implementation":"flash_attention_2"}'

# Example: override vLLM init args with 65% GPU memory
--backend-kwargs '{"gpu_memory_utilization":0.65}'
```

#### CUDA Device Notes

Because vLLM does not follow `cuda:0` style device selection, this demo selects GPUs by setting `CUDA_VISIBLE_DEVICES` via `--cuda-visible-devices`.

```bash
# Use GPU 0
--cuda-visible-devices 0

# Use GPU 1
--cuda-visible-devices 1
```

#### Timestamps Notes

Timestamps are only available when `--aligner-checkpoint` is provided. If you launch the demo without a forced aligner, the timestamps UI will be hidden automatically.

```bash
# No forced aligner
qwen-asr-demo --asr-checkpoint Qwen/Qwen3-ASR-1.7B

# With forced aligner
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B
```

#### HTTPS Notes

To avoid browser microphone permission issues after deploying the server, it is recommended/required to run the gradio service over HTTPS (especially when accessed remotely or behind modern browsers/gateways). Use `--ssl-certfile` and `--ssl-keyfile` to enable HTTPS. First, generate a private key and a self-signed certificate (valid for 365 days):

```bash
openssl req -x509 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes \
  -subj "/CN=localhost"
```

Then run the demo with HTTPS:

```bash
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --ip 0.0.0.0 --port 8000 \
  --ssl-certfile cert.pem \
  --ssl-keyfile key.pem \
  --no-ssl-verify
```

Then open `https://<your-ip>:8000` to use it. If your browser shows a warning, that’s expected for self-signed certificates. For production, use a real certificate.

### Streaming Demo

To experience Qwen3-ASR’s streaming transcription capability in a web UI, we provide a minimal Flask-based streaming demo. The demo captures microphone audio in the browser, resamples it to 16,000 Hz, and continuously pushes PCM chunks to the model. Run the demo with the following command:

```bash
qwen-asr-demo-streaming \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8000
```

Then open `http://<your-ip>:8000`, or access it via port forwarding in tools like VS Code.

## Deployment with vLLM

vLLM officially provides day-0 model support for Qwen3-ASR for efficient inference. 

### Installation
You can run Qwen3-ASR with vLLM nightly wheel or docker image. To install the nightly version of vLLM, we recommend using `uv` as the environment manager
```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
uv pip install "vllm[audio]" # For additional audio dependencies
```

### Online Serving
You can easily deploy Qwen3-ASR with vLLM by running the following command
```bash
vllm serve Qwen/Qwen3-ASR-1.7B
```
After the model server is successfully deployed, you can interact with it in multiple ways.

#### Using OpenAI SDK
```python
import base64
import httpx
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Create multimodal chat completion request
response = client.chat.completions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}
                    }
                }
            ]
        }
    ],
)

print(response.choices[0].message.content)
```
This model is also supported on vLLM with OpenAI transcription API.
```python
import httpx
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
audio_file = httpx.get(audio_url).content

transcription = client.audio.transcriptions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    file=audio_file,
)

print(transcription.text)
```

#### Using cURL
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "messages": [
    {"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}
    ]}
    ]
    }'
```

### Offline Inference
See the following example on using vLLM to run offline inference with Qwen3-ASR
```python
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
import base64
import requests

# Initialize the LLM
llm = LLM(
    model="Qwen/Qwen3-ASR-1.7B"
)

# Load audio
audio_asset = AudioAsset("winning_call")

# Create conversation with audio content
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": audio_asset.url}
            }
        ]
    }
]

sampling_params = SamplingParams(temperature=0.01, max_tokens=256)

# Run inference using .chat()
outputs = llm.chat(conversation, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```


## Fine Tuning

Please refer to [Qwen3-ASR-Finetuning](finetuning/) for detailed instructions on fine-tuning Qwen3-ASR.


## Docker

To make it easier to use our `qwen-asr` Python package, we provide a pre-built Docker image: [qwenllm/qwen3-asr](https://hub.docker.com/r/qwenllm/qwen3-asr). You only need to install the GPU driver and download the model files to run the code. Please follow the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to ensure Docker can access your GPU. If you are in Mainland China and have trouble reaching Docker Hub, you may use a registry mirror to accelerate image pulls.

First, pull the image and start a container:

```bash
LOCAL_WORKDIR=/path/to/your/workspace
HOST_PORT=8000
CONTAINER_PORT=80
docker run --gpus all --name qwen3-asr \
    -v /var/run/docker.sock:/var/run/docker.sock -p $HOST_PORT:$CONTAINER_PORT \
    --mount type=bind,source=$LOCAL_WORKDIR,target=/data/shared/Qwen3-ASR \
    --shm-size=4gb \
    -it qwenllm/qwen3-asr:latest
```

After running the command, you will enter the container’s bash shell. Your local workspace (**replace** `/path/to/your/workspace` **with the actual path**) will be mounted inside the container at `/data/shared/Qwen3-ASR`. Port `8000` on the host is mapped to port `80` in the container, so you can access services running in the container via `http://<host-ip>:8000`. Note that services inside the container must bind to `0.0.0.0` (not `127.0.0.1`) for port forwarding to work.

If you exit the container, you can start it again and re-enter it with:

```bash
docker start qwen3-asr
docker exec -it qwen3-asr bash
```

To remove the container completely, run:

```bash
docker rm -f qwen3-asr
```


## Evaluation

During evaluation, we ran inference for all models with `dtype=torch.bfloat16` and set `max_new_tokens=1024` using vLLM. Greedy search was used for all decoding, and none of the tests specified a language parameter. The detailed evaluation results are shown below.

<details>
<summary>ASR Benchmarks on Public Datasets (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th colspan="2" style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">English (en)</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Librispeech<br>clean | other</td>
      <td style="text-align: center;"><strong>1.39</strong> | 3.75</td>
      <td style="text-align: center;">2.89 | 3.56</td>
      <td style="text-align: center;">2.78 | 5.70</td>
      <td style="text-align: center;">1.51 | 3.97</td>
      <td style="text-align: center;">1.68 | 4.03</td>
      <td style="text-align: center;">2.11 | 4.55</td>
      <td style="text-align: center;">1.63 | <strong>3.38</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">GigaSpeech</td>
      <td style="text-align: center;">25.50</td>
      <td style="text-align: center;">9.37</td>
      <td style="text-align: center;">9.55</td>
      <td style="text-align: center;">9.76</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">8.88</td>
      <td style="text-align: center;"><strong>8.45</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-en</td>
      <td style="text-align: center;">9.08</td>
      <td style="text-align: center;">14.49</td>
      <td style="text-align: center;">13.78</td>
      <td style="text-align: center;">9.90</td>
      <td style="text-align: center;">9.90</td>
      <td style="text-align: center;">9.92</td>
      <td style="text-align: center;"><strong>7.39</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-en</td>
      <td style="text-align: center;"><strong>2.40</strong></td>
      <td style="text-align: center;">2.94</td>
      <td style="text-align: center;">6.31</td>
      <td style="text-align: center;">4.08</td>
      <td style="text-align: center;">5.49</td>
      <td style="text-align: center;">4.39</td>
      <td style="text-align: center;">3.35</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">MLS-en</td>
      <td style="text-align: center;">5.12</td>
      <td style="text-align: center;"><strong>3.68</strong></td>
      <td style="text-align: center;">7.09</td>
      <td style="text-align: center;">4.87</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">6.00</td>
      <td style="text-align: center;">4.58</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Tedlium</td>
      <td style="text-align: center;">7.69</td>
      <td style="text-align: center;">6.15</td>
      <td style="text-align: center;">4.91</td>
      <td style="text-align: center;">6.84</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>3.85<strong></td>
      <td style="text-align: center;"><strong>4.50</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">VoxPopuli</td>
      <td style="text-align: center;">10.29</td>
      <td style="text-align: center;">11.36</td>
      <td style="text-align: center;">12.12</td>
      <td style="text-align: center;">12.05</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>9.96<strong></td>
      <td style="text-align: center;"><strong>9.15</strong></td>
    </tr>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese (zh)</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech<br>net | meeting</td>
      <td style="text-align: center;">15.30 | 32.27</td>
      <td style="text-align: center;">14.43 | 13.47</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">9.86 | 19.11</td>
      <td style="text-align: center;">6.35 | -</td>
      <td style="text-align: center;">5.97 | 6.88</td>
      <td style="text-align: center;"><strong>4.97</strong> | <strong>5.88</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">AISHELL-2-test</td>
      <td style="text-align: center;">4.24</td>
      <td style="text-align: center;">11.62</td>
      <td style="text-align: center;">2.85</td>
      <td style="text-align: center;">5.06</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">3.15</td>
      <td style="text-align: center;"><strong>2.71</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">SpeechIO</td>
      <td style="text-align: center;">12.86</td>
      <td style="text-align: center;">5.30</td>
      <td style="text-align: center;">2.93</td>
      <td style="text-align: center;">7.56</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">3.44</td>
      <td style="text-align: center;"><strong>2.88</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-zh</td>
      <td style="text-align: center;">2.44</td>
      <td style="text-align: center;">2.71</td>
      <td style="text-align: center;">2.69</td>
      <td style="text-align: center;">4.09</td>
      <td style="text-align: center;">3.51</td>
      <td style="text-align: center;">2.88</td>
      <td style="text-align: center;"><strong>2.41</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-zh</td>
      <td style="text-align: center;">6.32</td>
      <td style="text-align: center;">7.70</td>
      <td style="text-align: center;">5.95</td>
      <td style="text-align: center;">12.91</td>
      <td style="text-align: center;">6.20</td>
      <td style="text-align: center;">6.89</td>
      <td style="text-align: center;"><strong>5.35</strong></td>
    </tr>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese Dialect</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">KeSpeech</td>
      <td style="text-align: center;">26.87</td>
      <td style="text-align: center;">24.71</td>
      <td style="text-align: center;">5.27</td>
      <td style="text-align: center;">28.79</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">7.08</td>
      <td style="text-align: center;"><strong>5.10</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-yue</td>
      <td style="text-align: center;">4.98</td>
      <td style="text-align: center;">9.43</td>
      <td style="text-align: center;">4.98</td>
      <td style="text-align: center;">9.18</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">5.79</td>
      <td style="text-align: center;"><strong>3.98</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-yue</td>
      <td style="text-align: center;">11.36</td>
      <td style="text-align: center;">18.76</td>
      <td style="text-align: center;">13.20</td>
      <td style="text-align: center;">16.23</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">9.50</td>
      <td style="text-align: center;"><strong>7.57</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-zh-tw</td>
      <td style="text-align: center;">6.32</td>
      <td style="text-align: center;">7.31</td>
      <td style="text-align: center;">4.06</td>
      <td style="text-align: center;">7.84</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">5.59</td>
      <td style="text-align: center;"><strong>3.77</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech-Yue<br>short | long</td>
      <td style="text-align: center;">15.62 | 25.29</td>
      <td style="text-align: center;">25.19 | 11.23</td>
      <td style="text-align: center;">9.74 | 11.40</td>
      <td style="text-align: center;">32.26 | 46.64</td>
      <td style="text-align: center;">- | -</td>
      <td style="text-align: center;">7.54 | 9.92</td>
      <td style="text-align: center;"><strong>5.82</strong> | <strong>8.85</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech-Chuan<br>easy | hard</td>
      <td style="text-align: center;">34.81 | 53.98</td>
      <td style="text-align: center;">43.79 | 67.30</td>
      <td style="text-align: center;"><strong>11.40<strong> | <strong>20.20</strong></td>
      <td style="text-align: center;">14.35 | 26.80</td>
      <td style="text-align: center;">- | -</td>
      <td style="text-align: center;">13.92 | 24.45</td>
      <td style="text-align: center;">11.99 | 21.63</td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>ASR Benchmarks on Internal Datasets (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Accented English</td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Accented English</td>
      <td style="text-align: center;">28.56</td>
      <td style="text-align: center;">23.85</td>
      <td style="text-align: center;">20.41</td>
      <td style="text-align: center;">21.30</td>
      <td style="text-align: center;">19.96</td>
      <td style="text-align: center;"><strong>16.62<strong></td>
      <td style="text-align: center;"><strong>16.07</strong></td>
    </tr>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese Mandarin</td>
    </tr>
    <tr>
      <td style="text-align: left;">Elders&Kids</td>
      <td style="text-align: center;">14.27</td>
      <td style="text-align: center;">36.93</td>
      <td style="text-align: center;">4.17</td>
      <td style="text-align: center;">10.61</td>
      <td style="text-align: center;">4.54</td>
      <td style="text-align: center;">4.48</td>
      <td style="text-align: center;"><strong>3.81</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">ExtremeNoise</td>
      <td style="text-align: center;">36.11</td>
      <td style="text-align: center;">29.06</td>
      <td style="text-align: center;">17.04</td>
      <td style="text-align: center;">63.17</td>
      <td style="text-align: center;">36.55</td>
      <td style="text-align: center;">17.88</td>
      <td style="text-align: center;"><strong>16.17</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">TongueTwister</td>
      <td style="text-align: center;">20.87</td>
      <td style="text-align: center;">4.97</td>
      <td style="text-align: center;">3.47</td>
      <td style="text-align: center;">16.63</td>
      <td style="text-align: center;">9.02</td>
      <td style="text-align: center;">4.06</td>
      <td style="text-align: center;"><strong>2.44</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Mandarin</td>
      <td style="text-align: center;">20.73</td>
      <td style="text-align: center;">12.50</td>
      <td style="text-align: center;">6.61</td>
      <td style="text-align: center;">14.01</td>
      <td style="text-align: center;">7.32</td>
      <td style="text-align: center;">7.06</td>
      <td style="text-align: center;"><strong>6.54</strong></td>
    </tr>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese Dialect</td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Cantonese</td>
      <td style="text-align: center;">16.05</td>
      <td style="text-align: center;">14.98</td>
      <td style="text-align: center;">7.56</td>
      <td style="text-align: center;">31.04</td>
      <td style="text-align: center;">5.85</td>
      <td style="text-align: center;"><strong>4.80<strong></td>
      <td style="text-align: center;"><strong>4.12</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Chinese Dialects</td>
      <td style="text-align: center;">45.37</td>
      <td style="text-align: center;">47.70</td>
      <td style="text-align: center;">19.85</td>
      <td style="text-align: center;">44.55</td>
      <td style="text-align: center;">19.41</td>
      <td style="text-align: center;"><strong>18.24<strong></td>
      <td style="text-align: center;"><strong>15.94</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>Dialect coverage:</strong> Results for <em>Dialog-Accented English</em> are averaged over 16 accents, and results for <em>Dialog-Chinese Dialects</em> are averaged over 22 Chinese dialects.</p>

</details>

<details>
<summary>Multilingual ASR Benchmarks (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GLM-ASR<br>-Nano-2512</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="6" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Open-sourced Benchmarks</td>
    </tr>
    <tr>
      <td style="text-align: left;">MLS</td>
      <td style="text-align: center;">13.32</td>
      <td style="text-align: center;">8.62</td>
      <td style="text-align: center;">28.70</td>
      <td style="text-align: center;">13.19</td>
      <td style="text-align: center;"><strong>8.55</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">CommonVoice</td>
      <td style="text-align: center;">19.40</td>
      <td style="text-align: center;">10.77</td>
      <td style="text-align: center;">17.25</td>
      <td style="text-align: center;">12.75</td>
      <td style="text-align: center;"><strong>9.18</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MLC-SLM</td>
      <td style="text-align: center;">34.93</td>
      <td style="text-align: center;">15.68</td>
      <td style="text-align: center;">29.94</td>
      <td style="text-align: center;">15.84</td>
      <td style="text-align: center;"><strong>12.74</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs</td>
      <td style="text-align: center;">16.08</td>
      <td style="text-align: center;">5.27</td>
      <td style="text-align: center;">10.03</td>
      <td style="text-align: center;">7.57</td>
      <td style="text-align: center;"><strong>4.90</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs<sup>†</sup></td>
      <td style="text-align: center;">20.05</td>
      <td style="text-align: center;">6.85</td>
      <td style="text-align: center;">31.89</td>
      <td style="text-align: center;">10.37</td>
      <td style="text-align: center;"><strong>6.62</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs<sup>††</sup></td>
      <td style="text-align: center;">24.83</td>
      <td style="text-align: center;"><strong>8.16</strong></td>
      <td style="text-align: center;">47.84</td>
      <td style="text-align: center;">21.80</td>
      <td style="text-align: center;">12.60</td>
    </tr>
    <tr>
      <td colspan="6" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Qwen-ASR Internal Benchmarks</td>
    </tr>
    <tr>
      <td style="text-align: left;">News-Multilingual</td>
      <td style="text-align: center;">49.40</td>
      <td style="text-align: center;">14.80</td>
      <td style="text-align: center;">65.07</td>
      <td style="text-align: center;">17.39</td>
      <td style="text-align: center;"><strong>12.80</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>Language coverage:</strong> <em>MLS</em> includes 8 languages: {da, de, en, es, fr, it, pl, pt}.<br><em>CommonVoice</em> includes 13 languages: {en, zh, yue, zh_TW, ar, de, es, fr, it, ja, ko, pt, ru}.<br><em>MLC-SLM</em> includes 11 languages: {en, fr, de, it, pt, es, ja, ko, ru, th, vi}.<br><em>Fleurs</em> includes 12 languages: {en, zh, yue, ar, de, es, fr, it, ja, ko, pt, ru }.<br><em>Fleurs<sup>†</sup></em> includes 8 additional languages beyond Fleurs: {hi, id, ms, nl, pl, th, tr, vi}.<br><em>Fleurs<sup>††</sup></em> includes 10 additional languages beyond Fleurs<sup>†</sup>: {cs, da, el, fa, fi, fil, hu, mk, ro, sv}.<br><em>News-Multilingual</em> includes 15 languages: {ar, de, es, fr, hi, id, it, ja, ko, nl, pl, pt, ru, th, vi}.</p>

</details>

<details>
<summary>Language Identification Accuracy (%) ↑</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">Whisper-large-v3</th>
      <th style="text-align: center;">Qwen3-ASR-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">MLS</td>
      <td style="text-align: center;"><strong>99.9</strong></td>
      <td style="text-align: center;">99.3</td>
      <td style="text-align: center;"><strong>99.9</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">CommonVoice</td>
      <td style="text-align: center;">92.7</td>
      <td style="text-align: center;"><strong>98.2<strong></td>
      <td style="text-align: center;"><strong>98.7</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MLC-SLM</td>
      <td style="text-align: center;">89.2</td>
      <td style="text-align: center;"><strong>92.7<strong></td>
      <td style="text-align: center;"><strong>94.1</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs</td>
      <td style="text-align: center;">94.6</td>
      <td style="text-align: center;"><strong>97.1<strong></td>
      <td style="text-align: center;"><strong>98.7</strong></td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">94.1</td>
      <td style="text-align: center;"><strong>96.8<strong></td>
      <td style="text-align: center;"><strong>97.9</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>Language coverage:</strong> The language sets follow Multilingual ASR Benchmarks. Here, Fleurs corresponds to Fleurs<sup>††</sup> in Multilingual ASR Benchmarks and covers 30 languages.</p>

</details>

<details>
<summary>Singing Voice & Song Transcription (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR<br>-1.0</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR-MLT<br>-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="7" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Singing</td>
    </tr>
    <tr>
      <td style="text-align: left;">M4Singer</td>
      <td style="text-align: center;">16.77</td>
      <td style="text-align: center;">20.88</td>
      <td style="text-align: center;">7.88</td>
      <td style="text-align: center;">13.58</td>
      <td style="text-align: center;">7.29</td>
      <td style="text-align: center;"><strong>5.98</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MIR-1k-vocal</td>
      <td style="text-align: center;">11.87</td>
      <td style="text-align: center;">9.85</td>
      <td style="text-align: center;">6.56</td>
      <td style="text-align: center;">11.71</td>
      <td style="text-align: center;">8.17</td>
      <td style="text-align: center;"><strong>6.25</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Opencpop</td>
      <td style="text-align: center;">7.93</td>
      <td style="text-align: center;">6.49</td>
      <td style="text-align: center;">3.80</td>
      <td style="text-align: center;">9.52</td>
      <td style="text-align: center;"><strong>2.98</strong></td>
      <td style="text-align: center;">3.08</td>
    </tr>
    <tr>
      <td style="text-align: left;">Popcs</td>
      <td style="text-align: center;">32.84</td>
      <td style="text-align: center;">15.13</td>
      <td style="text-align: center;">8.97</td>
      <td style="text-align: center;">13.77</td>
      <td style="text-align: center;">9.42</td>
      <td style="text-align: center;"><strong>8.52</strong></td>
    </tr>
    <tr>
      <td colspan="7" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Songs with BGM</td>
    </tr>
    <tr>
      <td style="text-align: left;">EntireSongs-en</td>
      <td style="text-align: center;">30.71</td>
      <td style="text-align: center;"><strong>12.18</strong></td>
      <td style="text-align: center;">33.51</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">14.60</td>
    </tr>
    <tr>
      <td style="text-align: left;">EntireSongs-zh</td>
      <td style="text-align: center;">34.86</td>
      <td style="text-align: center;">18.68</td>
      <td style="text-align: center;">23.99</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;"><strong>13.91</strong></td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>ASR Inference Mode Performance (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;">Model</th>
      <th style="text-align: left;">Infer. Mode</th>
      <th style="text-align: center;">Librispeech</th>
      <th style="text-align: center;">Fleurs-en</th>
      <th style="text-align: center;">Fleurs-zh</th>
      <th style="text-align: center;">Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="text-align: left; vertical-align: middle;">Qwen3-ASR-1.7B</td>
      <td style="text-align: left;">Offline</td>
      <td style="text-align: center;">1.63 | 3.38</td>
      <td style="text-align: center;">3.35</td>
      <td style="text-align: center;">2.41</td>
      <td style="text-align: center;">2.69</td>
    </tr>
    <tr>
      <td style="text-align: left;">Streaming</td>
      <td style="text-align: center;">1.95 | 4.51</td>
      <td style="text-align: center;">4.02</td>
      <td style="text-align: center;">2.84</td>
      <td style="text-align: center;">3.33</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td rowspan="2" style="text-align: left; vertical-align: middle;">Qwen3-ASR-0.6B</td>
      <td style="text-align: left;">Offline</td>
      <td style="text-align: center;">2.11 | 4.55</td>
      <td style="text-align: center;">4.39</td>
      <td style="text-align: center;">2.88</td>
      <td style="text-align: center;">3.48</td>
    </tr>
    <tr>
      <td style="text-align: left;">Streaming</td>
      <td style="text-align: center;">2.54 | 6.27</td>
      <td style="text-align: center;">5.38</td>
      <td style="text-align: center;">3.40</td>
      <td style="text-align: center;">4.40</td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>Forced Alignment Benchmarks (AAS ms ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">Monotonic-Aligner</th>
      <th style="text-align: center;">NFA</th>
      <th style="text-align: center;">WhisperX</th>
      <th style="text-align: center;">Qwen3-ForcedAligner-0.6B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">MFA-Labeled Raw</td>
    </tr>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">161.1</td>
      <td style="text-align: center;">109.8</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>33.1</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">107.5</td>
      <td style="text-align: center;">92.1</td>
      <td style="text-align: center;"><strong>37.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">100.7</td>
      <td style="text-align: center;">145.3</td>
      <td style="text-align: center;"><strong>41.7</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">122.7</td>
      <td style="text-align: center;">165.1</td>
      <td style="text-align: center;"><strong>46.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">142.7</td>
      <td style="text-align: center;">155.5</td>
      <td style="text-align: center;"><strong>75.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>37.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>38.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">200.7</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>40.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">124.7</td>
      <td style="text-align: center;">108.0</td>
      <td style="text-align: center;"><strong>36.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">161.1</td>
      <td style="text-align: center;">129.8</td>
      <td style="text-align: center;">133.2</td>
      <td style="text-align: center;"><strong>42.9</strong></td>
    </tr>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">MFA-Labeled Concat-300s</td>
    </tr>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">1742.4</td>
      <td style="text-align: center;">235.0</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>36.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">226.7</td>
      <td style="text-align: center;">227.2</td>
      <td style="text-align: center;"><strong>58.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">230.6</td>
      <td style="text-align: center;">2052.2</td>
      <td style="text-align: center;"><strong>53.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">220.3</td>
      <td style="text-align: center;">993.4</td>
      <td style="text-align: center;"><strong>62.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">290.5</td>
      <td style="text-align: center;">5719.4</td>
      <td style="text-align: center;"><strong>81.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>81.3</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>50.0</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">283.3</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>43.0</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">240.2</td>
      <td style="text-align: center;">4549.9</td>
      <td style="text-align: center;"><strong>39.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Cross-lingual</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>34.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">1742.4</td>
      <td style="text-align: center;">246.7</td>
      <td style="text-align: center;">2708.4</td>
      <td style="text-align: center;"><strong>52.9</strong></td>
    </tr>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Human-Labeled</td>
    </tr>
    <tr>
      <td style="text-align: left;">Raw</td>
      <td style="text-align: center;">49.9</td>
      <td style="text-align: center;">88.6</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>27.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Raw-Noisy</td>
      <td style="text-align: center;">53.3</td>
      <td style="text-align: center;">89.5</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>41.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Concat-60s</td>
      <td style="text-align: center;">51.1</td>
      <td style="text-align: center;">86.7</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>25.3</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Concat-300s</td>
      <td style="text-align: center;">410.8</td>
      <td style="text-align: center;">140.0</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>24.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Concat-Cross-lingual</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">141.3</td>
      <td style="text-align: center;">101.2</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>32.4</strong></td>
    </tr>
  </tbody>
</table>

</details>


## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@article{Qwen3-ASR,
  title={Qwen3-ASR Technical Report},
  author={Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, Junyang Lin},
  journal={arXiv preprint arXiv:2601.21337},
  year={2026}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen3-ASR&type=Date)](https://star-history.com/#QwenLM/Qwen3-ASR&Date)

<br>
