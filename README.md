# Qwen3-ASR · Mac 课堂转写工具（Apple Silicon 专用）

基于 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的本地语音转写工具，**仅支持 Apple Silicon Mac（M1/M2/M3…）**，用于录制并转写课堂上老师与学生的对话。完整的模型介绍、评测、vLLM/CUDA 部署等请见[上游仓库](https://github.com/QwenLM/Qwen3-ASR)。

功能：

- **命令行**：现场录音（`Ctrl+C` 结束）或转写已有音频/视频文件，支持逐词/逐字时间戳
- **网页工具**：点击按钮录音、识别文字**边录边刷新**、上传文件转写、结果自动保存并可下载
- 中英等 30 种语言 + 22 种中文方言，自动语言检测（适合中英混说的课堂）

> 说明：模型只做「语音识别 + 时间戳」，**不做说话人分离（diarization）**，无法自动区分「哪句是老师、哪句是学生」；时间戳可按时间顺序回看对话。

## 支持的模型 / 后端

脚本按模型名自动选择后端，也可用 `--backend mlx / transformers` 手动指定：

| 模型 | 后端 | 说明 |
|---|---|---|
| `mlx-community/Qwen3-ASR-0.6B-8bit`（**默认，推荐**） | [MLX](https://github.com/Blaizzy/mlx-audio)（`mlx-audio`） | Apple 官方 ML 框架，速度最快、依赖最少（无需 torch/transformers） |
| `mlx-community/Qwen3-ASR-1.7B-8bit` / `1.7B-4bit` | MLX | 识别质量更高，16GB 内存也能流畅运行 |
| `Qwen/Qwen3-ASR-0.6B-hf` / `1.7B-hf` | 🤗 transformers 原生（MPS） | 需 PyTorch ≥ 2.4 + 含 `qwen3_asr` 支持的 transformers（源码开发版） |

时间戳对齐模型（加 `--timestamps` 时自动加载）：`mlx-community/Qwen3-ForcedAligner-0.6B-8bit` / `Qwen/Qwen3-ForcedAligner-0.6B-hf`。

## 安装

一键安装（需先装 [Homebrew](https://brew.sh)）：

```bash
git clone https://github.com/earthonliner/Qwen3-ASR.git
cd Qwen3-ASR

bash scripts/setup_mac.sh              # 默认 MLX 后端（推荐）
# BACKEND=hf bash scripts/setup_mac.sh # 改用 transformers 原生（-hf）后端
```

手动安装（MLX 后端）只需：

```bash
brew install ffmpeg portaudio
conda create -n qwen3-asr python=3.12 -y && conda activate qwen3-asr
pip install -U mlx-audio librosa soundfile sounddevice flask
```

> 环境必须是 **arm64 原生** Python（检查：`python -c "import platform; print(platform.machine())"` 应输出 `arm64`）。模型首次运行时自动从 HuggingFace 下载，国内可先 `export HF_ENDPOINT=https://hf-mirror.com`。

## 网页工具

```bash
python scripts/web_app.py
# 浏览器打开 http://127.0.0.1:8000
```

- **录音转写**：点击按钮开始录音，识别文字边录边刷新（紫色为当前段落临时结果，满 `--segment-seconds`（默认 25 秒）后固化）；停止后自动保存录音 wav 与转写 txt
- **文件转写**：上传 wav / mp3 / m4a / mp4 等文件，显示并自动保存识别文本
- **历史记录**：查看/下载所有已保存的转写结果（默认存到 `./recordings/`）

常用参数：`--model`、`--port`（默认 8000）、`--host 0.0.0.0`（局域网访问；注意浏览器仅在 localhost/HTTPS 下允许录音）、`--output-dir`。

> 「实时」为分段增量转写：浏览器每 3 秒上传音频块，服务端反复转写当前段落刷新结果，延迟约数秒。推理串行化，多人同时使用会排队。

## 命令行

```bash
# 现场录音（下课按 Ctrl+C 结束），自动检测语言 + 逐词时间戳
python scripts/record_and_transcribe.py --record --timestamps

# 强制中文识别更稳
python scripts/record_and_transcribe.py --record --language Chinese

# 转写已有文件（用 1.7B-4bit 质量更高）
python scripts/record_and_transcribe.py --audio ./lesson.m4a --model mlx-community/Qwen3-ASR-1.7B-4bit --timestamps
```

| 参数 | 说明 |
|---|---|
| `--record` / `--audio PATH` | 现场录音 / 转写已有文件（二选一） |
| `--duration SECONDS` | 录音最长秒数（不填则录到 `Ctrl+C`） |
| `--model` | 模型名或本地目录（默认 `mlx-community/Qwen3-ASR-0.6B-8bit`） |
| `--language` | 强制语言（如 `Chinese`/`English`）；不填自动检测 |
| `--timestamps` | 输出逐词/逐字时间戳 |
| `--chunk-seconds` | 长音频分段时长（默认 30 秒） |
| `--output-dir` | 保存目录（默认 `./recordings`） |

也可直接用 `mlx-audio` 的 Python API：

```python
from mlx_audio.stt import load

model = load("mlx-community/Qwen3-ASR-1.7B-4bit")
result = model.generate("path_to_audio.wav", language="Chinese")
print(result.text)
```

## 常见问题

- **首次运行慢？** 需下载模型权重（约 1GB）并预热，之后明显变快。国内先设 `export HF_ENDPOINT=https://hf-mirror.com`。
- **无法录音？** 在 **系统设置 → 隐私与安全性 → 麦克风** 里允许终端（或 IDE）访问；`brew install portaudio` 后重装 `pip install -U sounddevice`。
- **`No module named 'mlx_audio'` 或 MLX 装不上？** `pip install -U mlx-audio`；确认 Python 环境是 arm64 原生（Rosetta/x86_64 环境装不了）。
- **MLX 报 `AttributeError: 'str' object has no attribute '__module__'`？** `mlx-lm` 与新版 transformers 的兼容问题，本仓库脚本已内置补丁，拉取最新代码即可；也可 `pip install -U mlx-lm`。
- **网页工具报 `There is no Stream(gpu, N) in current thread`？** MLX 计算流绑定线程，本仓库已把 MLX 推理固定到专用线程，拉取最新代码即可。
- **-hf 后端报 `KeyError: 'qwen3_asr'`？** transformers 版本没有 Qwen3-ASR 原生支持，安装开发版：`pip install -U "git+https://github.com/huggingface/transformers"`。
- **-hf 后端提示 `Disabling PyTorch because PyTorch >= 2.4 is required`？** 升级：`pip install -U "torch>=2.4" torchaudio`。若 pip 说找不到 ≥2.4 的版本，说明环境是 x86_64/Rosetta，需要新建 arm64 原生环境（`CONDA_SUBDIR=osx-arm64 conda create -n qwen3-asr python=3.12 -y`）。
- **内存吃紧？** 用默认 0.6B-8bit 模型，调小 `--chunk-seconds`，或关闭 `--timestamps`。

## 许可

Apache-2.0（见 [LICENSE](LICENSE)）。模型与上游代码版权归 Alibaba Qwen 团队所有，引用方式见[上游仓库](https://github.com/QwenLM/Qwen3-ASR#citation)。
