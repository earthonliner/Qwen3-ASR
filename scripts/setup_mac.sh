#!/usr/bin/env bash
#
# setup_mac.sh — 在 Apple Silicon Mac（如 MacBook Air M2 / 16GB）上一键安装 Qwen3-ASR-0.6B 的运行环境。
#
# 该脚本会：
#   1. 检查是否为 Apple Silicon（arm64）
#   2. 通过 Homebrew 安装系统依赖（ffmpeg、sox、portaudio）
#   3. 创建并激活一个独立的 conda / venv Python 3.12 环境
#   4. 安装 PyTorch（自带 MPS/Metal 支持）、transformers/qwen-asr 以及录音所需的 sounddevice
#   5. 可选：从 ModelScope 预先下载 0.6B 模型（适合中国大陆网络）
#
# 默认安装 -hf（🤗 transformers 原生）路线，对应 Qwen/Qwen3-ASR-0.6B-hf。
#
# 用法：
#   bash scripts/setup_mac.sh                     # 默认安装（-hf / 原生 transformers）
#   USE_HF=0 bash scripts/setup_mac.sh            # 改用非 -hf 的 qwen-asr 包版本
#   USE_MODELSCOPE=1 bash scripts/setup_mac.sh    # 额外用 ModelScope 预下载模型（推荐国内用户）
#   ENV_NAME=my-asr bash scripts/setup_mac.sh     # 自定义 conda 环境名
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-qwen3-asr}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
USE_HF="${USE_HF:-1}"                 # 1=使用 -hf 原生 transformers；0=使用 qwen-asr 包
USE_MODELSCOPE="${USE_MODELSCOPE:-0}"
MODEL_DIR="${MODEL_DIR:-./models}"

if [[ "${USE_HF}" == "1" ]]; then
  MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-0.6B-hf}"
  ALIGNER_ID="${ALIGNER_ID:-Qwen/Qwen3-ForcedAligner-0.6B-hf}"
else
  MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"
  ALIGNER_ID="${ALIGNER_ID:-Qwen/Qwen3-ForcedAligner-0.6B}"
fi

info()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
error() { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" 1>&2; }

# ---------------------------------------------------------------------------
# 1. 平台检查
# ---------------------------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  error "该脚本仅适用于 macOS。当前系统：$(uname -s)"
  exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
  warn "检测到架构为 ${ARCH}，本指南针对 Apple Silicon（arm64，如 M1/M2/M3）优化。Intel Mac 只能用 CPU 运行，速度较慢。"
fi

# ---------------------------------------------------------------------------
# 2. Homebrew 与系统依赖
# ---------------------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  warn "未检测到 Homebrew。请先安装 Homebrew：https://brew.sh"
  warn "安装命令："
  warn '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  error "请安装 Homebrew 后重新运行本脚本。"
  exit 1
fi

info "通过 Homebrew 安装系统依赖（ffmpeg / sox / portaudio）……"
brew install ffmpeg sox portaudio || warn "部分 brew 包安装失败，如果后续出错请手动重试。"

# ---------------------------------------------------------------------------
# 3. Python 环境（优先 conda，否则退回 python3 -m venv）
# ---------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
  info "使用 conda 创建 Python ${PYTHON_VERSION} 环境：${ENV_NAME}"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
  else
    info "conda 环境 ${ENV_NAME} 已存在，跳过创建。"
  fi
  conda activate "${ENV_NAME}"
  ACTIVATE_HINT="conda activate ${ENV_NAME}"
else
  warn "未检测到 conda，改用 python3 -m venv 创建虚拟环境 .venv-${ENV_NAME}"
  python3 -m venv ".venv-${ENV_NAME}"
  # shellcheck disable=SC1091
  source ".venv-${ENV_NAME}/bin/activate"
  ACTIVATE_HINT="source .venv-${ENV_NAME}/bin/activate"
fi

python -m pip install -U pip wheel

# ---------------------------------------------------------------------------
# 4. 安装 Python 依赖
# ---------------------------------------------------------------------------
# PyTorch 在 macOS 上的官方 wheel 已内置 MPS（Metal）后端，直接 pip 安装即可。
# 新版 transformers 需要 PyTorch >= 2.4，否则会禁用 PyTorch 导致模型无法加载。
info "安装 PyTorch >= 2.4（含 Apple Silicon MPS 支持）……"
python -m pip install -U "torch>=2.4" torchaudio

if [[ "${USE_HF}" == "1" ]]; then
  info "安装 transformers（-hf 模型的原生支持）……"
  # -hf 模型需要含 Qwen3-ASR 原生支持的 transformers。优先装稳定版，
  # 若正式版尚未包含该支持，请改用开发版（见下方提示）。
  python -m pip install -U "transformers>=4.57.6" accelerate
  info "安装音频处理依赖（librosa / soundfile / nagisa / soynlp）……"
  python -m pip install -U librosa soundfile nagisa soynlp
else
  info "安装 qwen-asr（非 -hf 的包版本）……"
  python -m pip install -U qwen-asr
fi

info "安装录音依赖 sounddevice……"
python -m pip install -U sounddevice

# 注意：vLLM / flash-attn 不支持 macOS，切勿在 Mac 上安装它们。

# ---------------------------------------------------------------------------
# 5. 可选：用 ModelScope 预下载模型（国内网络更快）
# ---------------------------------------------------------------------------
if [[ "${USE_MODELSCOPE}" == "1" ]]; then
  info "使用 ModelScope 预下载模型到 ${MODEL_DIR} ……"
  python -m pip install -U modelscope
  mkdir -p "${MODEL_DIR}"
  modelscope download --model "${MODEL_ID}"   --local_dir "${MODEL_DIR}/$(basename "${MODEL_ID}")"
  modelscope download --model "${ALIGNER_ID}" --local_dir "${MODEL_DIR}/$(basename "${ALIGNER_ID}")"
  info "模型已下载到 ${MODEL_DIR}。运行脚本时用 --model ${MODEL_DIR}/$(basename "${MODEL_ID}") 指向本地目录。"
fi

# ---------------------------------------------------------------------------
# 完成
# ---------------------------------------------------------------------------
cat <<EOF

============================================================
✅ 安装完成！

下次使用前，请先激活环境：
    ${ACTIVATE_HINT}

试运行（转写一个音频文件）：
    python scripts/record_and_transcribe.py --audio /path/to/audio.wav --language Chinese

现场录制课堂并转写（讲到结束时按 Ctrl+C 停止录音）：
    python scripts/record_and_transcribe.py --record --output-dir ./recordings --timestamps

默认使用的 ASR 模型：${MODEL_ID}
（脚本会根据模型名是否带 -hf 自动选择后端；如需切换可用 --model / --backend。）

如果你用 ModelScope 预下载了模型，加 --model 指向本地目录，例如：
    python scripts/record_and_transcribe.py --record --model ${MODEL_DIR}/$(basename "${MODEL_ID}")

提示：
- 如果直接从 HuggingFace 下载模型很慢，可先设置镜像：
      export HF_ENDPOINT=https://hf-mirror.com
- 若使用 -hf 模型时报错找不到 AutoModelForMultimodalLM / apply_transcription_request，
  说明 transformers 版本过旧，请安装开发版：
      pip install -U git+https://github.com/huggingface/transformers
============================================================
EOF
