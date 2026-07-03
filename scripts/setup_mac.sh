#!/usr/bin/env bash
#
# setup_mac.sh — 在 Apple Silicon Mac（M1/M2/M3…）上一键安装 Qwen3-ASR 课堂转写工具的运行环境。
#
# 该脚本会：
#   1. 检查是否为 macOS + Apple Silicon（arm64）
#   2. 通过 Homebrew 安装系统依赖（ffmpeg、portaudio）
#   3. 创建并激活一个 arm64 原生的 conda / venv Python 3.12 环境
#   4. 安装所选后端的依赖（默认 MLX；可选 transformers 原生 -hf）
#   5. 可选：预下载模型权重
#
# 用法：
#   bash scripts/setup_mac.sh                     # 默认安装 MLX 后端（推荐）
#   BACKEND=hf bash scripts/setup_mac.sh          # 安装 transformers 原生（-hf）后端
#   USE_MODELSCOPE=1 bash scripts/setup_mac.sh    # 额外用 ModelScope 预下载模型（国内用户，仅 -hf 模型有）
#   ENV_NAME=my-asr bash scripts/setup_mac.sh     # 自定义 conda 环境名
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-qwen3-asr}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
BACKEND="${BACKEND:-mlx}"             # mlx（推荐）| hf（transformers 原生）
USE_MODELSCOPE="${USE_MODELSCOPE:-0}"
MODEL_DIR="${MODEL_DIR:-./models}"

if [[ "${BACKEND}" == "mlx" ]]; then
  MODEL_ID="${MODEL_ID:-mlx-community/Qwen3-ASR-0.6B-8bit}"
  ALIGNER_ID="${ALIGNER_ID:-mlx-community/Qwen3-ForcedAligner-0.6B-8bit}"
else
  MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-0.6B-hf}"
  ALIGNER_ID="${ALIGNER_ID:-Qwen/Qwen3-ForcedAligner-0.6B-hf}"
fi

info()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
error() { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" 1>&2; }

# ---------------------------------------------------------------------------
# 1. 平台检查（仅支持 macOS + Apple Silicon）
# ---------------------------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  error "该仓库仅支持 macOS（Apple Silicon）。当前系统：$(uname -s)"
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  error "该仓库仅支持 Apple Silicon（arm64，如 M1/M2/M3）。当前架构：$(uname -m)"
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Homebrew 与系统依赖
# ---------------------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  error "未检测到 Homebrew。请先安装（https://brew.sh）后重新运行本脚本："
  error '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  exit 1
fi

info "通过 Homebrew 安装系统依赖（ffmpeg / portaudio）……"
brew install ffmpeg portaudio || warn "部分 brew 包安装失败，如果后续出错请手动重试。"

# ---------------------------------------------------------------------------
# 3. Python 环境（arm64 原生；优先 conda，否则退回 python3 -m venv）
# ---------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
  info "使用 conda 创建 arm64 原生 Python ${PYTHON_VERSION} 环境：${ENV_NAME}"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    CONDA_SUBDIR=osx-arm64 conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
  else
    info "conda 环境 ${ENV_NAME} 已存在，跳过创建。"
  fi
  conda activate "${ENV_NAME}"
  conda config --env --set subdir osx-arm64 || true
  ACTIVATE_HINT="conda activate ${ENV_NAME}"
else
  warn "未检测到 conda，改用 python3 -m venv 创建虚拟环境 .venv-${ENV_NAME}"
  python3 -m venv ".venv-${ENV_NAME}"
  # shellcheck disable=SC1091
  source ".venv-${ENV_NAME}/bin/activate"
  ACTIVATE_HINT="source .venv-${ENV_NAME}/bin/activate"
fi

python -m pip install -U pip wheel

# 校验环境是否为 arm64 原生（x86_64/Rosetta 环境装不了 MLX，torch 也只能到 2.2.2）
PY_MACHINE="$(python -c 'import platform; print(platform.machine())')"
info "Python 架构：${PY_MACHINE}"
if [[ "$PY_MACHINE" != "arm64" ]]; then
  error "当前 Python 是 ${PY_MACHINE}（Intel/Rosetta），无法使用 MLX / 新版 PyTorch。"
  error "请安装原生 arm64 的 conda（推荐 Miniforge：brew install miniforge）后重试，"
  error "或手动创建原生环境：CONDA_SUBDIR=osx-arm64 conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y"
  exit 1
fi

# ---------------------------------------------------------------------------
# 4. 安装 Python 依赖
# ---------------------------------------------------------------------------
info "安装通用依赖（librosa / soundfile / sounddevice / flask）……"
python -m pip install -U librosa soundfile sounddevice flask

if [[ "${BACKEND}" == "mlx" ]]; then
  info "安装 MLX 后端（mlx-audio）……"
  python -m pip install -U mlx-audio
else
  # transformers 原生（-hf）后端：需要 PyTorch >= 2.4 + 含 qwen3_asr 支持的 transformers
  info "安装 PyTorch >= 2.4（含 Apple Silicon MPS 支持）……"
  python -m pip install -U "torch>=2.4" torchaudio
  # 很多已发布的 transformers 尚未包含 qwen3_asr，直接安装源码开发版最稳
  TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-git+https://github.com/huggingface/transformers}"
  info "安装 transformers：${TRANSFORMERS_SPEC}"
  python -m pip install -U "${TRANSFORMERS_SPEC}" accelerate
  if ! python -c "from transformers import Qwen3ASRForConditionalGeneration" >/dev/null 2>&1; then
    warn "当前 transformers 仍不认识 qwen3_asr；如运行报错，请确认已安装源码开发版。"
  fi
  info "安装日语/韩语时间戳依赖（nagisa / soynlp）……"
  python -m pip install -U nagisa soynlp
fi

# ---------------------------------------------------------------------------
# 5. 可选：预下载模型
# ---------------------------------------------------------------------------
if [[ "${USE_MODELSCOPE}" == "1" ]]; then
  if [[ "${BACKEND}" == "mlx" ]]; then
    warn "mlx-community 模型主要发布在 HuggingFace；若 ModelScope 上没有同名仓库，下载会失败。"
    warn "失败时请改用：HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ${MODEL_ID} --local-dir ${MODEL_DIR}/$(basename "${MODEL_ID}")"
  fi
  info "使用 ModelScope 预下载模型到 ${MODEL_DIR} ……"
  python -m pip install -U modelscope
  mkdir -p "${MODEL_DIR}"
  modelscope download --model "${MODEL_ID}"   --local_dir "${MODEL_DIR}/$(basename "${MODEL_ID}")"
  modelscope download --model "${ALIGNER_ID}" --local_dir "${MODEL_DIR}/$(basename "${ALIGNER_ID}")"
fi

# ---------------------------------------------------------------------------
# 完成
# ---------------------------------------------------------------------------
cat <<EOF

============================================================
✅ 安装完成！（后端：${BACKEND}，默认模型：${MODEL_ID}）

下次使用前，请先激活环境：
    ${ACTIVATE_HINT}

网页工具（录音准实时转写 + 文件上传转写）：
    python scripts/web_app.py
    # 浏览器打开 http://127.0.0.1:8000

命令行：现场录制课堂并转写（结束时按 Ctrl+C）：
    python scripts/record_and_transcribe.py --record --timestamps

命令行：转写一个音频/视频文件：
    python scripts/record_and_transcribe.py --audio /path/to/audio.wav --language Chinese

提示：模型首次运行时自动从 HuggingFace 下载；国内可先设置镜像：
    export HF_ENDPOINT=https://hf-mirror.com
============================================================
EOF
