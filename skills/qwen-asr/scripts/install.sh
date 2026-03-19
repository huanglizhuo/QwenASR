#!/usr/bin/env bash
set -euo pipefail

REPO="huanglizhuo/QwenASR"
INSTALL_DIR="${HOME}/.local/bin"
MODEL_DIR="${HOME}/.openclaw/tools/qwen-asr/qwen3-asr-0.6b"

# --- 1. Install binary ---
if command -v qwen-asr &>/dev/null; then
  echo "qwen-asr is already installed: $(command -v qwen-asr)"
else
  # Detect platform
  OS="$(uname -s)"
  ARCH="$(uname -m)"

  case "${OS}-${ARCH}" in
    Darwin-arm64)  TARGET="aarch64-apple-darwin" ;;
    Linux-x86_64)  TARGET="x86_64-unknown-linux-gnu" ;;
    *)
      echo "No pre-built binary for ${OS}-${ARCH}."
      echo "Install from source: cargo install qwen-asr-cli"
      exit 1
      ;;
  esac

  # Get latest qwen-asr-cli release tag
  echo "Fetching latest release..."
  TAG=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases" \
    | grep -o '"tag_name": *"qwen-asr-cli-v[^"]*"' \
    | head -1 \
    | sed 's/"tag_name": *"//;s/"//')

  if [ -z "$TAG" ]; then
    echo "Could not find a qwen-asr-cli release."
    echo "Install from source: cargo install qwen-asr-cli"
    exit 1
  fi

  VERSION="${TAG#qwen-asr-cli-v}"
  ARCHIVE="qwen-asr-${VERSION}-${TARGET}.tar.gz"
  URL="https://github.com/${REPO}/releases/download/${TAG}/${ARCHIVE}"

  echo "Downloading ${ARCHIVE}..."
  TMPDIR="$(mktemp -d)"
  trap 'rm -rf "$TMPDIR"' EXIT

  if ! curl -fSL -o "${TMPDIR}/${ARCHIVE}" "$URL"; then
    echo "Download failed. No pre-built binary for your platform in this release."
    echo "Install from source: cargo install qwen-asr-cli"
    exit 1
  fi

  # Extract to install dir
  mkdir -p "$INSTALL_DIR"
  tar -xzf "${TMPDIR}/${ARCHIVE}" -C "$INSTALL_DIR"
  chmod +x "${INSTALL_DIR}/qwen-asr"

  echo "Installed qwen-asr to ${INSTALL_DIR}/qwen-asr"

  # Check if INSTALL_DIR is in PATH
  if ! echo "$PATH" | tr ':' '\n' | grep -qx "$INSTALL_DIR"; then
    echo ""
    echo "NOTE: ${INSTALL_DIR} is not in your PATH."
    echo "Add it with:  export PATH=\"${INSTALL_DIR}:\$PATH\""
  fi
fi

# --- 2. Download model ---
if [ -d "$MODEL_DIR" ] && [ -f "${MODEL_DIR}/model.safetensors" ]; then
  echo "Model already downloaded at ${MODEL_DIR}"
else
  echo "Downloading qwen3-asr-0.6b model..."
  mkdir -p "$(dirname "$MODEL_DIR")"
  qwen-asr download qwen3-asr-0.6b --output "$MODEL_DIR"
  echo "Model downloaded to ${MODEL_DIR}"
fi

echo ""
echo "Setup complete! Test with:"
echo "  qwen-asr -d ${MODEL_DIR} -i <audio-file> --silent"
