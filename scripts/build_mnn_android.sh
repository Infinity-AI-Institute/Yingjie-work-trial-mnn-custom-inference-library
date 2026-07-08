#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MNN="$ROOT/third_party/MNN"
BUILD="$MNN/build_android_arm64_llm_sep"

if [[ ! -d "$MNN" ]]; then
  git clone --depth 1 https://github.com/alibaba/MNN.git "$MNN"
fi

cd "$MNN"
git fetch --depth 1 origin 0bff03cbef43c783f44e41484b9f8a0b28bd758d
git checkout 0bff03cbef43c783f44e41484b9f8a0b28bd758d

ANDROID_HOME="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
if [[ -z "$ANDROID_HOME" ]]; then
  echo "Set ANDROID_HOME or ANDROID_SDK_ROOT" >&2
  exit 2
fi
ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$(find "$ANDROID_HOME/ndk" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)}"

cmake -S "$MNN" -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_LLM=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_OPENCL=ON \
  -DMNN_VULKAN=ON \
  -DMNN_ARM82=ON \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
cmake --build "$BUILD" --parallel

