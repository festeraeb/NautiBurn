#!/bin/bash
# Fast link of moe_histodb_scrape against the existing T440 CUDA-60 llama.cpp libs.
# Not a full llama.cpp rebuild.
set -euo pipefail
ROOT="${LLAMA_ROOT:-/home/cesarops/src/llama.cpp-mainline}"
BIN="${LLAMA_BIN:-$ROOT/build-cuda-60/bin}"
OUT="${1:-/home/cesarops/NautiBurn/b1/moe-histodb-scrape}"
SRC="/home/cesarops/NautiBurn/b1/moe_histodb_scrape.cpp"

g++ -O2 -std=c++17 \
  -I"$ROOT" -I"$ROOT/include" -I"$ROOT/common" -I"$ROOT/ggml/include" \
  -o "$OUT" "$SRC" \
  -L"$BIN" -lllama-common -lllama -lggml -lggml-cuda -lggml-cpu -lggml-base \
  -Wl,-rpath,"$BIN" \
  -L/usr/local/cuda-12.8/lib64 -lcudart -lcublas -lcublasLt \
  -lssl -lcrypto -lgomp -pthread

echo "built $OUT"
ldd "$OUT" | head -20
