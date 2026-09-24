#!/bin/bash
# Ivy Bridge flags: cesarops2 is a Xeon E5-2450 v2 (AVX, no AVX2, no FMA, no BMI2).
set -euo pipefail
ROOT="/home/cesarops/src/llama.cpp-mainline"
IVY="$ROOT/build-cuda-60-ivy/bin"
OUT="/home/cesarops/NautiBurn/b5/olmoe-layer0-oracle"
g++ -O2 -std=c++17 -march=ivybridge -fno-finite-math-only \
  -I"$ROOT/ggml/include" \
  -o "$OUT" \
  /home/cesarops/NautiBurn/b5/olmoe_layer0_oracle.cpp \
  -L"$IVY" -lggml -lggml-cuda -lggml-cpu -lggml-base \
  -Wl,-rpath,"$IVY" \
  -L/usr/local/cuda-12.8/lib64 -lcudart -lcublas -lcublasLt
echo "built $OUT"
