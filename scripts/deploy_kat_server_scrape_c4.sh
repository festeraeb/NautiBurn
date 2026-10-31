#!/usr/bin/env bash
# Deploy rebuilt llama-cuda60 libs from T440 → c4 and restart Kat with scrape env.
# Brief restart only; restores kat-coder on :5017.
set -euo pipefail
SRC="${SRC:-/home/cesarops/src/llama.cpp-mainline/build-cuda-60/bin}"
DST_HOST="${DST_HOST:-cesarops4}"
DST="${DST:-/home/cesarops/llama-cuda60/bin}"
SCRAPE_DIR="${SCRAPE_DIR:-/tmp/nautiburn-llama-scrape/kat}"
MODEL_ID="${MODEL_ID:-kat-coder-v2.5-dev-q8_0}"

need=(libllama-common.so libllama-common.so.0 libllama-server-impl.so llama-server)
for f in "${need[@]}"; do
  test -e "$SRC/$f" || { echo "missing $SRC/$f"; exit 1; }
done

echo "rsync libs → ${DST_HOST}:${DST}"
rsync -az --info=progress2 \
  "$SRC/libllama-common.so"* \
  "$SRC/libllama-server-impl.so" \
  "$SRC/llama-server" \
  "${DST_HOST}:${DST}/"

ssh "$DST_HOST" "mkdir -p '$SCRAPE_DIR'"

echo "restart kat-coder with NAUTIBURN_SCRAPE_DIR=$SCRAPE_DIR"
ssh "$DST_HOST" bash -s <<EOF
set -euo pipefail
export LD_LIBRARY_PATH=/home/cesarops/llama-cuda60/bin:/home/cesarops/llama-cuda60/cuda12
export CUDA_VISIBLE_DEVICES=GPU-c012cc70-3f06-1718-84e2-073aa275f0af,GPU-2d66f961-cba7-717f-1bd7-a04fff1fc8ab,GPU-0c4b4f5d-5f54-46db-be25-f6ffdf75ac79
export NAUTIBURN_SCRAPE_DIR='$SCRAPE_DIR'
export NAUTIBURN_MODEL_ID='$MODEL_ID'
export NAUTIBURN_CLASS_ID=llama-server-live
export NAUTIBURN_DTYPE=q8_0
# stop existing session if present
tmux has-session -t kat-coder 2>/dev/null && tmux kill-session -t kat-coder || true
sleep 2
tmux new-session -d -s kat-coder -- bash -lc '
  export LD_LIBRARY_PATH=/home/cesarops/llama-cuda60/bin:/home/cesarops/llama-cuda60/cuda12
  export CUDA_VISIBLE_DEVICES=GPU-c012cc70-3f06-1718-84e2-073aa275f0af,GPU-2d66f961-cba7-717f-1bd7-a04fff1fc8ab,GPU-0c4b4f5d-5f54-46db-be25-f6ffdf75ac79
  export NAUTIBURN_SCRAPE_DIR=$SCRAPE_DIR
  export NAUTIBURN_MODEL_ID=$MODEL_ID
  export NAUTIBURN_CLASS_ID=llama-server-live
  export NAUTIBURN_DTYPE=q8_0
  exec /home/cesarops/llama-cuda60/bin/llama-server \\
    -m /home/cesarops/models/Kwaipilot_KAT-Coder-V2.5-Dev-Q8_0.gguf \\
    --host 127.0.0.1 --port 5017 \\
    -ngl 99 -sm layer -ts 1,1,1 -fit off \\
    -c 512 -np 1 -t 8 -fa off \\
    --alias KAT-Coder-V2.5-Dev-Q8_0 \\
    > /home/cesarops/models/kat-coder-server.log 2>&1
'
# wait for health
for i in \$(seq 1 90); do
  if curl -sf -m 2 http://127.0.0.1:5017/health >/dev/null; then
    echo "kat health ok after \${i}s"
    exit 0
  fi
  sleep 2
done
echo "kat failed to become healthy" >&2
tail -40 /home/cesarops/models/kat-coder-server.log >&2 || true
exit 1
EOF
echo "c4 kat scrape deploy done"
