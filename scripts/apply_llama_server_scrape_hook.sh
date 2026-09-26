#!/usr/bin/env bash
# Apply NautiBurn llama-server MoE scrape adapter into a llama.cpp tree.
# Adapter only — does not turn llama.cpp into the Burn engine.
set -euo pipefail
ROOT="${1:-/home/cesarops/src/llama.cpp-mainline}"
SRC="$(cd "$(dirname "$0")/.." && pwd)/b1"
cp "$SRC/nautiburn_moe_server_scrape.h" "$ROOT/common/"
cp "$SRC/nautiburn_moe_server_scrape.cpp" "$ROOT/common/"
if ! grep -q 'nautiburn_moe_server_scrape.cpp' "$ROOT/common/CMakeLists.txt"; then
  python3 - <<'PY' "$ROOT/common/CMakeLists.txt"
import pathlib, sys
p = pathlib.Path(sys.argv[1])
t = p.read_text()
needle = "    ngram-mod.h\n"
insert = needle + "    nautiburn_moe_server_scrape.cpp\n    nautiburn_moe_server_scrape.h\n"
if "nautiburn_moe_server_scrape.cpp" not in t:
    if needle not in t:
        raise SystemExit("CMakeLists.txt: ngram-mod.h anchor missing")
    p.write_text(t.replace(needle, insert, 1))
print("CMakeLists.txt: added nautiburn_moe_server_scrape sources")
PY
fi
# server-context include + enable + flush — idempotent markers
CTX="$ROOT/tools/server/server-context.cpp"
if ! grep -q 'nautiburn_moe_server_scrape.h' "$CTX"; then
  sed -i '/#include "log.h"/a #include "nautiburn_moe_server_scrape.h"' "$CTX"
fi
if ! grep -q 'nautiburn_moe_server_scrape_maybe_enable' "$CTX"; then
  python3 - <<'PY' "$CTX"
import pathlib, sys
p = pathlib.Path(sys.argv[1])
t = p.read_text()
old = """        llama_init = common_init_from_params(params_base);"""
new = """        // Optional NautiBurn MoE scrape adapter (env-gated; no-op unless set).
        nautiburn_moe_server_scrape_maybe_enable(params_base);

        llama_init = common_init_from_params(params_base);"""
if old not in t:
    raise SystemExit("server-context.cpp: common_init_from_params anchor missing")
p.write_text(t.replace(old, new, 1))
print("server-context.cpp: wired maybe_enable")
PY
fi
if ! grep -q 'nautiburn_moe_server_scrape_flush' "$CTX"; then
  python3 - <<'PY' "$CTX"
import pathlib, sys
p = pathlib.Path(sys.argv[1])
t = p.read_text()
old = """    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();"""
new = """    void send_final_response(server_slot & slot) {
        // Flush any remaining MoE scrape rows from this generation.
        nautiburn_moe_server_scrape_flush();

        auto res = std::make_unique<server_task_result_cmpl_final>();"""
if old not in t:
    raise SystemExit("server-context.cpp: send_final_response anchor missing")
p.write_text(t.replace(old, new, 1))
print("server-context.cpp: wired flush")
PY
fi
echo "applied nautiburn server scrape adapter -> $ROOT"
