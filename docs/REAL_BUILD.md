# REAL_BUILD: NautiBurn (what was faked, what a real measurement looks like)

Branch `cleanup/no-sim-2026-10-28`, branched from `cursor/moe-llama-scrape-6889` (the current branch;
the remote default is `main`).

## Faked results and what they were meant to show

| Removed | What it did | Meant to show |
|---|---|---|
| `scripts/orchestrate_moe_prefetch_architecture.py` + `artifacts/prefetch_architecture/benchmark_canary_results.json` | Monte-Carlo "canary benchmark": sampled experts from a frequency table with `random.choices`, decided "predicted" with `random.random() < 0.515`, charged made-up microsecond constants for PCIe DMA and CPU fallback, and printed "5.4 -> 10.1 tok/s, +85%". No model ran. | That a lookahead prefetcher hides expert transfer time on dual Tesla P100. |
| `scripts/nautik9_moe_rotation_runner.py` + `artifacts/nautik9_moe_rotation/{rotation_results.json,NAUTIK9_MOE_ROTATION_REPORT.md}` | Hard-coded "clustering_similarity" and "p_value" per model family, "domain accuracy" = that number plus `hash(domain + family) % 100` jitter, and speedups computed from constants. No router was run on any prompt. | That a small sibling MoE's routing predicts a large sibling's experts, giving +16 to +21% throughput. |

## What a real build needs

1. Predictor accuracy: run the small and the large model on the same real prompts (several hundred,
   across domains), scrape which experts fire per layer per token (the existing scraper and histodb do
   this), and score the predictor against the large model's measured trace. Report hit rate with the
   number of prompts and tokens, and compare to the copy-previous-token baseline (which the page says
   already scores 0.56 on the one OLMoE trace).
2. Timing: measure per-layer decode time and PCIe transfer time on the actual card with the actual
   expert size (CUDA events or the engine's own log), not constants.
3. Speedup: only by running the same model and prompts with and without the prefetch hook inside the
   engine (Nauti-Strata `/v1/telemetry/prefetch` counters exist for this), several reps, and quoting the
   engine's own tok/s.

## Numbers that are allowed to be claimed

Only measured traces and engine-logged tok/s with their sample sizes. The current honest statement is the
page's: shadow mode, one small OLMoE trace, simple baselines are hard to beat, no speedup claim.

## Kept, with caveats (not removed)

* `crates/nautiburn-shadow/fixtures/devices.stub.json` is labelled `stub: true` and the consume loops
  use it, so "which device already holds the experts" in shadow logs is against an example layout,
  not live placement. Replace with a real device map before quoting placement results.
* `crates/nautiburn-layer0` runs real kernels on real OLMoE weights with synthetic hidden states
  (stated in its header); its expert ids are a kernel check, not a routing trace of a real prompt.
