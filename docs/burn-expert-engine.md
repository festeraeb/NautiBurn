# Burn expert engine

The crate is `crates/nautiburn-engine`. It is the hook and the JSON handoff. The GEMM call site is still `gemm_cubecl` (`Tensor::matmul`) in `nautiburn-layer0`. On a P100 the scheduled dtype is `f16`. A launch that needs more than 49152 bytes of shared memory is refused. The known CubeCL FP16 expert GEMM footprint, 69632 bytes, is refused. This crate does not rewrite that launch as FP32.

Both T440 P100s were occupied when this crate was added (GPU0 about 13033 MiB used, GPU1 about 12241 MiB used). cesarops4's three P100s are the Mixtral server. No new FP16 kernel was launched here.

## What the hook records

MoE-Beyond writes `Layer ID`, `Batch Number`, `Token`, `Activated Expert IDs`, and a token embedding. The hook stores the first four, plus the phase and the token index. Embeddings are not in the B1 trace, so they are not invented. moe-viz consumes the same per-token expert list (`token_index`, `layer`, `expert_ids`).

Counts are a layer-by-expert matrix, prefill and decode apart, which is the MoE-Beyond expert activation matrix. A next-layer joint is updated when the same token is seen at layer L and then layer L+1. A job name keeps its own matrix so a later pass can preload by job type.

`ingest` of `b1/trace.tsv` counts 227 rows: 211 prefill (1688 slots) and 16 decode (128 slots).

## JSON handoff

`route` reads an `ExpertPass` and writes `NextExperts`. The expert list is exactly the model's k when a source has counts. Order: next-layer joint from the experts just seen, then that job's histogram at the next layer, then the phase frequency at the next layer. If those are empty and `data/olmoe-expert-calibration.json` is loaded, the next ids are the experts with the largest router-row L2 at that layer (`weight_calibration`). An empty source with no calibration returns no ids.

`activation_bytes` is the FP16 hidden state for `context_tokens` (`tokens * 2048 * 2` on OLMoE). `slab_bytes` is the expert weights the handoff names. `copy_ms_estimate` divides `slab_bytes` by the plan's 12 GB/s. That is a planning figure, not a measured memcpy. The 13 ms figure is the rough Gen3 PCIe x16 budget for loading experts onto a card. Eight OLMoE FP16 experts are 100663296 bytes, 8.39 ms at that planning rate, inside the budget. The plan's two Mixtral Q4 experts, 200000000 bytes, are 16.67 ms, outside it. The card layout is not chosen here. It follows from how fast the CPU path gets and how often the prediction is right.

## CMoE and ExpertWeaver

CMoE partitions a dense FFN by activation frequency into shared neurons and routed neurons, then builds a router from those statistics. The useful piece for this engine is the frequency split. The histogram already stores it. The carving script is not ported.

ExpertWeaver clusters GLU activation patterns and sets a shared-versus-routed ratio from their coefficient of variation. Those activations are still not in the B1 trace, so that split is not in the forward.

The checkpoint also did not ship expert-weight calibration. `scripts/olmoe_expert_calibration.py` reads `olmoe-1b-7b-0924-f16.gguf` and writes `data/olmoe-expert-calibration.json`: 16 layers, 64 experts, 1024 rows. Each row is the router-row L2 of `ffn_gate_inp` plus the RMS of the FP16 gate, up, and down projections. Layer 0 expert 0 measured `router_l2` 1.4803899175344386. Across the file, router L2 runs from 0.46006687831362936 to 2.0849092499447957, and the three RMS fields stay near 0.015. `route` attaches those scales to the experts it names.

The learned MoE-Beyond transformer (the 97.5% figure in their paper, on DeepSeek-V2-Lite traces) is not loaded. No hit rate is claimed for it on OLMoE.
