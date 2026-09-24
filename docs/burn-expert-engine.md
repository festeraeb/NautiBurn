# Burn expert engine

The crate is `crates/nautiburn-engine`. It is the hook and the JSON handoff. The GEMM call site is still `gemm_cubecl` (`Tensor::matmul`) in `nautiburn-layer0`. On a P100 the scheduled dtype is `f16`. A launch that needs more than 49152 bytes of shared memory is refused. The known CubeCL FP16 expert GEMM footprint, 69632 bytes, is refused. This crate does not rewrite that launch as FP32.

Both T440 P100s were occupied when this crate was added (GPU0 about 13033 MiB used, GPU1 about 12241 MiB used). cesarops4's three P100s are the Mixtral server. No new FP16 kernel was launched here.

## What the hook records

MoE-Beyond writes `Layer ID`, `Batch Number`, `Token`, `Activated Expert IDs`, and a token embedding. The hook stores the first four, plus the phase and the token index. Embeddings are not in the B1 trace, so they are not invented. moe-viz consumes the same per-token expert list (`token_index`, `layer`, `expert_ids`).

Counts are a layer-by-expert matrix, prefill and decode apart, which is the MoE-Beyond expert activation matrix. A next-layer joint is updated when the same token is seen at layer L and then layer L+1. A job name keeps its own matrix so a later pass can preload by job type.

`ingest` of `b1/trace.tsv` counts 227 rows: 211 prefill (1688 slots) and 16 decode (128 slots).

## JSON handoff

`route` reads an `ExpertPass` and writes `NextExperts`. The expert list is exactly the model's k when a source has counts. Order: next-layer joint from the experts just seen, then that job's histogram at the next layer, then the phase frequency at the next layer. An empty source returns no ids.

`activation_bytes` is the FP16 hidden state for `context_tokens` (`tokens * 2048 * 2` on OLMoE). `slab_bytes` is the expert weights the handoff names. `copy_ms_estimate` divides `slab_bytes` by the plan's 12 GB/s. That is a planning figure, not a measured memcpy. The target is 13 ms. Eight OLMoE FP16 experts are 100663296 bytes, 8.39 ms, inside the target. The plan's two Mixtral Q4 experts, 200000000 bytes, are 16.67 ms, outside it. Those layouts (OpenVINO on the T440 and cesarops4, two hot cards plus a third guess card) are not built. The JSON is what a later tuner reads.

## CMoE and ExpertWeaver

CMoE partitions a dense FFN by activation frequency into shared neurons and routed neurons, then builds a router from those statistics. The useful piece for this engine is the frequency split. The histogram already stores it. The carving script is not ported.

ExpertWeaver clusters GLU activation patterns and sets a shared-versus-routed ratio from their coefficient of variation. That needs the calibration activations. This engine does not have them, so ExpertWeaver is not in the forward.

The learned MoE-Beyond transformer (the 97.5% figure in their paper, on DeepSeek-V2-Lite traces) is not loaded. No hit rate is claimed for it on OLMoE.
