# Burn expert engine

The crate is `crates/nautiburn-engine`. It is the hook and the JSON handoff. The GEMM call site is still `gemm_cubecl` (`Tensor::matmul`) in `nautiburn-layer0`. On a P100 the scheduled dtype is `f16`. A launch that needs more than 49152 bytes of shared memory is refused. The known CubeCL FP16 expert GEMM footprint, 69632 bytes, is refused. This crate does not rewrite that launch as FP32.

Both T440 P100s were occupied when this crate was added (GPU0 about 13033 MiB used, GPU1 about 12241 MiB used). cesarops4's three P100s are the Mixtral server. No new FP16 kernel was launched here.

## What the hook records

MoE-Beyond writes `Layer ID`, `Batch Number`, `Token`, `Activated Expert IDs`, and a token embedding. The hook stores the first four, plus the phase and the token index. Embeddings are not in the B1 trace, so they are not invented. moe-viz consumes the same per-token expert list (`token_index`, `layer`, `expert_ids`).

Counts are a layer-by-expert matrix, prefill and decode apart, which is the MoE-Beyond expert activation matrix. A next-layer joint is updated when the same token is seen at layer L and then layer L+1. A job name keeps its own matrix so a later pass can preload by job type.

`ingest` of `b1/trace.tsv` counts 227 rows: 211 prefill (1688 slots) and 16 decode (128 slots).

## JSON handoff

`route` reads an `ExpertPass` and writes `NextExperts`. The expert list is exactly the model's k when a source has counts. Order: next-layer joint from the experts just seen, then that job's histogram at the next layer, then the phase frequency at the next layer. If those are empty and `data/olmoe-expert-calibration.json` is loaded, the next ids are the experts with the largest router-row L2 at that layer (`weight_calibration`). An empty source with no calibration returns no ids.

`activation_bytes` is the FP16 hidden state for `context_tokens` (`tokens * 2048 * 2` on OLMoE). `slab_bytes` is the expert weights the handoff names. `copy_ms_estimate` divides `slab_bytes` by the plan's 12 GB/s. That is a planning figure, not a measured memcpy. The miss path is the next section.

## Same model on the Xeon

Two computers, T440 and cesarops4, each with its Xeon Gold. Each may use all of its cores. The CPU holds the same model the GPUs are running, all experts in system RAM, because the experts are the same. A different toy model is not the test.

A preload miss must not leave the GPU idle waiting for weights. The job (the work, including the context that has to move) runs on the CPU copy, or is passed to a card that already has the experts. Weights are not what moves in that pass.

13 ms is the user's average from other machines that predict MoE routing: how far ahead a prediction must be before a load is even possible. It is not a proof that preload cannot work. Four GPUs holding the most likely experts might mean no CPU drop and no speculative preload. Splitting a model across cards and passing the work is the other path. Occasional CPU fallback still has to beat ordinary routing. That comparison is not measured yet.

## What the histograms already count

`b1/trace.tsv` is one OLMoE forward. 227 data rows, each with eight expert ids from `ffn_moe_topk`. 211 of those rows are prefill (1688 ids) and 16 are decode (128 ids). Layers run from 0 through 15. `ingest` adds one count per id into a layer-by-expert matrix, prefill and decode apart, and a next-layer joint when the same token is seen at layer L and then L+1. Those are counts of ids in this trace. They are not a hit rate.

`data/olmoe-expert-calibration.json` is the same blob as commit `ca70c45` on `cursor/burn-fp16-expert-engine-ef2f`, and that blob is already on this branch. It has 1024 rows: 16 layers by 64 experts. Each row is the router-row L2 of `ffn_gate_inp` plus the RMS of the FP16 gate, up, and down projections. It is a weight table. It is not an activation histogram and not a hit rate.

## Context versus expert weight

File sizes and header fields for the GGUFs the live servers have open are in `data/cpu-miss-context.md`. Context cost stays the formula `tokens * hidden * dtype_bytes`. How many tokens must move is not known. No token count and no millisecond is filled in there.

## CMoE and ExpertWeaver

CMoE partitions a dense FFN by activation frequency into shared neurons and routed neurons, then builds a router from those statistics. The useful piece for this engine is the frequency split. The histogram already stores it. The carving script is not ported.

ExpertWeaver clusters GLU activation patterns and sets a shared-versus-routed ratio from their coefficient of variation. Those activations are still not in the B1 trace, so that split is not in the forward.

The checkpoint also did not ship expert-weight calibration. `scripts/olmoe_expert_calibration.py` reads `olmoe-1b-7b-0924-f16.gguf` and writes `data/olmoe-expert-calibration.json`: 16 layers, 64 experts, 1024 rows. Each row is the router-row L2 of `ffn_gate_inp` plus the RMS of the FP16 gate, up, and down projections. Layer 0 expert 0 measured `router_l2` 1.4803899175344386. Across the file, router L2 runs from 0.46006687831362936 to 2.0849092499447957, and the three RMS fields stay near 0.015. `route` attaches those scales to the experts it names.

The learned MoE-Beyond transformer (the 97.5% figure in their paper, on DeepSeek-V2-Lite traces) is not loaded. No hit rate is claimed for it on OLMoE.

OpenVINO is the CPU path on the two Xeon Golds, device CPU. `scripts/olmoe_openvino_cpu.py` builds three separate f32 MatMul models and compiles them with `compile_model(model, "CPU")`. OpenVINO `2026.4.0-22959-99c81491cc3-releases/2026/4` reported `FULL_DEVICE_NAME` as the Xeon on each run. The local T440 is an Intel Xeon Gold 6226 (`avx512f` present, `avx512_vnni` present, `avx512_bf16` absent). Its measured `expert_ms` values, 20 repeats, are 2.985007800452877 at M=1 and 6.127933099924121 at M=8 (`data/openvino-cpu-t440.jsonl`). cesarops4 is an Intel Xeon Gold 6148 (`avx512f` present, `avx512_vnni` absent, `avx512_bf16` absent). Its measured `expert_ms` values are 3.562492200580891 at M=1 and 6.093012299970724 at M=8 (`data/openvino-cpu-cesarops4.jsonl`).
