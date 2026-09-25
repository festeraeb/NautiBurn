# Context versus expert weight

Read from the GGUF each live server had open. Header only: `scripts/gguf_header_shape.py` stops at the tensor-info table (`header_end` bytes into the file). Tensor payloads were not read.

Byte size of a tensor is `(product(dims) / block_size) * type_size`. Block sizes are the ggml static asserts (`ggml_half` is 2 bytes): Q4_K is 144 bytes per 256 values, Q5_K is 176 per 256, Q6_K is 210 per 256, Q8_0 is 34 per 32, F32 is 4 per 1. A routed expert's bytes for one tensor are that size divided by `expert_count` when `expert_count` is one of the dimensions.

Context cost, still symbolic:

```
context_bytes = tokens * hidden * dtype_bytes
```

`hidden` below is the header's `embedding_length`. `tokens` is how many tokens have to move. `dtype_bytes` is the width of the state that moves. Neither `tokens` nor `dtype_bytes` was measured. No millisecond is computed.

## T440 Tiel-Coder, pid 1658536, port 5230

Cmdline and `cesarops-tilecoder.service` both open `/mnt/raid0/models/Qwen3.6-35B-A3B-Q5_K_M.gguf`. `-ngl 99`, `--tensor-split 1,1`. File size 25913185920 bytes. Header ends at byte 10990692. Architecture `qwen35moe`. `block_count` 41. `embedding_length` 2048. `expert_count` 256. `expert_used_count` 8. `expert_feed_forward_length` 512. `expert_shared_feed_forward_length` 512.

Routed tensors are `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps`. Gate and up are the same shape on every layer, `[2048, 512, 256]`. Down is `[512, 2048, 256]`. The ggml type is not the same on every layer, so one number does not cover the file.

| Layers | Gate and up | Down | One routed expert, gate+up+down |
|---|---|---|---|
| 0–4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34–39 | Q5_K, 720896 bytes each | Q6_K, 860160 bytes | 2301952 bytes |
| 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33 | Q5_K, 720896 bytes each | Q5_K, 720896 bytes | 2162688 bytes |
| 40 | Q8_0, 1114112 bytes each | Q8_0, 1114112 bytes | 3342336 bytes |

The shared-expert tensors `ffn_gate_shexp`, `ffn_up_shexp`, and `ffn_down_shexp` are 2D Q8_0, 1114112 bytes each, same on all 41 layers. Three of them are 3342336 bytes. That sum is the three tensors, not a slice of the 256 routed experts. `blk.0.ffn_gate_inp_shexp.weight` is F32 `[2048]`, 8192 bytes, and is not in that sum.

`hidden` for this file is 2048. Context bytes stay `tokens * 2048 * dtype_bytes`.

## T440 embedder, pid 108158, port 5210

Cmdline opens `/codebase/models/embeddinggemma-300m.Q8_0.gguf` with `-ngl 0`. The unit is `cesarops-t440-embed-cpu.service`. File size 328577440 bytes. Header ends at byte 6530463. Architecture `gemma-embedding`. `block_count` 24. `embedding_length` 768. `feed_forward_length` 1152. No `expert_count` key. No tensor name contains `exps`, `shexp`, or `expert`, so there is no per-expert size. Layer 0 dense FFN weights, not experts: `ffn_down` `[1152, 768]` Q8_0 940032 bytes, `ffn_gate` and `ffn_up` `[768, 1152]` Q8_0 940032 bytes each.

`hidden` for this file is 768. Context bytes stay `tokens * 768 * dtype_bytes`.

## cesarops4 Mixtral, pid 4137576, port 5017

Cmdline opens `/home/cesarops/models/mixtral-8x7b-instruct-v0.1-Q4_K_M-worthdoing.gguf`. `-ngl 99`, `-sm layer`, `-ts 1,1,1`. File size 28448468000 bytes. Header ends at byte 740380. Architecture `llama`. Name in the header: `Mixtral 8x7B Instruct v0.1`. `block_count` 32. `embedding_length` 4096. `expert_count` 8. `expert_used_count` 2. `feed_forward_length` 14336. There is no `expert_feed_forward_length` key.

Gate and up, `ffn_gate_exps` and `ffn_up_exps`, are Q4_K `[4096, 14336, 8]` on all 32 layers: 33030144 bytes per expert per tensor. Down, `ffn_down_exps`, is `[14336, 4096, 8]` and the type changes by layer.

| Layers | Down | One routed expert, gate+up+down |
|---|---|---|
| 0–3, 6, 9, 12, 15, 18, 21, 24, 27–31 | Q6_K, 48168960 bytes | 114229248 bytes |
| 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26 | Q4_K, 33030144 bytes | 99090432 bytes |

No `shexp` tensors. `hidden` for this file is 4096. Context bytes stay `tokens * 4096 * dtype_bytes`.

## cesarops4 SmolLM, pid 3198908, port 5002

Cmdline and `cesarops4-smollm3.service` both open `/home/cesarops/models/HuggingFaceTB_SmolLM3-3B-Q8_0.gguf`. `-ngl 99`. File size 3275575104 bytes. Header ends at byte 7843620. Architecture `smollm3`. `block_count` 36. `embedding_length` 2048. `feed_forward_length` 11008. No `expert_count` key. No tensor name contains `exps`, `shexp`, or `expert`, so there is no per-expert size. Layer 0 dense FFN weights, not experts: `ffn_down` `[11008, 2048]` Q8_0 23953408 bytes, `ffn_gate` and `ffn_up` `[2048, 11008]` Q8_0 23953408 bytes each.

`hidden` for this file is 2048. Context bytes stay `tokens * 2048 * dtype_bytes`.
