# Thom translator route

Date: 2026-09-24. Host: t440cesarops. This note is the translator route. It is not milestone B4.

## Turn

Forge already translates a Thom turn inside `ornith_intent_translate` (`repos/Forge/src/context_injector.rs`), which runs at the start of `executive_injector_preflight` before the thinker. The turn is `[THOM]` or any message that contains `translate` (`thom_nim::is_thom_translate_turn`). Other operator messages stay on the local draft endpoint.

## Route

The live hop is the llama-server already running on this host. It is pid 3341393, `http://127.0.0.1:5240`, model id `QwQ-32B-IQ2_XXS`, file `/mnt/raid0/models/huihui-ai_QwQ-32B-abliterated-IQ2_XXS.gguf`, on T440 GPU0. `GET /health` returned ok and `GET /v1/models` returned that id. A 48-token chat call returned reasoning text (229 chars) and empty content with finish reason `length`, so the server generates. `inference_client::complete_prompt` is the client, the same one the draft translator already uses, and it merges reasoning into the completion when content is empty.

`THOM_LOCAL_URL` overrides that base. An empty value turns the local seat off. The default is `http://127.0.0.1:5240`. The model id is whatever that server reports. No new model was started.

The named cluster role `ornith15-translator` (`Ornith-1.5-35B-A3B-Q4_K_M` at `http://192.168.4.34:5200`) did not answer. `http://127.0.0.1:5203` (the configured draft endpoint) did not answer. The n8n webhook `http://127.0.0.1:5678/webhook/llm-translate` returned 404. That webhook is an output normalizer, not a generative translator.

If the local hop errors or returns empty text, the turn falls through to `nauti_nodes::adapter::nim::NimRuntime` (`/codebase/repos/nauti-nodes/src/adapter/nim.rs`). Forge does not grow a second NIM client. `NimRuntime::infer` posts `/v1/chat/completions`. NIM is the fallback.

Try order, first non-empty completion wins:

| Family | Model id | Override |
|---|---|---|
| DeepSeek | `deepseek-ai/deepseek-v4-pro` | `NIM_MODEL_DEEPSEEK` |
| Lightning | `nvidia/nemotron-3.5-lightning-30b-a3b` | `NIM_MODEL_NVIDIA_LIGHTNING` |
| Kimi | `moonshotai/kimi-k3` | `NIM_MODEL_KIMI` |
| GLM | `z-ai/glm-5.2` | `NIM_MODEL_GLM`, else `NIM_THINKER_B_MODEL` |

The prompt is the existing Ornith / Tom translator prompt. The reply is still wrapped as `[ORNITH INTENT TRANSLATION — handoff to thinker]` plus the original message. If every model fails, the local draft translator remains.

## Environment

`NIM_BASE_URL` and `NIM_API_KEY` are both set in this process environment. The key is not written here. The base host is `integrate.api.nvidia.com`. `NimRuntime` appends `/v1/...`, so a base that already ends in `/v1` is stripped to one `/v1` segment before it is stored on `NimConfig`.

If either variable is empty at call time, `translate_operator_message` returns `missing NIM_BASE_URL` or `missing NIM_API_KEY` and does not invent a key.

## Not the MoE token path

`thom_nim` is not called from expert dispatch, PAMP prefetch, the decode loop, or `nim://cloud-structured`. A local expert miss does not call this endpoint.

## Process

Source is `repos/Forge/src/thom_nim.rs` on branch `cursor/thom-nim-route-ef2f`, commit `96f3dc9`, wired from `ornith_intent_translate`. The local seat is tried first. `cargo test --release --bin cesarops-forge-v3 -- thom_nim` ran 5 tests: 5 passed. Those tests lock the T440 default `http://127.0.0.1:5240` and the NIM try order DeepSeek, Lightning, Kimi, GLM. Forge has no origin remote, so this commit stays local.

The live unit is `cesarops-forge-v3.service`. The release binary was rebuilt from this tree (`CARGO_TARGET_DIR=/data/cargo-target/forge-thom-nim`, `cargo test --release --bin cesarops-forge-v3 -- thom_nim` 5 passed, then `cargo build --release --bin cesarops-forge-v3`, exit 0) and installed onto `/data/cargo-target/release/cesarops-forge-v3` after a stop, because the running file was busy. `systemctl start` at Thu 2026-09-24 20:47:35 UTC. Running pid **1300339**. `ActiveState=active`. `http://127.0.0.1:9100/health` returned status ok. The running executable contains `thom_local_translator` and `http://127.0.0.1:5240`, and the four NIM model ids in increasing offset order DeepSeek, Lightning, Kimi, GLM. A `[THOM]` turn calls the local llama-server first. NIM remains the fallback, in that order. `NIM_BASE_URL` and `NIM_API_KEY` are set in that process. Values are not written here.

GPU0 stayed up (llama-server pids 108158 and 3341393). QwQ on :5240 still returns health ok. GPU1 was not used.

Drop-in `/etc/systemd/system/cesarops-forge-v3.service.d/thom-nim.conf` loads `EnvironmentFile=-/home/cesarops/.config/forge/thom-nim.env` (mode 600, outside this repo). That file is not committed.

GPU0 stayed up through the restart (llama-server pids 108158 and 3341393). GPU1 was not used.
