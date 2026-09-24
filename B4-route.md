# Thom translator route

Date: 2026-09-24. Host: t440cesarops. This note is the translator route. It is not milestone B4.

## Turn

Forge already translates a Thom turn inside `ornith_intent_translate` (`repos/Forge/src/context_injector.rs`), which runs at the start of `executive_injector_preflight` before the thinker. The turn is `[THOM]` or any message that contains `translate` (`thom_nim::is_thom_translate_turn`). Other operator messages stay on the local draft endpoint.

## Route

That turn calls `nauti_nodes::adapter::nim::NimRuntime` (`/codebase/repos/nauti-nodes/src/adapter/nim.rs`). Forge does not grow a second NIM client. `NimRuntime::infer` posts `/v1/chat/completions`.

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

Source is `repos/Forge/src/thom_nim.rs` on branch `cursor/thom-nim-route-ef2f`, commit `3f4ec53` (`3f4ec53b2f2fb4c0ccb31203bf442ea52346a6de`), wired from `ornith_intent_translate`. `cargo test --bin cesarops-forge-v3` ran the four `thom_nim` tests: 4 passed. Those tests lock the try order DeepSeek, Lightning, Kimi, GLM.

The live unit is `cesarops-forge-v3.service`. The release binary was built from that commit (`CARGO_TARGET_DIR=/data/cargo-target/forge-thom-nim`, `cargo build --release --bin cesarops-forge-v3`, exit 0) and copied onto `/data/cargo-target/release/cesarops-forge-v3`, which is the unit `ExecStart`. `systemctl restart cesarops-forge-v3.service` at Thu 2026-09-24 10:33:19 UTC. Running pid **4040128**. `ActiveState=active`. `http://127.0.0.1:9100/health` returned status ok. The running executable contains `thom_nim_translator` and the four model ids in increasing offset order: DeepSeek, Lightning, Kimi, GLM. A `[THOM]` turn therefore calls `ornith_intent_translate` → `thom_nim::translate_operator_message` → `NimRuntime::infer` in that order. `NIM_BASE_URL` and `NIM_API_KEY` are set in that process. Values are not written here.

Drop-in `/etc/systemd/system/cesarops-forge-v3.service.d/thom-nim.conf` loads `EnvironmentFile=-/home/cesarops/.config/forge/thom-nim.env` (mode 600, outside this repo). That file is not committed.

GPU0 stayed up through the restart (llama-server pids 108158 and 3341393). GPU1 was not used.
