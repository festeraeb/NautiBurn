// Env-gated MoE expert scrape for llama-server (adapter only).
// Same tensor gate as moe_histodb_scrape.cpp: ffn_moe_topk-* / ffn_moe_argsort-*.
// Enable with NAUTIBURN_SCRAPE_DIR (and preferably NAUTIBURN_MODEL_ID).
#pragma once

#include "common.h"

#include "ggml.h"

#include <cstddef>

struct ggml_tensor;

// If NAUTIBURN_SCRAPE_DIR is set, installs cb_eval on params and returns true.
// Safe to call once before common_init_from_params. No-op when env unset.
bool nautiburn_moe_server_scrape_maybe_enable(common_params & params);

// Flush any buffered activations to scrape JSON under NAUTIBURN_SCRAPE_DIR.
// Call at end of a generation (e.g. send_final_response) so the last decode
// lands even without a following layer-0 wrap.
void nautiburn_moe_server_scrape_flush(void);

// ggml_backend_sched_eval_callback — also installed via maybe_enable.
bool nautiburn_moe_server_scrape_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);
