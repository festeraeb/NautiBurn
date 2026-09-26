// Emit the B5 numeric oracle that nautiburn-layer0 `compare` consumes.
//
// Sibling of olmoe_expert_scrape.cpp. That tool scrapes *routing* (top-k expert
// ids) into a TSV; this one scrapes *activations* into the raw little-endian
// blobs `compare` reads. Both run the same model on the same prompt; neither
// synthesizes anything.
//
// Files written into NAUTIBURN_ORACLE_DIR (raw, no header, host order):
//
//   hidden.f32          128 x 2048      f32  layer-0 residual stream, pre-MoE
//   ffn_gate_inp.f32     64 x 2048      f32  router weight
//   oracle_out.f32      128 x 2048      f32  layer-0 MoE output (llama.cpp)
//   oracle_ids.i32      128 x 8         i32  layer-0 top-8 expert ids
//
// Expert weight blobs (ffn_{gate,up,down}_exps.f32) are NOT written here; they
// are static graph inputs already produced by the Rust `dump` subcommand. This
// file closes the activation half of the fixture only.
//
// Everything is captured off real graph tensors via cb_eval. If a tensor is
// absent the tool fails closed with a nonzero exit rather than writing a
// partial oracle.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int N_TOKENS = 128;   // must match N_TOKENS in layer0/src/main.rs
static const int N_EMBD   = 2048;  // must match N_EMBD
static const int N_EXPERT = 64;    // must match N_EXPERT
static const int N_USED   = 8;     // must match N_USED
static const int LAYER    = 0;     // layer 0 only

// One captured tensor, converted to f32 (or i32) on the host.
struct Blob {
    bool  seen = false;
    std::vector<float>   f;
    std::vector<int32_t> i;
};

struct Oracle {
    std::string phase;
    std::vector<llama_token> batch;
    Blob hidden;      // ffn_norm output / MoE input
    Blob gate_inp;    // router weight
    Blob out;         // ffn_moe_out
    Blob topk_direct; // dedicated ffn_moe_topk tensor, authoritative
    Blob topk_argsort; // first-k-of-argsort, diagnostic fallback
    Blob logits;      // ffn_moe_logits (f32, n_expert x n_tok)

    // Tokens accumulated so far across ubatch callback invocations. The graph
    // callback fires once per ubatch, so this counts up to N_TOKENS.
    int64_t tokens_done = 0;
    int observed = 0;
};

static bool has_prefix(const char * name, const char * p) {
    return name != nullptr && std::strncmp(name, p, std::strlen(p)) == 0;
}

static int layer_of(const char * name, const char * p) {
    return std::atoi(name + std::strlen(p));
}

// Append one call's [rows, cols] slice into `dst` at token offset `token_base`.
//
// The slice may be float or int; exactly one of the two vectors is non-empty,
// which is how the element type is decided. Note this is NOT `!f.empty()`
// acting as the int flag: an i32 slice leaves `f` empty and fills `i`.
//
// The callback can fire once per ubatch, so every chunk is written at its own
// token offset, making the final buffer contiguous and token-major.
static void accumulate_rows(Blob * dst, const std::vector<float> & f,
                            const std::vector<int32_t> & i,
                            int64_t cols, int64_t token_base) {
    if (!i.empty()) {
        if (dst->i.empty()) {
            dst->i.assign((size_t) (N_TOKENS * cols), 0);
        }
        for (size_t r = 0; r * (size_t) cols + cols <= i.size(); ++r) {
            const size_t at = (size_t) (token_base + r) * (size_t) cols;
            if (at + (size_t) cols > dst->i.size()) {
                break;
            }
            for (int64_t c = 0; c < cols; ++c) {
                dst->i[at + (size_t) c] = i[r * (size_t) cols + (size_t) c];
            }
        }
    } else {
        if (dst->f.empty()) {
            dst->f.assign((size_t) (N_TOKENS * cols), 0.0f);
        }
        for (size_t r = 0; r * (size_t) cols + cols <= f.size(); ++r) {
            const size_t at = (size_t) (token_base + r) * (size_t) cols;
            if (at + (size_t) cols > dst->f.size()) {
                break;
            }
            for (int64_t c = 0; c < cols; ++c) {
                dst->f[at + (size_t) c] = f[r * (size_t) cols + (size_t) c];
            }
        }
    }
    dst->seen = true;
}

// Pull a tensor to host memory, converted to f32/i32. Returns false on type
// or shape mismatch rather than guessing.
static bool grab(ggml_tensor * t, Blob * dst, bool as_int, int64_t want_elems) {
    const ggml_type gt = t->type;
    if (!as_int && gt != GGML_TYPE_F32 && gt != GGML_TYPE_F16 && gt != GGML_TYPE_BF16) {
        LOG_ERR("skip %s: type %s is not f32/f16/bf16\n", t->name, ggml_type_name(gt));
        return false;
    }
    if (as_int && gt != GGML_TYPE_I32) {
        LOG_ERR("skip %s: type %s is not i32\n", t->name, ggml_type_name(gt));
        return false;
    }

    const size_t n_bytes = ggml_nbytes(t);
    std::vector<uint8_t> buf;
    const uint8_t * data = nullptr;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = static_cast<const uint8_t *>(t->data);
    } else {
        buf.resize(n_bytes);
        ggml_backend_tensor_get(t, buf.data(), 0, n_bytes);
        data = buf.data();
    }

    if (as_int) {
        if (want_elems > 0 && (int64_t) ggml_nelements(t) != want_elems) {
            LOG_ERR("skip %s: %lld elems != %lld\n", t->name,
                    (long long) ggml_nelements(t), (long long) want_elems);
            return false;
        }
        // Walk with nb[] rather than assuming contiguity. ffn_moe_topk is a
        // strided view into the argsort buffer (ne=[8,128] but nb[1]=256, i.e.
        // a 64-element row pitch), so a flat read silently interleaves the
        // wrong experts. The f32 path below already respects strides.
        const int64_t n0 = t->ne[0];
        dst->i.resize((size_t) ggml_nelements(t));
        for (int64_t c1 = 0; c1 < t->ne[1]; ++c1) {
            for (int64_t c2 = 0; c2 < t->ne[2]; ++c2) {
                for (int64_t c3 = 0; c3 < t->ne[3]; ++c3) {
                    const uint8_t * src = data
                        + (size_t) c1 * t->nb[1]
                        + (size_t) c2 * t->nb[2]
                        + (size_t) c3 * t->nb[3];
                    const size_t idx =
                        (size_t) ((c3 * t->ne[2] + c2) * t->ne[1] + c1) * (size_t) n0;
                    for (int64_t e = 0; e < n0; ++e) {
                        int32_t v = 0;
                        std::memcpy(&v, src + (size_t) e * t->nb[0], sizeof(v));
                        dst->i[idx + (size_t) e] = v;
                    }
                }
            }
        }
    } else {
        if (want_elems > 0 && (int64_t) ggml_nelements(t) != want_elems) {
            LOG_ERR("skip %s: %lld elems != %lld\n", t->name,
                    (long long) ggml_nelements(t), (long long) want_elems);
            return false;
        }
        const int64_t n0 = t->ne[0];
        dst->f.resize((size_t) ggml_nelements(t));

        if (gt == GGML_TYPE_F32) {
            // F32 is the target type, so ggml supplies no to_float for it.
            // Copy each row by stride instead of assuming contiguity.
            for (int64_t c1 = 0; c1 < t->ne[1]; ++c1) {
                for (int64_t c2 = 0; c2 < t->ne[2]; ++c2) {
                    for (int64_t c3 = 0; c3 < t->ne[3]; ++c3) {
                        const uint8_t * src = data
                            + (size_t) c1 * t->nb[1]
                            + (size_t) c2 * t->nb[2]
                            + (size_t) c3 * t->nb[3];
                        float * dstp = dst->f.data()
                            + (size_t) ((c3 * t->ne[2] + c2) * t->ne[1] + c1) * (size_t) n0;
                        std::memcpy(dstp, src, (size_t) n0 * sizeof(float));
                    }
                }
            }
        } else {
            const ggml_type_traits * tr = ggml_get_type_traits(gt);
            if (tr == nullptr || tr->to_float == nullptr) {
                LOG_ERR("skip %s: no to_float for %s\n", t->name, ggml_type_name(gt));
                return false;
            }
            for (int64_t c1 = 0; c1 < t->ne[1]; ++c1) {
                for (int64_t c2 = 0; c2 < t->ne[2]; ++c2) {
                    for (int64_t c3 = 0; c3 < t->ne[3]; ++c3) {
                        const uint8_t * src = data
                            + (size_t) c1 * t->nb[1]
                            + (size_t) c2 * t->nb[2]
                            + (size_t) c3 * t->nb[3];
                        const size_t idx =
                            (size_t) ((c3 * t->ne[2] + c2) * t->ne[1] + c1) * (size_t) n0;
                        tr->to_float(src, dst->f.data() + idx, n0);
                    }
                }
            }
        }
    }
    dst->seen = true;
    return true;
}

// cb_eval: opt into every tensor so the graph is built exactly as the normal
// run builds it, then capture only the layer-0 nodes we care about.
static bool cb_eval(ggml_tensor * t, bool ask, void * user) {
    auto * o = static_cast<Oracle *>(user);
    if (ask) {
        return true;
    }

    const char * n = t->name;
    const int64_t n_tok = (int64_t) N_TOKENS;

    // Diagnostic: NAUTIBURN_DEBUG_TENSORS=1 lists every tensor the callback
    // sees, so tensor names/shapes can be confirmed from the running graph
    // rather than guessed from the model file.
    if (getenv("NAUTIBURN_DEBUG_TENSORS") != nullptr) {
        LOG_INF("dbg tensor %-28s type=%-4s ne=[%lld,%lld] nb=[%zu,%zu]\n", n,
                ggml_type_name(t->type),
                (long long) t->ne[0], (long long) t->ne[1],
                (size_t) t->nb[0], (size_t) t->nb[1]);
    }

    // Hidden stream: ffn_norm-<L> output feeds the MoE block.
    if (has_prefix(n, "ffn_norm-") && layer_of(n, "ffn_norm-") == LAYER &&
        t->ne[1] == n_tok && !o->hidden.seen) {
        o->observed++;
        if (grab(t, &o->hidden, false, n_tok * N_EMBD)) {
            LOG_INF("captured hidden from %s\n", n);
        }
        return true;
    }

    // Router weight is a static tensor (not per-layer).
    if (has_prefix(n, "ffn_gate_inp") && !o->gate_inp.seen && t->ne[0] == N_EXPERT) {
        o->observed++;
        if (grab(t, &o->gate_inp, false, (int64_t) N_EXPERT * N_EMBD)) {
            LOG_INF("captured gate_inp from %s\n", n);
        }
        return true;
    }

    // Router logits: ne[0] = n_expert, ne[1] = n_tok. This is the router's
    // real output, so `compare` can check the routing math directly instead of
    // inferring it from the weight matrix.
    if (has_prefix(n, "ffn_moe_logits-") && layer_of(n, "ffn_moe_logits-") == LAYER &&
        t->ne[0] == N_EXPERT && t->ne[1] == n_tok && !o->logits.seen) {
        o->observed++;
        if (grab(t, &o->logits, false, (int64_t) N_EXPERT * n_tok)) {
            LOG_INF("captured router logits from %s\n", n);
        }
        return true;
    }

    // MoE block output.
    if (has_prefix(n, "ffn_moe_out-") && layer_of(n, "ffn_moe_out-") == LAYER &&
        t->ne[1] == n_tok && !o->out.seen) {
        o->observed++;
        if (grab(t, &o->out, false, n_tok * N_EMBD)) {
            LOG_INF("captured moe_out from %s\n", n);
        }
        return true;
    }

    // Top-k ids.
    //
    // The dedicated ffn_moe_topk tensor is authoritative, but it is evaluated
    // AFTER ffn_moe_argsort in graph order, so a single shared buffer lets
    // argsort win the race. Capture each into its own blob.
    //
    // Both are captured per ubatch and accumulated at the token offset: the
    // callback fires once per ubatch, so a 128-token prompt with ubatch 8
    // yields 16 invocations of 8 tokens each.
    if (has_prefix(n, "ffn_moe_topk-") && layer_of(n, "ffn_moe_topk-") == LAYER &&
        t->ne[0] == N_USED && t->ne[1] <= n_tok) {
        o->observed++;
        Blob chunk;
        if (grab(t, &chunk, true, 0)) {
            const int64_t base = o->tokens_done;
            accumulate_rows(&o->topk_direct, chunk.f, chunk.i, N_USED, base);
            o->tokens_done += t->ne[1];
            LOG_INF("captured topk chunk from %s tokens=%lld base=%lld\n", n,
                    (long long) t->ne[1], (long long) base);
        }
        return true;
    }
    if (has_prefix(n, "ffn_moe_argsort-") && layer_of(n, "ffn_moe_argsort-") == LAYER &&
        t->ne[0] == N_EXPERT && t->ne[1] <= n_tok) {
        o->observed++;
        Blob chunk;
        if (grab(t, &chunk, true, 0)) {
            const int64_t base = o->tokens_done;
            // Keep only the first k of each row: that is the top-k set.
            chunk.i.resize((size_t) (t->ne[1] * N_USED));
            accumulate_rows(&o->topk_argsort, chunk.f, chunk.i, N_USED, base);
            LOG_INF("captured topk candidate from %s argsort tokens=%lld base=%lld\n",
                    n, (long long) t->ne[1], (long long) base);
        }
        return true;
    }

    return true;
}

static bool write_f32(const std::string & path, const std::vector<float> & v) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) { LOG_ERR("cannot write %s\n", path.c_str()); return false; }
    // f32 is already host-order little-endian on x86; write raw.
    const size_t n = std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    if (n != v.size()) { LOG_ERR("short write %s\n", path.c_str()); return false; }
    return true;
}

static bool write_i32(const std::string & path, const std::vector<int32_t> & v) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) { LOG_ERR("cannot write %s\n", path.c_str()); return false; }
    const size_t n = std::fwrite(v.data(), sizeof(int32_t), v.size(), f);
    std::fclose(f);
    if (n != v.size()) { LOG_ERR("short write %s\n", path.c_str()); return false; }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * dir      = std::getenv("NAUTIBURN_ORACLE_DIR");
    const char * gpu_uuid = std::getenv("NAUTIBURN_GPU_UUID");
    const char * weight   = std::getenv("NAUTIBURN_WEIGHT");
    const char * dtype    = std::getenv("NAUTIBURN_DTYPE");
    const char * quant    = std::getenv("NAUTIBURN_QUANT");
    if (!dir || !gpu_uuid || !weight || !dtype || !quant) {
        LOG_ERR("NAUTIBURN_ORACLE_DIR, NAUTIBURN_GPU_UUID, NAUTIBURN_WEIGHT, "
                "NAUTIBURN_DTYPE, NAUTIBURN_QUANT are required\n");
        return 2;
    }

    Oracle oracle;
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &oracle;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init model\n");
        llama_backend_free();
        return 3;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("empty prompt\n");
        llama_backend_free();
        return 4;
    }
    if ((int) tokens.size() != N_TOKENS) {
        // Fail closed: a short or long prompt would silently misalign every
        // row of the 128-token oracle.
        LOG_ERR("prompt tokenized to %zu, need exactly %d\n", tokens.size(), N_TOKENS);
        llama_backend_free();
        return 4;
    }

    oracle.phase = "prefill";
    oracle.batch = tokens;
    if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(tokens.data()),
                                            (int32_t) tokens.size()))) {
        LOG_ERR("llama_decode failed during prefill\n");
        llama_backend_free();
        return 6;
    }

    LOG_INF("observed=%d hidden=%d gate_inp=%d out=%d topk=%d tokens=%lld\n",
            oracle.observed, (int) oracle.hidden.seen, (int) oracle.gate_inp.seen,
            (int) oracle.out.seen, (int) oracle.topk_direct.seen,
            (long long) oracle.tokens_done);

    // Fail closed: never leave a half-written oracle that `compare` would
    // silently read as valid.
    //
    // ffn_gate_inp is deliberately NOT required. The router weight is a static
    // leaf tensor, so it never appears in the graph callback at all (confirmed
    // via NAUTIBURN_DEBUG_TENSORS=1: only computed nodes are seen). The
    // converter already writes ffn_gate_inp.f32 from router.f32, so the dumper
    // must not clobber or require it. ffn_moe_logits below is the router's
    // actual output and is a stronger check of the routing math anyway.
    struct { const char * name; const Blob * b; size_t want; } req[] = {
        {"hidden",  &oracle.hidden, (size_t) N_TOKENS * N_EMBD},
        {"moe_out", &oracle.out,    (size_t) N_TOKENS * N_EMBD},
    };
    for (auto & r : req) {
        if (!r.b->seen) {
            LOG_ERR("activation %s never observed - not writing a partial oracle\n", r.name);
            llama_backend_free();
            return 7;
        }
        if (r.b->f.size() != r.want) {
            LOG_ERR("activation %s has %zu elems, need %zu\n",
                    r.name, r.b->f.size(), r.want);
            llama_backend_free();
            return 7;
        }
    }
    if (!oracle.topk_direct.seen) {
        LOG_ERR("top-k ids never observed - not writing a partial oracle\n");
        llama_backend_free();
        return 7;
    }
    // Every token must have been captured; a partial accumulation would leave
    // zero-filled ids that `compare` would read as expert 0.
    if (oracle.tokens_done != (int64_t) N_TOKENS) {
        LOG_ERR("captured %lld of %d tokens - not writing a partial oracle\n",
                (long long) oracle.tokens_done, (int) N_TOKENS);
        llama_backend_free();
        return 7;
    }

    const std::string d(dir);
    bool ok = true;
    ok = ok && write_f32(d + "/hidden.f32",       oracle.hidden.f);
    ok = ok && write_f32(d + "/oracle_out.f32",    oracle.out.f);
    ok = ok && write_i32(d + "/oracle_ids.i32",    oracle.topk_direct.i);
    if (oracle.logits.seen) {
        ok = ok && write_f32(d + "/oracle_logits.f32", oracle.logits.f);
    }
    // ffn_gate_inp.f32 is owned by the converter (it comes from router.f32).
    // The dumper must not rewrite it, and must not claim to have produced it.

    if (!ok) {
        LOG_ERR("failed to write oracle files\n");
        llama_backend_free();
        return 8;
    }

    LOG_INF("wrote oracle to %s (hidden=%zu gate_inp=%zu out=%zu topk=%zu)\n",
            dir, oracle.hidden.f.size(), oracle.gate_inp.f.size(),
            oracle.out.f.size(), oracle.topk_direct.i.size());
    LOG_INF("NOTE: expert weight blobs (ffn_{gate,up,down}_exps.f32) are NOT written\n"
            "      here. They are static inputs already produced by the Rust `dump`\n"
            "      subcommand; convert those f16 files to f32 before `compare`.\n");

    llama_backend_free();
    return 0;
}

