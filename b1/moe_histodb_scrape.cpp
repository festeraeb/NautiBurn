// Side-channel MoE expert scrape → histodb scrape JSON v1.
// Same gate as b1/olmoe_expert_scrape.cpp: copy ffn_moe_topk-* (or first-k of
// ffn_moe_argsort-*) via cb_eval. Adapter only — not a Burn engine.
//
// Required env:
//   NAUTIBURN_MODEL_ID  histodb model_id
//   NAUTIBURN_OUT       prefill JSON path; decode written beside as *-decode.json
// Optional:
//   NAUTIBURN_CLASS_ID  class_id (default: llama-moe-sidechannel)
//   NAUTIBURN_DTYPE     requests.dtype (default: gguf)
//   NAUTIBURN_DECODE_N  greedy decode tokens after prefill (default: 1)

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
#include <ctime>
#include <string>
#include <vector>

struct Row {
    int layer = 0;
    int token_index = 0;
    llama_token token_id = -1;
    std::vector<int32_t> experts;
};

struct Scrape {
    std::string phase;
    std::vector<llama_token> batch;
    std::vector<Row> topk;
    std::vector<Row> argsort;
    std::vector<uint8_t> scratch;
    int observed = 0;
};

static bool name_prefix(const char * name, const char * prefix) {
    return name != nullptr && std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

static int layer_from_name(const char * name, const char * prefix) {
    return std::atoi(name + std::strlen(prefix));
}

static int32_t read_i32(const uint8_t * data, const ggml_tensor * t, int64_t i0, int64_t i1) {
    const size_t off = static_cast<size_t>(i0) * t->nb[0] + static_cast<size_t>(i1) * t->nb[1];
    int32_t v = 0;
    std::memcpy(&v, data + off, sizeof(v));
    return v;
}

static bool cb_eval(ggml_tensor * t, bool ask, void * user) {
    auto * scrape = static_cast<Scrape *>(user);
    const bool topk = name_prefix(t->name, "ffn_moe_topk-");
    const bool argsort = name_prefix(t->name, "ffn_moe_argsort-");
    if (ask) {
        return topk || argsort;
    }
    if (!topk && !argsort) {
        return true;
    }
    scrape->observed++;
    if (t->type != GGML_TYPE_I32) {
        LOG_ERR("skip %s type %s\n", t->name, ggml_type_name(t->type));
        return true;
    }

    const bool host = ggml_backend_buffer_is_host(t->buffer);
    const size_t n_bytes = ggml_nbytes(t);
    const uint8_t * data = nullptr;
    if (host) {
        data = static_cast<const uint8_t *>(t->data);
    } else {
        scrape->scratch.resize(n_bytes);
        ggml_backend_tensor_get(t, scrape->scratch.data(), 0, n_bytes);
        data = scrape->scratch.data();
    }

    const int64_t n_exp = t->ne[0];
    const int64_t n_tok = t->ne[1];
    const int64_t k = argsort ? std::min<int64_t>(n_exp, 8) : n_exp;
    const char * prefix = topk ? "ffn_moe_topk-" : "ffn_moe_argsort-";

    for (int64_t tok = 0; tok < n_tok; ++tok) {
        Row row;
        row.layer = layer_from_name(t->name, prefix);
        row.token_index = static_cast<int>(tok);
        row.token_id = (tok >= 0 && static_cast<size_t>(tok) < scrape->batch.size())
            ? scrape->batch[static_cast<size_t>(tok)]
            : -1;
        row.experts.reserve(static_cast<size_t>(k));
        for (int64_t e = 0; e < k; ++e) {
            row.experts.push_back(read_i32(data, t, e, tok));
        }
        if (topk) {
            scrape->topk.push_back(std::move(row));
        } else {
            scrape->argsort.push_back(std::move(row));
        }
    }
    return true;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
    return out;
}

static bool write_scrape_json(
    const char * path,
    const char * model_id,
    const char * request_id,
    const char * class_id,
    const char * dtype,
    const char * phase,
    const std::vector<llama_token> & token_ids,
    const std::vector<Row> & rows) {
    FILE * out = std::fopen(path, "w");
    if (!out) {
        LOG_ERR("cannot write %s\n", path);
        return false;
    }
    std::fprintf(out,
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"model_id\": \"%s\",\n"
        "  \"request_id\": \"%s\",\n"
        "  \"class_id\": \"%s\",\n"
        "  \"source\": \"scrape\",\n"
        "  \"phase\": \"%s\",\n"
        "  \"n_tokens\": %zu,\n"
        "  \"dtype\": \"%s\",\n"
        "  \"token_ids\": [",
        json_escape(model_id).c_str(),
        json_escape(request_id).c_str(),
        json_escape(class_id).c_str(),
        phase,
        token_ids.size(),
        json_escape(dtype).c_str());
    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (i) {
            std::fputc(',', out);
        }
        std::fprintf(out, "%d", token_ids[i]);
    }
    std::fprintf(out, "],\n  \"activations\": [\n");
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row & row = rows[i];
        std::fprintf(out, "    {\"layer\": %d, \"token_pos\": %d, \"expert_ids\": [",
            row.layer, row.token_index);
        for (size_t e = 0; e < row.experts.size(); ++e) {
            if (e) {
                std::fputc(',', out);
            }
            std::fprintf(out, "%d", row.experts[e]);
        }
        std::fprintf(out, "]}%s\n", (i + 1 < rows.size()) ? "," : "");
    }
    std::fprintf(out, "  ]\n}\n");
    std::fclose(out);
    LOG_INF("scrape_json=%s rows=%zu phase=%s\n", path, rows.size(), phase);
    return !rows.empty();
}

static bool decode_batch(llama_context * ctx, Scrape & scrape, const std::vector<llama_token> & batch, const char * phase) {
    scrape.phase = phase;
    scrape.batch = batch;
    scrape.topk.clear();
    scrape.argsort.clear();
    if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(batch.data()), static_cast<int32_t>(batch.size())))) {
        LOG_ERR("llama_decode failed during %s\n", phase);
        return false;
    }
    return true;
}

static const std::vector<Row> & preferred_rows(const Scrape & scrape) {
    return scrape.topk.empty() ? scrape.argsort : scrape.topk;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * model_id = std::getenv("NAUTIBURN_MODEL_ID");
    const char * out_path = std::getenv("NAUTIBURN_OUT");
    const char * class_id = std::getenv("NAUTIBURN_CLASS_ID");
    const char * dtype = std::getenv("NAUTIBURN_DTYPE");
    const char * decode_n_env = std::getenv("NAUTIBURN_DECODE_N");
    if (!model_id || !out_path) {
        LOG_ERR("NAUTIBURN_MODEL_ID and NAUTIBURN_OUT are required\n");
        return 2;
    }
    if (!class_id) {
        class_id = "llama-moe-sidechannel";
    }
    if (!dtype) {
        dtype = "gguf";
    }
    int decode_n = 1;
    if (decode_n_env && *decode_n_env) {
        decode_n = std::atoi(decode_n_env);
        if (decode_n < 0) {
            decode_n = 0;
        }
    }

    Scrape scrape;
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &scrape;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
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

    const long stamp = static_cast<long>(std::time(nullptr));
    char req_prefill[128];
    char req_decode[128];
    std::snprintf(req_prefill, sizeof(req_prefill), "nb-llama-%ld-prefill", stamp);
    std::snprintf(req_decode, sizeof(req_decode), "nb-llama-%ld-decode", stamp);

    if (!decode_batch(ctx, scrape, tokens, "prefill")) {
        llama_backend_free();
        return 6;
    }
    const std::vector<Row> & prefill_rows = preferred_rows(scrape);
    if (!write_scrape_json(out_path, model_id, req_prefill, class_id, dtype, "prefill", tokens, prefill_rows)) {
        llama_backend_free();
        return 8;
    }

    if (decode_n > 0) {
        std::vector<llama_token> decode_tokens;
        decode_tokens.reserve(static_cast<size_t>(decode_n));
        std::vector<Row> decode_rows;
        for (int i = 0; i < decode_n; ++i) {
            const int32_t idx = (i == 0)
                ? static_cast<int32_t>(tokens.size()) - 1
                : -1;
            const float * logits = llama_get_logits_ith(ctx, idx);
            if (logits == nullptr) {
                LOG_ERR("no logits for decode step %d\n", i);
                break;
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            llama_token next = 0;
            float best = logits[0];
            for (int32_t t = 1; t < n_vocab; ++t) {
                if (logits[t] > best) {
                    best = logits[t];
                    next = t;
                }
            }
            std::vector<llama_token> one{next};
            if (!decode_batch(ctx, scrape, one, "decode")) {
                llama_backend_free();
                return 7;
            }
            decode_tokens.push_back(next);
            const std::vector<Row> & rows = preferred_rows(scrape);
            // Relabel token_pos to absolute decode index for histodb.
            for (const Row & src : rows) {
                Row copy = src;
                copy.token_index = static_cast<int>(decode_tokens.size()) - 1;
                copy.token_id = next;
                decode_rows.push_back(std::move(copy));
            }
        }
        if (!decode_rows.empty()) {
            std::string dec_path = out_path;
            const auto dot = dec_path.rfind('.');
            if (dot == std::string::npos) {
                dec_path += "-decode.json";
            } else {
                dec_path = dec_path.substr(0, dot) + "-decode.json";
            }
            if (!write_scrape_json(dec_path.c_str(), model_id, req_decode, class_id, dtype, "decode", decode_tokens, decode_rows)) {
                llama_backend_free();
                return 9;
            }
        }
    }

    llama_backend_free();
    return 0;
}
