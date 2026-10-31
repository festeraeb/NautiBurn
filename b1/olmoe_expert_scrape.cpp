// Read OLMoE expert ids from llama.cpp graph tensors.
// Rows are copied from ffn_moe_topk-* (or the first k of ffn_moe_argsort-*
// when the top-k view is not a compute node). Nothing is synthesized.

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

struct Row {
    std::string phase;
    int layer;
    int token_index;
    llama_token token_id;
    std::string tensor;
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
    LOG_INF("tensor %s ne0=%lld ne1=%lld type=%s\n", t->name, (long long) t->ne[0], (long long) t->ne[1], ggml_type_name(t->type));
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

    // ggml_argsort_top_k views the first ne[0] ids of a descending argsort.
    const int64_t n_exp = t->ne[0];
    const int64_t n_tok = t->ne[1];
    const int64_t k = argsort ? std::min<int64_t>(n_exp, 8) : n_exp;
    const char * prefix = topk ? "ffn_moe_topk-" : "ffn_moe_argsort-";

    for (int64_t tok = 0; tok < n_tok; ++tok) {
        Row row;
        row.phase = scrape->phase;
        row.layer = layer_from_name(t->name, prefix);
        row.token_index = static_cast<int>(tok);
        row.token_id = (tok >= 0 && static_cast<size_t>(tok) < scrape->batch.size())
            ? scrape->batch[static_cast<size_t>(tok)]
            : -1;
        row.tensor = topk ? "ffn_moe_topk" : "ffn_moe_argsort";
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

static void write_rows(FILE * out, const std::vector<Row> & rows) {
    for (const Row & row : rows) {
        std::fprintf(out, "%s\t%d\t%d\t%d\t%s\t",
            row.phase.c_str(), row.layer, row.token_index, row.token_id, row.tensor.c_str());
        for (size_t i = 0; i < row.experts.size(); ++i) {
            if (i) {
                std::fputc(',', out);
            }
            std::fprintf(out, "%d", row.experts[i]);
        }
        std::fputc('\n', out);
    }
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

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * trace_path = std::getenv("NAUTIBURN_TRACE");
    const char * gpu_uuid = std::getenv("NAUTIBURN_GPU_UUID");
    const char * weight = std::getenv("NAUTIBURN_WEIGHT");
    const char * dtype = std::getenv("NAUTIBURN_DTYPE");
    const char * quant = std::getenv("NAUTIBURN_QUANT");
    if (!trace_path || !gpu_uuid || !weight || !dtype || !quant) {
        LOG_ERR("NAUTIBURN_TRACE, NAUTIBURN_GPU_UUID, NAUTIBURN_WEIGHT, NAUTIBURN_DTYPE, NAUTIBURN_QUANT are required\n");
        return 2;
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

    FILE * out = std::fopen(trace_path, "w");
    if (!out) {
        LOG_ERR("cannot write %s\n", trace_path);
        llama_backend_free();
        return 5;
    }

    std::fprintf(out, "# nautiburn-b1-trace v1\n");
    std::fprintf(out, "dtype: %s\n", dtype);
    std::fprintf(out, "quant: %s\n", quant);
    std::fprintf(out, "gpu_uuid: %s\n", gpu_uuid);
    std::fprintf(out, "weight: %s\n", weight);
    std::fprintf(out, "source_checkpoint_dtype: BF16\n");
    std::fprintf(out, "cast: allenai OLMoE-1B-7B-0924 BF16 checkpoint published as F16 GGUF\n");
    std::fprintf(out, "model: allenai/OLMoE-1B-7B-0924\n");
    std::fprintf(out, "phase_field: prefill|decode\n");
    std::fprintf(out, "columns: phase layer token_index token_id tensor expert_ids\n");

    LOG_INF("prompt tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("token %zu %d\n", i, tokens[i]);
        std::fprintf(out, "llama_token: %zu %d\n", i, tokens[i]);
    }

    if (!decode_batch(ctx, scrape, tokens, "prefill")) {
        std::fclose(out);
        llama_backend_free();
        return 6;
    }
    const std::vector<Row> & prefill_rows = scrape.topk.empty() ? scrape.argsort : scrape.topk;
    write_rows(out, prefill_rows);
    const int prefill_observed = scrape.observed;
    LOG_INF("prefill rows topk=%zu argsort=%zu observed=%d\n", scrape.topk.size(), scrape.argsort.size(), scrape.observed);

    const float * logits = llama_get_logits_ith(ctx, static_cast<int32_t>(tokens.size()) - 1);
    llama_token next = 0;
    if (logits != nullptr) {
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        float best = logits[0];
        for (int32_t i = 1; i < n_vocab; ++i) {
            if (logits[i] > best) {
                best = logits[i];
                next = i;
            }
        }
        std::vector<llama_token> one{next};
        if (!decode_batch(ctx, scrape, one, "decode")) {
            std::fclose(out);
            llama_backend_free();
            return 7;
        }
        const std::vector<Row> & decode_rows = scrape.topk.empty() ? scrape.argsort : scrape.topk;
        write_rows(out, decode_rows);
        LOG_INF("decode token %d rows topk=%zu argsort=%zu\n", next, scrape.topk.size(), scrape.argsort.size());
    } else {
        LOG_ERR("no logits after prefill\n");
    }

    std::fprintf(out, "prefill_observed_nodes: %d\n", prefill_observed);
    std::fclose(out);
    llama_backend_free();
    return prefill_rows.empty() ? 8 : 0;
}
