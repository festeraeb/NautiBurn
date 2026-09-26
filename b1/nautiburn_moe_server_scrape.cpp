// Env-gated MoE expert scrape for llama-server → histodb scrape JSON v1.
// Adapter only: does not become a Burn engine.
// Same tensor gate as moe_histodb_scrape.cpp: ffn_moe_topk-* / ffn_moe_argsort-*.

#include "nautiburn_moe_server_scrape.h"

#include "ggml-backend.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct Row {
    int layer = 0;
    int token_pos = 0;
    std::vector<int32_t> expert_ids;
};

struct ScrapeState {
    std::mutex mu;
    std::vector<Row> topk;
    std::vector<Row> argsort;
    std::vector<uint8_t> scratch;
    int last_layer = -1;
    int max_layer = -1;
    std::atomic<uint64_t> seq{0};
    std::string scrape_dir;
    std::string model_id;
    std::string class_id;
    std::string dtype;
    bool enabled = false;
};

static ScrapeState g_state;

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

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"':  out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:   out += c; break;
        }
    }
    return out;
}

static const std::vector<Row> & preferred_rows_locked() {
    return g_state.topk.empty() ? g_state.argsort : g_state.topk;
}

static bool buffer_empty_locked() {
    return g_state.topk.empty() && g_state.argsort.empty();
}

static int max_token_pos_locked() {
    int m = 0;
    for (const Row & r : preferred_rows_locked()) {
        if (r.token_pos > m) {
            m = r.token_pos;
        }
    }
    return m;
}

static bool write_scrape_locked(const std::vector<Row> & rows, const char * phase) {
    if (rows.empty() || g_state.scrape_dir.empty()) {
        return false;
    }
    const uint64_t n = ++g_state.seq;
    const long stamp = static_cast<long>(std::time(nullptr));
    char path[1024];
    char req_id[128];
    std::snprintf(path, sizeof(path), "%s/run-%ld-%llu.json",
        g_state.scrape_dir.c_str(), stamp, static_cast<unsigned long long>(n));
    std::snprintf(req_id, sizeof(req_id), "nb-server-%ld-%llu",
        stamp, static_cast<unsigned long long>(n));

    FILE * out = std::fopen(path, "w");
    if (!out) {
        LOG_ERR("nautiburn scrape: cannot write %s\n", path);
        return false;
    }

    const int nt = [&]() {
        int m = 0;
        for (const Row & r : rows) {
            if (r.token_pos + 1 > m) {
                m = r.token_pos + 1;
            }
        }
        return m;
    }();

    std::fprintf(out,
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"model_id\": \"%s\",\n"
        "  \"request_id\": \"%s\",\n"
        "  \"class_id\": \"%s\",\n"
        "  \"source\": \"scrape\",\n"
        "  \"phase\": \"%s\",\n"
        "  \"n_tokens\": %d,\n"
        "  \"dtype\": \"%s\",\n"
        "  \"token_ids\": [],\n"
        "  \"activations\": [\n",
        json_escape(g_state.model_id).c_str(),
        json_escape(req_id).c_str(),
        json_escape(g_state.class_id).c_str(),
        phase,
        nt,
        json_escape(g_state.dtype).c_str());

    for (size_t i = 0; i < rows.size(); ++i) {
        const Row & row = rows[i];
        std::fprintf(out, "    {\"layer\": %d, \"token_pos\": %d, \"expert_ids\": [",
            row.layer, row.token_pos);
        for (size_t e = 0; e < row.expert_ids.size(); ++e) {
            if (e) {
                std::fputc(',', out);
            }
            std::fprintf(out, "%d", row.expert_ids[e]);
        }
        std::fprintf(out, "]}%s\n", (i + 1 < rows.size()) ? "," : "");
    }
    std::fprintf(out, "  ]\n}\n");
    std::fclose(out);
    LOG_INF("nautiburn scrape: wrote %s rows=%zu phase=%s\n", path, rows.size(), phase);
    return true;
}

static void flush_locked(const char * phase) {
    const std::vector<Row> & rows = preferred_rows_locked();
    if (rows.empty()) {
        g_state.topk.clear();
        g_state.argsort.clear();
        g_state.last_layer = -1;
        g_state.max_layer = -1;
        return;
    }
    // Copy before clear — preferred_rows_locked references the buffers.
    std::vector<Row> copy = rows;
    write_scrape_locked(copy, phase);
    g_state.topk.clear();
    g_state.argsort.clear();
    g_state.last_layer = -1;
    g_state.max_layer = -1;
}

} // namespace

bool nautiburn_moe_server_scrape_maybe_enable(common_params & params) {
    const char * dir = std::getenv("NAUTIBURN_SCRAPE_DIR");
    if (!dir || !*dir) {
        return false;
    }

    g_state.scrape_dir = dir;
    const char * mid = std::getenv("NAUTIBURN_MODEL_ID");
    g_state.model_id = (mid && *mid) ? mid : "unknown-moe";
    const char * cid = std::getenv("NAUTIBURN_CLASS_ID");
    g_state.class_id = (cid && *cid) ? cid : "llama-server-live";
    const char * dt = std::getenv("NAUTIBURN_DTYPE");
    g_state.dtype = (dt && *dt) ? dt : "gguf";
    g_state.enabled = true;

    params.cb_eval = nautiburn_moe_server_scrape_cb_eval;
    params.cb_eval_user_data = &g_state;

    LOG_INF("nautiburn scrape: enabled dir=%s model_id=%s class_id=%s\n",
        g_state.scrape_dir.c_str(), g_state.model_id.c_str(), g_state.class_id.c_str());
    return true;
}

void nautiburn_moe_server_scrape_flush(void) {
    if (!g_state.enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_state.mu);
    const char * phase = (max_token_pos_locked() > 0) ? "prefill" : "decode";
    flush_locked(phase);
}

bool nautiburn_moe_server_scrape_cb_eval(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (!g_state.enabled) {
        return false;
    }
    const bool topk = name_prefix(t->name, "ffn_moe_topk-");
    const bool argsort = name_prefix(t->name, "ffn_moe_argsort-");
    if (ask) {
        return topk || argsort;
    }
    if (!topk && !argsort) {
        return true;
    }
    if (t->type != GGML_TYPE_I32) {
        return true;
    }

    std::lock_guard<std::mutex> lock(g_state.mu);

    const int layer = layer_from_name(t->name, topk ? "ffn_moe_topk-" : "ffn_moe_argsort-");

    // New decode graph: layer wraps back toward 0 after a prior high layer.
    if (!buffer_empty_locked() && g_state.max_layer >= 0 && layer < g_state.last_layer && layer <= 1) {
        flush_locked(max_token_pos_locked() > 0 ? "prefill" : "decode");
    }

    const bool host = ggml_backend_buffer_is_host(t->buffer);
    const size_t n_bytes = ggml_nbytes(t);
    const uint8_t * data = nullptr;
    if (host) {
        data = static_cast<const uint8_t *>(t->data);
    } else {
        g_state.scratch.resize(n_bytes);
        ggml_backend_tensor_get(t, g_state.scratch.data(), 0, n_bytes);
        data = g_state.scratch.data();
    }

    const int64_t n_exp = t->ne[0];
    const int64_t n_tok = t->ne[1];
    const int64_t k = argsort ? std::min<int64_t>(n_exp, 8) : n_exp;

    for (int64_t tok = 0; tok < n_tok; ++tok) {
        Row row;
        row.layer = layer;
        row.token_pos = static_cast<int>(tok);
        row.expert_ids.reserve(static_cast<size_t>(k));
        for (int64_t e = 0; e < k; ++e) {
            row.expert_ids.push_back(read_i32(data, t, e, tok));
        }
        if (topk) {
            g_state.topk.push_back(std::move(row));
        } else {
            g_state.argsort.push_back(std::move(row));
        }
    }

    g_state.last_layer = layer;
    if (layer > g_state.max_layer) {
        g_state.max_layer = layer;
    }
    return true;
}
