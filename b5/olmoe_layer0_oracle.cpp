// llama.cpp oracle for one OLMoE layer-0 MoE block.
//
// Same op sequence as llama_model_olmoe::graph / build_moe_ffn:
// mul_mat(gate), softmax, argsort_top_k, mul_mat_id up, mul_mat_id gate,
// swiglu_split, mul_mat_id down, mul by the softmax weights, add across k.
// norm_w is false and expert_weights_scale is left at 0, matching olmoe.cpp.
// Weights stay F16. No Q4.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static constexpr int64_t N_TOKENS = 128;
static constexpr int64_t N_EMBD = 2048;
static constexpr int64_t N_FF = 1024;
static constexpr int64_t N_EXPERT = 64;
static constexpr int64_t N_USED = 8;

static void die(const char * msg) {
    std::fprintf(stderr, "oracle: %s\n", msg);
    std::exit(1);
}

static void write_file(const std::string & path, const void * data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        die("open for write failed");
    }
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    if (!out) {
        die("write failed");
    }
}

struct Loaded {
    enum ggml_type type;
    std::vector<uint8_t> raw;
    std::vector<float> f32;
};

static Loaded read_tensor(FILE * file, const gguf_context * ctx, const char * name, const int64_t * ne, int n_dim) {
    const int64_t id = gguf_find_tensor(ctx, name);
    if (id < 0) {
        std::fprintf(stderr, "oracle: missing tensor %s\n", name);
        std::exit(1);
    }
    const enum ggml_type type = gguf_get_tensor_type(ctx, id);
    if (type != GGML_TYPE_F16 && type != GGML_TYPE_F32) {
        std::fprintf(stderr, "oracle: %s type is %s\n", name, ggml_type_name(type));
        std::exit(1);
    }
    const int64_t * got = gguf_get_tensor_ne(ctx, id);
    int64_t n_elem = 1;
    for (int i = 0; i < n_dim; ++i) {
        if (got[i] != ne[i]) {
            std::fprintf(stderr, "oracle: %s ne[%d]=%lld want %lld\n", name, i, (long long) got[i], (long long) ne[i]);
            std::exit(1);
        }
        n_elem *= ne[i];
    }
    const size_t size = gguf_get_tensor_size(ctx, id);
    const size_t off = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, id);
    if (fseeko(file, static_cast<off_t>(off), SEEK_SET) != 0) {
        die("seek failed");
    }
    Loaded loaded;
    loaded.type = type;
    loaded.raw.resize(size);
    if (std::fread(loaded.raw.data(), 1, size, file) != size) {
        die("short read");
    }
    loaded.f32.resize(static_cast<size_t>(n_elem));
    if (type == GGML_TYPE_F32) {
        std::memcpy(loaded.f32.data(), loaded.raw.data(), size);
    } else {
        const auto * src = reinterpret_cast<const ggml_fp16_t *>(loaded.raw.data());
        for (int64_t i = 0; i < n_elem; ++i) {
            loaded.f32[static_cast<size_t>(i)] = ggml_fp16_to_fp32(src[i]);
        }
    }
    std::printf("tensor %s type=%s bytes=%zu\n", name, ggml_type_name(type), size);
    return loaded;
}

static std::vector<float> frozen_hidden() {
    // Deterministic. Each token is scaled to RMS 1, the scale of a post-norm residual.
    std::vector<float> h(static_cast<size_t>(N_EMBD * N_TOKENS));
    uint32_t state = 20260924u;
    for (int64_t token = 0; token < N_TOKENS; ++token) {
        double sum = 0.0;
        for (int64_t i = 0; i < N_EMBD; ++i) {
            state = state * 1664525u + 1013904223u;
            const float u = static_cast<float>(state >> 8) / static_cast<float>(1 << 24);
            const float v = (u * 2.0f) - 1.0f;
            h[static_cast<size_t>(i + token * N_EMBD)] = v;
            sum += static_cast<double>(v) * static_cast<double>(v);
        }
        const float scale = static_cast<float>(std::sqrt(static_cast<double>(N_EMBD) / sum));
        for (int64_t i = 0; i < N_EMBD; ++i) {
            h[static_cast<size_t>(i + token * N_EMBD)] *= scale;
        }
    }
    return h;
}

int main(int argc, char ** argv) {
    const char * gguf_path = argc > 1 ? argv[1] : "/mnt/t440/codebase/models/olmoe-1b-7b-0924-f16.gguf";
    const char * out_dir = argc > 2 ? argv[2] : "/tmp/nautiburn-b5/run";

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        die("ggml_backend_cuda_init failed");
    }
    char desc[256] = {};
    ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
    std::printf("oracle_backend=%s device0=%s\n", ggml_backend_name(backend), desc);

    struct ggml_context * meta = nullptr;
    struct gguf_init_params params = { true, &meta };
    struct gguf_context * gguf = gguf_init_from_file(gguf_path, params);
    if (!gguf) {
        die("gguf open failed");
    }
    FILE * file = std::fopen(gguf_path, "rb");
    if (!file) {
        die("gguf fopen failed");
    }

    const int64_t ne_gate_inp[2] = { N_EMBD, N_EXPERT };
    const int64_t ne_up[3] = { N_EMBD, N_FF, N_EXPERT };
    const int64_t ne_down[3] = { N_FF, N_EMBD, N_EXPERT };
    auto gate_inp = read_tensor(file, gguf, "blk.0.ffn_gate_inp.weight", ne_gate_inp, 2);
    auto gate_exps = read_tensor(file, gguf, "blk.0.ffn_gate_exps.weight", ne_up, 3);
    auto up_exps = read_tensor(file, gguf, "blk.0.ffn_up_exps.weight", ne_up, 3);
    auto down_exps = read_tensor(file, gguf, "blk.0.ffn_down_exps.weight", ne_down, 3);
    std::fclose(file);
    gguf_free(gguf);
    ggml_free(meta);

    auto hidden = frozen_hidden();
    write_file(std::string(out_dir) + "/hidden.f32", hidden.data(), hidden.size() * sizeof(float));
    write_file(std::string(out_dir) + "/ffn_gate_inp.f32", gate_inp.f32.data(), gate_inp.f32.size() * sizeof(float));
    write_file(std::string(out_dir) + "/ffn_gate_exps.f32", gate_exps.f32.data(), gate_exps.f32.size() * sizeof(float));
    write_file(std::string(out_dir) + "/ffn_up_exps.f32", up_exps.f32.data(), up_exps.f32.size() * sizeof(float));
    write_file(std::string(out_dir) + "/ffn_down_exps.f32", down_exps.f32.data(), down_exps.f32.size() * sizeof(float));
    std::printf("host_copy_bytes=%zu\n", gate_inp.raw.size() + gate_exps.raw.size() + up_exps.raw.size() + down_exps.raw.size());

    const size_t ctx_size = 64 * 1024 * 1024;
    struct ggml_init_params ip = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        die("ggml_init failed");
    }

    struct ggml_tensor * t_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, N_TOKENS);
    struct ggml_tensor * t_gate_inp = ggml_new_tensor_2d(ctx, gate_inp.type, N_EMBD, N_EXPERT);
    struct ggml_tensor * t_gate = ggml_new_tensor_3d(ctx, gate_exps.type, N_EMBD, N_FF, N_EXPERT);
    struct ggml_tensor * t_up = ggml_new_tensor_3d(ctx, up_exps.type, N_EMBD, N_FF, N_EXPERT);
    struct ggml_tensor * t_down = ggml_new_tensor_3d(ctx, down_exps.type, N_FF, N_EMBD, N_EXPERT);

    struct ggml_tensor * logits = ggml_mul_mat(ctx, t_gate_inp, t_hidden);
    struct ggml_tensor * probs = ggml_soft_max(ctx, logits);
    struct ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, N_USED);
    struct ggml_tensor * ids = ggml_cont(ctx, selected);
    struct ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, N_EXPERT, N_TOKENS);
    struct ggml_tensor * weights = ggml_get_rows(ctx, probs_3d, selected);
    struct ggml_tensor * cur = ggml_reshape_3d(ctx, t_hidden, N_EMBD, 1, N_TOKENS);
    struct ggml_tensor * up = ggml_mul_mat_id(ctx, t_up, cur, selected);
    struct ggml_tensor * gate = ggml_mul_mat_id(ctx, t_gate, cur, selected);
    struct ggml_tensor * mid = ggml_swiglu_split(ctx, gate, up);
    struct ggml_tensor * experts = ggml_mul_mat_id(ctx, t_down, mid, selected);
    experts = ggml_mul(ctx, experts, weights);

    struct ggml_tensor * parts[N_USED];
    for (int64_t i = 0; i < N_USED; ++i) {
        parts[i] = ggml_view_2d(ctx, experts, N_EMBD, N_TOKENS, experts->nb[2], i * experts->nb[1]);
    }
    struct ggml_tensor * moe = parts[0];
    for (int64_t i = 1; i < N_USED; ++i) {
        moe = ggml_add(ctx, moe, parts[i]);
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        die("alloc failed");
    }
    ggml_backend_tensor_set(t_hidden, hidden.data(), 0, hidden.size() * sizeof(float));
    ggml_backend_tensor_set(t_gate_inp, gate_inp.raw.data(), 0, gate_inp.raw.size());
    ggml_backend_tensor_set(t_gate, gate_exps.raw.data(), 0, gate_exps.raw.size());
    ggml_backend_tensor_set(t_up, up_exps.raw.data(), 0, up_exps.raw.size());
    ggml_backend_tensor_set(t_down, down_exps.raw.data(), 0, down_exps.raw.size());

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(gf, moe);
    ggml_build_forward_expand(gf, ids);
    const enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "oracle: compute status %d\n", (int) status);
        std::exit(1);
    }

    std::vector<float> out(static_cast<size_t>(N_EMBD * N_TOKENS));
    std::vector<int32_t> id_out(static_cast<size_t>(N_USED * N_TOKENS));
    ggml_backend_tensor_get(moe, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_tensor_get(ids, id_out.data(), 0, id_out.size() * sizeof(int32_t));
    for (int32_t id : id_out) {
        if (id < 0 || id >= N_EXPERT) {
            die("expert id out of range");
        }
    }
    write_file(std::string(out_dir) + "/oracle_out.f32", out.data(), out.size() * sizeof(float));
    write_file(std::string(out_dir) + "/oracle_ids.i32", id_out.data(), id_out.size() * sizeof(int32_t));
    std::printf("oracle_out_bytes=%zu oracle_ids=%zu first_id=%d\n", out.size() * sizeof(float), id_out.size(), id_out[0]);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
