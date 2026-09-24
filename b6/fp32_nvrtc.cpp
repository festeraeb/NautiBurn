// One expert-shaped FP32 GEMM, JIT through CUDA 12.8 nvrtc.
// Qwen1.5-MoE routed expert: K=2048, N=1408. No DP4A, no int4 MMA.
#include <cuda.h>
#include <nvrtc.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_NVRTC(x)                                                         \
    do {                                                                       \
        nvrtcResult _e = (x);                                                  \
        if (_e != NVRTC_SUCCESS) {                                             \
            std::fprintf(stderr, "nvrtc %s\n", nvrtcGetErrorString(_e));       \
            return 1;                                                          \
        }                                                                      \
    } while (0)
#define CHECK_CU(x)                                                            \
    do {                                                                       \
        CUresult _e = (x);                                                     \
        if (_e != CUDA_SUCCESS) {                                              \
            const char *s = nullptr;                                           \
            cuGetErrorString(_e, &s);                                          \
            std::fprintf(stderr, "cuda %s\n", s ? s : "?");                     \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static const char *kSrc = R"(
extern "C" __global__ void gemm_f32_row(const float *a, const float *b, float *c, int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float *col = b + (size_t)n * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += a[k] * col[k];
    c[n] = acc;
}
)";

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static void print_nvrtc_maps() {
    FILE *f = std::fopen("/proc/self/maps", "r");
    if (!f) return;
    char line[1024];
    while (std::fgets(line, sizeof line, f)) {
        if (std::strstr(line, "libnvrtc")) std::fputs(line, stdout);
    }
    std::fclose(f);
}

int main() {
    constexpr int K = 2048;
    constexpr int N = 1408;
    constexpr int Q4K_BLOCK = 144;
    constexpr int Q4K_WEIGHTS = 256;
    const size_t fp32_bytes = (size_t)K * N * sizeof(float);
    const size_t q4_bytes = (size_t)K * N * Q4K_BLOCK / Q4K_WEIGHTS;

    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    char name[128] = {};
    CHECK_CU(cuDeviceGetName(name, sizeof name, dev));
    std::printf("device=%s K=%d N=%d fp32_bytes=%zu q4_bytes=%zu\n", name, K, N, fp32_bytes, q4_bytes);

    nvrtcProgram prog;
    CHECK_NVRTC(nvrtcCreateProgram(&prog, kSrc, "gemm_f32_row.cu", 0, nullptr, nullptr));
    const char *opts[] = {"--gpu-architecture=compute_60", "-fmad=true"};
    nvrtcResult compile = nvrtcCompileProgram(prog, 2, opts);
    size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    if (log_size > 1) {
        std::string log(log_size, '\0');
        nvrtcGetProgramLog(prog, log.data());
        std::fputs(log.c_str(), stderr);
    }
    if (compile != NVRTC_SUCCESS) return 1;
    size_t ptx_size = 0;
    CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptx_size));
    std::vector<char> ptx(ptx_size);
    CHECK_NVRTC(nvrtcGetPTX(prog, ptx.data()));
    int major = 0, minor = 0;
    nvrtcVersion(&major, &minor);
    std::printf("nvrtc_version=%d.%d arch=compute_60\n", major, minor);
    print_nvrtc_maps();

    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));
    CUmodule mod;
    CHECK_CU(cuModuleLoadData(&mod, ptx.data()));
    CUfunction fn;
    CHECK_CU(cuModuleGetFunction(&fn, mod, "gemm_f32_row"));

    std::vector<float> h_a(K), h_b((size_t)K * N), h_c(N);
    for (int k = 0; k < K; ++k) h_a[k] = (k % 17) * 0.01f;
    for (size_t i = 0; i < h_b.size(); ++i) h_b[i] = ((int)(i % 13) - 6) * 0.01f;

    CUdeviceptr d_a, d_b, d_c, d_q4;
    CHECK_CU(cuMemAlloc(&d_a, K * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_b, fp32_bytes));
    CHECK_CU(cuMemAlloc(&d_c, N * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_q4, q4_bytes));
    std::vector<char> h_q4(q4_bytes, 1);

    auto time_copy = [&](CUdeviceptr dst, const void *src, size_t n, const char *label) {
        // warmup
        CHECK_CU(cuMemcpyHtoD(dst, src, n));
        CHECK_CU(cuCtxSynchronize());
        const int reps = 20;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) CHECK_CU(cuMemcpyHtoD(dst, src, n));
        CHECK_CU(cuCtxSynchronize());
        double ms = ms_since(t0) / reps;
        std::printf("%s_ms=%.6f bytes=%zu\n", label, ms, n);
        return 0;
    };
    if (time_copy(d_q4, h_q4.data(), q4_bytes, "h2d_q4")) return 1;
    if (time_copy(d_b, h_b.data(), fp32_bytes, "h2d_fp32")) return 1;
    CHECK_CU(cuMemcpyHtoD(d_a, h_a.data(), K * sizeof(float)));

    int n = N;
    int k = K;
    void *args[] = {&d_a, &d_b, &d_c, &k, &n};
    auto launch = [&]() {
        return cuLaunchKernel(fn, (N + 127) / 128, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr);
    };
    CHECK_CU(launch());
    CHECK_CU(cuCtxSynchronize());
    const int greps = 50;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < greps; ++i) CHECK_CU(launch());
    CHECK_CU(cuCtxSynchronize());
    double gpu_ms = ms_since(t0) / greps;
    std::printf("gpu_fp32_gemm_ms=%.6f\n", gpu_ms);

    t0 = std::chrono::steady_clock::now();
    const int creps = 5;
    for (int r = 0; r < creps; ++r) {
        for (int col = 0; col < N; ++col) {
            const float *bcol = h_b.data() + (size_t)col * K;
            float acc = 0.f;
            for (int kk = 0; kk < K; ++kk) acc += h_a[kk] * bcol[kk];
            h_c[col] = acc;
        }
    }
    double cpu_ms = ms_since(t0) / creps;
    std::printf("cpu_fp32_gemm_ms=%.6f\n", cpu_ms);

    CHECK_CU(cuMemcpyDtoH(h_c.data(), d_c, sizeof(float)));
    std::printf("sample_c0=%.6f\n", h_c[0]);
    std::printf("choice_inputs q4_copy_plus_gpu_vs_cpu recorded above\n");
    return 0;
}
