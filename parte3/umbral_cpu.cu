// umbral_cpu.cu
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
  cudaError_t _e = (x); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(EXIT_FAILURE); \
  } \
} while(0)

// ==================== GPU ====================
__global__ void umbralizar(const int* in, int* out, int N, int umbral) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (in[i] >= umbral) ? 1 : 0;
}

// ==================== CPU ====================
void umbralizar_cpu(const int *in, int *out, int N, int umbral) {
    for (int i = 0; i < N; ++i) out[i] = (in[i] >= umbral) ? 1 : 0;
}

int main(int argc, char** argv) {
    const int N       = (argc > 1) ? atoi(argv[1]) : (1 << 20); // 1M por defecto
    const int UMBRAL  = (argc > 2) ? atoi(argv[2]) : 500000;
    const int blockSz = (argc > 3) ? atoi(argv[3]) : 256;

    size_t bytes = (size_t)N * sizeof(int);

    // Host
    int *h_in  = (int*)malloc(bytes);
    int *h_out = (int*)malloc(bytes);
    int *h_ref = (int*)malloc(bytes);
    if (!h_in || !h_out || !h_ref) {
        fprintf(stderr, "Fallo al reservar memoria en host\n");
        return 1;
    }
    for (int i = 0; i < N; ++i) h_in[i] = i; // datos simples

    // ---------- CPU ----------
    auto t0 = std::chrono::high_resolution_clock::now();
    umbralizar_cpu(h_in, h_ref, N, UMBRAL);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- GPU ----------
    int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    auto th2d0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    auto th2d1 = std::chrono::high_resolution_clock::now();

    int gridSz = (N + blockSz - 1) / blockSz;

    // (opcional) warm-up
    umbralizar<<<gridSz, blockSz>>>(d_in, d_out, N, UMBRAL);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto tk0 = std::chrono::high_resolution_clock::now();
    umbralizar<<<gridSz, blockSz>>>(d_in, d_out, N, UMBRAL);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto tk1 = std::chrono::high_resolution_clock::now();

    auto td2h0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    auto td2h1 = std::chrono::high_resolution_clock::now();

    double ms_h2d  = std::chrono::duration<double, std::milli>(th2d1 - th2d0).count();
    double ms_kern = std::chrono::duration<double, std::milli>(tk1   - tk0).count();
    double ms_d2h  = std::chrono::duration<double, std::milli>(td2h1 - td2h0).count();

    // ---------- Verificación ----------
    bool ok = true;
    for (int i = 0; i < 10 && i < N; ++i) {
        if (h_out[i] != h_ref[i]) { ok = false; break; }
    }

    printf("N=%d, UMBRAL=%d, block=%d, grid=%d\n", N, UMBRAL, blockSz, gridSz);
    printf("CPU: %.3f ms | H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Resultado %s\n",
           ms_cpu, ms_h2d, ms_kern, ms_d2h, ok ? "OK" : "MAL");

    // Muestra
    for (int i = 0; i < 5; ++i) {
        printf("in[%d]=%d -> CPU=%d, GPU=%d\n", i, h_in[i], h_ref[i], h_out[i]);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);

    return ok ? 0 : 1;
}
