#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                         \
do {                                                                             \
    cudaError_t _e = (call);                                                     \
    if (_e != cudaSuccess) {                                                     \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
                cudaGetErrorString(_e));                                         \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
} while (0)

__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
    // N por defecto ~1M; puedes pasar N, blockSize por línea de comandos.
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 20);
    int blockSize = (argc > 2) ? atoi(argv[2]) : 256;
    size_t bytes = (size_t)N * sizeof(float);

    // Declarar e inicializar en host
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Fallo al reservar memoria en host\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = (float)i * 0.001f;
    }

    // Reservar en device
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copiar host -> device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Configurar grid/block
    int gridSize = (N + blockSize - 1) / blockSize;

    // medir tiempo del kernel con eventos
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Lanzar kernel en GPU
    vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Copiar device -> host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Validación rápida
    bool ok = true;
    for (int i = 0; i < 10 && i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) { ok = false; break; }
    }

    printf("N=%d | grid=%d block=%d | tiempo kernel = %.3f ms | resultado %s\n",
           N, gridSize, blockSize, ms, ok ? "OK" : "MAL");

    // Imprime algunos resultados
    for (int i = 0; i < 5 && i < N; ++i) {
        printf("C[%d] = %.6f\n", i, h_C[i]);
    }

    // Limpieza
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
