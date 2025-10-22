#include <cstdio>
#include <cuda_runtime.h>

// Kernel: escribe 1 en out[i] si in[i] >= umbral, de lo contrario 0.
__global__ void umbralizar(const int *in, int *out, int N, int umbral) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // índice global
    if (i < N) {                                     // control de límites
        out[i] = (in[i] >= umbral) ? 1 : 0;          // umbralización
    }
}

int main() {
    // ----- Parámetros del problema -----
    const int N = 32;
    const int UMBRAL = 50;

    int *h_in  = new int[N];
    int *h_out = new int[N];

    for (int i = 0; i < N; ++i) {
        h_in[i] = i * 5;
    }

    int *d_in  = nullptr;
    int *d_out = nullptr;

    // ----- Reserva de memoria en GPU -----
    cudaMalloc((void**)&d_in,  N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));

    // ----- Copia Host -> Device -----
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // ----- Configuración de lanzamiento -----
    int blockSize = 128;
    int gridSize  = (N + blockSize - 1) / blockSize;

    // ----- Lanzamiento del kernel -----
    umbralizar<<<gridSize, blockSize>>>(d_in, d_out, N, UMBRAL);

    // Sincroniza para esperar a que termine el kernel
    cudaDeviceSynchronize();

    // ----- Copia Device -> Host -----
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // ----- Verificación -----
    printf("Umbral = %d\n", UMBRAL);
    for (int i = 0; i < N; ++i) {
        printf("in[%2d] = %3d  -> out[%2d] = %d\n", i, h_in[i], i, h_out[i]);
    }

    // ----- Limpieza -----
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    printf("Finalizado (host).\n");
    return 0;
}
