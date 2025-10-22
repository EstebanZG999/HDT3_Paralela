#include <cstdio>
#include <cuda_runtime.h>


// Kernel: escribe 1 en out[i] si in[i] >= umbral, de lo contrario 0.
// TODO(1): agrega el calificador correcto para que esta función se ejecute en la GPU.
// TODO(2): completa los parámetros que necesita el kernel (entrada, salida, N y umbral).
______ void umbralizar(/* TODO: puntero entrada, puntero salida, tamaño N, int umbral */) {
    // TODO(3): calcula el índice global i usando blockIdx.x, blockDim.x y threadIdx.x.
    int i = /* TODO: índice global */;
    // TODO(4): control de límites para evitar accesos fuera de rango.
    if (/* TODO: condición de límites */) {
        // TODO(5): escribe 1 si in[i] >= umbral, si no 0, en out[i].
        // Sugerencia: out[i] = (in[i] >= umbral) ? 1 : 0;
        /* TODO */
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


    // ----- Copia Host -> Device -----


    // ----- Configuración de lanzamiento -----

    int blockSize = /* TODO: p.ej., 128 */;
    int gridSize  = /* TODO: fórmula de techo para cubrir N */;

    // ----- Lanzamiento del kernel -----
    // Ejemplo: umbralizar<<<gridSize, blockSize>>>(d_in, d_out, N, UMBRAL);
    

    // (Recomendado) Sincroniza para esperar a que termine el kernel.
    // TODO(11): usa cudaDeviceSynchronize();
    /* TODO */



    // ----- Copia Device -> Host -----
    
    /* TODO */

    // ----- Verificación rápida -----
    printf("Umbral = %d\n", UMBRAL);
    for (int i = 0; i < N; ++i) {
        // Esperado: 1 si h_in[i] >= UMBRAL, 0 si no.
        printf("in[%2d] = %3d  -> out[%2d] = %d\n", i, h_in[i], i, h_out[i]);
    }

    // ----- Limpieza -----
    /* TODO */
    /* TODO */

    delete[] h_in;
    delete[] h_out;

    printf("Finalizado (host).\n");
    return 0;
}