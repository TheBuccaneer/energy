#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while(0)

// Einfacher GPU-Kernel um die GPU zu beschäftigen
__global__ void dummyKernel(float *data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = val * 1.0001f + 0.0001f;
        }
        data[idx] = val;
    }
}

void printBandwidth(const char* direction, size_t bytes, float ms) {
    float gb = bytes / (1024.0f * 1024.0f * 1024.0f);
    float bandwidth = gb / (ms / 1000.0f);
    printf("[%s] %.2f GB in %.2f ms = %.2f GB/s\n", direction, gb, ms, bandwidth);
}

int main(int argc, char** argv) {
    // Standard: 2 GB pro Transfer
    size_t transferSize = 2ULL * 1024 * 1024 * 1024;
    
    if (argc > 1) {
        transferSize = (size_t)(atof(argv[1]) * 1024 * 1024 * 1024);
    }
    
    printf("=== PCIe Bandwidth Benchmark ===\n");
    printf("Transfer-Größe: %.2f GB\n", transferSize / (1024.0f * 1024.0f * 1024.0f));
    printf("Drücke Ctrl+C zum Beenden\n\n");
    
    // Speicher allokieren - PINNED für maximale Geschwindigkeit!
    float *h_data;
    float *d_data;
    CUDA_CHECK(cudaHostAlloc(&h_data, transferSize, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&d_data, transferSize));
    
    // Host-Daten initialisieren
    for (size_t i = 0; i < transferSize / sizeof(float); i++) {
        h_data[i] = (float)i;
    }
    
    // Events für Zeitmessung
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int iteration = 0;
    
    while (1) {
        iteration++;
        printf("\n--- Iteration %d ---\n", iteration);
        
        // ===== UPLOAD (Host -> GPU) =====
        printf("Starte UPLOAD...\n");
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, transferSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float uploadMs;
        CUDA_CHECK(cudaEventElapsedTime(&uploadMs, start, stop));
        printBandwidth("UPLOAD", transferSize, uploadMs);
        
        // GPU-Arbeit simulieren
        int numElements = transferSize / sizeof(float);
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, numElements);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Pause zwischen Upload und Download
        printf("Warte 3 Sekunden...\n");
        sleep(3);
        
        // ===== DOWNLOAD (GPU -> Host) =====
        printf("Starte DOWNLOAD...\n");
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(h_data, d_data, transferSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float downloadMs;
        CUDA_CHECK(cudaEventElapsedTime(&downloadMs, start, stop));
        printBandwidth("DOWNLOAD", transferSize, downloadMs);
        
        // Längere Pause vor nächster Iteration
        printf("Warte 5 Sekunden vor nächster Iteration...\n");
        sleep(5);
    }
    
    // Programm wird mit Ctrl+C beendet
    return 0;
}