#!/bin/bash
# GPU Stabilisierungs-Script

set -e

# Optionen
LOCK_CLOCKS="${LOCK_CLOCKS:-0}"  # 1=Clocks locken, 0=nicht locken
WARMUP_DURATION="${WARMUP_DURATION:-60}"

echo "=== GPU Stabilization (Safe & Robust) ==="
echo "Clock locking: $([[ "$LOCK_CLOCKS" == "1" ]] && echo "ENABLED" || echo "DISABLED")"
echo ""

# Persistence Mode aktivieren
sudo nvidia-smi -pm 1

# GPU konfigurieren
for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    gpu_name=$(nvidia-smi -i $gpu_id --query-gpu=name --format=csv,noheader)
    echo "Configuring GPU $gpu_id: $gpu_name"
    
    # Power Limit setzen (konservativ)
    if [[ "$gpu_name" == *"3090"* ]]; then
        sudo nvidia-smi -i $gpu_id -pl 350 2>/dev/null || echo "  Warning: Power limit not supported"
    elif [[ "$gpu_name" == *"1080"* ]]; then
        sudo nvidia-smi -i $gpu_id -pl 250 2>/dev/null || echo "  Warning: Power limit not supported"
    fi
    
    # Clock Locks (nur wenn aktiviert)
    if [[ "$LOCK_CLOCKS" == "1" ]]; then
        echo "  Locking clocks..."
        if [[ "$gpu_name" == *"3090"* ]]; then
            sudo nvidia-smi -i $gpu_id -lgc 1695 2>/dev/null || echo "  Warning: GPU clock lock not supported"
            sudo nvidia-smi -i $gpu_id -lmc 9750 2>/dev/null || echo "  Warning: Memory clock lock not supported"
        elif [[ "$gpu_name" == *"1080"* ]]; then
            sudo nvidia-smi -i $gpu_id -lgc 1480 2>/dev/null || echo "  Warning: GPU clock lock not supported"
            sudo nvidia-smi -i $gpu_id -lmc 5505 2>/dev/null || echo "  Warning: Memory clock lock not supported"
        fi
    fi
done

echo ""
echo "Current GPU status:"
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory --format=csv

# Snapshots für Audit
echo ""
echo "Saving initial state snapshot..."
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE > warmup_before.log 2>/dev/null || true

# Warmup-Tool kompilieren
echo ""
echo "Preparing warmup tool..."

if [ ! -f "./gpu_warmup" ]; then
    echo "Compiling GPU warmup (cuBLAS)..."
    
    cat > gpu_warmup.cu << 'CUDA_CODE'
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error: %d (line %d)\n", status, __LINE__); \
            exit(1); \
        } \
    } while(0)

void checkTemperature(int *targetTemp) {
    static int target = -1;
    
    if (target == -1) {
        FILE *fp = popen("nvidia-smi --query-gpu=temperature.gpu_target --format=csv,noheader,nounits", "r");
        if (fp && fscanf(fp, "%d", &target) == 1) {
            pclose(fp);
        } else {
            target = 83;
        }
        *targetTemp = target;
    }
    
    FILE *fp = popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits", "r");
    if (fp) {
        int temp;
        if (fscanf(fp, "%d", &temp) == 1) {
            printf(" [Temp: %d°C / Target: %d°C]", temp, target);
            if (temp >= target - 2) {
                printf(" ⚠ Near target!");
            }
        }
        pclose(fp);
    }
}

int main(int argc, char** argv) {
    int duration = 60;
    int n = 8192;
    int gpu_id = 0;
    
    if (argc > 1) {
        duration = atoi(argv[1]);
    }
    if (argc > 2) {
        gpu_id = atoi(argv[2]);
    }
    
    CUDA_CHECK(cudaSetDevice(gpu_id));
    
    printf("GPU Warmup (cuBLAS SGEMM) - GPU %d - %d seconds\n\n", gpu_id, duration);
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    size_t size = n * n * sizeof(float);
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    float *h_init = (float*)malloc(size);
    for (size_t i = 0; i < n * n; i++) {
        h_init[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_A, h_init, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_init, size, cudaMemcpyHostToDevice));
    free(h_init);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    time_t start_time = time(NULL);
    int iterations = 0;
    int last_printed = -1;
    int targetTemp = -1;
    
    while (time(NULL) - start_time < duration) {
        CUBLAS_CHECK(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n
        ));
        
        iterations++;
        
        if (iterations % 50 == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            int elapsed = time(NULL) - start_time;
            
            if (elapsed != last_printed) {
                printf("\r%d/%d seconds (%d iters)", elapsed, duration, iterations);
                checkTemperature(&targetTemp);
                fflush(stdout);
                last_printed = elapsed;
            }
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n\nWarmup complete! (%d iterations)\n", iterations);
    
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}
CUDA_CODE

    nvcc -o gpu_warmup gpu_warmup.cu -lcublas
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to compile warmup tool"
        exit 1
    fi
    echo "Compiled successfully!"
fi

# Warmup ausführen
echo ""
echo "Starting warmup for all GPUs..."

for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    gpu_name=$(nvidia-smi -i $gpu_id --query-gpu=name --format=csv,noheader)
    echo ""
    echo "=== Warming up GPU $gpu_id: $gpu_name ==="
    # CUDA_VISIBLE_DEVICES remappt auf Device 0 im Prozess!
    CUDA_VISIBLE_DEVICES=$gpu_id ./gpu_warmup $WARMUP_DURATION 0
done

# Final status
echo ""
echo "Saving final state snapshot..."
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE > warmup_after.log 2>/dev/null || true

echo ""
echo "Final GPU status:"
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory --format=csv
echo ""
echo "✓ GPUs ready for measurements!"
echo "  (Snapshots saved to warmup_before.log and warmup_after.log)"