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
