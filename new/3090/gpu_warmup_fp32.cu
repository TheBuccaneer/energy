#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call) do { const cudaError_t s=(call); if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); } while(0)
#define CUBLAS_CHECK(call) do { const cublasStatus_t s=(call); if(s!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure"); } while(0)

int main(int argc, char** argv) {
    try {
        const int duration = argc > 1 ? std::max(1, std::atoi(argv[1])) : 60;
        const int n = 4096;
        const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
        float *a=nullptr, *b=nullptr, *c=nullptr;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c), bytes));
        CUDA_CHECK(cudaMemset(a, 1, bytes));
        CUDA_CHECK(cudaMemset(b, 2, bytes));
        cublasHandle_t handle{};
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        const float alpha=1.0f, beta=0.0f;
        const auto start=std::chrono::steady_clock::now();
        long long iterations=0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() < duration) {
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
                         &alpha, b, CUDA_R_32F,n, a, CUDA_R_32F,n,
                         &beta, c, CUDA_R_32F,n,
                         CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
            ++iterations;
            if ((iterations % 8) == 0) CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Warmup complete: " << iterations << " GEMMs\n";
        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(c));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << '\n';
        return 2;
    }
}
