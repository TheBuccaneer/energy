// main.cu - CUDA GEMM Energy Benchmark with cuBLAS Strided Batched
// Compile: nvcc -O3 -std=c++17 -o gemm_bench main.cu -lcublas -lnvidia-ml
// Usage: ./gemm_bench [--test|-t] [--output|-o <path>]

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <ctime>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <map>
#include <locale>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdint>
#include <cmath>

// ============================================================================
// Configuration
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    MAX_BATCH_SIZE   = 250000;
constexpr int    MACRO_REPEATS    = 50;

// Problem sizes and their batch_counts (canonical)
// batch_count = parallel GEMM instances per cuBLAS call (FIXED)
// batches = how many times to repeat the call (ADAPTIVE, calculated to reach ~1s)
static const std::map<int, std::vector<int>> SIZE_TO_BATCH_COUNTS = {
    {64,    {512, 1024}},
    {128,   {256, 512}},
    {256,   {128, 256}},
    {512,   {64, 128}},
    {1024,  {32, 64}},
    {2048,  {16, 32}},
    {4096,  {4, 8}},
    {8192,  {1, 2}},
    {16384, {1}}
};

static const std::vector<int> GEMM_SIZES = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
static const int MAX_SIZE = 16384;

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d: status=%d\n", __FILE__, __LINE__, \
                (int)status); \
        if (status == CUBLAS_STATUS_INVALID_VALUE) { \
            fprintf(stderr, "  CUBLAS_STATUS_INVALID_VALUE - Check parameters!\n"); \
        } \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_NVML(call) do { \
    nvmlReturn_t ret = call; \
    if (ret != NVML_SUCCESS) { \
        fprintf(stderr, "NVML Error at %s:%d: %s\n", __FILE__, __LINE__, \
                nvmlErrorString(ret)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// Utility Functions
// ============================================================================

std::string getTimestamp() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

void ensureDirectoryExists(const char* filepath) {
    namespace fs = std::filesystem;
    fs::path file_path(filepath);
    if (file_path.has_parent_path()) {
        fs::create_directories(file_path.parent_path());
    }
}

bool fileExists(const char* filepath) {
    struct stat buffer;
    return (stat(filepath, &buffer) == 0);
}

void initializeMatrix(float* mat, int n, int lda, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * lda + j] = dist(gen);
        }
    }
}

// ============================================================================
// Capacity Management for d_C
// ============================================================================

void ensureCapacity(float** d_C, size_t* cap_bytes, size_t needed_bytes) {
    if (needed_bytes > *cap_bytes) {
        // Overflow guard (64-bit check)
        if (needed_bytes > (size_t(1) << 40)) {  // > 1 TB
            fprintf(stderr, "ERROR: Requested allocation too large: %zu bytes\n", needed_bytes);
            exit(EXIT_FAILURE);
        }
        
        // Dynamic VRAM guard: Check available GPU memory (95% of free)
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
        
        size_t max_allowed = static_cast<size_t>(free_bytes * 0.95);  // 95% of currently free memory
        
        if (needed_bytes > max_allowed) {
            fprintf(stderr, "WARNING: Allocation exceeds 95%% of free GPU memory:\n");
            fprintf(stderr, "  Requested: %zu bytes (%.2f MB)\n", needed_bytes, needed_bytes / (1024.0 * 1024.0));
            fprintf(stderr, "  Free: %zu bytes (%.2f MB)\n", free_bytes, free_bytes / (1024.0 * 1024.0));
            fprintf(stderr, "  Max recommended (95%% of free): %zu bytes (%.2f MB)\n", max_allowed, max_allowed / (1024.0 * 1024.0));
            fprintf(stderr, "  Attempting allocation anyway...\n");
            // Don't exit - try to allocate and let CUDA fail if really out of memory
        }
        
        // Free old allocation if exists
        if (*d_C != nullptr) {
            CHECK_CUDA(cudaFree(*d_C));
            *d_C = nullptr;  // Set to nullptr after freeing
            *cap_bytes = 0;
        }
        
        // Allocate new buffer - if this fails, CHECK_CUDA will exit
        cudaError_t err = cudaMalloc((void**)d_C, needed_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMalloc failed for %zu bytes (%.2f MB): %s\n",
                    needed_bytes, needed_bytes / (1024.0 * 1024.0),
                    cudaGetErrorString(err));
            fprintf(stderr, "  Free memory: %zu bytes (%.2f MB)\n",
                    free_bytes, free_bytes / (1024.0 * 1024.0));
            exit(EXIT_FAILURE);
        }
        *cap_bytes = needed_bytes;
        
        fprintf(stderr, "INFO: Allocated d_C to %zu bytes (%.2f MB), %.1f%% of free VRAM\n",
                needed_bytes, needed_bytes / (1024.0 * 1024.0), 
                (needed_bytes * 100.0) / free_bytes);
    }
}

// ============================================================================
// NVML Helper Functions
// ============================================================================

struct GPUTelemetry {
    unsigned int pcie_gen;
    unsigned int pcie_width;
    unsigned int sm_clock;
    unsigned int mem_clock;
    unsigned int temp;
    unsigned long long throttle_reasons;  // NVML bitmask: 0=none, 1=thermal, 2=power, etc.
};

std::string getGPUName(nvmlDevice_t device) {
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    CHECK_NVML(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
    std::string full_name(name);
    
    // Extract short model name (e.g., "RTX 3090" from "NVIDIA GeForce RTX 3090")
    // Remove common prefixes
    const char* prefixes[] = {"NVIDIA GeForce ", "NVIDIA Tesla ", "NVIDIA ", "AMD Radeon ", "AMD "};
    for (const char* prefix : prefixes) {
        size_t pos = full_name.find(prefix);
        if (pos == 0) {
            return full_name.substr(strlen(prefix));
        }
    }
    return full_name;  // Return as-is if no prefix matched
}

unsigned long long getGPUEnergy(nvmlDevice_t device) {
    unsigned long long energy_mj = 0;
    nvmlReturn_t ret = nvmlDeviceGetTotalEnergyConsumption(device, &energy_mj);
    if (ret != NVML_SUCCESS) {
        return 0;  // Not supported
    }
    return energy_mj;
}

GPUTelemetry getGPUTelemetry(nvmlDevice_t device) {
    GPUTelemetry telem;
    CHECK_NVML(nvmlDeviceGetCurrPcieLinkGeneration(device, &telem.pcie_gen));
    CHECK_NVML(nvmlDeviceGetCurrPcieLinkWidth(device, &telem.pcie_width));
    CHECK_NVML(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &telem.sm_clock));
    CHECK_NVML(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &telem.mem_clock));
    CHECK_NVML(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &telem.temp));
    
    // Get throttle reasons (bitmask)
    nvmlReturn_t ret = nvmlDeviceGetCurrentClocksThrottleReasons(device, &telem.throttle_reasons);
    if (ret != NVML_SUCCESS) {
        telem.throttle_reasons = 0;  // Not supported or error
    }
    
    return telem;
}

// ============================================================================
// Auto-Batch Determination (Strided-Batched) - ADAPTIVE like main.cu
// ============================================================================

struct BatchResult {
    int batches;
    bool below_target;
};

BatchResult determineBatchSize(cublasHandle_t handle, float* d_A, float* d_B, float** d_C,
                               float* h_A, float* h_B, float* h_C,
                               size_t* cap_bytes, int n, int lda, int batch_count,
                               float target_seconds, cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int batches = 1;
    
    long long strideA = 0;
    long long strideB = 0;
    long long strideC = (long long)lda * (long long)n;
    
    // Need to allocate lda×n per batch for correct striding
    size_t needed_bytes = sizeof(float) * (size_t)lda * (size_t)n * (size_t)batch_count;
    ensureCapacity(d_C, cap_bytes, needed_bytes);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warmup: 2 calls
    for (int w = 0; w < 2; w++) {
        CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                               n, n, n, &alpha,
                                               d_A, lda, strideA,
                                               d_B, lda, strideB,
                                               &beta,
                                               *d_C, lda, strideC,
                                               batch_count));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // ADAPTIVE: Iteratively double batches until target_seconds is reached (like main.cu)
    while (batches <= MAX_BATCH_SIZE) {
        // Measure E2E time for current batches (with H2D + compute + D2H)
        size_t src_pitch = size_t(lda) * sizeof(float);
        size_t dst_pitch = size_t(lda) * sizeof(float);
        size_t width_in_bytes = size_t(n) * sizeof(float);
        size_t height = size_t(n);
        
        CHECK_CUDA(cudaEventRecord(start, stream));
        
        CHECK_CUDA(cudaMemcpy2DAsync(d_A, dst_pitch, h_A, src_pitch,
                                     width_in_bytes, height,
                                     cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpy2DAsync(d_B, dst_pitch, h_B, src_pitch,
                                     width_in_bytes, height,
                                     cudaMemcpyHostToDevice, stream));
        
        // Run 'batches' iterations
        for (int b = 0; b < batches; b++) {
            CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                                   n, n, n, &alpha,
                                                   d_A, lda, strideA,
                                                   d_B, lda, strideB,
                                                   &beta,
                                                   *d_C, lda, strideC,
                                                   batch_count));
        }
        
        CHECK_CUDA(cudaMemcpy2DAsync(h_C, src_pitch, *d_C, dst_pitch,
                                     width_in_bytes, height,
                                     cudaMemcpyDeviceToHost, stream));
        
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float elapsed = ms / 1000.0f;
        
        // Check if we reached target
        if (elapsed >= target_seconds) {
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            return {batches, false};  // Target reached
        }
        
        // Check if we hit max limit
        if (batches >= MAX_BATCH_SIZE) {
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            return {batches, true};  // Below target (hit max batch limit)
        }
        
        // Double batches for next iteration
        batches = std::min(batches * 2, MAX_BATCH_SIZE);
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return {batches, false};
}

// ============================================================================
// CSV Writing Functions
// ============================================================================

void writeCSVHeader(std::ofstream& csv) {
    csv << "timestamp,run_id_global,run_id_per_size,device_name,num_threads,problem_size,batches,batch_count,"
        << "gpu_e2e_time_s,gpu_kernel_time_s,wall_time_s,"
        << "total_energy_j,energy_per_batch_j,energy_per_second_j,energy_per_flop_j,"
        << "time_per_gemm_ms_kernel,time_per_gemm_ms_e2e,"
        << "flops_total,gflops_per_s,avg_power_w,"
        << "below_target,"
        << "pcie_gen,pcie_width,sm_clock_mhz,mem_clock_mhz,temp_c,throttle_reasons\n";
}

void writeCSVRow(std::ofstream& csv, int run_id_global, int run_id_per_size,
                 const std::string& device_name, const std::string& num_threads,
                 int n, int batches, int batch_count,
                 float gpu_time_s, float kernel_time_s, float wall_time_s,
                 double energy_j, double avg_power_w, bool below_target,
                 const GPUTelemetry& telem) {
    
    // Calculate metrics
    double flops_per_gemm = 2.0 * double(n) * double(n) * double(n);
    double total_instances = double(batches) * double(batch_count);  // Total GEMM instances
    double flops_total = flops_per_gemm * total_instances;
    
    double energy_per_batch = (batches > 0) ? (energy_j / batches) : 0.0;  // Energy per call
    double energy_per_second = (wall_time_s > 0) ? (energy_j / wall_time_s) : 0.0;
    double energy_per_flop = (flops_total > 0) ? (energy_j / flops_total) : 0.0;
    
    double time_per_gemm_ms_kernel = (total_instances > 0) ? (kernel_time_s * 1e3 / total_instances) : 0.0;
    double time_per_gemm_ms_e2e = (total_instances > 0) ? (gpu_time_s * 1e3 / total_instances) : 0.0;
    
    double gflops_per_s = (kernel_time_s > 0) ? (flops_total / kernel_time_s / 1e9) : 0.0;
    
    // Write row
    csv << getTimestamp() << ","
        << run_id_global << ","
        << run_id_per_size << ","
        << device_name << ","
        << num_threads << ","  // Empty for GPU
        << n << ","
        << batches << ","
        << batch_count << ","
        << std::scientific << std::setprecision(6)
        << gpu_time_s << ","
        << kernel_time_s << ","
        << wall_time_s << ","
        << energy_j << ","
        << energy_per_batch << ","
        << energy_per_second << ","
        << energy_per_flop << ","
        << std::fixed << std::setprecision(6)
        << time_per_gemm_ms_kernel << ","
        << time_per_gemm_ms_e2e << ","
        << std::scientific << std::setprecision(6)
        << flops_total << ","
        << std::fixed << std::setprecision(2)
        << gflops_per_s << ","
        << avg_power_w << ","
        << (below_target ? 't' : 'f') << ","
        << telem.pcie_gen << ","
        << telem.pcie_width << ","
        << telem.sm_clock << ","
        << telem.mem_clock << ","
        << telem.temp << ","
        << "0x" << std::hex << telem.throttle_reasons << std::dec
        << "\n";
}

// ============================================================================
// Cleanup Helper
// ============================================================================

void cleanup_and_exit(cudaEvent_t start_event, cudaEvent_t stop_event,
                      cudaEvent_t start_kernel, cudaEvent_t stop_kernel,
                      float* d_A, float* d_B, float* d_C,
                      float* h_A, float* h_B, float* h_C,
                      cudaStream_t stream, cublasHandle_t handle,
                      std::ofstream& csv_file) {
    std::cout << "\nTest mode: 5 rows written. Exiting...\n";
    
    csv_file.close();
    
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_CUDA(cudaEventDestroy(start_kernel));
    CHECK_CUDA(cudaEventDestroy(stop_kernel));
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    if (d_C != nullptr) {
        CHECK_CUDA(cudaFree(d_C));
    }
    
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUBLAS(cublasDestroy(handle));
    nvmlShutdown();
    
    exit(EXIT_SUCCESS);
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char** argv) {
    bool test_mode = false;
    std::string output_file = "./data/raw/gpu_gemm_measurements_strided.csv";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0 || strcmp(argv[i], "-t") == 0) {
            test_mode = true;
            output_file = "./data/raw/gpu_gemm_test_strided.csv";
        } else if ((strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) && i + 1 < argc) {
            output_file = argv[++i];
        }
    }
    
    if (test_mode) {
        std::cout << "TEST MODE: Will write 5 rows and exit\n";
    }
    std::cout << "Output file: " << output_file << "\n\n";
    
    // Initialize NVML
    CHECK_NVML(nvmlInit());
    
    nvmlDevice_t nvml_device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &nvml_device));
    
    std::string device_name = getGPUName(nvml_device);
    std::cout << "GPU: " << device_name << "\n";
    
    // Check energy measurement support
    unsigned long long test_energy = getGPUEnergy(nvml_device);
    if (test_energy == 0) {
        std::cerr << "Warning: Energy measurement not supported on this GPU\n";
    }
    
    // Initialize CUDA
    CHECK_CUDA(cudaSetDevice(0));
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));  // Disable TF32
    
    // Allocate pinned host memory (maximum size for all problem sizes)
    size_t max_bytes = sizeof(float) * size_t(MAX_SIZE) * size_t(MAX_SIZE);
    
    float *h_A, *h_B, *h_C;
    CHECK_CUDA(cudaMallocHost((void**)&h_A, max_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_B, max_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_C, max_bytes));
    
    // Initialize matrices (full MAX_SIZE × MAX_SIZE)
    initializeMatrix(h_A, MAX_SIZE, MAX_SIZE, 42);
    initializeMatrix(h_B, MAX_SIZE, MAX_SIZE, 43);
    initializeMatrix(h_C, MAX_SIZE, MAX_SIZE, 44);
    
    // Allocate device memory (maximum size for A, B)
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, max_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, max_bytes));
    
    // d_C will be dynamically sized via ensureCapacity
    d_C = nullptr;
    size_t d_C_capacity_bytes = 0;
    
    // CSV setup
    int run_id_global = 1;
    int total_rows = 0;
    
    ensureDirectoryExists(output_file.c_str());
    bool write_header = !fileExists(output_file.c_str());
    
    std::ofstream csv_file(output_file, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Error: CANNOT open file: " << output_file << "\n";
        return EXIT_FAILURE;
    }
    
    if (write_header) {
        writeCSVHeader(csv_file);
    }
    
    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    cudaEvent_t start_kernel, stop_kernel;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));
    CHECK_CUDA(cudaEventCreate(&start_kernel));
    CHECK_CUDA(cudaEventCreate(&stop_kernel));
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // ========================================================================
    // Main measurement loop: sweep over all sizes
    // ========================================================================
    
    std::cout << "Starting measurements...\n\n";
    
    // Outer loop: over problem sizes
    for (int n : GEMM_SIZES) {
        auto it = SIZE_TO_BATCH_COUNTS.find(n);
        if (it == SIZE_TO_BATCH_COUNTS.end()) {
            std::cerr << "ERROR: No batch_counts defined for size " << n << "\n";
            continue;
        }
        
        const std::vector<int>& batch_counts = it->second;
        int run_id_per_size = 1;  // Reset counter for each new problem size
        
        // Inner loop: over batch_counts for this size
        for (int batch_count : batch_counts) {
            std::cout << "GEMM size " << n << "x" << n << " (batch_count=" << batch_count << ")\n";
            
            long long strideA = 0;
            long long strideB = 0;
            long long strideC = (long long)MAX_SIZE * (long long)n;
            
            // ADAPTIVE: Determine batches for this batch_count (like main.cu)
            std::cout << "  Determine batch size (adaptive)..." << std::flush;
            BatchResult batch_result = determineBatchSize(handle, d_A, d_B, &d_C,
                                                          h_A, h_B, h_C,
                                                          &d_C_capacity_bytes,
                                                          n, MAX_SIZE, batch_count,
                                                          TARGET_RUNTIME_S, stream);
            int batches = batch_result.batches;
            bool below_target_size = batch_result.below_target;
            
            std::cout << " using " << batches << " batches";
            if (below_target_size) {
                std::cout << " (!) below target";
            }
            std::cout << "\n";
            
            // MACRO_REPEATS measurements with the determined configuration
            for (int rep = 0; rep < MACRO_REPEATS; rep++) {
            size_t src_pitch = size_t(MAX_SIZE) * sizeof(float);
            size_t dst_pitch = size_t(MAX_SIZE) * sizeof(float);
            size_t width_in_bytes = size_t(n) * sizeof(float);
            size_t height = size_t(n);
            
            auto wall_start = std::chrono::steady_clock::now();
            unsigned long long energy_before = getGPUEnergy(nvml_device);
            
            CHECK_CUDA(cudaEventRecord(start_event, stream));
            
            CHECK_CUDA(cudaMemcpy2DAsync(d_A, dst_pitch,
                                         h_A, src_pitch,
                                         width_in_bytes, height,
                                         cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemcpy2DAsync(d_B, dst_pitch,
                                         h_B, src_pitch,
                                         width_in_bytes, height,
                                         cudaMemcpyHostToDevice, stream));
            
            CHECK_CUDA(cudaEventRecord(start_kernel, stream));
            
            for (int b = 0; b < batches; b++) {
                CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                                       n, n, n, &alpha,
                                                       d_A, MAX_SIZE, strideA,
                                                       d_B, MAX_SIZE, strideB,
                                                       &beta,
                                                       d_C, MAX_SIZE, strideC,
                                                       batch_count));
            }
            
            CHECK_CUDA(cudaEventRecord(stop_kernel, stream));
            
            CHECK_CUDA(cudaMemcpy2DAsync(h_C, src_pitch,
                                         d_C, dst_pitch,
                                         width_in_bytes, height,
                                         cudaMemcpyDeviceToHost, stream));
            
            CHECK_CUDA(cudaEventRecord(stop_event, stream));
            
            CHECK_CUDA(cudaDeviceSynchronize());
            
            unsigned long long energy_after = getGPUEnergy(nvml_device);
            auto wall_end = std::chrono::steady_clock::now();
            
            float gpu_ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start_event, stop_event));
            float gpu_time_s = gpu_ms / 1000.0f;
            
            float kernel_ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
            float kernel_time_s = kernel_ms / 1000.0f;
            
            std::chrono::duration<double> wall_duration = wall_end - wall_start;
            float wall_time_s = wall_duration.count();
            
            double energy_j = 0.0;
            double avg_power_w = 0.0;
            
            if (energy_after > energy_before) {
                unsigned long long energy_mj = energy_after - energy_before;
                energy_j = energy_mj / 1000.0;
                avg_power_w = energy_j / wall_time_s;
            }
            
            bool below_target = (gpu_time_s < TARGET_RUNTIME_S);
            
            GPUTelemetry telem = getGPUTelemetry(nvml_device);
            
            writeCSVRow(csv_file, run_id_global, run_id_per_size, device_name, "", n, batches, batch_count,
                       gpu_time_s, kernel_time_s, wall_time_s, energy_j, avg_power_w, 
                       below_target, telem);
            csv_file.flush();
            
            run_id_global++;
            run_id_per_size++;
            
            ++total_rows;
            if (test_mode && total_rows >= 5) {
                cleanup_and_exit(start_event, stop_event, start_kernel, stop_kernel,
                               d_A, d_B, d_C, h_A, h_B, h_C, stream, handle, csv_file);
            }
            
            char check = below_target ? '!' : '+';
            std::cout << "  " << check << " Run " << (rep + 1) << "/" 
                     << MACRO_REPEATS << ": "
                     << std::fixed << std::setprecision(3) 
                     << gpu_time_s << "s (GPU) "
                     << wall_time_s << "s (wall) | "
                     << "E=" << std::setprecision(1) << energy_j << "J "
                     << "P=" << std::setprecision(0) << avg_power_w << "W "
                     << "T=" << telem.temp << "°C";
            if (telem.throttle_reasons != 0) {
                std::cout << " THROTTLE=0x" << std::hex << telem.throttle_reasons << std::dec;
            }
            if (below_target) {
                std::cout << " (!)";
            }
            std::cout << "\n";
        }  // End of MACRO_REPEATS loop
        
        std::cout << "  Cooling down for 30 seconds\n";
        std::this_thread::sleep_for(std::chrono::seconds(30));
        
        // Free d_C after each batch_count to avoid memory issues on low-VRAM GPUs (e.g., RTX 5050)
        if (d_C != nullptr) {
            CHECK_CUDA(cudaFree(d_C));
            d_C = nullptr;
            d_C_capacity_bytes = 0;
            std::cout << "  Released d_C memory for next batch_count\n";
        }
        
        std::cout << "\n";
        }  // End of inner loop (batch_counts)
    }  // End of outer loop (problem sizes)
    
    std::cout << "\n\nBenchmark complete!\n";
    std::cout << "Results saved to: " << output_file << "\n\n";
    
    // Cleanup
    csv_file.close();
    
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_CUDA(cudaEventDestroy(start_kernel));
    CHECK_CUDA(cudaEventDestroy(stop_kernel));
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUBLAS(cublasDestroy(handle));
    nvmlShutdown();
    
    return EXIT_SUCCESS;
}
