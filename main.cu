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
constexpr int    BATCH_COUNT      = 8;  // Hardcoded strided-batched count

// GEMM sizes only (2^x steps: 64 to 16384)
static const int GEMM_SIZES[] = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
};
static const int NUM_SIZES = sizeof(GEMM_SIZES) / sizeof(GEMM_SIZES[0]);
static const int MAX_SIZE = *std::max_element(std::begin(GEMM_SIZES), std::end(GEMM_SIZES));

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
        
        // RAM guard (check against total GPU memory)
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        if (needed_bytes > prop.totalGlobalMem / 2) {  // Safety margin: use max 50% of GPU RAM
            fprintf(stderr, "ERROR: Allocation exceeds 50%% GPU memory: %zu bytes (available: %zu)\n",
                    needed_bytes, prop.totalGlobalMem);
            exit(EXIT_FAILURE);
        }
        
        // Free old allocation if exists
        if (*d_C != nullptr) {
            CHECK_CUDA(cudaFree(*d_C));
        }
        
        // Allocate new buffer
        CHECK_CUDA(cudaMalloc((void**)d_C, needed_bytes));
        *cap_bytes = needed_bytes;
        
        fprintf(stderr, "INFO: Reallocated d_C to %zu bytes (%.2f MB)\n",
                needed_bytes, needed_bytes / (1024.0 * 1024.0));
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
// Auto-Batch Determination (Strided-Batched)
// ============================================================================

struct BatchResult {
    int batches;
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
    
    // Measure E2E time for single call (with H2D + compute + D2H)
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
    
    CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                           n, n, n, &alpha,
                                           d_A, lda, strideA,
                                           d_B, lda, strideB,
                                           &beta,
                                           *d_C, lda, strideC,
                                           batch_count));
    
    CHECK_CUDA(cudaMemcpy2DAsync(h_C, src_pitch, *d_C, dst_pitch,
                                 width_in_bytes, height,
                                 cudaMemcpyDeviceToHost, stream));
    
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float t_call = ms / 1000.0f;
    
    batches = (int)std::ceil(target_seconds / t_call);
    batches = std::max(1, batches);
    
    if (batches > MAX_BATCH_SIZE) {
        batches = MAX_BATCH_SIZE;
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return {batches};
}

// ============================================================================
// CSV Writing Functions
// ============================================================================

void writeCSVHeader(std::ofstream& csv) {
    csv << "timestamp,run_id_global,run_id_per_size,device_name,num_threads,problem_size,batches,"
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
        << std::scientific << std::setprecision(6)
        << gpu_time_s << ","
        << kernel_time_s << ","
        << wall_time_s << ","
        << energy_j << ","
        << energy_per_batch << ","
        << energy_per_second << ","
        << energy_per_flop << ","
        << time_per_gemm_ms_kernel << ","
        << time_per_gemm_ms_e2e << ","
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
    
    exit(EXIT_SUCCESS);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    bool test_mode = false;
    std::string output_file = "data/GPU/gemm_bench_RTX_3090.csv";
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test" || arg == "-t") {
            test_mode = true;
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            }
        }
    }
    
    // Print configuration
    std::cout << "\n=== CUDA GEMM Benchmark (Strided-Batched) ===\n";
    std::cout << "Batch Count (per call): " << BATCH_COUNT << "\n";
    std::cout << "Target Runtime: " << TARGET_RUNTIME_S << " seconds\n";
    std::cout << "Macro Repeats: " << MACRO_REPEATS << "\n";
    std::cout << "Output: " << output_file << "\n";
    if (test_mode) {
        std::cout << "TEST MODE: Will stop after 5 measurements\n";
    }
    std::cout << "\n";
    
    // Initialize CUDA
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found\n";
        return EXIT_FAILURE;
    }
    
    CHECK_CUDA(cudaSetDevice(0));
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n\n";
    
    // Initialize NVML
    CHECK_NVML(nvmlInit());
    nvmlDevice_t nvml_device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &nvml_device));
    std::string device_name = getGPUName(nvml_device);
    
    // Create cuBLAS handle and stream
    cublasHandle_t handle;
    cudaStream_t stream;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    
    // Set PEDANTIC_MATH (disable TF32)
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    
    // Allocate host memory (pinned) - maximum size
    float *h_A, *h_B, *h_C;
    size_t max_bytes = size_t(MAX_SIZE) * size_t(MAX_SIZE) * sizeof(float);
    CHECK_CUDA(cudaMallocHost((void**)&h_A, max_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_B, max_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_C, max_bytes));
    
    // Initialize matrices (once)
    initializeMatrix(h_A, MAX_SIZE, MAX_SIZE, 42);
    initializeMatrix(h_B, MAX_SIZE, MAX_SIZE, 123);
    
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
    
    for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
        int n = GEMM_SIZES[size_idx];
        int run_id_per_size = 1;
        
        std::cout << "GEMM size " << n << "x" << n << " (batch_count=" << BATCH_COUNT << ")\n";
        
        // Strides for strided-batched
        long long strideA = 0;  // Broadcast A
        long long strideB = 0;  // Broadcast B
        long long strideC = (long long)MAX_SIZE * (long long)n;
        
        // Determine batch size for this matrix size
        std::cout << "  Determine batch size..." << std::flush;
        BatchResult batch_result = determineBatchSize(handle, d_A, d_B, &d_C,
                                                      h_A, h_B, h_C,
                                                      &d_C_capacity_bytes,
                                                      n, MAX_SIZE, BATCH_COUNT,
                                                      TARGET_RUNTIME_S, stream);
        int batches = batch_result.batches;
        
        std::cout << " using " << batches << " batches\n";
        
        // Run MACRO_REPEATS measurements
        for (int rep = 0; rep < MACRO_REPEATS; rep++) {
            // ================================================================
            // E2E Measurement Start
            // ================================================================
            
            // Prepare 2D copy parameters for n×n submatrix
            size_t src_pitch = size_t(MAX_SIZE) * sizeof(float);
            size_t dst_pitch = size_t(MAX_SIZE) * sizeof(float);
            size_t width_in_bytes = size_t(n) * sizeof(float);
            size_t height = size_t(n);
            
            auto wall_start = std::chrono::steady_clock::now();
            unsigned long long energy_before = getGPUEnergy(nvml_device);
            
            // GPU E2E timing starts HERE (before H2D)
            CHECK_CUDA(cudaEventRecord(start_event, stream));
            
            // H2D transfers (pinned, async) - 2D copy for upper-left n×n submatrix
            CHECK_CUDA(cudaMemcpy2DAsync(d_A, dst_pitch,
                                         h_A, src_pitch,
                                         width_in_bytes, height,
                                         cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemcpy2DAsync(d_B, dst_pitch,
                                         h_B, src_pitch,
                                         width_in_bytes, height,
                                         cudaMemcpyHostToDevice, stream));
            
            // Kernel timing starts HERE (after H2D)
            CHECK_CUDA(cudaEventRecord(start_kernel, stream));
            
            // GPU kernel (strided-batched calls)
            for (int b = 0; b < batches; b++) {
                CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                                       n, n, n, &alpha,
                                                       d_B, MAX_SIZE, strideB,
                                                       d_A, MAX_SIZE, strideA,
                                                       &beta,
                                                       d_C, MAX_SIZE, strideC,
                                                       BATCH_COUNT));
            }
            
            // Kernel timing ends HERE (before D2H)
            CHECK_CUDA(cudaEventRecord(stop_kernel, stream));
            
            // D2H transfer - only first instance C[0] (offset 0)
            CHECK_CUDA(cudaMemcpy2DAsync(h_C, src_pitch,
                                         d_C, dst_pitch,  // Offset 0 = first instance
                                         width_in_bytes, height,
                                         cudaMemcpyDeviceToHost, stream));
            
            // GPU E2E timing ends HERE (after D2H)
            CHECK_CUDA(cudaEventRecord(stop_event, stream));
            
            // Synchronize and measure
            CHECK_CUDA(cudaDeviceSynchronize());
            
            unsigned long long energy_after = getGPUEnergy(nvml_device);
            auto wall_end = std::chrono::steady_clock::now();
            
            // ================================================================
            // E2E Measurement End
            // ================================================================
            
            // Calculate timings
            float gpu_ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start_event, stop_event));
            float gpu_time_s = gpu_ms / 1000.0f;
            
            float kernel_ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
            float kernel_time_s = kernel_ms / 1000.0f;
            
            std::chrono::duration<double> wall_duration = wall_end - wall_start;
            float wall_time_s = wall_duration.count();
            
            // Calculate energy (convert mJ to J) and power
            double energy_j = 0.0;
            double avg_power_w = 0.0;
            
            if (energy_after > energy_before) {
                unsigned long long energy_mj = energy_after - energy_before;
                energy_j = energy_mj / 1000.0;  // mJ to J
                avg_power_w = energy_j / wall_time_s;
            }
            
            // Check if this run is below target
            bool below_target = (gpu_time_s < TARGET_RUNTIME_S);
            
            // Get GPU telemetry
            GPUTelemetry telem = getGPUTelemetry(nvml_device);
            
            // Write to CSV
            writeCSVRow(csv_file, run_id_global, run_id_per_size, device_name, "", n, batches, BATCH_COUNT,
                       gpu_time_s, kernel_time_s, wall_time_s, energy_j, avg_power_w, 
                       below_target, telem);
            csv_file.flush();
            
            // Increment counters
            run_id_global++;
            run_id_per_size++;
            
            // Test mode check
            ++total_rows;
            if (test_mode && total_rows >= 5) {
                cleanup_and_exit(start_event, stop_event, start_kernel, stop_kernel,
                               d_A, d_B, d_C, h_A, h_B, h_C, stream, handle, csv_file);
            }
            
            // Console progress
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
        }
        
        // Cooling pause after all 50 runs (except after last problem size)
        if (size_idx < NUM_SIZES - 1) {
            std::cout << "  Cooling down for 30 seconds\n";
            std::this_thread::sleep_for(std::chrono::seconds(30));
        }
        
        std::cout << "\n";
    }
    
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
