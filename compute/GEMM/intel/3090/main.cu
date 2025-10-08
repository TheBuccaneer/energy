// main.cu - CUDA GEMM Energy Benchmark for RTX 3090
// Compile: nvcc -O3 -std=c++17 -o gemm_bench main.cu -lcublas -lnvidia-ml
// Usage: ./gemm_bench

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
#include <random>
#include <algorithm>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>

// ============================================================================
// Configuration - RTX 3090 only (hardcoded)
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    MAX_BATCH_SIZE   = 200000;
constexpr int    MACRO_REPEATS    = 5;

// GEMM sizes only
static const int GEMM_SIZES[] = {
    64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896,
    1024, 1152, 1280, 1408, 1536
};
static const int NUM_SIZES = sizeof(GEMM_SIZES) / sizeof(GEMM_SIZES[0]);
static const int MAX_SIZE = *std::max_element(std::begin(GEMM_SIZES), std::end(GEMM_SIZES));

static const char* OUTPUT_FILE = "data/raw/energy_benchmark_rtx3090.csv";

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
        fprintf(stderr, "cuBLAS Error at %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
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

std::string getHostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return std::string(hostname);
    }
    return "unknown";
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
// NVML Helper Functions
// ============================================================================

struct GPUTelemetry {
    unsigned int pcie_gen;
    unsigned int pcie_width;
    unsigned int sm_clock;
    unsigned int mem_clock;
    unsigned int temp;
};

std::string getGPUName(nvmlDevice_t device) {
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    CHECK_NVML(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
    return std::string(name);
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
    return telem;
}

// ============================================================================
// Auto-Batch Determination
// ============================================================================

struct BatchResult {
    int batches;
    bool below_target;
};

BatchResult determineBatchSize(cublasHandle_t handle, float* d_A, float* d_B, float* d_C,
                               int n, int lda, float target_seconds, cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int batch = 1;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    while (batch <= MAX_BATCH_SIZE) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        
        for (int b = 0; b < batch; b++) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                     n, n, n, &alpha, d_B, lda, d_A, lda, 
                                     &beta, d_C, lda));
        }
        
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float elapsed = ms / 1000.0f;
        
        if (elapsed >= target_seconds) {
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            return {batch, false};  // Target reached
        }
        
        if (batch >= MAX_BATCH_SIZE) {
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            return {batch, true};  // Below target (hit max batch limit)
        }
        
        batch = std::min(batch * 2, MAX_BATCH_SIZE);
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return {batch, false};
}

// ============================================================================
// CSV Output Functions
// ============================================================================

void writeCSVHeader(std::ofstream& file) {
    file << "timestamp,host,gpu_name,matrix_size,mode,batches,"
         << "seconds_target,seconds_gpu,seconds_wall,"
         << "energy_j,avg_power_w,below_target,"
         << "pcie_gen_current,pcie_width_current,"
         << "clocks_sm_mhz,clocks_mem_mhz,temp_c\n";
}

void writeCSVRow(std::ofstream& file, const std::string& host, 
                 const std::string& gpu_name, int n, int batches,
                 float gpu_time_s, float wall_time_s,
                 double energy_j, double avg_power_w, bool below_target,
                 const GPUTelemetry& telem) {
    file << getTimestamp() << ","
         << host << ","
         << gpu_name << ","
         << n << ","
         << "e2e" << ","
         << batches << ","
         << TARGET_RUNTIME_S << ","
         << std::fixed << std::setprecision(4) << gpu_time_s << ","
         << wall_time_s << ","
         << std::setprecision(3) << energy_j << ","
         << std::setprecision(1) << avg_power_w << ","
         << (below_target ? 1 : 0) << ","
         << telem.pcie_gen << ","
         << telem.pcie_width << ","
         << telem.sm_clock << ","
         << telem.mem_clock << ","
         << telem.temp << "\n";
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main() {
    // Initialize CUDA device 0
    CHECK_CUDA(cudaSetDevice(0));
    
    // Initialize NVML
    CHECK_NVML(nvmlInit());
    nvmlDevice_t nvml_device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &nvml_device));
    
    std::string gpu_name = getGPUName(nvml_device);
    std::string hostname = getHostname();
    
    std::cout << "========================================\n";
    std::cout << "CUDA GEMM Energy Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "System:         " << hostname << "\n";
    std::cout << "GPU:            " << gpu_name << "\n";
    std::cout << "Target runtime: " << TARGET_RUNTIME_S << "s\n";
    std::cout << "Macro repeats:  " << MACRO_REPEATS << "\n";
    std::cout << "Matrix sizes:   " << NUM_SIZES << " sizes (64-1536)\n";
    std::cout << "Output:         " << OUTPUT_FILE << "\n";
    std::cout << "========================================\n\n";
    
    // Create cuBLAS handle and disable TF32
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));  // TF32 OFF
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    
    // Allocate pinned host memory for maximum size (reuse for all sizes)
    const size_t max_elements = MAX_SIZE * MAX_SIZE;
    const size_t max_bytes = max_elements * sizeof(float);
    
    float *h_A, *h_B, *h_C;
    CHECK_CUDA(cudaMallocHost(&h_A, max_bytes));
    CHECK_CUDA(cudaMallocHost(&h_B, max_bytes));
    CHECK_CUDA(cudaMallocHost(&h_C, max_bytes));
    
    std::cout << "Allocated pinned host buffers: " 
              << (3 * max_bytes / (1024*1024)) << " MB\n";
    
    // Allocate device memory for maximum size (reuse for all sizes)
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, max_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, max_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, max_bytes));
    
    std::cout << "Allocated device buffers: " 
              << (3 * max_bytes / (1024*1024)) << " MB\n\n";
    
    // Initialize matrices once for maximum size
    std::cout << "Initializing host matrices...\n";
    initializeMatrix(h_A, MAX_SIZE, MAX_SIZE, 42);
    initializeMatrix(h_B, MAX_SIZE, MAX_SIZE, 43);
    
    // Prepare CSV output
    ensureDirectoryExists(OUTPUT_FILE);
    bool write_header = !fileExists(OUTPUT_FILE);
    
    std::ofstream csv_file(OUTPUT_FILE, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Cannot open output file: " << OUTPUT_FILE << "\n";
        return EXIT_FAILURE;
    }
    
    if (write_header) {
        writeCSVHeader(csv_file);
    }
    
    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // ========================================================================
    // Main measurement loop: sweep over all sizes
    // ========================================================================
    
    std::cout << "Starting measurements...\n";
    std::cout << "========================================\n\n";
    
    for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
        int n = GEMM_SIZES[size_idx];
        
        std::cout << "GEMM size " << n << "x" << n << "\n";
        
        // Determine batch size for this matrix size
        std::cout << "  Determining batch size... " << std::flush;
        BatchResult batch_result = determineBatchSize(handle, d_A, d_B, d_C, n, MAX_SIZE, 
                                                      TARGET_RUNTIME_S, stream);
        int batches = batch_result.batches;
        bool below_target_size = batch_result.below_target;
        
        std::cout << "using " << batches << " batches";
        if (below_target_size) {
            std::cout << " (!) below target";
        }
        std::cout << "\n";
        
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
            
            // H2D transfers (pinned, async) - 2D copy for upper-left n×n submatrix
            CHECK_CUDA(cudaMemcpy2DAsync(d_A, dst_pitch,
                                         h_A, src_pitch,
                                         width_in_bytes, height,
                                         cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemcpy2DAsync(d_B, dst_pitch,
                                         h_B, src_pitch,
                                         width_in_bytes, height,
                                         cudaMemcpyHostToDevice, stream));
            
            // GPU kernel timing
            CHECK_CUDA(cudaEventRecord(start_event, stream));
            
            for (int b = 0; b < batches; b++) {
                CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                        n, n, n, &alpha, 
                                        d_B, MAX_SIZE,  // B first (transposed)
                                        d_A, MAX_SIZE,  // then A (transposed)
                                        &beta, 
                                        d_C, MAX_SIZE));  // ldc = MAX_SIZE
            }
            
            CHECK_CUDA(cudaEventRecord(stop_event, stream));
            
            // D2H transfer (2D copy for realistic E2E)
            CHECK_CUDA(cudaMemcpy2DAsync(h_C, src_pitch,
                                         d_C, dst_pitch,
                                         width_in_bytes, height,
                                         cudaMemcpyDeviceToHost, stream));
            
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
            writeCSVRow(csv_file, hostname, gpu_name, n, batches,
                       gpu_time_s, wall_time_s, energy_j, avg_power_w, 
                       below_target, telem);
            csv_file.flush();
            
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
            if (below_target) {
                std::cout << " (!)";
            }
            std::cout << "\n";
        }
        
        std::cout << "\n";
    }
    
    std::cout << "========================================\n";
    std::cout << "Benchmark complete!\n";
    std::cout << "Results saved to: " << OUTPUT_FILE << "\n";
    std::cout << "Total measurements: " << (NUM_SIZES * MACRO_REPEATS) << "\n";
    std::cout << "========================================\n";
    
    // Cleanup
    csv_file.close();
    
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    
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