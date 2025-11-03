// main.cu - CUDA GEMM Energy Benchmark for RTX 3090
// Compile: nvcc -O3 -std=c++17 -o gemm_bench main.cu -lcublas -lnvidia-ml
// Usage: ./gemm_bench [--test|-t]

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
#include <cstdlib>  // NEW: for setenv()

// ============================================================================
// Configuration - RTX 3090 only (hardcoded)
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    MAX_BATCH_SIZE   = 250000;
constexpr int    MACRO_REPEATS    = 50;  // CHANGED: from 5 to 50

// GEMM sizes only (2^x steps: 64 to 16384)
static const int GEMM_SIZES[] = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
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
    unsigned long long throttle_reasons;  // NVML bitmask: 0=none, 1=thermal, 2=power, etc.
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
    
    // Get throttle reasons (bitmask)
    nvmlReturn_t ret = nvmlDeviceGetCurrentClocksThrottleReasons(device, &telem.throttle_reasons);
    if (ret != NVML_SUCCESS) {
        telem.throttle_reasons = 0;  // Not supported or error
    }
    
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
        // Note: During batch determination, we measure kernel-only time
        // In actual measurements, events will include H2D/D2H transfers
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
// CSV Writing Functions
// ============================================================================

void writeCSVHeader(std::ofstream& csv) {
    csv << "timestamp,"
        << "hostname,"
        << "gpu_name,"
        << "problem_size,"
        << "batches,"
        << "gpu_time_s,"
        << "wall_time_s,"
        << "energy_j,"
        << "avg_power_w,"
        << "below_target,"
        << "pcie_gen,"
        << "pcie_width,"
        << "sm_clock_mhz,"
        << "mem_clock_mhz,"
        << "temp_c,"
        << "throttle_reasons\n";
}

void writeCSVRow(std::ofstream& csv, const std::string& hostname, const std::string& gpu_name,
                 int n, int batches, float gpu_time, float wall_time, 
                 double energy, double power, bool below_target, 
                 const GPUTelemetry& telem) {
    csv << getTimestamp() << ","
        << hostname << ","
        << gpu_name << ","
        << n << ","
        << batches << ","
        << std::fixed << std::setprecision(6) << gpu_time << ","
        << std::fixed << std::setprecision(6) << wall_time << ","
        << std::fixed << std::setprecision(6) << energy << ","
        << std::fixed << std::setprecision(2) << power << ","
        << (below_target ? "true" : "false") << ","
        << telem.pcie_gen << ","
        << telem.pcie_width << ","
        << telem.sm_clock << ","
        << telem.mem_clock << ","
        << telem.temp << ","
        << "0x" << std::hex << telem.throttle_reasons << std::dec << "\n";
}

// ============================================================================
// Cleanup Helper (for test mode early exit)
// ============================================================================

void cleanup_and_exit(cudaEvent_t start_event, cudaEvent_t stop_event,
                     float* d_A, float* d_B, float* d_C,
                     float* h_A, float* h_B, float* h_C,
                     cudaStream_t stream, cublasHandle_t handle,
                     std::ofstream& csv_file) {
    std::cout << "\n========================================\n";
    std::cout << "Test mode: 5 rows written, exiting...\n";
    std::cout << "========================================\n";
    
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
    
    exit(EXIT_SUCCESS);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {  // CHANGED: signature to accept args
    // NEW: Parse command line arguments for test mode
    bool test_mode = false;
    int total_rows = 0;
    
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--test") || !strcmp(argv[i], "-t")) {
            test_mode = true;
            std::cout << "Test mode enabled: will write 5 rows and exit\n";
        }
    }
    
    std::cout << "========================================\n";
    std::cout << "CUDA GEMM Energy Benchmark (RTX 3090)\n";
    std::cout << "========================================\n\n";
    
    // NEW: Disable TF32 globally (best-effort)
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
    
    // Initialize CUDA
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " CUDA device(s)\n";
    
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found!\n";
        return EXIT_FAILURE;
    }
    
    // Use first device (RTX 3090)
    CHECK_CUDA(cudaSetDevice(0));
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "Using device: " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB\n\n";
    
    // Initialize NVML
    CHECK_NVML(nvmlInit());
    
    nvmlDevice_t nvml_device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &nvml_device));
    
    std::string hostname = getHostname();
    std::string gpu_name = getGPUName(nvml_device);
    
    std::cout << "Hostname: " << hostname << "\n";
    std::cout << "GPU Name: " << gpu_name << "\n\n";
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // NEW: Disable TF32 in cuBLAS (strict FP32)
#ifdef CUBLAS_PEDANTIC_MATH
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    std::cout << "TF32 disabled (pedantic FP32)\n";  // NEW: one-time stdout
#else
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    std::cout << "TF32 disabled (default math mode)\n";  // NEW: one-time stdout
#endif
    
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
            
            // GPU kernel (batches)
            for (int b = 0; b < batches; b++) {
                CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                        n, n, n, &alpha, 
                                        d_B, MAX_SIZE,  // B first (transposed)
                                        d_A, MAX_SIZE,  // then A (transposed)
                                        &beta, 
                                        d_C, MAX_SIZE));  // ldc = MAX_SIZE
            }
            
            // D2H transfer (2D copy for realistic E2E)
            CHECK_CUDA(cudaMemcpy2DAsync(h_C, src_pitch,
                                         d_C, dst_pitch,
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
            csv_file.flush();  // UNCHANGED: already present
            
            // NEW: Test mode check
            ++total_rows;
            if (test_mode && total_rows >= 5) {
                cleanup_and_exit(start_event, stop_event, d_A, d_B, d_C,
                               h_A, h_B, h_C, stream, handle, csv_file);
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
