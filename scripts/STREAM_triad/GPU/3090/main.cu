// main.cu - CUDA STREAM Triad Energy Benchmark for RTX 3090
// Operation: a[i] = b[i] + scalar * c[i]
// Compile: nvcc -O3 -std=c++17 -o stream_bench main.cu -lnvidia-ml
// Usage: ./stream_bench [--test|-t] [--output|-o <path>] [--device|-d <id>]

#include <cuda_runtime.h>
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

// ============================================================================
// Configuration - RTX 3090 only (hardcoded)
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    MAX_BATCH_SIZE   = 250000;
constexpr int    MACRO_REPEATS    = 50;

// STREAM Triad scalar constant
constexpr float SCALAR = 3.0f;

// Problem sizes: vector lengths (matching CPU STREAM exactly)
static const long long PROBLEM_SIZES[] = {
    1000000LL, 2000000LL, 4000000LL, 8000000LL, 16000000LL, 32000000LL,
    64000000LL, 128000000LL, 256000000LL
};
static const int NUM_SIZES = sizeof(PROBLEM_SIZES) / sizeof(PROBLEM_SIZES[0]);
static const long long MAX_N = *std::max_element(std::begin(PROBLEM_SIZES), std::end(PROBLEM_SIZES));

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

void initializeVector(float* vec, long long n, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (long long i = 0; i < n; i++) {
        vec[i] = dist(gen);
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
    unsigned long long throttle_reasons;
};

std::string getGPUName(nvmlDevice_t device) {
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    CHECK_NVML(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
    std::string full_name(name);
    
    const char* prefixes[] = {"NVIDIA GeForce ", "NVIDIA Tesla ", "NVIDIA ", "AMD Radeon ", "AMD "};
    for (const char* prefix : prefixes) {
        size_t pos = full_name.find(prefix);
        if (pos == 0) {
            return full_name.substr(strlen(prefix));
        }
    }
    return full_name;
}

unsigned long long getGPUEnergy(nvmlDevice_t device) {
    unsigned long long energy_mj = 0;
    nvmlReturn_t ret = nvmlDeviceGetTotalEnergyConsumption(device, &energy_mj);
    if (ret != NVML_SUCCESS) {
        return 0;
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
    
    nvmlReturn_t ret = nvmlDeviceGetCurrentClocksThrottleReasons(device, &telem.throttle_reasons);
    if (ret != NVML_SUCCESS) {
        telem.throttle_reasons = 0;
    }
    
    return telem;
}

// ============================================================================
// STREAM Triad CUDA Kernel: a[i] = b[i] + scalar * c[i]
// ============================================================================

__global__ void stream_triad_kernel(float* __restrict__ a, 
                                     const float* __restrict__ b,
                                     const float* __restrict__ c,
                                     float scalar, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = b[idx] + scalar * c[idx];
    }
}

// ============================================================================
// Auto-Batch Determination
// ============================================================================

struct BatchResult {
    int batches;
    bool below_target;
};

BatchResult determineBatchSize(float* d_a, float* d_b, float* d_c,
                               long long n, float target_seconds, cudaStream_t stream) {
    int batch = 1;
    
    // Kernel launch config
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    while (batch <= MAX_BATCH_SIZE) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        
        for (int b = 0; b < batch; b++) {
            stream_triad_kernel<<<numBlocks, blockSize, 0, stream>>>(d_a, d_b, d_c, SCALAR, n);
        }
        
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float elapsed = ms / 1000.0f;
        
        if (elapsed >= target_seconds) {
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            return {batch, false};
        }
        
        if (batch >= MAX_BATCH_SIZE) {
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            return {batch, true};
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
        << "run_id_global,"
        << "run_id_per_size,"
        << "device_name,"
        << "num_threads,"
        << "problem_size,"
        << "batches,"
        << "gpu_e2e_time_s,"
        << "gpu_kernel_time_s,"
        << "wall_time_s,"
        << "total_energy_j,"
        << "energy_per_batch_j,"
        << "energy_per_second_j,"
        << "energy_per_flop_j,"
        << "time_per_gemm_ms_kernel,"
        << "time_per_gemm_ms_e2e,"
        << "flops_total,"
        << "gflops_per_s,"
        << "avg_power_w,"
        << "below_target,"
        << "pcie_gen,"
        << "pcie_width,"
        << "sm_clock_mhz,"
        << "mem_clock_mhz,"
        << "temp_c,"
        << "throttle_reasons\n";
}

void writeCSVRow(std::ofstream& csv, int run_id_global, int run_id_per_size,
                 const std::string& device_name, const std::string& num_threads,
                 long long n, int batches, 
                 float gpu_e2e_time, float gpu_kernel_time, float wall_time, 
                 double total_energy, double avg_power, bool below_target, 
                 const GPUTelemetry& telem) {
    // Calculate derived metrics
    double energy_per_batch = (batches > 0) ? (total_energy / batches) : 0.0;
    double energy_per_second = (wall_time > 0) ? (total_energy / wall_time) : 0.0;
    
    // Calculate FLOPs (for STREAM Triad: 2 * N per operation: mul + add)
    double flops_per_op = 2.0 * n;
    double flops_total = flops_per_op * batches;
    double gflops_per_s = (gpu_kernel_time > 0) ? (flops_total / gpu_kernel_time / 1e9) : 0.0;
    
    double energy_per_flop = (flops_total > 0) ? (total_energy / flops_total) : 0.0;
    
    double time_per_gemm_ms_kernel = (batches > 0) ? (1e3 * gpu_kernel_time / batches) : 0.0;
    double time_per_gemm_ms_e2e = (batches > 0) ? (1e3 * gpu_e2e_time / batches) : 0.0;
    
    csv.imbue(std::locale::classic());
    
    csv << getTimestamp() << ","
        << run_id_global << ","
        << run_id_per_size << ","
        << device_name << ","
        << num_threads << ","
        << n << ","
        << batches << ","
        << std::fixed << std::setprecision(6) << gpu_e2e_time << ","
        << std::fixed << std::setprecision(6) << gpu_kernel_time << ","
        << std::fixed << std::setprecision(6) << wall_time << ","
        << std::fixed << std::setprecision(6) << total_energy << ","
        << std::scientific << std::setprecision(6) << energy_per_batch << ","
        << std::fixed << std::setprecision(6) << energy_per_second << ","
        << std::scientific << std::setprecision(6) << energy_per_flop << ","
        << std::fixed << std::setprecision(6) << time_per_gemm_ms_kernel << ","
        << std::fixed << std::setprecision(6) << time_per_gemm_ms_e2e << ","
        << std::scientific << std::setprecision(6) << flops_total << ","
        << std::fixed << std::setprecision(2) << gflops_per_s << ","
        << std::fixed << std::setprecision(2) << avg_power << ","
        << (below_target ? "t" : "f") << ","
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
                     cudaEvent_t start_kernel, cudaEvent_t stop_kernel,
                     float* d_a, float* d_b, float* d_c,
                     float* h_a, float* h_b, float* h_c,
                     cudaStream_t stream,
                     std::ofstream& csv_file) {
    std::cout << "\n Test mode: 5 rows written! \n";
    
    csv_file.close();
    
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_CUDA(cudaEventDestroy(start_kernel));
    CHECK_CUDA(cudaEventDestroy(stop_kernel));
    
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    
    free(h_a);
    free(h_b);
    free(h_c);
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    nvmlShutdown();
    
    exit(EXIT_SUCCESS);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Parse command line arguments
    bool test_mode = false;
    int total_rows = 0;
    int device_id = 0;
    std::string output_file = "data/raw/STREAM_3090.csv";
    
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--test") || !strcmp(argv[i], "-t")) {
            test_mode = true;
            std::cout << "Test mode enabled\n";
        } else if ((!strcmp(argv[i], "--output") || !strcmp(argv[i], "-o")) && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((!strcmp(argv[i], "--device") || !strcmp(argv[i], "-d")) && i + 1 < argc) {
            device_id = atoi(argv[++i]);
        }
    }
    
    std::cout << "\nCUDA STREAM Triad Energy Benchmark (RTX 3090)\n";
    std::cout << "Operation: a[i] = b[i] + " << SCALAR << " * c[i]\n";
    
    // Initialize CUDA
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " CUDA device(s)\n";
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return EXIT_FAILURE;
    }
    
    if (device_id >= device_count) {
        std::cerr << "Invalid device ID: " << device_id << "\n";
        return EXIT_FAILURE;
    }
    
    CHECK_CUDA(cudaSetDevice(device_id));
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    
    std::cout << "Using device " << device_id << ": " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB\n\n";
    
    // Initialize NVML
    CHECK_NVML(nvmlInit());
    
    nvmlDevice_t nvml_device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(device_id, &nvml_device));
    
    std::string device_name = getGPUName(nvml_device);
    
    std::cout << "Device Name: " << device_name << "\n\n";
    
    // Initialize run ID counters
    int run_id_global = 1;
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Calculate memory requirements for maximum size
    // STREAM Triad needs 3 arrays: a (output), b (input), c (input)
    const size_t max_bytes = MAX_N * sizeof(float);
    
    // Check if we have enough GPU memory
    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    size_t required_device_mem = 3 * max_bytes;
    
    if (required_device_mem > free_mem) {
        std::cerr << "Warning: Not enough GPU memory for largest size.\n";
        std::cerr << "  Required: " << (required_device_mem / (1024*1024)) << " MB\n";
        std::cerr << "  Available: " << (free_mem / (1024*1024)) << " MB\n";
    }
    
    // Allocate host memory (regular malloc, not pinned)
    float *h_a = nullptr, *h_b = nullptr, *h_c = nullptr;
    
    h_a = (float*)malloc(max_bytes);
    h_b = (float*)malloc(max_bytes);
    h_c = (float*)malloc(max_bytes);
    
    if (!h_a || !h_b || !h_c) {
        std::cerr << "Failed to allocate host memory\n";
        free(h_a); free(h_b); free(h_c);
        return EXIT_FAILURE;
    }
    
    std::cout << "Allocated host buffers: " 
              << (3 * max_bytes / (1024*1024)) << " MB\n";
    
    // Allocate device memory
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaError_t alloc_err;
    
    alloc_err = cudaMalloc(&d_a, max_bytes);
    if (alloc_err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for d_a\n";
        cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
        return EXIT_FAILURE;
    }
    
    alloc_err = cudaMalloc(&d_b, max_bytes);
    if (alloc_err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for d_b\n";
        cudaFree(d_a);
        cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
        return EXIT_FAILURE;
    }
    
    alloc_err = cudaMalloc(&d_c, max_bytes);
    if (alloc_err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for d_c\n";
        cudaFree(d_a); cudaFree(d_b);
        cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
        return EXIT_FAILURE;
    }
    
    std::cout << "Allocated device buffers: " 
              << (3 * max_bytes / (1024*1024)) << " MB\n\n";
    
    // Initialize vectors (seed=42 for b, seed=43 for c, like CPU)
    std::cout << "Initializing host vectors (seed=42 for b, seed=43 for c)\n";
    initializeVector(h_b, MAX_N, 42);
    initializeVector(h_c, MAX_N, 43);
    std::fill(h_a, h_a + MAX_N, 0.0f);
    
    // Prepare CSV output
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
    
    // Kernel launch config
    int blockSize = 256;
    
    // ========================================================================
    // Main measurement loop
    // ========================================================================
    
    std::cout << "Starting measurements...\n\n";
    
    for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
        long long n = PROBLEM_SIZES[size_idx];
        int run_id_per_size = 1;
        
        std::cout << "STREAM Triad size " << n << " elements\n";
        
        size_t size_bytes = n * sizeof(float);
        if (size_bytes > max_bytes) {
            std::cout << "  SKIPPING: Size exceeds allocated buffer\n\n";
            continue;
        }
        
        int numBlocks = (n + blockSize - 1) / blockSize;
        
        // Determine batch size
        std::cout << "  Determine batch size..." << std::flush;
        
        // Pre-transfer data to device for batch determination
        CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, size_bytes, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_c, h_c, size_bytes, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        BatchResult batch_result = determineBatchSize(d_a, d_b, d_c, n, TARGET_RUNTIME_S, stream);
        int batches = batch_result.batches;
        bool below_target_size = batch_result.below_target;
        
        std::cout << " using " << batches << " batches";
        if (below_target_size) {
            std::cout << " (!) below target";
        }
        std::cout << "\n";
        
        // Run MACRO_REPEATS measurements
        for (int rep = 0; rep < MACRO_REPEATS; rep++) {
            // ================================================================
            // E2E Measurement Start
            // ================================================================
            
            auto wall_start = std::chrono::steady_clock::now();
            unsigned long long energy_before = getGPUEnergy(nvml_device);
            
            // GPU E2E timing starts HERE (before H2D)
            CHECK_CUDA(cudaEventRecord(start_event, stream));
            
            // H2D transfers (b and c only; a is output)
            CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, size_bytes, cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(d_c, h_c, size_bytes, cudaMemcpyHostToDevice, stream));
            
            // Kernel timing starts HERE (after H2D)
            CHECK_CUDA(cudaEventRecord(start_kernel, stream));
            
            // GPU kernel (batches)
            for (int b = 0; b < batches; b++) {
                stream_triad_kernel<<<numBlocks, blockSize, 0, stream>>>(d_a, d_b, d_c, SCALAR, n);
            }
            
            // Kernel timing ends HERE (before D2H)
            CHECK_CUDA(cudaEventRecord(stop_kernel, stream));
            
            // D2H transfer (a only, the output)
            CHECK_CUDA(cudaMemcpyAsync(h_a, d_a, size_bytes, cudaMemcpyDeviceToHost, stream));
            
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
            
            // Calculate energy and power
            double energy_j = 0.0;
            double avg_power_w = 0.0;
            
            if (energy_after > energy_before) {
                unsigned long long energy_mj = energy_after - energy_before;
                energy_j = energy_mj / 1000.0;
                avg_power_w = energy_j / wall_time_s;
            }
            
            // Check if this run is below target (based on E2E time, like GEMM)
            bool below_target = (gpu_time_s < TARGET_RUNTIME_S);
            
            // Get GPU telemetry
            GPUTelemetry telem = getGPUTelemetry(nvml_device);
            
            // Write to CSV
            writeCSVRow(csv_file, run_id_global, run_id_per_size, device_name, "", n, batches,
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
                               d_a, d_b, d_c, h_a, h_b, h_c, stream, csv_file);
            }
            
            // Console progress
            char check = below_target ? '!' : '+';
            std::cout << "  " << check << " Run " << (rep + 1) << "/" 
                     << MACRO_REPEATS << ": "
                     << std::fixed << std::setprecision(3) 
                     << kernel_time_s << "s (kernel) "
                     << gpu_time_s << "s (e2e) "
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
        
        // Cooling pause (except after last size)
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
    
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    
    free(h_a);
    free(h_b);
    free(h_c);
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    nvmlShutdown();
    
    return EXIT_SUCCESS;
}
