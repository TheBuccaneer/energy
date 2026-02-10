// ============================================================================
// GPU Reduction (DOT with ones) Energy Benchmark with CUDA
// Variante A: Each of 50 runs per config writes its own CSV row (no aggregation)
// ============================================================================
// Compile: nvcc -O3 -std=c++17 -o reduction_gpu main.cu -lcublas -lnvidia-ml
// Usage: ./reduction_gpu [--test|-t] [--output|-o <path>] [--device|-d <index>]

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
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <cmath>

// ============================================================================
// Configuration
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    MAX_BATCH_SIZE   = 1000000;
constexpr int    REPEATS          = 50;  // 50 measurements per configuration

// Problem sizes (vector lengths in number of elements)
static const std::vector<int> PROBLEM_SIZES = {
    1000000, 2000000, 4000000, 8000000, 16000000, 32000000,
    64000000, 128000000, 256000000
};

static const int MAX_N = 256000000;

static const char* DEFAULT_OUTPUT_FILE = "reduction_gpu.csv";

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

void initializeVector(float* vec, int n, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; i++) {
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
    
    // Remove common prefixes for cleaner output
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
        return 0;  // Not supported
    }
    return energy_mj;
}

GPUTelemetry getGPUTelemetry(nvmlDevice_t device) {
    GPUTelemetry telem = {};
    
    nvmlDeviceGetCurrPcieLinkGeneration(device, &telem.pcie_gen);
    nvmlDeviceGetCurrPcieLinkWidth(device, &telem.pcie_width);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &telem.sm_clock);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &telem.mem_clock);
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &telem.temp);
    nvmlDeviceGetCurrentClocksThrottleReasons(device, &telem.throttle_reasons);
    
    return telem;
}

// ============================================================================
// CSV Output Functions (CSV_COLUMNS.md format with 26 columns)
// ============================================================================

void writeCSVHeader(std::ofstream& file) {
    file << "timestamp,run_id_global,run_id_per_size,device_name,num_threads,problem_size,"
         << "batches,gpu_e2e_time_s,gpu_kernel_time_s,wall_time_s,total_energy_j,"
         << "energy_per_batch_j,energy_per_second_j,energy_per_flop_j,"
         << "time_per_gemm_ms_kernel,time_per_gemm_ms_e2e,flops_total,gflops_per_s,"
         << "avg_power_w,below_target,pcie_gen,pcie_width,sm_clock_mhz,mem_clock_mhz,"
         << "temp_c,throttle_reasons\n";
}

void writeCSVRow(std::ofstream& file, int run_id_global, int run_id_per_size,
                const std::string& device_name, int problem_size, int batches,
                float gpu_e2e_time_s, float gpu_kernel_time_s, float wall_time_s,
                double total_energy_j, bool below_target, const GPUTelemetry& telem) {
    // Compute derived metrics
    // For reduction: 1 FLOP per element (addition)
    double total_instances = static_cast<double>(batches);
    double flops_total = static_cast<double>(problem_size) * total_instances;
    
    double energy_per_batch_j = (total_energy_j >= 0 && batches > 0) ? 
        (total_energy_j / batches) : -1.0;
    double energy_per_second_j = (total_energy_j >= 0 && wall_time_s > 0) ? 
        (total_energy_j / wall_time_s) : -1.0;
    double energy_per_flop_j = (total_energy_j >= 0 && flops_total > 0) ? 
        (total_energy_j / flops_total) : -1.0;
    
    double time_per_gemm_ms_kernel = (batches > 0) ? 
        (1e3 * gpu_kernel_time_s / batches) : 0.0;
    double time_per_gemm_ms_e2e = (batches > 0) ? 
        (1e3 * gpu_e2e_time_s / batches) : 0.0;
    
    double gflops_per_s = (gpu_kernel_time_s > 0) ? 
        (flops_total / gpu_kernel_time_s / 1e9) : 0.0;
    
    double avg_power_w = (total_energy_j >= 0 && wall_time_s > 0) ? 
        (total_energy_j / wall_time_s) : -1.0;
    
    // Write CSV row
    file << getTimestamp() << ","
         << run_id_global << ","
         << run_id_per_size << ","
         << device_name << ","
         << "," // num_threads (empty for GPU)
         << problem_size << ","
         << batches << ","
         << std::fixed << std::setprecision(6)
         << gpu_e2e_time_s << ","
         << gpu_kernel_time_s << ","
         << wall_time_s << ","
         << total_energy_j << ","
         << std::scientific << std::setprecision(6)
         << energy_per_batch_j << ","
         << std::fixed << std::setprecision(6)
         << energy_per_second_j << ","
         << std::scientific << std::setprecision(6)
         << energy_per_flop_j << ","
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
// Adaptive Batch Size Determination (Iterative, like CPU version)
// ============================================================================

int determineBatchSize(cublasHandle_t handle, float* d_x, float* d_ones,
                       float* d_result, int n, cudaStream_t stream) {
    const double MIN_TIME = TARGET_RUNTIME_S;        // Minimum for stable measurement
    const double MAX_TIME = TARGET_RUNTIME_S * 3.0;  // Maximum to avoid thermal drift
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // 1) WARMUP (not measured!) - critical for CUDA/cuBLAS initialization
    for (int w = 0; w < 5; w++) {
        CHECK_CUBLAS(cublasSdot(handle, n, d_x, 1, d_ones, 1, d_result));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 2) Adaptive starting point based on problem size
    int batches;
    if (n >= 128000000) {
        batches = 50;
    } else if (n >= 64000000) {
        batches = 100;
    } else if (n >= 16000000) {
        batches = 500;
    } else if (n >= 4000000) {
        batches = 2000;
    } else {
        batches = 5000;
    }
    
    // 3) Iteratively scale until we're in the target window
    while (batches <= MAX_BATCH_SIZE) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        
        for (int b = 0; b < batches; b++) {
            CHECK_CUBLAS(cublasSdot(handle, n, d_x, 1, d_ones, 1, d_result));
        }
        
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float elapsed_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double measured_time = elapsed_ms / 1000.0;
        
        // Guard against near-zero measurements (timer granularity issues)
        // Just scale up aggressively and retry
        if (measured_time < 1e-6) {
            batches = std::min(batches * 10, MAX_BATCH_SIZE);
            continue;
        }
        
        // If we're in the acceptable range, use this batch size
        if (measured_time >= MIN_TIME) {
            // If we're under max_time, perfect
            if (measured_time <= MAX_TIME) {
                CHECK_CUDA(cudaEventDestroy(start));
                CHECK_CUDA(cudaEventDestroy(stop));
                return batches;
            }
            // If we exceeded max_time, scale back proportionally
            int scaled_batches = static_cast<int>(batches * (MIN_TIME / measured_time));
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            return std::max(1, scaled_batches);
        }
        
        // Scale up: estimate how many batches needed for min_time
        double time_per_batch = measured_time / batches;
        int needed_batches = static_cast<int>(std::ceil((MIN_TIME * 1.05) / time_per_batch));
        
        // Make sure we make progress (at least 2x)
        if (needed_batches <= batches) {
            needed_batches = batches * 2;
        }
        
        batches = std::min(needed_batches, MAX_BATCH_SIZE);
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Hit max batch size without reaching min_time - accept it
    return MAX_BATCH_SIZE;
}

// ============================================================================
// Main Benchmark Function
// ============================================================================

int run_benchmark(const char* output_file, bool test_mode, int device_index) {
    // Set CUDA device
    CHECK_CUDA(cudaSetDevice(device_index));
    
    // Initialize NVML
    CHECK_NVML(nvmlInit());
    nvmlDevice_t nvml_device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(device_index, &nvml_device));
    
    std::string device_name = getGPUName(nvml_device);
    
    std::cout << "=== GPU Reduction Benchmark (Adaptive Batching) ===" << std::endl;
    std::cout << "Device: " << device_name << " (index " << device_index << ")" << std::endl;
    std::cout << "Target runtime per measurement: " << TARGET_RUNTIME_S << "s" << std::endl;
    std::cout << "Repetitions per configuration: " << REPEATS << " (each writes a CSV row)" << std::endl;
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    
    // Allocate device memory for maximum problem size
    std::cout << "\nAllocating device memory (max size: " << MAX_N 
              << " elements = " << (MAX_N * sizeof(float) / (1024.0 * 1024.0)) 
              << " MB)..." << std::endl;
    
    float* d_x = nullptr;
    float* d_ones = nullptr;
    float* d_result = nullptr;
    
    CHECK_CUDA(cudaMalloc(&d_x, MAX_N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ones, MAX_N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    
    // Allocate and initialize host memory
    float* h_x = nullptr;
    float* h_ones = nullptr;
    float* h_result = nullptr;
    
    CHECK_CUDA(cudaMallocHost(&h_x, MAX_N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_ones, MAX_N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_result, sizeof(float)));
    
    std::cout << "Initializing vectors..." << std::endl;
    initializeVector(h_x, MAX_N, 42);
    std::fill(h_ones, h_ones + MAX_N, 1.0f);
    
    // Copy to device (one-time for initial setup)
    std::cout << "Copying to device..." << std::endl;
    CHECK_CUDA(cudaMemcpy(d_x, h_x, MAX_N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ones, h_ones, MAX_N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Open CSV file
    ensureDirectoryExists(output_file);
    bool write_header = !fileExists(output_file);
    std::ofstream csv_file(output_file, std::ios::app);
    
    if (!csv_file.is_open()) {
        std::cerr << "Error: Cannot open file: " << output_file << std::endl;
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
    
    // Determine which configurations to test
    std::vector<int> sizes_to_test;
    if (test_mode) {
        // Test mode: only first 2 sizes
        sizes_to_test.push_back(PROBLEM_SIZES[0]);
        sizes_to_test.push_back(PROBLEM_SIZES[1]);
        std::cout << "\n*** TEST MODE: Only " << sizes_to_test.size() << " sizes ***" << std::endl;
    } else {
        // Full benchmark
        sizes_to_test = PROBLEM_SIZES;
    }
    
    int run_id_global = 0;
    
    std::cout << "\nStarting measurements...\n" << std::endl;
    
    // Main measurement loop: outer loop over problem sizes
    for (int n : sizes_to_test) {
        std::cout << "========================================" << std::endl;
        std::cout << "Problem size: " << n << " elements" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Adaptive batch size determination (iterative, like CPU version)
        std::cout << "  Determining batch size... " << std::flush;
        int batches = determineBatchSize(handle, d_x, d_ones, d_result, n, stream);
        std::cout << "using " << batches << " batches" << std::endl;
        
        int run_id_per_size = 0;
        
        // Run REPEATS measurements with this batch size
        for (int rep = 0; rep < REPEATS; rep++) {
            run_id_global++;
            run_id_per_size++;
            
            // Wall clock start
            auto wall_start = std::chrono::steady_clock::now();
            
            // Read energy before
            unsigned long long energy_before = getGPUEnergy(nvml_device);
            
            // E2E timing: includes H2D + kernel + D2H
            CHECK_CUDA(cudaEventRecord(start_event, stream));
            
            // H2D copy per run (for consistency with E2E measurement)
            CHECK_CUDA(cudaMemcpyAsync(d_x, h_x, n * sizeof(float), 
                                      cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(d_ones, h_ones, n * sizeof(float), 
                                      cudaMemcpyHostToDevice, stream));
            
            // Kernel timing: only the reduction operations
            CHECK_CUDA(cudaEventRecord(start_kernel, stream));
            
            for (int b = 0; b < batches; b++) {
                CHECK_CUBLAS(cublasSdot(handle, n, d_x, 1, d_ones, 1, d_result));
            }
            
            CHECK_CUDA(cudaEventRecord(stop_kernel, stream));
            
            // D2H copy of result
            CHECK_CUDA(cudaMemcpyAsync(h_result, d_result, sizeof(float), 
                                      cudaMemcpyDeviceToHost, stream));
            
            CHECK_CUDA(cudaEventRecord(stop_event, stream));
            
            // Synchronize
            CHECK_CUDA(cudaDeviceSynchronize());
            
            // Read energy after
            unsigned long long energy_after = getGPUEnergy(nvml_device);
            
            // Wall clock end
            auto wall_end = std::chrono::steady_clock::now();
            
            // Compute times
            float e2e_ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&e2e_ms, start_event, stop_event));
            float gpu_e2e_time_s = e2e_ms / 1000.0f;
            
            float kernel_ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
            float gpu_kernel_time_s = kernel_ms / 1000.0f;
            
            std::chrono::duration<double> wall_duration = wall_end - wall_start;
            float wall_time_s = wall_duration.count();
            
            // Compute energy
            double total_energy_j = -1.0;
            if (energy_after > energy_before) {
                unsigned long long energy_mj = energy_after - energy_before;
                total_energy_j = energy_mj / 1000.0;
            }
            
            // Read telemetry
            GPUTelemetry telem = getGPUTelemetry(nvml_device);
            
            // Check if actual runtime is below target (with 5% tolerance)
            bool below_target = (gpu_kernel_time_s < TARGET_RUNTIME_S * 0.95);
            
            // Write CSV row for this single run
            writeCSVRow(csv_file, run_id_global, run_id_per_size, device_name,
                       n, batches, gpu_e2e_time_s, gpu_kernel_time_s, wall_time_s,
                       total_energy_j, below_target, telem);
            csv_file.flush();
            
            // Console output for EVERY measurement (like CPU version)
            double gflops = (gpu_kernel_time_s > 0) ? 
                (static_cast<double>(n) * batches / gpu_kernel_time_s / 1e9) : 0.0;
            double power = (total_energy_j >= 0 && wall_time_s > 0) ? 
                (total_energy_j / wall_time_s) : -1.0;
            
            char check = below_target ? '!' : '+';
            std::cout << "  " << check << " Run " << rep + 1 << "/" << REPEATS << ": "
                     << std::fixed << std::setprecision(3) 
                     << gpu_kernel_time_s << "s (kernel) "
                     << gpu_e2e_time_s << "s (E2E)";
            
            if (total_energy_j >= 0) {
                std::cout << " | E=" << std::setprecision(1) << total_energy_j << "J";
            }
            if (power >= 0) {
                std::cout << " P=" << std::setprecision(0) << power << "W";
            }
            std::cout << " | " << std::setprecision(2) << gflops << " GFLOPS";
            std::cout << " | T=" << telem.temp << "°C";
            if (telem.throttle_reasons != 0) {
                std::cout << " [THROTTLE]";
            }
            if (below_target) {
                std::cout << " (!)";
            }
            std::cout << std::endl;
        }
        
        // Cooldown (except for last size)
        if (n != sizes_to_test.back()) {
            std::cout << "  Cooling down for 30 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(30));
        }
        
        std::cout << std::endl;
    }
    
    csv_file.close();
    
    std::cout << "========================================" << std::endl;
    std::cout << "Benchmark complete!" << std::endl;
    std::cout << "Results saved to: " << output_file << std::endl;
    std::cout << "Total CSV rows (individual runs): " << run_id_global << std::endl;
    int total_configs = sizes_to_test.size();
    std::cout << "Configurations tested: " << total_configs 
              << " (" << sizes_to_test.size() << " sizes × " << REPEATS << " runs)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_CUDA(cudaEventDestroy(start_kernel));
    CHECK_CUDA(cudaEventDestroy(stop_kernel));
    
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_ones));
    CHECK_CUDA(cudaFree(d_result));
    
    CHECK_CUDA(cudaFreeHost(h_x));
    CHECK_CUDA(cudaFreeHost(h_ones));
    CHECK_CUDA(cudaFreeHost(h_result));
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUBLAS(cublasDestroy(handle));
    nvmlShutdown();
    
    return EXIT_SUCCESS;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    bool test_mode = false;
    const char* output_file = DEFAULT_OUTPUT_FILE;
    int device_index = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test" || arg == "-t") {
            test_mode = true;
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((arg == "--device" || arg == "-d") && i + 1 < argc) {
            device_index = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -t, --test           Test mode (reduced configurations with console output)\n"
                      << "  -o, --output FILE    Output CSV file (default: " << DEFAULT_OUTPUT_FILE << ")\n"
                      << "  -d, --device INDEX   CUDA device index (default: 0)\n"
                      << "  -h, --help           Show this help\n"
                      << "\nHardcoded configurations:\n"
                      << "  Problem sizes: ";
            for (size_t i = 0; i < PROBLEM_SIZES.size(); i++) {
                std::cout << PROBLEM_SIZES[i];
                if (i < PROBLEM_SIZES.size() - 1) std::cout << ", ";
            }
            std::cout << "\n  Repetitions per config: " << REPEATS << " (each run writes a CSV row)\n";
            std::cout << "  Target runtime: " << TARGET_RUNTIME_S << "s\n";
            return 0;
        }
    }
    
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "Test mode: " << (test_mode ? "yes" : "no") << std::endl;
    std::cout << "Device index: " << device_index << std::endl;
    
    return run_benchmark(output_file, test_mode, device_index);
}

// ============================================================================
// Key Changes from Original (to match CPU version)
// ============================================================================
/*
 * [UPDATED] determineBatchSize(): Now uses iterative scaling like CPU:
 *           - Warmup (5 cublasSdot calls, not measured)
 *           - Adaptive starting point based on problem size
 *           - Iterative loop until MIN_TIME (1.0s) is reached
 *           - Uses std::ceil with 5% buffer for scaling
 *           - Guard against near-zero measurements
 *           - Scales back if exceeding MAX_TIME (3.0s)
 *           - Simplified signature (no H2D/D2H in calibration, kernel-only)
 *
 * [UPDATED] Console output: Every measurement now prints a line (like CPU)
 *           - Shows +/! prefix (+ = OK, ! = below target)
 *           - Shows kernel time, E2E time, energy, power, GFLOPS, temperature
 *
 * [UPDATED] below_target: Now based on actual measured kernel time 
 *           (< 0.95 * TARGET) instead of batch limit check
 *
 * [UNCHANGED] Everything else:
 *           - NVML energy measurement logic
 *           - CSV format and columns (26 columns per CSV_COLUMNS.md)
 *           - Problem sizes
 *           - 50 repetitions per configuration
 *           - CUDA Events for E2E and kernel timing
 *           - H2D/D2H in actual measurement runs
 */
