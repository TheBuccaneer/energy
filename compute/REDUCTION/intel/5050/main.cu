// reduction_gpu_cublas.cu - CUDA Reduction (DOT with ones) Energy Benchmark
// Compile: nvcc -O3 -std=c++17 -o reduction_gpu_cublas reduction_gpu_cublas.cu -lcublas -lnvidia-ml
// Usage: ./reduction_gpu_cublas

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
#include <vector>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>

// ============================================================================
// Configuration - Hardcoded
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    REPEATS = 5;

// N values in elements (19 sizes)
static const int N_SIZES[] = {
    1000000, 1500000, 2000000, 3000000, 4000000, 6000000,
    8000000, 12000000, 16000000, 24000000, 32000000, 48000000,
    64000000, 96000000, 128000000, 160000000, 192000000,
    256000000, 384000000
};
static const int NUM_SIZES = sizeof(N_SIZES) / sizeof(N_SIZES[0]);
static const int MAX_N = *std::max_element(std::begin(N_SIZES), std::end(N_SIZES));

static const char* OUTPUT_FILE = "data/raw/reduction_gpu_cublas.csv";

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
    unsigned int pcie_rx_kbs;
    unsigned int pcie_tx_kbs;
    unsigned int sm_clock;
    unsigned int mem_clock;
    unsigned int temp;
    unsigned long long throttle_reasons;
};

std::string getGPUName(nvmlDevice_t device) {
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    CHECK_NVML(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
    return std::string(name);
}

std::string getDriverVersion(nvmlDevice_t device) {
    char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    CHECK_NVML(nvmlSystemGetDriverVersion(version, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));
    return std::string(version);
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
    
    // Get PCIe throughput (KB/s)
    nvmlReturn_t ret_rx = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &telem.pcie_rx_kbs);
    nvmlReturn_t ret_tx = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &telem.pcie_tx_kbs);
    if (ret_rx != NVML_SUCCESS) telem.pcie_rx_kbs = 0;
    if (ret_tx != NVML_SUCCESS) telem.pcie_tx_kbs = 0;
    
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
// Determine passes for target runtime
// ============================================================================

int determinePassesKernel(cublasHandle_t handle, float* d_x, float* d_ones, 
                          float* d_result, int n, float target_seconds, 
                          cudaStream_t stream) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Set pointer mode to DEVICE for kernel-only measurement
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    
    // Measure one pass
    CHECK_CUDA(cudaEventRecord(start, stream));
    CHECK_CUBLAS(cublasSdot(handle, n, d_x, 1, d_ones, 1, d_result));
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float t_one = ms / 1000.0f;
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Fudge factor to better hit target
    int passes = (int)std::ceil(target_seconds / t_one * 1.05);
    return std::max(1, passes);
}

int determinePassesE2E(cublasHandle_t handle, float* d_x, float* d_ones,
                       float* h_x, int n, float target_seconds,
                       cudaStream_t stream) {
    size_t n_bytes = n * sizeof(float);
    float h_result = 0.0f;
    
    // Set pointer mode to HOST for E2E measurement
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    
    auto start = std::chrono::steady_clock::now();
    
    // H2D transfer
    CHECK_CUDA(cudaMemcpyAsync(d_x, h_x, n_bytes, cudaMemcpyHostToDevice, stream));
    
    // DOT operation
    CHECK_CUBLAS(cublasSdot(handle, n, d_x, 1, d_ones, 1, &h_result));
    
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    float t_one = elapsed.count();
    
    // Fudge factor to better hit target
    int passes = (int)std::ceil(target_seconds / t_one * 1.05);
    return std::max(1, passes);
}

// ============================================================================
// CSV Output Functions
// ============================================================================

void writeCSVHeader(std::ofstream& file) {
    // Original fields from GEMM
    file << "timestamp,host,gpu_name,matrix_size,mode,batches,"
         << "seconds_target,seconds_gpu,seconds_wall,"
         << "energy_j,avg_power_w,below_target,"
         // New fields for reduction
         << "workload,impl,dtype,N,passes_kernel,passes_e2e,"
         << "seconds_kernel,energy_kernel_j,"
         << "avg_power_w_kernel,avg_power_w_e2e,"
         << "bytes_total,bw_gb_s,"
         << "time_mode,energy_mode,includes_transfer,"
         << "device_name,driver_version,"
         << "pcie_gen_current,pcie_width_current,pcie_rx_kbs,pcie_tx_kbs,"
         << "clocks_sm_mhz,clocks_mem_mhz,temp_c,throttle_reasons,notes\n";
}

void writeCSVRow(std::ofstream& file, const std::string& host,
                 const std::string& gpu_name, int n, 
                 int passes_kernel, int passes_e2e,
                 const std::string& mode,  // "kernel" or "e2e"
                 float seconds_kernel, float seconds_e2e,
                 double energy_kernel_j, double energy_e2e_j,
                 size_t bytes_total, double bw_gb_s,
                 const GPUTelemetry& telem,
                 const std::string& driver_version) {
    
    bool is_kernel_mode = (mode == "kernel");
    
    // Mode-specific values
    float seconds_gpu = seconds_kernel;  // Always kernel time for comparability
    float seconds_wall = is_kernel_mode ? seconds_kernel : seconds_e2e;
    
    // For kernel mode: energy_j = -1 (only energy_kernel_j is valid)
    // For e2e mode: energy_j = e2e_energy_j
    double energy_j = is_kernel_mode ? -1.0 : energy_e2e_j;
    
    // Compute power separately for each mode (always)
    double avg_power_w_kernel = -1.0;
    double avg_power_w_e2e = -1.0;
    
    if (energy_kernel_j >= 0 && seconds_kernel > 0) {
        avg_power_w_kernel = energy_kernel_j / seconds_kernel;
    }
    if (energy_e2e_j >= 0 && seconds_e2e > 0) {
        avg_power_w_e2e = energy_e2e_j / seconds_e2e;
    }
    
    // For avg_power_w field (mode-dependent)
    double avg_power_w = is_kernel_mode ? avg_power_w_kernel : avg_power_w_e2e;
    
    bool below_target = (seconds_gpu < TARGET_RUNTIME_S);
    
    file << getTimestamp() << ","
         << host << ","
         << gpu_name << ","
         << "0" << ","  // matrix_size (not applicable, use 0)
         << mode << ","
         << (is_kernel_mode ? passes_kernel : passes_e2e) << ","  // batches field
         << std::fixed << std::setprecision(2) << TARGET_RUNTIME_S << ","
         << std::setprecision(4) << seconds_gpu << ","
         << seconds_wall << ","
         << std::setprecision(3) << energy_j << ","
         << std::setprecision(1) << avg_power_w << ","
         << (below_target ? 1 : 0) << ","
         // New fields
         << "reduction" << ","
         << "cublas" << ","
         << "fp32" << ","
         << n << ","
         << passes_kernel << ","
         << passes_e2e << ","
         << std::setprecision(4) << seconds_kernel << ","
         << std::setprecision(3) << energy_kernel_j << ","
         << std::setprecision(1) << avg_power_w_kernel << ","
         << avg_power_w_e2e << ","
         << bytes_total << ","
         << std::setprecision(2) << bw_gb_s << ","
         << mode << ","  // time_mode
         << mode << ","  // energy_mode
         << (is_kernel_mode ? 0 : 1) << ","  // includes_transfer
         << gpu_name << ","
         << driver_version << ","
         << telem.pcie_gen << ","
         << telem.pcie_width << ","
         << telem.pcie_rx_kbs << ","
         << telem.pcie_tx_kbs << ","
         << telem.sm_clock << ","
         << telem.mem_clock << ","
         << telem.temp << ","
         << telem.throttle_reasons << ","
         << "" << "\n";  // notes empty
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
    std::string driver_version = getDriverVersion(nvml_device);
    std::string hostname = getHostname();
    
    std::cout << "========================================\n";
    std::cout << "CUDA Reduction Energy Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "System:         " << hostname << "\n";
    std::cout << "GPU:            " << gpu_name << "\n";
    std::cout << "Driver:         " << driver_version << "\n";
    std::cout << "Target runtime: " << TARGET_RUNTIME_S << "s\n";
    std::cout << "Repeats:        " << REPEATS << "\n";
    std::cout << "N sizes:        " << NUM_SIZES << " sizes (1M-384M)\n";
    std::cout << "Output:         " << OUTPUT_FILE << "\n";
    std::cout << "========================================\n\n";
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    
    // Allocate pinned host memory for maximum size
    const size_t max_bytes = MAX_N * sizeof(float);
    
    float *h_x;
    CHECK_CUDA(cudaMallocHost(&h_x, max_bytes));
    
    std::cout << "Allocated pinned host buffer: "
              << (max_bytes / (1024*1024)) << " MB\n";
    
    // Allocate device memory for x, ones, and result scalar
    float *d_x, *d_ones, *d_result;
    CHECK_CUDA(cudaMalloc(&d_x, max_bytes));
    CHECK_CUDA(cudaMalloc(&d_ones, max_bytes));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    
    std::cout << "Allocated device buffers: "
              << (2 * max_bytes / (1024*1024)) << " MB\n\n";
    
    // Initialize and upload ONES vector (persistent on GPU)
    std::cout << "Initializing and uploading ONES vector...\n";
    std::vector<float> h_ones_vec(MAX_N, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_ones, h_ones_vec.data(), max_bytes, cudaMemcpyHostToDevice));
    
    // Initialize host x vector once
    std::cout << "Initializing host x vector...\n";
    initializeVector(h_x, MAX_N, 42);
    
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
    
    // ========================================================================
    // Main measurement loop: sweep over all N sizes
    // ========================================================================
    
    std::cout << "Starting measurements...\n";
    std::cout << "========================================\n\n";
    
    for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
        int n = N_SIZES[size_idx];
        size_t n_bytes = n * sizeof(float);
        
        std::cout << "N = " << n << " elements (" 
                  << (n_bytes / (1024*1024)) << " MB)\n";
        
        // Upload x for this N (use first n elements)
        CHECK_CUDA(cudaMemcpy(d_x, h_x, n_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Mini warmups (2 passes, outside measurement)
        float dummy_h_result = 0.0f;
        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        for (int w = 0; w < 2; w++) {
            CHECK_CUBLAS(cublasSdot(handle, n, d_x, 1, d_ones, 1, &dummy_h_result));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Determine passes separately for kernel and E2E
        std::cout << "  Determining passes... " << std::flush;
        int passes_kernel = determinePassesKernel(handle, d_x, d_ones, d_result, 
                                                  n, TARGET_RUNTIME_S, stream);
        int passes_e2e = determinePassesE2E(handle, d_x, d_ones, h_x,
                                           n, TARGET_RUNTIME_S, stream);
        std::cout << "kernel=" << passes_kernel << ", e2e=" << passes_e2e << "\n";
        
        // Bytes for each mode
        // Kernel: DOT reads TWO vectors (x AND ones)
        size_t bytes_total_kernel = 2ULL * n * sizeof(float) * passes_kernel;
        // E2E: only H2D(x) per pass, ONES is persistent
        size_t bytes_total_e2e = 1ULL * n * sizeof(float) * passes_e2e;
        
        // Storage for repeat measurements
        std::vector<float> kernel_times;
        std::vector<float> e2e_times;
        std::vector<double> kernel_energies;
        std::vector<double> e2e_energies;
        
        // Run REPEATS measurements
        for (int rep = 0; rep < REPEATS; rep++) {
            
            // ================================================================
            // KERNEL-ONLY Measurement (DEVICE pointer mode, Events only)
            // ================================================================
            
            CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
            
            // Start timing with CUDA event
            CHECK_CUDA(cudaEventRecord(start_event, stream));
            
            // Start energy measurement (immediately after event)
            unsigned long long energy_before_kernel = getGPUEnergy(nvml_device);
            
            // Execute DOT operations
            for (int p = 0; p < passes_kernel; p++) {
                CHECK_CUBLAS(cublasSdot(handle, n, d_x, 1, d_ones, 1, d_result));
            }
            
            // Stop timing with CUDA event
            CHECK_CUDA(cudaEventRecord(stop_event, stream));
            CHECK_CUDA(cudaEventSynchronize(stop_event));
            
            // Stop energy measurement (immediately after sync)
            unsigned long long energy_after_kernel = getGPUEnergy(nvml_device);
            
            float kernel_ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start_event, stop_event));
            float kernel_time_s = kernel_ms / 1000.0f;
            
            double kernel_energy_j = -1.0;  // Default if not supported
            if (energy_after_kernel > 0 && energy_before_kernel > 0 && 
                energy_after_kernel >= energy_before_kernel) {
                kernel_energy_j = (energy_after_kernel - energy_before_kernel) / 1000.0;
            }
            
            kernel_times.push_back(kernel_time_s);
            kernel_energies.push_back(kernel_energy_j);
            
            // ================================================================
            // E2E Measurement (HOST pointer mode, Wall-Clock)
            // Note: E2E bandwidth is PCIe-limited. For PCIe 3.0 x8,
            //       typical H2D bandwidth is ~6-7 GB/s (theoretical ~7.9 GB/s)
            // ================================================================
            
            CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
            
            unsigned long long energy_before_e2e = getGPUEnergy(nvml_device);
            auto wall_start = std::chrono::steady_clock::now();
            
            // Loop: per pass do H2D → DOT → D2H
            for (int p = 0; p < passes_e2e; p++) {
                float h_result = 0.0f;
                
                // H2D transfer of x
                CHECK_CUDA(cudaMemcpyAsync(d_x, h_x, n_bytes, cudaMemcpyHostToDevice, stream));
                
                // DOT operation (result goes to host directly with HOST pointer mode)
                CHECK_CUBLAS(cublasSdot(handle, n, d_x, 1, d_ones, 1, &h_result));
                
                // Note: D2H of result is implicit with HOST pointer mode
            }
            
            // Synchronize before stopping measurements
            CHECK_CUDA(cudaDeviceSynchronize());
            
            auto wall_end = std::chrono::steady_clock::now();
            unsigned long long energy_after_e2e = getGPUEnergy(nvml_device);
            
            std::chrono::duration<double> wall_duration = wall_end - wall_start;
            float e2e_time_s = wall_duration.count();
            
            double e2e_energy_j = -1.0;  // Default if not supported
            if (energy_after_e2e > 0 && energy_before_e2e > 0 &&
                energy_after_e2e >= energy_before_e2e) {
                e2e_energy_j = (energy_after_e2e - energy_before_e2e) / 1000.0;
            }
            
            e2e_times.push_back(e2e_time_s);
            e2e_energies.push_back(e2e_energy_j);
            
            // Compute power for console output
            double power_kernel = (kernel_energy_j >= 0 && kernel_time_s > 0) 
                                  ? (kernel_energy_j / kernel_time_s) : -1.0;
            double power_e2e = (e2e_energy_j >= 0 && e2e_time_s > 0) 
                              ? (e2e_energy_j / e2e_time_s) : -1.0;
            
            // Console progress
            std::cout << "  + Run " << (rep + 1) << "/" << REPEATS << ": "
                     << "kernel=" << std::fixed << std::setprecision(3) << kernel_time_s << "s";
            if (kernel_energy_j >= 0) {
                std::cout << " E=" << std::setprecision(1) << kernel_energy_j << "J";
                if (power_kernel >= 0) {
                    std::cout << " P=" << std::setprecision(0) << power_kernel << "W";
                }
            }
            std::cout << " | e2e=" << std::setprecision(3) << e2e_time_s << "s";
            if (e2e_energy_j >= 0) {
                std::cout << " E=" << std::setprecision(1) << e2e_energy_j << "J";
                if (power_e2e >= 0) {
                    std::cout << " P=" << std::setprecision(0) << power_e2e << "W";
                }
            }
            std::cout << "\n";
        }
        
        // Compute medians
        std::sort(kernel_times.begin(), kernel_times.end());
        std::sort(e2e_times.begin(), e2e_times.end());
        std::sort(kernel_energies.begin(), kernel_energies.end());
        std::sort(e2e_energies.begin(), e2e_energies.end());
        
        float median_kernel_time = kernel_times[REPEATS / 2];
        float median_e2e_time = e2e_times[REPEATS / 2];
        double median_kernel_energy = kernel_energies[REPEATS / 2];
        double median_e2e_energy = e2e_energies[REPEATS / 2];
        
        // Compute bandwidth separately for each mode
        double bw_gb_s_kernel = (bytes_total_kernel / (1e9)) / median_kernel_time;
        double bw_gb_s_e2e = (bytes_total_e2e / (1e9)) / median_e2e_time;
        
        // Get telemetry
        GPUTelemetry telem = getGPUTelemetry(nvml_device);
        
        // Write TWO CSV rows (kernel + e2e) with respective passes and bytes
        writeCSVRow(csv_file, hostname, gpu_name, n, passes_kernel, passes_e2e, "kernel",
                   median_kernel_time, median_e2e_time,
                   median_kernel_energy, median_e2e_energy,
                   bytes_total_kernel, bw_gb_s_kernel, telem, driver_version);
        
        writeCSVRow(csv_file, hostname, gpu_name, n, passes_kernel, passes_e2e, "e2e",
                   median_kernel_time, median_e2e_time,
                   median_kernel_energy, median_e2e_energy,
                   bytes_total_e2e, bw_gb_s_e2e, telem, driver_version);
        
        csv_file.flush();
        
        std::cout << "  → Median: kernel=" << std::setprecision(3) << median_kernel_time << "s, "
                  << "e2e=" << std::setprecision(3) << median_e2e_time << "s\n"
                  << "     BW_kernel=" << std::setprecision(1) << bw_gb_s_kernel << " GB/s, "
                  << "BW_e2e=" << bw_gb_s_e2e << " GB/s\n"
                  << "     PCIe: Gen" << telem.pcie_gen << " x" << telem.pcie_width 
                  << " (RX=" << std::setprecision(1) << (telem.pcie_rx_kbs / 1024.0) << " MB/s, "
                  << "TX=" << (telem.pcie_tx_kbs / 1024.0) << " MB/s)\n\n";
    }
    
    std::cout << "========================================\n";
    std::cout << "Benchmark complete!\n";
    std::cout << "Results saved to: " << OUTPUT_FILE << "\n";
    std::cout << "Total measurements: " << (NUM_SIZES * REPEATS * 2) << "\n";
    std::cout << "========================================\n";
    
    // Cleanup
    csv_file.close();
    
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_ones));
    CHECK_CUDA(cudaFree(d_result));
    
    CHECK_CUDA(cudaFreeHost(h_x));
    
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUBLAS(cublasDestroy(handle));
    nvmlShutdown();
    
    return EXIT_SUCCESS;
}