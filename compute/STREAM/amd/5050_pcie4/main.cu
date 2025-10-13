// stream_triad_sweep.cu - CUDA STREAM Triad with Size Sweep and NVML Energy
// Compile: nvcc -O3 -std=c++17 -lnvidia-ml -o stream_triad_sweep main.cu
// Optional Double: nvcc -O3 -std=c++17 -DUSE_DOUBLE -lnvidia-ml -o stream_triad_sweep_fp64 main.cu

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
#include <algorithm>
#include <vector>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>

// ============================================================================
// Data Type Configuration
// ============================================================================

#ifdef USE_DOUBLE
using real = double;
constexpr const char* DTYPE_STR = "fp64";
constexpr size_t BYTES_PER_ITER = 24ULL;  // 2 reads + 1 write, 8 bytes each
#else
using real = float;
constexpr const char* DTYPE_STR = "fp32";
constexpr size_t BYTES_PER_ITER = 12ULL;  // 2 reads + 1 write, 4 bytes each
#endif

// ============================================================================
// Configuration
// ============================================================================

// Size sweep list
#ifdef USE_DOUBLE
static constexpr size_t SIZES[] = {
    1ULL << 20,  // 2^20 = 1M
    1ULL << 22,  // 2^22 = 4M
    1ULL << 24,  // 2^24 = 16M
    1ULL << 26,  // 2^26 = 64M
    1ULL << 27,  // 2^27 = 128M
    1ULL << 28   // 2^28 = 256M (~6 GiB for FP64)
};
#else
static constexpr size_t SIZES[] = {
    1ULL << 20,  // 2^20 = 1M
    1ULL << 22,  // 2^22 = 4M
    1ULL << 24,  // 2^24 = 16M
    1ULL << 26,  // 2^26 = 64M
    1ULL << 27,  // 2^27 = 128M
    1ULL << 28,  // 2^28 = 256M
    1ULL << 29   // 2^29 = 512M (~6 GiB for FP32)
};
#endif

static constexpr int REPEATS = 50;
static constexpr double TARGET_S = 1.1;
static constexpr double SAFETY_FACTOR = 1.02;
static constexpr double OOM_SAFETY = 0.85;
static constexpr real SCALAR_Q = 3.0;

const char* CSV_PATH = "data/raw/stream_triad_sweep.csv";

// Launch configuration
constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define NVML_CHECK(call) do { \
    nvmlReturn_t err = call; \
    if (err != NVML_SUCCESS) { \
        fprintf(stderr, "NVML Error at %s:%d: %s\n", __FILE__, __LINE__, \
                nvmlErrorString(err)); \
    } \
} while(0)

// ============================================================================
// NVML Helper
// ============================================================================

struct NVMLContext {
    nvmlDevice_t device;
    bool initialized = false;
    bool energy_supported = false;
    
    bool init() {
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS) {
            std::cerr << "NVML Init failed: " << nvmlErrorString(result) << "\n";
            return false;
        }
        
        result = nvmlDeviceGetHandleByIndex(0, &device);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get NVML device handle\n";
            nvmlShutdown();
            return false;
        }
        
        // Test if TotalEnergy is supported
        unsigned long long test_energy;
        result = nvmlDeviceGetTotalEnergyConsumption(device, &test_energy);
        energy_supported = (result == NVML_SUCCESS);
        
        initialized = true;
        return true;
    }
    
    unsigned long long getTotalEnergyMJ() {
        if (!initialized || !energy_supported) return 0;
        unsigned long long energy_mj = 0;
        nvmlReturn_t result = nvmlDeviceGetTotalEnergyConsumption(device, &energy_mj);
        if (result != NVML_SUCCESS) return 0;
        return energy_mj;
    }
    
    void shutdown() {
        if (initialized) {
            nvmlShutdown();
            initialized = false;
        }
    }
};

// ============================================================================
// Host Memory Helper
// ============================================================================

struct HostBuf {
    void* p{nullptr};
    bool pinned{false};
};

static void host_alloc(HostBuf& hb, size_t bytes) {
    if (cudaMallocHost(&hb.p, bytes) == cudaSuccess) {
        hb.pinned = true;
        return;
    }
    hb.p = std::malloc(bytes);
    if (!hb.p) {
        fprintf(stderr, "Host alloc failed (%zu bytes)\n", bytes);
        std::exit(EXIT_FAILURE);
    }
}

static void host_free(HostBuf& hb) {
    if (!hb.p) return;
    if (hb.pinned) cudaFreeHost(hb.p);
    else std::free(hb.p);
    hb.p = nullptr;
    hb.pinned = false;
}

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

std::string readableBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

bool checkVRAMAvailable(size_t N) {
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t required_bytes = 3 * N * sizeof(real);
    size_t safe_threshold = static_cast<size_t>(free_bytes * OOM_SAFETY);
    return required_bytes <= safe_threshold;
}

// ============================================================================
// STREAM Triad Kernel
// ============================================================================

__global__ void triad_kernel(real* __restrict__ a, 
                            const real* __restrict__ b, 
                            const real* __restrict__ c, 
                            real q, 
                            size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < N; i += stride) {
        a[i] = b[i] + q * c[i];
    }
}

// ============================================================================
// Device Info
// ============================================================================

struct DeviceInfo {
    std::string name;
    size_t total_global_mem;
    int cc_major;
    int cc_minor;
    int driver_version;
    int sm_count;
};

DeviceInfo getDeviceInfo() {
    DeviceInfo info;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    info.name = std::string(prop.name);
    info.total_global_mem = prop.totalGlobalMem;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    
    CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
    
    return info;
}

// ============================================================================
// GPU Runtime Info (per measurement)
// ============================================================================

struct GPURuntimeInfo {
    unsigned int temp_c = 0;
    unsigned int clocks_sm_mhz = 0;
    unsigned int clocks_mem_mhz = 0;
    unsigned long long throttle_reasons = 0;
};

GPURuntimeInfo getGPURuntimeInfo(nvmlDevice_t device, bool nvml_ok) {
    GPURuntimeInfo info;
    if (!nvml_ok) return info;
    
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temp_c);
    nvmlDeviceGetClock(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &info.clocks_sm_mhz);
    nvmlDeviceGetClock(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &info.clocks_mem_mhz);
    nvmlDeviceGetCurrentClocksThrottleReasons(device, &info.throttle_reasons);
    
    return info;
}

// ============================================================================
// CSV Output - Matching reduction schema
// ============================================================================

void writeCSVHeader(std::ofstream& file) {
    file << "timestamp,host,gpu_name,matrix_size,mode,batches,seconds_target,"
         << "seconds_gpu,seconds_wall,energy_j,avg_power_w,below_target,workload,"
         << "impl,dtype,N,passes_kernel,passes_e2e,seconds_kernel,energy_kernel_j,"
         << "avg_power_w_kernel,avg_power_w_e2e,bytes_total,bw_gb_s,time_mode,"
         << "energy_mode,includes_transfer,device_name,driver_version,"
         << "pcie_gen_current,pcie_width_current,pcie_rx_kbs,pcie_tx_kbs,"
         << "clocks_sm_mhz,clocks_mem_mhz,temp_c,throttle_reasons,notes\n";
}

void writeCSVRow(std::ofstream& file,
                const std::string& timestamp,
                const std::string& hostname,
                const DeviceInfo& dev_info,
                size_t N,
                size_t passes,
                int repeat_idx,
                double seconds_kernel,
                double energy_j,
                double avg_power_w,
                size_t bytes_total,
                double bw_gb_s,
                const GPURuntimeInfo& runtime,
                int grid_size,
                const std::string& extra_notes) {
    
    int below_target = (seconds_kernel >= TARGET_S) ? 0 : 1;
    
    file << timestamp << ","
         << hostname << ","
         << dev_info.name << ","
         << "0" << ","  // matrix_size (not applicable for STREAM)
         << "kernel" << ","  // mode
         << repeat_idx << ","  // batches -> repeat index
         << std::fixed << std::setprecision(2) << TARGET_S << ","
         << std::setprecision(4) << seconds_kernel << ","
         << seconds_kernel << ","  // seconds_wall = seconds_gpu for kernel-only
         << std::setprecision(3) << energy_j << ","
         << std::setprecision(1) << avg_power_w << ","
         << below_target << ","
         << "stream_triad" << ","
         << "cuda" << ","
         << DTYPE_STR << ","
         << N << ","
         << passes << ","  // passes_kernel
         << passes << ","  // passes_e2e (same for kernel-only)
         << std::setprecision(4) << seconds_kernel << ","
         << std::setprecision(3) << energy_j << ","
         << std::setprecision(1) << avg_power_w << ","
         << avg_power_w << ","  // avg_power_w_e2e (same)
         << bytes_total << ","
         << std::setprecision(2) << bw_gb_s << ","
         << "kernel" << ","  // time_mode
         << "kernel" << ","  // energy_mode
         << "0" << ","  // includes_transfer
         << dev_info.name << ","
         << dev_info.driver_version << ","
         << "0,0,0,0,"  // pcie stats (not tracked for STREAM)
         << runtime.clocks_sm_mhz << ","
         << runtime.clocks_mem_mhz << ","
         << runtime.temp_c << ","
         << runtime.throttle_reasons << ","
         << "sweep;repeats=" << REPEATS << ";passes=" << passes 
         << ";grid=" << grid_size << ";block=" << BLOCK_SIZE 
         << ";dtype=" << DTYPE_STR << extra_notes << "\n";
}

void writeSkippedRow(std::ofstream& file,
                    const std::string& timestamp,
                    const std::string& hostname,
                    const DeviceInfo& dev_info,
                    size_t N) {
    file << timestamp << "," << hostname << "," << dev_info.name << ","
         << "0,kernel,0,0.00,0.0000,0.0000,0.000,0.0,1,"
         << "stream_triad,cuda," << DTYPE_STR << ","
         << N << ",0,0,0.0000,0.000,0.0,0.0,0,0.00,"
         << "kernel,kernel,0," << dev_info.name << "," << dev_info.driver_version << ","
         << "0,0,0,0,0,0,0,0,skip_oom\n";
}

// ============================================================================
// Calibration with warm-up
// ============================================================================

size_t calibrateWithWarmup(real* d_a, const real* d_b, const real* d_c, 
                          real q, size_t N, int grid_size, cudaStream_t stream) {
    // Untimed warm-up pass
    triad_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_c, q, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Timed calibration pass
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start, stream));
    triad_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_c, q, N);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_single;
    CUDA_CHECK(cudaEventElapsedTime(&ms_single, start, stop));
    double t_pass = ms_single / 1000.0;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // Calculate passes with safety factor to ensure >= 1.0s
    size_t passes = std::max(static_cast<size_t>(1), 
                            static_cast<size_t>(std::ceil((TARGET_S / t_pass) * SAFETY_FACTOR)));
    return passes;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Initialize NVML
    NVMLContext nvml;
    bool nvml_ok = nvml.init();
    if (!nvml_ok) {
        std::cerr << "⚠️  NVML not available, energy measurements will be zero\n";
    } else {
        std::cout << "✓ NVML initialized, energy tracking enabled\n";
    }
    
    // Get device info
    DeviceInfo dev_info = getDeviceInfo();
    std::string hostname = getHostname();
    
    std::cout << "========================================\n";
    std::cout << "STREAM Triad - Size Sweep + NVML Energy\n";
    std::cout << "========================================\n";
    std::cout << "Device:         " << dev_info.name << "\n";
    std::cout << "Compute Cap:    " << dev_info.cc_major << "." << dev_info.cc_minor << "\n";
    std::cout << "Driver:         " << dev_info.driver_version << "\n";
    std::cout << "SMs:            " << dev_info.sm_count << "\n";
    std::cout << "Total Memory:   " << readableBytes(dev_info.total_global_mem) << "\n";
    std::cout << "Data type:      " << DTYPE_STR << " (" << sizeof(real) << " bytes)\n";
    std::cout << "Bytes/iter:     " << BYTES_PER_ITER << " (STREAM)\n";
    std::cout << "Target runtime: " << TARGET_S << "s\n";
    std::cout << "Repeats/size:   " << REPEATS << "\n";
    std::cout << "Sizes:          " << sizeof(SIZES)/sizeof(SIZES[0]) << "\n";
    std::cout << "Output:         " << CSV_PATH << "\n";
    std::cout << "========================================\n\n";
    
    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Prepare CSV
    ensureDirectoryExists(CSV_PATH);
    bool write_header = !fileExists(CSV_PATH);
    
    std::ofstream csv_file(CSV_PATH, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Cannot open " << CSV_PATH << "\n";
        return EXIT_FAILURE;
    }
    
    if (write_header) {
        writeCSVHeader(csv_file);
    }
    
    // Create CUDA events
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    
    // ========================================================================
    // SIZE SWEEP
    // ========================================================================
    
    constexpr size_t num_sizes = sizeof(SIZES) / sizeof(SIZES[0]);
    
    for (size_t size_idx = 0; size_idx < num_sizes; size_idx++) {
        size_t N = SIZES[size_idx];
        
        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "Size " << (size_idx + 1) << "/" << num_sizes 
                  << ": N = " << N << " (2^" << static_cast<int>(std::log2(N)) << ")\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        // Check VRAM
        if (!checkVRAMAvailable(N)) {
            size_t free_bytes, total_bytes;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            std::cout << "⚠️  SKIPPING: OOM (need " << readableBytes(3*N*sizeof(real)) 
                      << ", free " << readableBytes(free_bytes) << ")\n";
            writeSkippedRow(csv_file, getTimestamp(), hostname, dev_info, N);
            csv_file.flush();
            continue;
        }
        
        // Allocate memory
        const size_t vec_bytes = N * sizeof(real);
        
        HostBuf h_a, h_b, h_c;
        host_alloc(h_a, vec_bytes);
        host_alloc(h_b, vec_bytes);
        host_alloc(h_c, vec_bytes);
        
        real* h_a_ptr = static_cast<real*>(h_a.p);
        real* h_b_ptr = static_cast<real*>(h_b.p);
        real* h_c_ptr = static_cast<real*>(h_c.p);
        for (size_t i = 0; i < N; i++) {
            h_a_ptr[i] = 0.0;
            h_b_ptr[i] = 1.0;
            h_c_ptr[i] = 2.0;
        }
        
        real *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_b, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_c, vec_bytes));
        
        CUDA_CHECK(cudaMemcpy(d_a, h_a.p, vec_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.p, vec_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c, h_c.p, vec_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Grid size
        int grid_nom = static_cast<int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        int grid_size = std::min(grid_nom, dev_info.sm_count * 32);
        
        // Calibrate with warm-up
        std::cout << "Calibrating (with warm-up)... " << std::flush;
        size_t passes = calibrateWithWarmup(d_a, d_b, d_c, SCALAR_Q, N, grid_size, stream);
        std::cout << passes << " passes (grid=" << grid_size << ")\n";
        
        size_t bytes_total = BYTES_PER_ITER * N * passes;
        
        // Measurement loop
        std::cout << "Measuring " << REPEATS << " runs...\n";
        
        for (int run = 0; run < REPEATS; run++) {
            // Reset array a (outside measurement)
            CUDA_CHECK(cudaMemsetAsync(d_a, 0, vec_bytes, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            // Energy start
            unsigned long long energy_start_mj = nvml.getTotalEnergyMJ();
            
            // Timing start
            CUDA_CHECK(cudaEventRecord(start_event, stream));
            
            // Kernel loop
            for (size_t p = 0; p < passes; p++) {
                triad_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_c, SCALAR_Q, N);
            }
            
            // Timing stop
            CUDA_CHECK(cudaEventRecord(stop_event, stream));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            // Energy stop
            unsigned long long energy_stop_mj = nvml.getTotalEnergyMJ();
            
            // Get runtime info
            GPURuntimeInfo runtime = getGPURuntimeInfo(nvml.device, nvml_ok);
            
            // Calculate metrics
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
            double seconds_kernel = ms / 1000.0;
            
            double energy_j = 0.0;
            double avg_power_w = 0.0;
            std::string extra_notes = "";
            
            if (nvml_ok && nvml.energy_supported && energy_stop_mj >= energy_start_mj) {
                energy_j = (energy_stop_mj - energy_start_mj) / 1000.0;
                avg_power_w = energy_j / seconds_kernel;
            } else if (!nvml_ok || !nvml.energy_supported) {
                extra_notes = ";no_energy";
            }
            
            double bw_gb_s = (bytes_total / 1e9) / seconds_kernel;
            
            // Write CSV
            writeCSVRow(csv_file, getTimestamp(), hostname, dev_info, N, passes, 
                       run, seconds_kernel, energy_j, avg_power_w, bytes_total, 
                       bw_gb_s, runtime, grid_size, extra_notes);
            csv_file.flush();
            
            // Console (every 10th)
            if (run == 0 || run == REPEATS-1 || (run+1) % 10 == 0) {
                std::cout << "  [" << std::setw(2) << (run+1) << "/" << REPEATS << "] "
                          << std::fixed << std::setprecision(4) << seconds_kernel << "s";
                if (energy_j > 0) {
                    std::cout << " E=" << std::setprecision(1) << energy_j << "J";
                    if (avg_power_w > 0) {
                        std::cout << " P=" << std::setprecision(0) << avg_power_w << "W";
                    }
                }
                std::cout << " | BW=" << std::setprecision(2) << bw_gb_s << " GB/s\n";
            }
        }
        
        std::cout << "✓ Complete\n";
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        host_free(h_a);
        host_free(h_b);
        host_free(h_c);
    }
    
    // Final cleanup
    csv_file.close();
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvml.shutdown();
    
    std::cout << "\n========================================\n";
    std::cout << "✓ Benchmark complete!\n";
    std::cout << "Results: " << CSV_PATH << "\n";
    std::cout << "========================================\n";
    
    return EXIT_SUCCESS;
}

// Compile:
// nvcc -O3 -std=c++17 -lnvidia-ml -o stream_triad_sweep main.cu
// nvcc -O3 -std=c++17 -DUSE_DOUBLE -lnvidia-ml -o stream_triad_sweep_fp64 main.cu
