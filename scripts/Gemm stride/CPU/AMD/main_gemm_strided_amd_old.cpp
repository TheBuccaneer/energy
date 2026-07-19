// ============================================================================
// CPU Strided-Batched GEMM Benchmark
// 1:1 reproduction of GPU measurement logic (main.cu) for CPU
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cstring>
#include <algorithm>

// BLAS header (OpenBLAS)
#include <cblas.h>

// Linux-specific headers for RAPL, CPU info, temp
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cstdint>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

// ============================================================================
// Configuration Constants
// ============================================================================

constexpr float TARGET_RUNTIME_S = 1.0f;
constexpr int COOLDOWN_MS = 60000; // 30s cooldown like GPU
constexpr int REPETITIONS = 50;    // 50 measurements per (n, batch_count)
constexpr int MAX_SIZE = 16384;

// Hardcoded thread counts for Intel CPU (as per CSV_COLUMNS.md)
const std::vector<int> THREAD_COUNTS = {1, 2, 4, 8, 10, 16, 20, 32, 64};

// Problem sizes
const std::vector<int> PROBLEM_SIZES = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

// batch_count mapping per problem size (same as GPU)
const std::vector<std::vector<int>> BATCH_COUNT_MAP = {
    {512, 1024},    // 64
    {256, 512},     // 128
    {128, 256},     // 256
    {64, 128},      // 512
    {32, 64},       // 1024
    {16, 32},       // 2048
    {4, 8},         // 4096
    {1, 2},         // 8192
    {1}             // 16384
};

// ============================================================================
// Helper Functions
// ============================================================================

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

std::string get_cpu_model() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string model = line.substr(pos + 2);
                // Remove "Intel(R)" / "AMD" prefixes for cleaner output
                size_t intel_pos = model.find("Intel(R)");
                if (intel_pos != std::string::npos) {
                    model = model.substr(intel_pos + 9);
                }
                size_t amd_pos = model.find("AMD");
                if (amd_pos != std::string::npos) {
                    model = model.substr(amd_pos + 4);
                }
                // Trim whitespace
                model.erase(0, model.find_first_not_of(" \t"));
                model.erase(model.find_last_not_of(" \t") + 1);
                return model;
            }
        }
    }
    return "Unknown CPU";
}

int get_cpu_freq_mhz() {
    std::ifstream freq_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    if (freq_file.is_open()) {
        int freq_khz;
        freq_file >> freq_khz;
        return freq_khz / 1000; // Convert kHz to MHz
    }
    return -1;
}

int get_cpu_temp_c() {
    // Try common hwmon paths (coretemp for Intel, k10temp for AMD)
    std::vector<std::string> temp_paths = {
        "/sys/class/hwmon/hwmon0/temp1_input",
        "/sys/class/hwmon/hwmon1/temp1_input",
        "/sys/class/hwmon/hwmon2/temp1_input"
    };
    
    for (const auto& path : temp_paths) {
        std::ifstream temp_file(path);
        if (temp_file.is_open()) {
            int temp_millidegrees;
            temp_file >> temp_millidegrees;
            return temp_millidegrees / 1000; // Convert to Celsius
        }
    }
    return -1;
}

// ============================================================================
// RAPL Energy Measurement
// ============================================================================

struct RAPLReader {
    std::vector<std::string> energy_paths;
    bool available;
    
    RAPLReader() : available(false) {
        // Find all RAPL package domains
        for (int i = 0; i < 16; ++i) {
            std::string base_path = "/sys/class/powercap/intel-rapl/intel-rapl:" + std::to_string(i);
            std::string name_path = base_path + "/name";
            
            std::ifstream name_file(name_path);
            if (!name_file.is_open()) break;
            
            std::string name;
            std::getline(name_file, name);
            
            if (name == "package-" + std::to_string(i) || name.find("package") != std::string::npos) {
                std::string energy_path = base_path + "/energy_uj";
                std::ifstream test(energy_path);
                if (test.is_open()) {
                    energy_paths.push_back(energy_path);
                }
            }
        }
        
        if (!energy_paths.empty()) {
            available = true;
        } else {
            std::cerr << "WARNING: RAPL not available. Make sure:\n"
                      << "  1. You're on Linux with Intel/AMD CPU\n"
                      << "  2. powercap module loaded: sudo modprobe intel_rapl_msr\n"
                      << "  3. Permissions set: sudo chmod -R a+r /sys/class/powercap/intel-rapl/\n"
                      << "Energy measurements will be set to -1.0\n" << std::endl;
        }
    }
    
    long long read_energy_uj() {
        if (!available) return -1;
        
        long long total = 0;
        for (const auto& path : energy_paths) {
            std::ifstream energy_file(path);
            if (energy_file.is_open()) {
                long long energy;
                energy_file >> energy;
                total += energy;
            }
        }
        return total;
    }
    
    double read_energy_j() {
        long long uj = read_energy_uj();
        return (uj < 0) ? -1.0 : (uj * 1e-6);
    }
};

// ============================================================================
// Perf Event Counter Group (cycles, instructions, cache-misses)
// ============================================================================

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

struct PerfReadFormat {
    uint64_t nr;
    uint64_t values[3];
};

class PerfGroupCounter {
public:
    PerfGroupCounter() : fd_leader_(-1), fd_instructions_(-1), fd_cache_misses_(-1), available_(false) {}
    
    ~PerfGroupCounter() {
        close_fds();
    }
    
    bool open() {
        close_fds();
        
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(pe));
        pe.size = sizeof(pe);
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        pe.inherit = 1;
        pe.read_format = PERF_FORMAT_GROUP;
        
        pe.type = PERF_TYPE_HARDWARE;
        pe.config = PERF_COUNT_HW_CPU_CYCLES;
        fd_leader_ = perf_event_open(&pe, 0, -1, -1, 0);
        if (fd_leader_ < 0) {
            available_ = false;
            return false;
        }
        
        pe.disabled = 0;
        pe.config = PERF_COUNT_HW_INSTRUCTIONS;
        fd_instructions_ = perf_event_open(&pe, 0, -1, fd_leader_, 0);
        if (fd_instructions_ < 0) {
            close_fds();
            available_ = false;
            return false;
        }
        
        pe.config = PERF_COUNT_HW_CACHE_MISSES;
        fd_cache_misses_ = perf_event_open(&pe, 0, -1, fd_leader_, 0);
        if (fd_cache_misses_ < 0) {
            close_fds();
            available_ = false;
            return false;
        }
        
        available_ = true;
        return true;
    }
    
    void start() {
        if (!available_) return;
        ioctl(fd_leader_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fd_leader_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }
    
    void stop() {
        if (!available_) return;
        ioctl(fd_leader_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    }
    
    void read(int64_t& cycles, int64_t& instructions, int64_t& cache_misses) {
        if (!available_) {
            cycles = -1;
            instructions = -1;
            cache_misses = -1;
            return;
        }
        
        PerfReadFormat data;
        memset(&data, 0, sizeof(data));
        ssize_t ret = ::read(fd_leader_, &data, sizeof(data));
        
        if (ret != sizeof(data) || data.nr != 3) {
            cycles = -1;
            instructions = -1;
            cache_misses = -1;
            return;
        }
        
        cycles = static_cast<int64_t>(data.values[0]);
        instructions = static_cast<int64_t>(data.values[1]);
        cache_misses = static_cast<int64_t>(data.values[2]);
    }
    
    bool is_available() const { return available_; }
    
private:
    void close_fds() {
        if (fd_cache_misses_ >= 0) { ::close(fd_cache_misses_); fd_cache_misses_ = -1; }
        if (fd_instructions_ >= 0) { ::close(fd_instructions_); fd_instructions_ = -1; }
        if (fd_leader_ >= 0) { ::close(fd_leader_); fd_leader_ = -1; }
        available_ = false;
    }
    
    int fd_leader_;
    int fd_instructions_;
    int fd_cache_misses_;
    bool available_;
};

// ============================================================================
// Matrix Operations
// ============================================================================

void fill_matrix(float* mat, int rows, int cols, int ld, float value) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i * ld + j] = value;
        }
    }
}

// ============================================================================
// Batched GEMM Wrapper
// ============================================================================

void build_pointer_arrays(
    float* A, float* B, float* C,
    int n, int ldc, int batch_count,
    std::vector<const float*>& A_array,
    std::vector<const float*>& B_array,
    std::vector<float*>& C_array)
{
    A_array.resize(batch_count);
    B_array.resize(batch_count);
    C_array.resize(batch_count);
    
    // Broadcast A and B (all point to same matrices)
    for (int i = 0; i < batch_count; ++i) {
        A_array[i] = A;
        B_array[i] = B;
        C_array[i] = C + (long long)i * ldc * n; // Strided C
    }
}

void run_batched_gemm(
    const std::vector<const float*>& A_array,
    const std::vector<const float*>& B_array,
    std::vector<float*>& C_array,
    int n, int lda, int batch_count)
{
    // OpenBLAS: Loop over individual sgemm calls
    // This is semantically identical to GPU's strided batched GEMM
    for (int i = 0; i < batch_count; ++i) {
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            n, n, n,
            1.0f,
            A_array[i], lda,
            B_array[i], lda,
            0.0f,
            C_array[i], lda
        );
    }
}

// ============================================================================
// Batch Size Determination
// ============================================================================

int determine_batch_size(
    float* A, float* B, float* C,
    int n, int lda, int ldc, int batch_count,
    float target_runtime_s)
{
    std::vector<const float*> A_array;
    std::vector<const float*> B_array;
    std::vector<float*> C_array;
    
    build_pointer_arrays(A, B, C, n, ldc, batch_count, A_array, B_array, C_array);
    
    // Warmups
    int warmup_count = (n <= 256) ? 8 : 2;
    for (int i = 0; i < warmup_count; ++i) {
        run_batched_gemm(A_array, B_array, C_array, n, lda, batch_count);
    }
    
    // Adaptive batch determination (like GPU version)
    int batches = 1;
    const int MAX_ITERATIONS = 20;
    
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        // Measure with current batch count
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < batches; ++i) {
            run_batched_gemm(A_array, B_array, C_array, n, lda, batch_count);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double t_total = std::chrono::duration<double>(end - start).count();
        
        // If we exceeded target, we're done
        if (t_total >= target_runtime_s) {
            return batches;
        }
        
        // Otherwise, estimate how many more batches we need
        double t_per_batch = t_total / batches;
        int estimated_batches = (int)std::ceil(target_runtime_s / t_per_batch);
        
        // Increase batches (but not by more than 10x at once to avoid overshooting)
        batches = std::min(estimated_batches, batches * 10);
        
        // Safety: ensure we make progress
        if (batches <= 1) batches = 2;
    }
    
    // If we still haven't reached target after MAX_ITERATIONS, return what we have
    return batches;
}

// ============================================================================
// Measurement Run
// ============================================================================

struct MeasurementResult {
    double kernel_time_s;
    double e2e_time_s;
    double wall_time_s;
    double total_energy_j;
    int batches;
    bool below_target;
    int64_t cpu_cycles;
    int64_t cpu_instructions;
    int64_t cpu_cache_misses;
};

MeasurementResult run_measurement(
    float* A, float* B, float* C,
    int n, int lda, int ldc, int batch_count,
    int batches, RAPLReader& rapl, PerfGroupCounter& perf_counter)
{
    MeasurementResult result;
    result.batches = batches;
    
    std::vector<const float*> A_array;
    std::vector<const float*> B_array;
    std::vector<float*> C_array;
    
    build_pointer_arrays(A, B, C, n, ldc, batch_count, A_array, B_array, C_array);
    
    // Energy measurement start
    double energy_before = rapl.read_energy_j();
    
    // Timing start
    auto start = std::chrono::high_resolution_clock::now();
    
    // Start perf counters - exactly around the batch loop
    perf_counter.start();
    
    // Run batches
    for (int i = 0; i < batches; ++i) {
        run_batched_gemm(A_array, B_array, C_array, n, lda, batch_count);
    }
    
    // Stop perf counters
    perf_counter.stop();
    
    // Timing end
    auto end = std::chrono::high_resolution_clock::now();
    
    // Energy measurement end
    double energy_after = rapl.read_energy_j();
    
    // Read perf counter values
    perf_counter.read(result.cpu_cycles, result.cpu_instructions, result.cpu_cache_misses);
    
    // Calculate times (all identical for CPU)
    result.wall_time_s = std::chrono::duration<double>(end - start).count();
    result.kernel_time_s = result.wall_time_s;
    result.e2e_time_s = result.wall_time_s;
    
    // Calculate energy
    if (energy_before >= 0 && energy_after >= 0) {
        result.total_energy_j = energy_after - energy_before;
    } else {
        result.total_energy_j = -1.0;
    }
    
    result.below_target = (result.wall_time_s < TARGET_RUNTIME_S);
    
    return result;
}

// ============================================================================
// CSV Output
// ============================================================================

void write_csv_header(std::ofstream& file) {
    file << "timestamp,run_id_global,run_id_per_size,device_name,num_threads,problem_size,"
         << "batches,batch_count,"
         << "gpu_e2e_time_s,gpu_kernel_time_s,wall_time_s,"
         << "total_energy_j,energy_per_batch_j,energy_per_second_j,energy_per_flop_j,"
         << "time_per_gemm_ms_kernel,time_per_gemm_ms_e2e,"
         << "flops_total,gflops_per_s,avg_power_w,below_target,"
         << "pcie_gen,pcie_width,sm_clock_mhz,mem_clock_mhz,temp_c,throttle_reasons,"
         << "cpu_cycles,cpu_instructions,cpu_ipc,cpu_cache_misses\n";
}

void write_csv_row(
    std::ofstream& file,
    int run_id_global,
    int run_id_per_size,
    const std::string& device_name,
    int num_threads,
    int problem_size,
    int batch_count,
    const MeasurementResult& result)
{
    // Calculate metrics
    long long total_instances = (long long)result.batches * batch_count;
    double flops_total = 2.0 * std::pow(problem_size, 3) * total_instances;
    
    double energy_per_batch_j = (result.total_energy_j >= 0) 
        ? (result.total_energy_j / result.batches) : -1.0;
    
    double energy_per_second_j = (result.total_energy_j >= 0)
        ? (result.total_energy_j / result.wall_time_s) : -1.0;
    
    double energy_per_flop_j = (result.total_energy_j >= 0)
        ? (result.total_energy_j / flops_total) : -1.0;
    
    double time_per_gemm_ms_kernel = 1e3 * result.kernel_time_s / total_instances;
    double time_per_gemm_ms_e2e = 1e3 * result.e2e_time_s / total_instances;
    
    double gflops_per_s = flops_total / result.kernel_time_s / 1e9;
    
    double avg_power_w = (result.total_energy_j >= 0)
        ? (result.total_energy_j / result.wall_time_s) : -1.0;
    
    char below_target = result.below_target ? 't' : 'f';
    
    int sm_clock_mhz = get_cpu_freq_mhz();
    int temp_c = get_cpu_temp_c();
    
    // Write row
    file << get_timestamp() << ","
         << run_id_global << ","
         << run_id_per_size << ","
         << device_name << ","
         << num_threads << ","
         << problem_size << ","
         << result.batches << ","
         << batch_count << ","
         << std::fixed << std::setprecision(6)
         << result.e2e_time_s << ","
         << result.kernel_time_s << ","
         << result.wall_time_s << ","
         << result.total_energy_j << ","
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
         << below_target << ","
         << "," // pcie_gen (empty for CPU)
         << "," // pcie_width (empty for CPU)
         << ((sm_clock_mhz > 0) ? std::to_string(sm_clock_mhz) : "") << ","
         << "," // mem_clock_mhz (empty for CPU)
         << ((temp_c > 0) ? std::to_string(temp_c) : "") << ","
         << ","  // throttle_reasons (empty for CPU)
         << result.cpu_cycles << ","
         << result.cpu_instructions << ","
         << std::fixed << std::setprecision(6)
         << ((result.cpu_cycles > 0 && result.cpu_instructions >= 0) 
             ? (static_cast<double>(result.cpu_instructions) / static_cast<double>(result.cpu_cycles)) 
             : -1.0) << ","
         << result.cpu_cache_misses << "\n";
}

// ============================================================================
// Main Benchmark Loop
// ============================================================================

void run_benchmark(const std::string& output_file, bool test_mode) {
    // Set OpenBLAS threads will be done per iteration
    std::cout << "Using OpenBLAS" << std::endl;
    
    // Get CPU info
    std::string cpu_model = get_cpu_model();
    std::cout << "CPU: " << cpu_model << std::endl;
    
    // Initialize RAPL
    RAPLReader rapl;
    
    // Initialize perf counters
    PerfGroupCounter perf_counter;
    bool perf_available = perf_counter.open();
    if (perf_available) {
        std::cout << "Perf counters: Available (cycles, instructions, cache-misses)" << std::endl;
    } else {
        std::cerr << "WARNING: Perf counters not accessible. cpu_cycles/instructions/ipc/cache_misses will be -1.\n"
                  << "Run as root or set /proc/sys/kernel/perf_event_paranoid to 1 or lower." << std::endl;
    }
    
    // Allocate matrices (Row-Major)
    // For strided batched GEMM, we need C large enough for:
    // max_batch_count * ldc * n_max
    // With n_max=16384, ldc=16384, max_batch_count=1024:
    // We need 1024 * 16384 * 64 = 1,073,741,824 elements for n=64, batch_count=1024
    // But we can reuse C buffer across different (n, batch_count) combinations
    // So we allocate for the worst case within our MAX_SIZE constraint
    
    // Actually, the issue is: we're using ldc=MAX_SIZE for all n!
    // For n=64 with batch_count=1024, we need: 1024 * MAX_SIZE * 64 elements
    // That's still too much. Let's use a smarter approach:
    // We'll allocate C dynamically per problem size
    
    std::cout << "Allocating base matrices..." << std::endl;
    std::cout << "  A, B: " << MAX_SIZE << "x" << MAX_SIZE 
              << " (" << (MAX_SIZE * MAX_SIZE * sizeof(float) / (1024.0 * 1024.0)) << " MB each)" << std::endl;
    
    float* A = new float[MAX_SIZE * MAX_SIZE];
    float* B = new float[MAX_SIZE * MAX_SIZE];
    
    fill_matrix(A, MAX_SIZE, MAX_SIZE, MAX_SIZE, 1.0f);
    fill_matrix(B, MAX_SIZE, MAX_SIZE, MAX_SIZE, 1.0f);
    
    // C will be allocated per problem size
    float* C = nullptr;
    
    // Open CSV file
    bool file_exists = (access(output_file.c_str(), F_OK) == 0);
    std::ofstream csv_file(output_file, std::ios::app);
    
    if (!file_exists) {
        write_csv_header(csv_file);
    }
    
    int run_id_global = 0;
    
    // Test mode: only smallest size
    std::vector<int> sizes_to_test = test_mode ? std::vector<int>{64} : PROBLEM_SIZES;
    std::vector<int> threads_to_test = test_mode ? std::vector<int>{1} : THREAD_COUNTS;
    
    // Benchmark loop
    for (int num_threads : threads_to_test) {
        // Set thread count for OpenBLAS
        openblas_set_num_threads(num_threads);
        
        std::cout << "\n=== Testing with " << num_threads << " threads ===" << std::endl;
        
        for (size_t size_idx = 0; size_idx < sizes_to_test.size(); ++size_idx) {
            int n = sizes_to_test[size_idx];
            const auto& batch_counts = BATCH_COUNT_MAP[size_idx];
            
            // Allocate C for this problem size
            // We need: max_batch_count_for_this_n * MAX_SIZE * n
            int max_bc = *std::max_element(batch_counts.begin(), batch_counts.end());
            long long c_size = (long long)max_bc * MAX_SIZE * n;
            
            if (C != nullptr) delete[] C;
            C = new float[c_size];
            std::fill(C, C + c_size, 0.0f);
            
            std::cout << "\n--- n=" << n << " (C buffer: " 
                      << (c_size * sizeof(float) / (1024.0 * 1024.0)) << " MB for up to " 
                      << max_bc << " batches) ---" << std::endl;
            
            int run_id_per_size = 0;
            
            for (int batch_count : batch_counts) {
                std::cout << "n=" << n << ", batch_count=" << batch_count << " ... " << std::flush;
                
                // Determine batches once
                int batches = determine_batch_size(A, B, C, n, MAX_SIZE, MAX_SIZE, batch_count, TARGET_RUNTIME_S);
                
                std::cout << "(" << batches << " batches per run, " << REPETITIONS << " reps)\n";
                
                // Run 50 repetitions
                for (int rep = 0; rep < REPETITIONS; ++rep) {
                    run_id_global++;
                    run_id_per_size++;
                    
                    // Run measurement
                    auto result = run_measurement(A, B, C, n, MAX_SIZE, MAX_SIZE, batch_count, batches, rapl, perf_counter);
                    
                    // Write to CSV
                    write_csv_row(csv_file, run_id_global, run_id_per_size, cpu_model, 
                                 num_threads, n, batch_count, result);
                    csv_file.flush();
                    
                    // Output every measurement (like GPU version)
                    long long total_instances = (long long)batches * batch_count;
                    double gflops = 2.0 * std::pow(n, 3) * total_instances / result.kernel_time_s / 1e9;
                    double avg_power_w = (result.total_energy_j >= 0) 
                        ? (result.total_energy_j / result.wall_time_s) : -1.0;
                    
                    std::cout << "  [" << (rep + 1) << "/" << REPETITIONS << "] "
                              << std::fixed << std::setprecision(2)
                              << result.wall_time_s << "s, "
                              << gflops << " GFLOPS, "
                              << ((result.total_energy_j >= 0) ? std::to_string((int)result.total_energy_j) : "N/A") << " J, "
                              << ((avg_power_w >= 0) ? std::to_string((int)avg_power_w) : "N/A") << " W"
                              << std::endl;
                }
                
                // Cooldown after all 50 repetitions (except last measurement point)
                if (!(size_idx == sizes_to_test.size() - 1 && batch_count == batch_counts.back())) {
                    std::cout << "  Cooldown 30s..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(COOLDOWN_MS));
                }
            }
        }
    }
    
    // Cleanup
    if (C != nullptr) delete[] C;
    delete[] A;
    delete[] B;
    
    csv_file.close();
    std::cout << "\nBenchmark complete! Results written to: " << output_file << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string output_file = "cpu_gemm_benchmark.csv";
    bool test_mode = false;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test" || arg == "-t") {
            test_mode = true;
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -t, --test           Test mode (only n=64, 1 thread)\n"
                      << "  -o, --output FILE    Output CSV file (default: cpu_gemm_benchmark.csv)\n"
                      << "  -h, --help           Show this help\n";
            return 0;
        }
    }
    
    std::cout << "=== CPU Strided-Batched GEMM Benchmark ===" << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "Test mode: " << (test_mode ? "yes" : "no") << std::endl;
    
    run_benchmark(output_file, test_mode);
    
    return 0;
}
