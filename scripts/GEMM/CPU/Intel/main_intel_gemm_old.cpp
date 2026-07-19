// main.cpp - CPU GEMM Energy Benchmark with Multi-Thread Support
//
// Build: g++ -O3 -march=native -std=c++17 -o cpu_bench main.cpp -lopenblas
// Usage: ./cpu_bench [--test|-t] [--output|-o <path>]
//
// Requirements:
// - OpenBLAS (CBLAS API)
// - CPU stabilization via 01_enable_CPU_Intel.sh (run before this script)
// - RAPL access for energy measurements (falls back to -1 if not available)

#include <cblas.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>
#include <vector>
#include <glob.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cmath>
#include <cstdint>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>

// ============================================================================
// Configuration
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    MAX_BATCH_SIZE   = 250000;
constexpr int    MACRO_REPEATS    = 50;  // 50 runs per size (same as GPU)
constexpr int    WARMUP_SIZE      = 512;

// GEMM sizes - same as GPU variant (2^x steps: 64 to 16384)
static const int GEMM_SIZES[] = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
};
static const int NUM_SIZES = sizeof(GEMM_SIZES) / sizeof(GEMM_SIZES[0]);
static const int MAX_SIZE = *std::max_element(std::begin(GEMM_SIZES), std::end(GEMM_SIZES));

// Thread counts to test
static const int THREAD_COUNTS[] = {1, 2, 4, 8, 10, 16, 20, 32, 64};
static const int NUM_THREAD_COUNTS = sizeof(THREAD_COUNTS) / sizeof(THREAD_COUNTS[0]);

// ============================================================================
// OpenBLAS Threading
// ============================================================================

extern "C" {
    void openblas_set_num_threads(int num_threads);
    int  openblas_get_num_threads();
}

// ============================================================================
// Perf Event Counter Group (cycles, instructions, cache-misses)
// ============================================================================

// Wrapper for perf_event_open syscall
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
                            }

                            // Data structure for reading group counters (read_format = PERF_FORMAT_GROUP)
                            struct PerfReadFormat {
                                uint64_t nr;            // Number of events
                                uint64_t values[3];     // cycles, instructions, cache-misses
                            };

                            // Hardware performance counter group
                            // Measures: cycles, instructions, cache-misses for the timed region only
                            // Uses group leader pattern to minimize multiplexing
                            class PerfGroupCounter {
                            public:
                                PerfGroupCounter() : fd_leader_(-1), fd_instructions_(-1), fd_cache_misses_(-1), available_(false) {}

                                ~PerfGroupCounter() {
                                    close_fds();
                                }

                                // Open perf events as a group (cycles as leader)
                                // Returns true if successful, false if perf_event_open not available
                                bool open() {
                                    close_fds();  // Ensure clean state

                                    struct perf_event_attr pe;
                                    memset(&pe, 0, sizeof(pe));
                                    pe.size = sizeof(pe);
                                    pe.disabled = 1;           // Start disabled, enable with ioctl
                                    pe.exclude_kernel = 1;     // User-space only
                                    pe.exclude_hv = 1;         // Exclude hypervisor
                                    pe.inherit = 1;            // Count child threads (OpenBLAS spawns threads)
                                    pe.read_format = PERF_FORMAT_GROUP;  // Read all counters together

                                    // Group leader: cycles
                                    pe.type = PERF_TYPE_HARDWARE;
                                    pe.config = PERF_COUNT_HW_CPU_CYCLES;
                                    fd_leader_ = perf_event_open(&pe, 0, -1, -1, 0);
                                    if (fd_leader_ < 0) {
                                        // perf_event_open not available (permission denied or not supported)
                                        available_ = false;
                                        return false;
                                    }

                                    // Group member: instructions
                                    pe.disabled = 0;  // Members start enabled when leader starts
                                    pe.config = PERF_COUNT_HW_INSTRUCTIONS;
                                    fd_instructions_ = perf_event_open(&pe, 0, -1, fd_leader_, 0);
                                    if (fd_instructions_ < 0) {
                                        close_fds();
                                        available_ = false;
                                        return false;
                                    }

                                    // Group member: cache-misses
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

                                // Reset and enable counters (call before measured region)
                                void start() {
                                    if (!available_) return;
                                    ioctl(fd_leader_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
                                    ioctl(fd_leader_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
                                }

                                // Disable counters (call after measured region)
                                void stop() {
                                    if (!available_) return;
                                    ioctl(fd_leader_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
                                }

                                // Read counter values after stop()
                                // Returns: cycles, instructions, cache_misses (-1 if not available)
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
                                        // Read failed or wrong number of events
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
                            // Utility Functions
                            // ============================================================================

                            std::string getTimestamp() {
                                auto now = std::time(nullptr);
                                auto tm = *std::localtime(&now);
                                std::ostringstream oss;
                                oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
                                return oss.str();
                            }

                            std::string getCPUModel() {
                                // Try lscpu first (more robust)
                                FILE* pipe = popen("lscpu 2>/dev/null | grep 'Model name:' | sed 's/Model name:\\s*//'", "r");
                                if (pipe) {
                                    char buffer[256];
                                    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                                        pclose(pipe);
                                        std::string model(buffer);
                                        // Remove trailing newline
                                        if (!model.empty() && model.back() == '\n') {
                                            model.pop_back();
                                        }
                                        if (!model.empty()) {
                                            return model;
                                        }
                                    }
                                    pclose(pipe);
                                }

                                // Fallback: /proc/cpuinfo
                                std::ifstream cpuinfo("/proc/cpuinfo");
                                std::string line;
                                while (std::getline(cpuinfo, line)) {
                                    if (line.find("model name") != std::string::npos) {
                                        size_t pos = line.find(':');
                                        if (pos != std::string::npos) {
                                            std::string model = line.substr(pos + 1);
                                            // Trim leading spaces
                                            model.erase(0, model.find_first_not_of(" \t"));
                                            return model;
                                        }
                                    }
                                }

                                return "Unknown CPU";
                            }

                            std::string getShortCPUName(const std::string& full_name) {
                                // Extract short CPU name (e.g., "Intel Xeon E5-2680 v4" from full name)
                                std::string name = full_name;

                                // Remove frequency info (@ X.XXGHz)
                                size_t at_pos = name.find('@');
                                if (at_pos != std::string::npos) {
                                    name = name.substr(0, at_pos);
                                }

                                // Remove (R), (TM), "CPU", "Processor" tokens
                                const char* tokens[] = {"(R)", "(TM)", " CPU ", " Processor"};
                                for (const char* token : tokens) {
                                    size_t pos;
                                    while ((pos = name.find(token)) != std::string::npos) {
                                        name.erase(pos, strlen(token));
                                    }
                                }

                                // Trim trailing spaces
                                size_t end = name.find_last_not_of(" \t\n\r");
                                if (end != std::string::npos) {
                                    name = name.substr(0, end + 1);
                                }

                                // Ensure "Intel" prefix (or AMD for AMD CPUs)
                                if (name.find("Intel") == std::string::npos && name.find("AMD") == std::string::npos) {
                                    // Check if it's Intel or AMD
                                    if (full_name.find("Intel") != std::string::npos) {
                                        name = "Intel " + name;
                                    } else if (full_name.find("AMD") != std::string::npos) {
                                        name = "AMD " + name;
                                    }
                                }

                                return name;
                            }

                            void ensureDirectoryExists(const char* filepath) {
                                std::string path(filepath);
                                size_t pos = path.find_last_of('/');
                                if (pos == std::string::npos) {
                                    return;  // No directory component
                                }

                                std::string dir = path.substr(0, pos);
                                if (dir.empty()) {
                                    return;
                                }

                                // Create directories recursively
                                std::string current = "";
                                size_t start = 0;

                                while (start < dir.length()) {
                                    size_t end = dir.find('/', start);
                                    if (end == std::string::npos) {
                                        end = dir.length();
                                    }

                                    if (end > start) {
                                        current += dir.substr(start, end - start) + "/";
                                        mkdir(current.c_str(), 0755);  // Ignore errors if exists
                                    }

                                    start = end + 1;
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
                            // RAPL Energy Measurement
                            // ============================================================================

                            std::vector<std::string> globPaths(const std::string& pattern) {
                                glob_t glob_result;
                                std::vector<std::string> paths;

                                if (glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result) == 0) {
                                    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
                                        paths.push_back(std::string(glob_result.gl_pathv[i]));
                                    }
                                }
                                globfree(&glob_result);
                                return paths;
                            }

                            std::string readFile(const std::string& path) {
                                std::ifstream file(path);
                                if (!file.is_open()) {
                                    return "";
                                }
                                std::string content;
                                std::getline(file, content);
                                return content;
                            }

                            double readRAPLEnergy() {
                                // Read sum of all package-zone energies from RAPL
                                // Returns energy in Joules, or -1.0 if not accessible

                                std::vector<std::string> candidates;

                                // Collect all RAPL zones
                                auto base_zones = globPaths("/sys/class/powercap/*rapl*");
                                candidates.insert(candidates.end(), base_zones.begin(), base_zones.end());

                                auto sub_zones = globPaths("/sys/class/powercap/*rapl*:*");
                                candidates.insert(candidates.end(), sub_zones.begin(), sub_zones.end());

                                auto subsub_zones = globPaths("/sys/class/powercap/*rapl*:*:*");
                                candidates.insert(candidates.end(), subsub_zones.begin(), subsub_zones.end());

                                // Filter for package-zones only (ignore dram, core, etc.)
                                std::vector<std::string> package_zones;
                                for (const auto& zone : candidates) {
                                    std::string name_path = zone + "/name";
                                    std::string name = readFile(name_path);
                                    if (name.find("package") != std::string::npos) {
                                        package_zones.push_back(zone);
                                    }
                                }

                                if (package_zones.empty()) {
                                    return -1.0;
                                }

                                // Sum energy from all package zones
                                double total_energy_uj = 0.0;
                                for (const auto& zone : package_zones) {
                                    std::string energy_path = zone + "/energy_uj";
                                    std::string energy_str = readFile(energy_path);
                                    if (!energy_str.empty()) {
                                        try {
                                            double energy_uj = std::stod(energy_str);
                                            total_energy_uj += energy_uj;
                                        } catch (...) {
                                            return -1.0;
                                        }
                                    } else {
                                        return -1.0;
                                    }
                                }

                                // Convert µJ to J
                                return total_energy_uj / 1e6;
                            }

                            // ============================================================================
                            // CPU Telemetry
                            // ============================================================================

                            double readCPUTemperature() {
                                // Try coretemp (Intel) first
                                auto coretemp_paths = globPaths("/sys/class/hwmon/hwmon*/temp*_input");
                                for (const auto& path : coretemp_paths) {
                                    // Check if this is a coretemp sensor
                                    std::string hwmon_dir = path.substr(0, path.rfind('/'));
                                    std::string name = readFile(hwmon_dir + "/name");
                                    if (name.find("coretemp") != std::string::npos || name.find("k10temp") != std::string::npos) {
                                        std::string temp_str = readFile(path);
                                        if (!temp_str.empty()) {
                                            try {
                                                // Temperature is in millidegrees Celsius
                                                return std::stod(temp_str) / 1000.0;
                                            } catch (...) {
                                                continue;
                                            }
                                        }
                                    }
                                }

                                return -1.0;
                            }

                            unsigned int readCPUClock() {
                                // Read current CPU frequency from sysfs (in kHz)
                                std::string freq_str = readFile("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
                                if (!freq_str.empty()) {
                                    try {
                                        unsigned int freq_khz = std::stoul(freq_str);
                                        return freq_khz / 1000;  // Convert to MHz
                                    } catch (...) {
                                        return 0;
                                    }
                                }
                                return 0;
                            }

                            // ============================================================================
                            // Batch Size Calibration
                            // ============================================================================

                            int determineBatchSize(float* h_A, float* h_B, float* h_C, int n, int lda, double target_time) {
                                const float alpha = 1.0f;
                                const float beta = 0.0f;

                                int batches = 1;

                                while (batches <= MAX_BATCH_SIZE) {
                                    auto start = std::chrono::steady_clock::now();
                                    for (int b = 0; b < batches; b++) {
                                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                                    n, n, n, alpha, h_A, lda, h_B, lda, beta, h_C, lda);
                                    }
                                    auto end = std::chrono::steady_clock::now();
                                    std::chrono::duration<double> elapsed = end - start;
                                    double measured_time = elapsed.count();

                                    // Guard: if measured_time is 0 or too small, just double batches
                                    if (measured_time <= 1e-9) {
                                        batches = std::min(batches * 2, MAX_BATCH_SIZE);
                                        continue;
                                    }

                                    // Done if >= target_time
                                    if (measured_time >= target_time) {
                                        return batches;
                                    }

                                    // Guard: if batches=1 already exceeds target, we can't go lower
                                    // (This case is handled above, but explicit for clarity)

                                    // Scale up: estimate how many batches needed for target_time
                                    double time_per_batch = measured_time / batches;
                                    int needed_batches = static_cast<int>(std::ceil(target_time / time_per_batch));

                                    // Ensure progress (at least 2x)
                                    if (needed_batches <= batches) {
                                        needed_batches = batches * 2;
                                    }

                                    batches = std::min(needed_batches, MAX_BATCH_SIZE);
                                }

                                // Hit MAX_BATCH_SIZE without reaching target_time
                                return MAX_BATCH_SIZE;
                            }

                            // ============================================================================
                            // CSV Output
                            // ============================================================================

                            void writeCSVHeader(std::ofstream& file) {
                                file << "timestamp,"
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
                                << "throttle_reasons,"
                                << "cpu_cycles,"
                                << "cpu_instructions,"
                                << "cpu_ipc,"
                                << "cpu_cache_misses\n";
                            }

                            void writeCSVRow(std::ofstream& file,
                                             int run_id_global, int run_id_per_size,
                                             const std::string& device_name, int num_threads,
                                             int n, int batches,
                                             double compute_time_s, double energy_j, double avg_power_w,
                                             bool below_target, unsigned int cpu_clock_mhz, double temp_c,
                                             int64_t cpu_cycles, int64_t cpu_instructions, int64_t cpu_cache_misses) {

                                // Compute derived metrics
                                double flops_total = 2.0 * n * n * n * batches;  // 2N³ per GEMM
                                double gflops_per_s = (flops_total / compute_time_s) / 1e9;
                                double energy_per_batch_j = (energy_j >= 0) ? (energy_j / batches) : -1.0;
                                double energy_per_second_j = (energy_j >= 0) ? (energy_j / compute_time_s) : -1.0;
                                double energy_per_flop_j = (energy_j >= 0) ? (energy_j / flops_total) : -1.0;
                                double time_per_gemm_ms = (compute_time_s / batches) * 1000.0;

                                // Compute cpu_ipc: only if both cycles and instructions are valid (>= 0) and cycles > 0
                                // Acceptance test: cpu_ipc should typically be ~0.2-3.0 for valid measurements
                                double cpu_ipc = -1.0;
                                if (cpu_cycles > 0 && cpu_instructions >= 0) {
                                    cpu_ipc = static_cast<double>(cpu_instructions) / static_cast<double>(cpu_cycles);
                                }

                                // Write CSV row (CPU has identical values for all time fields)
                                file << getTimestamp() << ","
                                << run_id_global << ","
                                << run_id_per_size << ","
                                << device_name << ","
                                << num_threads << ","
                                << n << ","
                                << batches << ","
                                << std::fixed << std::setprecision(6) << compute_time_s << ","
                                << std::fixed << std::setprecision(6) << compute_time_s << ","
                                << std::fixed << std::setprecision(6) << compute_time_s << ","
                                << std::fixed << std::setprecision(6) << energy_j << ","
                                << std::scientific << std::setprecision(6) << energy_per_batch_j << ","
                                << std::fixed << std::setprecision(6) << energy_per_second_j << ","
                                << std::scientific << std::setprecision(6) << energy_per_flop_j << ","
                                << std::fixed << std::setprecision(6) << time_per_gemm_ms << ","
                                << std::fixed << std::setprecision(6) << time_per_gemm_ms << ","
                                << std::scientific << std::setprecision(6) << flops_total << ","
                                << std::fixed << std::setprecision(2) << gflops_per_s << ","
                                << std::fixed << std::setprecision(2) << avg_power_w << ","
                                << (below_target ? "t" : "f") << ","
                                << ","  // pcie_gen (empty for CPU)
                                << ","  // pcie_width (empty for CPU)
                                << cpu_clock_mhz << ","
                                << ","  // mem_clock_mhz (empty for CPU)
                                << std::fixed << std::setprecision(0) << temp_c << ","
                                << ","  // throttle_reasons (empty for CPU)
                                // Columns 27-30: perf counters (acceptance test: -1 if perf not available, else >= 0)
                                << cpu_cycles << ","
                                << cpu_instructions << ","
                                << std::fixed << std::setprecision(6) << cpu_ipc << ","
                                << cpu_cache_misses << "\n";
                                             }

                                             // ============================================================================
                                             // Main Benchmark
                                             // ============================================================================

                                             int main(int argc, char* argv[]) {
                                                 // Parse command-line arguments
                                                 bool test_mode = false;
                                                 std::string output_file = "data/raw/energy_benchmark_cpu.csv";

                                                 for (int i = 1; i < argc; i++) {
                                                     if (strcmp(argv[i], "--test") == 0 || strcmp(argv[i], "-t") == 0) {
                                                         test_mode = true;
                                                     } else if ((strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) && i + 1 < argc) {
                                                         std::string user_path = argv[++i];

                                                         // Check if user_path ends with / or is . or ./
                                                         bool is_dir = false;
                                                         if (!user_path.empty()) {
                                                             is_dir = (user_path.back() == '/' || user_path == "." || user_path == "./");
                                                         }

                                                         // Check if it's an existing directory
                                                         struct stat st;
                                                         if (!is_dir && stat(user_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
                                                             is_dir = true;
                                                         }

                                                         if (is_dir) {
                                                             // It's a directory - append default filename
                                                             if (!user_path.empty() && user_path.back() != '/') {
                                                                 user_path += "/";
                                                             }
                                                             output_file = user_path + "energy_benchmark_cpu.csv";
                                                         } else {
                                                             // It's a file path
                                                             output_file = user_path;
                                                         }
                                                     }
                                                 }

                                                 if (test_mode) {
                                                     std::cout << "TEST MODE: Will run only 5 measurements and exit\n";
                                                 }

                                                 // Get system info
                                                 std::string cpu_model_full = getCPUModel();
                                                 std::string device_name = getShortCPUName(cpu_model_full);

                                                 // Test RAPL availability
                                                 double rapl_test = readRAPLEnergy();
                                                 bool rapl_available = (rapl_test >= 0);

                                                 if (!rapl_available) {
                                                     std::cerr << "\nWARNING: RAPL not accessible (permission denied).\n";
                                                     std::cerr << "Energy/power measurements will be -1.\n";
                                                     std::cerr << "Run as root or grant RAPL access to enable energy measurements.\n\n";
                                                 }

                                                 // Print configuration
                                                 std::cout << "========================================\n";
                                                 std::cout << "CPU GEMM Energy Benchmark (Multi-Thread)\n";
                                                 std::cout << "========================================\n";
                                                 std::cout << "CPU (full):     " << cpu_model_full << "\n";
                                                 std::cout << "Device name:    " << device_name << "\n";
                                                 std::cout << "Thread counts:  ";
                                                 for (int i = 0; i < NUM_THREAD_COUNTS; i++) {
                                                     std::cout << THREAD_COUNTS[i];
                                                     if (i < NUM_THREAD_COUNTS - 1) std::cout << ", ";
                                                 }
                                                 std::cout << "\n";
                                                 std::cout << "Target runtime: " << TARGET_RUNTIME_S << "s\n";
                                                 std::cout << "Macro repeats:  " << MACRO_REPEATS << " (per size, per thread count)\n";
                                                 std::cout << "Matrix sizes:   " << NUM_SIZES << " sizes (64-16384, 2^x steps)\n";
                                                 std::cout << "RAPL available: " << (rapl_available ? "Yes" : "No") << "\n";
                                                 std::cout << "Output:         " << output_file << "\n";
                                                 std::cout << "Total runs:     " << (NUM_THREAD_COUNTS * NUM_SIZES * MACRO_REPEATS) << "\n";
                                                 std::cout << "========================================\n\n";

                                                 // Allocate aligned host memory (64-byte aligned for cache line)
                                                 const size_t max_elements = MAX_SIZE * MAX_SIZE;
                                                 const size_t max_bytes = max_elements * sizeof(float);

                                                 float *h_A, *h_B, *h_C;
                                                 if (posix_memalign((void**)&h_A, 64, max_bytes) != 0 ||
                                                     posix_memalign((void**)&h_B, 64, max_bytes) != 0 ||
                                                     posix_memalign((void**)&h_C, 64, max_bytes) != 0) {
                                                     std::cerr << "Error: Failed to allocate aligned memory\n";
                                                 return EXIT_FAILURE;
                                                     }

                                                     std::cout << "Allocated aligned host buffers: "
                                                     << (3 * max_bytes / (1024*1024)) << " MB\n";

                                                     // Initialize matrices for maximum size
                                                     std::cout << "Initializing host matrices...\n";
                                                     initializeMatrix(h_A, MAX_SIZE, MAX_SIZE, 42);
                                                     initializeMatrix(h_B, MAX_SIZE, MAX_SIZE, 43);

                                                     // Warm-up: single SGEMM call with 1 thread
                                                     std::cout << "Running warm-up (512x512, 1 thread)...\n";
                                                     openblas_set_num_threads(1);
                                                     const float alpha = 1.0f;
                                                     const float beta = 0.0f;
                                                     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                                                 WARMUP_SIZE, WARMUP_SIZE, WARMUP_SIZE,
                                                                 alpha, h_A, MAX_SIZE, h_B, MAX_SIZE, beta, h_C, MAX_SIZE);
                                                     std::cout << "Warm-up complete.\n\n";

                                                     // Prepare CSV output - append mode (never overwrite)
                                                     ensureDirectoryExists(output_file.c_str());
                                                     bool write_header = !fileExists(output_file.c_str());

                                                     std::ofstream csv_file(output_file, std::ios::app);
                                                     if (!csv_file.is_open()) {
                                                         std::cerr << "Error: Cannot open output file: " << output_file << "\n";
                                                         free(h_A);
                                                         free(h_B);
                                                         free(h_C);
                                                         return EXIT_FAILURE;
                                                     }

                                                     if (write_header) {
                                                         writeCSVHeader(csv_file);
                                                     }

                                                     // ========================================================================
                                                     // Main measurement loop: sweep over thread counts, then sizes
                                                     // ========================================================================

                                                     std::cout << "Starting measurements...\n";
                                                     std::cout << "========================================\n\n";

                                                     // Initialize perf counter group (cycles, instructions, cache-misses)
                                                     // Opened once; inherit=1 ensures child threads (OpenBLAS workers) are counted
                                                     PerfGroupCounter perf_counter;
                                                     bool perf_available = perf_counter.open();

                                                     if (perf_available) {
                                                         std::cout << "Perf counters: Available (cycles, instructions, cache-misses)\n";
                                                     } else {
                                                         std::cerr << "\nWARNING: Perf counters not accessible (permission denied or not supported).\n";
                                                         std::cerr << "cpu_cycles, cpu_instructions, cpu_ipc, cpu_cache_misses will be -1.\n";
                                                         std::cerr << "Run as root or set /proc/sys/kernel/perf_event_paranoid to 1 or lower.\n\n";
                                                     }

                                                     int run_id_global = 1;
                                                     int total_rows = 0;

                                                     for (int thread_idx = 0; thread_idx < NUM_THREAD_COUNTS; thread_idx++) {
                                                         int num_threads = THREAD_COUNTS[thread_idx];
                                                         openblas_set_num_threads(num_threads);

                                                         std::cout << "\n=== THREAD COUNT: " << num_threads << " (confirmed: "
                                                         << openblas_get_num_threads() << ") ===\n\n";

                                                         for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
                                                             int n = GEMM_SIZES[size_idx];
                                                             int run_id_per_size = 1;  // Reset for each new problem size

                                                             std::cout << "GEMM size " << n << "x" << n << "\n";

                                                             // Determine batch size for this matrix size and thread count
                                                             std::cout << "  Determining batch size... " << std::flush;
                                                             int batches = determineBatchSize(h_A, h_B, h_C, n, MAX_SIZE, TARGET_RUNTIME_S);
                                                             std::cout << "using " << batches << " batches\n";

                                                             // Run MACRO_REPEATS measurements
                                                             for (int rep = 0; rep < MACRO_REPEATS; rep++) {
                                                                 // ============================================================
                                                                 // Measurement Start
                                                                 // ============================================================

                                                                 auto wall_start = std::chrono::steady_clock::now();
                                                                 double energy_before = readRAPLEnergy();

                                                                 // Start perf counters (reset + enable) - exactly around the batch loop
                                                                 perf_counter.start();

                                                                 // Execute batch - this is the ONLY code measured by perf counters
                                                                 for (int b = 0; b < batches; b++) {
                                                                     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                                                                 n, n, n, alpha, h_A, MAX_SIZE, h_B, MAX_SIZE,
                                                                                 beta, h_C, MAX_SIZE);
                                                                 }

                                                                 // Stop perf counters (disable) - end of measured region
                                                                 perf_counter.stop();

                                                                 double energy_after = readRAPLEnergy();
                                                                 auto wall_end = std::chrono::steady_clock::now();

                                                                 // ============================================================
                                                                 // Measurement End
                                                                 // ============================================================

                                                                 // Read perf counter values
                                                                 // Acceptance test: if perf available, values >= 0; if not, all -1
                                                                 int64_t cpu_cycles, cpu_instructions, cpu_cache_misses;
                                                                 perf_counter.read(cpu_cycles, cpu_instructions, cpu_cache_misses);

                                                                 // Calculate timing
                                                                 std::chrono::duration<double> wall_duration = wall_end - wall_start;
                                                                 double compute_time_s = wall_duration.count();

                                                                 // Calculate energy and power
                                                                 double energy_j = -1.0;
                                                                 double avg_power_w = -1.0;

                                                                 if (energy_before >= 0 && energy_after >= 0) {
                                                                     double delta = energy_after - energy_before;
                                                                     if (delta >= 0) {
                                                                         energy_j = delta;
                                                                         avg_power_w = energy_j / compute_time_s;
                                                                     }
                                                                 }

                                                                 // Read CPU telemetry
                                                                 double temp_c = readCPUTemperature();
                                                                 unsigned int cpu_clock_mhz = readCPUClock();

                                                                 // Check if below target
                                                                 bool below_target = (compute_time_s < TARGET_RUNTIME_S);

                                                                 // Write to CSV
                                                                 writeCSVRow(csv_file, run_id_global, run_id_per_size,
                                                                             device_name, num_threads, n, batches,
                                                                             compute_time_s, energy_j, avg_power_w,
                                                                             below_target, cpu_clock_mhz, temp_c,
                                                                             cpu_cycles, cpu_instructions, cpu_cache_misses);
                                                                 csv_file.flush();

                                                                 // Increment counters
                                                                 run_id_global++;
                                                                 run_id_per_size++;
                                                                 total_rows++;

                                                                 // Test mode: exit after 5 rows
                                                                 if (test_mode && total_rows >= 5) {
                                                                     std::cout << "\nTEST MODE: Reached 5 measurements, exiting.\n";
                                                                     csv_file.close();
                                                                     free(h_A);
                                                                     free(h_B);
                                                                     free(h_C);
                                                                     return EXIT_SUCCESS;
                                                                 }

                                                                 // Console progress
                                                                 char check = below_target ? '!' : '+';
                                                                 std::cout << "  " << check << " Run " << (rep + 1) << "/"
                                                                 << MACRO_REPEATS << ": "
                                                                 << std::fixed << std::setprecision(3) << compute_time_s << "s";

                                                                 if (energy_j >= 0) {
                                                                     std::cout << " | E=" << std::setprecision(1) << energy_j << "J";
                                                                 }
                                                                 if (avg_power_w >= 0) {
                                                                     std::cout << " P=" << std::setprecision(0) << avg_power_w << "W";
                                                                 }
                                                                 if (cpu_clock_mhz > 0) {
                                                                     std::cout << " F=" << cpu_clock_mhz << "MHz";
                                                                 }
                                                                 if (temp_c >= 0) {
                                                                     std::cout << " T=" << std::setprecision(1) << temp_c << "°C";
                                                                 }
                                                                 if (below_target) {
                                                                     std::cout << " (!)";
                                                                 }
                                                                 std::cout << "\n";
                                                             }

                                                             // Cooling pause after each size (except last in current thread count)
                                                             if (size_idx < NUM_SIZES - 1) {
                                                                 std::cout << "  Cooling down for 60 seconds...\n";
                                                                 std::this_thread::sleep_for(std::chrono::seconds(60));
                                                             }

                                                             std::cout << "\n";
                                                         }

                                                         // Longer cooling pause between thread counts (except after last)
                                                         if (thread_idx < NUM_THREAD_COUNTS - 1) {
                                                             std::cout << "=== Cooling down for 60 seconds before next thread count ===\n\n";
                                                             std::this_thread::sleep_for(std::chrono::seconds(60));
                                                         }
                                                     }

                                                     std::cout << "========================================\n";
                                                     std::cout << "Benchmark complete!\n";
                                                     std::cout << "Results saved to: " << output_file << "\n";
                                                     std::cout << "Total measurements: " << total_rows << "\n";
                                                     std::cout << "========================================\n";

                                                     // Cleanup
                                                     csv_file.close();
                                                     free(h_A);
                                                     free(h_B);
                                                     free(h_C);

                                                     return EXIT_SUCCESS;
                                             }
