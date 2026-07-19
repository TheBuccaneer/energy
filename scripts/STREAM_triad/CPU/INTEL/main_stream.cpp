// ============================================================================
// CPU STREAM Triad Energy Benchmark with Adaptive Batching
// Updated to match CSV_COLUMNS.md format and use adaptive batching logic
// Variante A: Each of 50 runs per config writes its own CSV row (no aggregation)
// ============================================================================
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o stream_cpu main.cpp -lpthread -lm
// Usage: ./stream_cpu [--test] [--output <path>]

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
#include <vector>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <omp.h>

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
                                    pe.inherit = 1;            // Count child threads (OpenMP workers)
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
                            // Configuration - Hardcoded
                            // ============================================================================

                            constexpr double TARGET_RUNTIME_S = 1.0;
                            constexpr int    REPEATS          = 50;  // 50 measurements per configuration (each writes a CSV row)
                            constexpr int    MAX_BATCH_SIZE   = 1000000;  // Upper limit for batches

                            // Hardcoded thread counts (as per CSV_COLUMNS.md)
                            static const int THREAD_COUNTS[] = {1, 2, 4, 8, 10, 16, 20, 32, 64};
                            static const int NUM_THREAD_CONFIGS = sizeof(THREAD_COUNTS) / sizeof(THREAD_COUNTS[0]);

                            // Problem sizes in number of elements (matching Reduction CPU exactly)
                            static const int PROBLEM_SIZES[] = {
                                1000000, 2000000, 4000000, 8000000, 16000000, 32000000,
                                64000000, 128000000, 256000000
                            };
                            static const int NUM_SIZES = sizeof(PROBLEM_SIZES) / sizeof(PROBLEM_SIZES[0]);
                            static const int MAX_N = *std::max_element(std::begin(PROBLEM_SIZES), std::end(PROBLEM_SIZES));

                            static const char* DEFAULT_OUTPUT_FILE = "stream_cpu_amd.csv";

                            // STREAM Triad scalar constant
                            static const float SCALAR = 3.0f;

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
                                std::ifstream cpuinfo("/proc/cpuinfo");
                                std::string line;
                                while (std::getline(cpuinfo, line)) {
                                    if (line.find("model name") != std::string::npos) {
                                        size_t pos = line.find(':');
                                        if (pos != std::string::npos) {
                                            std::string model = line.substr(pos + 2);
                                            // Remove vendor prefixes for cleaner output
                                            size_t intel_pos = model.find("Intel(R)");
                                            if (intel_pos != std::string::npos) {
                                                model = model.substr(intel_pos + 9);
                                            }
                                            size_t amd_pos = model.find("AMD");
                                            if (amd_pos != std::string::npos) {
                                                model = model.substr(amd_pos + 4);
                                            }
                                            model.erase(0, model.find_first_not_of(" \t"));
                                            model.erase(model.find_last_not_of(" \t") + 1);
                                            return model;
                                        }
                                    }
                                }
                                return "Unknown CPU";
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
                            // RAPL Energy Reading (powercap interface) - PER ZONE with Overflow Handling
                            // ============================================================================

                            struct RAPLZone {
                                std::string path;
                                std::string name;
                                unsigned long long max_energy_range_uj;
                            };

                            std::vector<RAPLZone> discoverRAPLZones() {
                                std::vector<RAPLZone> zones;
                                const char* base_path = "/sys/class/powercap";

                                DIR* dir = opendir(base_path);
                                if (!dir) {
                                    return zones;
                                }

                                struct dirent* entry;
                                while ((entry = readdir(dir)) != nullptr) {
                                    std::string name = entry->d_name;
                                    if (name.find("rapl") != std::string::npos) {
                                        std::string zone_path = std::string(base_path) + "/" + name;

                                        // Read zone name
                                        std::ifstream name_file(zone_path + "/name");
                                        std::string zone_name;
                                        if (name_file.is_open()) {
                                            std::getline(name_file, zone_name);
                                        }

                                        // Only package zones
                                        bool is_pkg = (zone_name.find("package") != std::string::npos);

                                        if (is_pkg) {
                                            RAPLZone zone;
                                            zone.path = zone_path;
                                            zone.name = zone_name;

                                            // Read max_energy_range_uj for overflow detection
                                            std::ifstream range_file(zone_path + "/max_energy_range_uj");
                                            if (range_file.is_open()) {
                                                range_file >> zone.max_energy_range_uj;
                                            } else {
                                                zone.max_energy_range_uj = 0;
                                            }

                                            zones.push_back(zone);
                                        }
                                    }
                                }
                                closedir(dir);
                                return zones;
                            }

                            // Read energy from all zones, return vector of readings (μJ)
                            std::vector<unsigned long long> readRAPLEnergyPerZone(const std::vector<RAPLZone>& zones) {
                                std::vector<unsigned long long> readings;

                                for (const auto& zone : zones) {
                                    std::ifstream energy_file(zone.path + "/energy_uj");
                                    if (energy_file.is_open()) {
                                        unsigned long long uj = 0;
                                        energy_file >> uj;
                                        readings.push_back(uj);
                                    } else {
                                        readings.push_back(0); // Failed read
                                    }
                                }

                                return readings;
                            }

                            // Compute total energy delta across all zones (with per-zone overflow handling)
                            double computeTotalEnergyDelta(const std::vector<unsigned long long>& before,
                                                           const std::vector<unsigned long long>& after,
                                                           const std::vector<RAPLZone>& zones) {
                                if (before.size() != after.size() || before.size() != zones.size()) {
                                    return -1.0;
                                }

                                double total_j = 0.0;
                                bool any_valid = false;

                                for (size_t i = 0; i < zones.size(); i++) {
                                    unsigned long long b = before[i];
                                    unsigned long long a = after[i];
                                    unsigned long long r = zones[i].max_energy_range_uj;

                                    // Skip zones with read failures or invalid range
                                    if (b == 0 || a == 0 || r == 0) {
                                        continue;
                                    }

                                    // Per-zone delta with overflow handling
                                    unsigned long long delta_uj = (a >= b) ? (a - b) : (a + r - b);
                                    total_j += delta_uj / 1e6;
                                    any_valid = true;
                                }

                                return any_valid ? total_j : -1.0;
                                                           }

                                                           // ============================================================================
                                                           // Temperature Reading (hwmon) - coretemp/k10temp
                                                           // ============================================================================

                                                           int readCPUTemperature() {
                                                               // Try common hwmon paths (coretemp for Intel, k10temp for AMD)
                                                               std::vector<std::string> temp_paths = {
                                                                   "/sys/class/hwmon/hwmon0/temp1_input",
                                                                   "/sys/class/hwmon/hwmon1/temp1_input",
                                                                   "/sys/class/hwmon/hwmon2/temp1_input",
                                                                   "/sys/class/hwmon/hwmon3/temp1_input"
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
                                                           // CPU Clock Frequency Reading
                                                           // ============================================================================

                                                           int readCPUFrequencyMHz() {
                                                               std::ifstream freq_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
                                                               if (freq_file.is_open()) {
                                                                   int freq_khz;
                                                                   freq_file >> freq_khz;
                                                                   return freq_khz / 1000; // Convert kHz to MHz
                                                               }
                                                               return -1;
                                                           }

                                                           // ============================================================================
                                                           // STREAM Triad Function: a[i] = b[i] + scalar * c[i]
                                                           // ============================================================================

                                                           // Runs 'batches' STREAM Triad operations (each of size n)
                                                           // Uses OpenMP for parallelization with explicit thread control
                                                           // Returns checksum to prevent dead-code elimination
                                                           float run_batched_stream_triad(float* __restrict__ a, const float* __restrict__ b,
                                                                                          const float* __restrict__ c, int n, int batches,
                                                                                          int num_threads) {
                                                               omp_set_num_threads(num_threads);

                                                               for (int batch = 0; batch < batches; batch++) {
                                                                   #pragma omp parallel for schedule(static)
                                                                   for (int i = 0; i < n; i++) {
                                                                       a[i] = b[i] + SCALAR * c[i];
                                                                   }
                                                               }

                                                               // Strong DCE prevention: parallel reduction over ALL elements of a[]
                                                               // This forces the compiler to actually compute and store every a[i]
                                                               float checksum = 0.0f;
                                                               #pragma omp parallel for reduction(+:checksum) schedule(static)
                                                               for (int i = 0; i < n; i++) {
                                                                   checksum += a[i];
                                                               }

                                                               return checksum;
                                                                                          }

                                                                                          // ============================================================================
                                                                                          // Adaptive Batch Size Determination (equivalent to Reduction logic)
                                                                                          // ============================================================================

                                                                                          int determine_batch_size(float* a, const float* b, const float* c, int n,
                                                                                                                   int num_threads, double target_runtime_s) {
                                                                                              int batches = 1;
                                                                                              volatile float sink = 0.0f;  // Consume checksum to prevent optimization

                                                                                              while (batches <= MAX_BATCH_SIZE) {
                                                                                                  auto start = std::chrono::steady_clock::now();
                                                                                                  sink += run_batched_stream_triad(a, b, c, n, batches, num_threads);
                                                                                                  auto end = std::chrono::steady_clock::now();
                                                                                                  std::chrono::duration<double> elapsed = end - start;
                                                                                                  double measured_time = elapsed.count();

                                                                                                  // Guard: if measured_time is 0 or too small, just double batches
                                                                                                  if (measured_time <= 1e-9) {
                                                                                                      batches = std::min(batches * 2, MAX_BATCH_SIZE);
                                                                                                      continue;
                                                                                                  }

                                                                                                  // Done if >= target_time
                                                                                                  if (measured_time >= target_runtime_s) {
                                                                                                      return batches;
                                                                                                  }

                                                                                                  // Scale up: estimate how many batches needed for target_time
                                                                                                  double time_per_batch = measured_time / batches;
                                                                                                  int needed_batches = static_cast<int>(std::ceil(target_runtime_s / time_per_batch));

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
                                                                                                                   // Single Measurement Run
                                                                                                                   // ============================================================================

                                                                                                                   struct MeasurementResult {
                                                                                                                       double time_s;
                                                                                                                       double energy_j;
                                                                                                                       int temp_c;
                                                                                                                       int64_t cpu_cycles;
                                                                                                                       int64_t cpu_instructions;
                                                                                                                       int64_t cpu_cache_misses;
                                                                                                                   };

                                                                                                                   MeasurementResult run_single_measurement(float* a, const float* b, const float* c,
                                                                                                                                                            int n, int batches, int num_threads,
                                                                                                                                                            const std::vector<RAPLZone>& rapl_zones,
                                                                                                                                                            PerfGroupCounter& perf_counter) {
                                                                                                                       MeasurementResult result;
                                                                                                                       volatile float sink = 0.0f;  // Consume checksum to prevent optimization

                                                                                                                       // ============================================================
                                                                                                                       // Measurement Start - consistent window for time, energy, perf
                                                                                                                       // ============================================================

                                                                                                                       auto wall_start = std::chrono::steady_clock::now();
                                                                                                                       auto energy_before = readRAPLEnergyPerZone(rapl_zones);

                                                                                                                       // Start perf counters (reset + enable) - exactly around the batch loop
                                                                                                                       perf_counter.start();

                                                                                                                       // Execute batched STREAM Triad - this is the ONLY code measured by perf counters
                                                                                                                       sink += run_batched_stream_triad(a, b, c, n, batches, num_threads);

                                                                                                                       // Stop perf counters (disable) - end of measured region
                                                                                                                       perf_counter.stop();

                                                                                                                       auto energy_after = readRAPLEnergyPerZone(rapl_zones);
                                                                                                                       auto wall_end = std::chrono::steady_clock::now();

                                                                                                                       // ============================================================
                                                                                                                       // Measurement End
                                                                                                                       // ============================================================

                                                                                                                       // Read perf counter values
                                                                                                                       perf_counter.read(result.cpu_cycles, result.cpu_instructions, result.cpu_cache_misses);

                                                                                                                       // Compute time
                                                                                                                       std::chrono::duration<double> duration = wall_end - wall_start;
                                                                                                                       result.time_s = duration.count();

                                                                                                                       // Compute energy
                                                                                                                       result.energy_j = computeTotalEnergyDelta(energy_before, energy_after, rapl_zones);

                                                                                                                       // Read temperature
                                                                                                                       result.temp_c = readCPUTemperature();

                                                                                                                       return result;
                                                                                                                                                            }

                                                                                                                                                            // ============================================================================
                                                                                                                                                            // CSV Output Functions (CSV_COLUMNS.md format with 26 columns + 4 perf counters)
                                                                                                                                                            // ============================================================================

                                                                                                                                                            void writeCSVHeader(std::ofstream& file) {
                                                                                                                                                                file << "timestamp,run_id_global,run_id_per_size,device_name,num_threads,problem_size,"
                                                                                                                                                                << "batches,gpu_e2e_time_s,gpu_kernel_time_s,wall_time_s,total_energy_j,"
                                                                                                                                                                << "energy_per_batch_j,energy_per_second_j,energy_per_flop_j,"
                                                                                                                                                                << "time_per_gemm_ms_kernel,time_per_gemm_ms_e2e,flops_total,gflops_per_s,"
                                                                                                                                                                << "avg_power_w,below_target,pcie_gen,pcie_width,sm_clock_mhz,mem_clock_mhz,"
                                                                                                                                                                << "temp_c,throttle_reasons,cpu_cycles,cpu_instructions,cpu_ipc,cpu_cache_misses\n";
                                                                                                                                                            }

                                                                                                                                                            void writeCSVRow(std::ofstream& file, int run_id_global, int run_id_per_size,
                                                                                                                                                                             const std::string& device_name, int num_threads, int problem_size,
                                                                                                                                                                             int batches, double time_s, double energy_j, int temp_c,
                                                                                                                                                                             bool below_target,
                                                                                                                                                                             int64_t cpu_cycles, int64_t cpu_instructions, int64_t cpu_cache_misses) {
                                                                                                                                                                // Compute derived metrics
                                                                                                                                                                // For STREAM Triad: 2 FLOPs per element (multiply + add): a[i] = b[i] + scalar * c[i]
                                                                                                                                                                double flops_total = 2.0 * static_cast<double>(problem_size) * batches;
                                                                                                                                                                double energy_per_batch_j = (batches > 0 && energy_j >= 0) ? (energy_j / batches) : -1.0;
                                                                                                                                                                double energy_per_second_j = (time_s > 0 && energy_j >= 0) ? (energy_j / time_s) : -1.0;
                                                                                                                                                                double energy_per_flop_j = (flops_total > 0 && energy_j >= 0) ? (energy_j / flops_total) : -1.0;
                                                                                                                                                                double time_per_gemm_ms = (batches > 0) ? (1e3 * time_s / batches) : 0.0;
                                                                                                                                                                double gflops_per_s = (time_s > 0) ? (flops_total / time_s / 1e9) : 0.0;
                                                                                                                                                                double avg_power_w = (time_s > 0 && energy_j >= 0) ? (energy_j / time_s) : -1.0;

                                                                                                                                                                // Compute cpu_ipc: only if both cycles and instructions are valid (>= 0) and cycles > 0
                                                                                                                                                                double cpu_ipc = -1.0;
                                                                                                                                                                if (cpu_cycles > 0 && cpu_instructions >= 0) {
                                                                                                                                                                    cpu_ipc = static_cast<double>(cpu_instructions) / static_cast<double>(cpu_cycles);
                                                                                                                                                                }

                                                                                                                                                                // Read current CPU frequency
                                                                                                                                                                int sm_clock_mhz = readCPUFrequencyMHz();

                                                                                                                                                                // For CPU: gpu_e2e_time_s, gpu_kernel_time_s, wall_time_s are all identical
                                                                                                                                                                file << getTimestamp() << ","
                                                                                                                                                                << run_id_global << ","
                                                                                                                                                                << run_id_per_size << ","
                                                                                                                                                                << device_name << ","
                                                                                                                                                                << num_threads << ","
                                                                                                                                                                << problem_size << ","
                                                                                                                                                                << batches << ","
                                                                                                                                                                << std::fixed << std::setprecision(6)
                                                                                                                                                                << time_s << ","  // gpu_e2e_time_s (identical to wall_time_s for CPU)
                                                                                                                                                                << time_s << ","  // gpu_kernel_time_s (identical to wall_time_s for CPU)
                                                                                                                                                                << time_s << ","  // wall_time_s
                                                                                                                                                                << energy_j << ","
                                                                                                                                                                << std::scientific << std::setprecision(6)
                                                                                                                                                                << energy_per_batch_j << ","
                                                                                                                                                                << std::fixed << std::setprecision(6)
                                                                                                                                                                << energy_per_second_j << ","
                                                                                                                                                                << std::scientific << std::setprecision(6)
                                                                                                                                                                << energy_per_flop_j << ","
                                                                                                                                                                << std::fixed << std::setprecision(6)
                                                                                                                                                                << time_per_gemm_ms << ","  // time_per_gemm_ms_kernel (identical to e2e for CPU)
                                                                                                                                                                << time_per_gemm_ms << ","  // time_per_gemm_ms_e2e
                                                                                                                                                                << std::scientific << std::setprecision(6)
                                                                                                                                                                << flops_total << ","
                                                                                                                                                                << std::fixed << std::setprecision(2)
                                                                                                                                                                << gflops_per_s << ","
                                                                                                                                                                << avg_power_w << ","
                                                                                                                                                                << (below_target ? 't' : 'f') << ","
                                                                                                                                                                << "," // pcie_gen (empty for CPU)
                                                                                                                                                                << "," // pcie_width (empty for CPU)
                                                                                                                                                                << ((sm_clock_mhz > 0) ? std::to_string(sm_clock_mhz) : "") << ","
                                                                                                                                                                << "," // mem_clock_mhz (empty for CPU)
                                                                                                                                                                << ((temp_c > 0) ? std::to_string(temp_c) : "") << ","
                                                                                                                                                                << "," // throttle_reasons (empty for CPU)
                                                                                                                                                                // Columns 27-30: perf counters
                                                                                                                                                                << cpu_cycles << ","
                                                                                                                                                                << cpu_instructions << ","
                                                                                                                                                                << std::fixed << std::setprecision(6) << cpu_ipc << ","
                                                                                                                                                                << cpu_cache_misses << "\n";
                                                                                                                                                                             }

                                                                                                                                                                             // ============================================================================
                                                                                                                                                                             // Main Benchmark Function
                                                                                                                                                                             // ============================================================================

                                                                                                                                                                             void run_benchmark(const char* output_file, bool test_mode) {
                                                                                                                                                                                 // Get CPU info
                                                                                                                                                                                 std::string cpu_model = getCPUModel();
                                                                                                                                                                                 std::cout << "=== CPU STREAM Triad Benchmark (Adaptive Batching) ===" << std::endl;
                                                                                                                                                                                 std::cout << "CPU: " << cpu_model << std::endl;
                                                                                                                                                                                 std::cout << "Target runtime per measurement: " << TARGET_RUNTIME_S << "s" << std::endl;
                                                                                                                                                                                 std::cout << "Repetitions per configuration: " << REPEATS << " (each writes a CSV row)" << std::endl;
                                                                                                                                                                                 std::cout << "STREAM Triad: a[i] = b[i] + " << SCALAR << " * c[i]" << std::endl;

                                                                                                                                                                                 // Initialize RAPL
                                                                                                                                                                                 std::vector<RAPLZone> rapl_zones = discoverRAPLZones();
                                                                                                                                                                                 if (rapl_zones.empty()) {
                                                                                                                                                                                     std::cerr << "WARNING: RAPL not available. Make sure:\n"
                                                                                                                                                                                     << "  1. You're on Linux with Intel/AMD CPU\n"
                                                                                                                                                                                     << "  2. powercap module loaded: sudo modprobe intel_rapl_msr\n"
                                                                                                                                                                                     << "  3. Permissions set: sudo chmod -R a+r /sys/class/powercap/\n"
                                                                                                                                                                                     << "Energy measurements will be set to -1.0\n" << std::endl;
                                                                                                                                                                                 } else {
                                                                                                                                                                                     std::cout << "RAPL: Found " << rapl_zones.size() << " package zone(s)" << std::endl;
                                                                                                                                                                                     for (const auto& zone : rapl_zones) {
                                                                                                                                                                                         std::cout << "  - " << zone.name << " (max_range: "
                                                                                                                                                                                         << (zone.max_energy_range_uj / 1e6) << " J)" << std::endl;
                                                                                                                                                                                     }
                                                                                                                                                                                 }

                                                                                                                                                                                 // Allocate vectors for maximum problem size
                                                                                                                                                                                 // STREAM Triad needs 3 arrays: a (output), b (input), c (input)
                                                                                                                                                                                 std::cout << "\nAllocating vectors (max size: " << MAX_N << " elements = "
                                                                                                                                                                                 << (3 * MAX_N * sizeof(float) / (1024.0 * 1024.0)) << " MB total)..." << std::endl;

                                                                                                                                                                                 float* a_max = nullptr;
                                                                                                                                                                                 float* b_max = nullptr;
                                                                                                                                                                                 float* c_max = nullptr;

                                                                                                                                                                                 if (posix_memalign((void**)&a_max, 64, MAX_N * sizeof(float)) != 0 ||
                                                                                                                                                                                     posix_memalign((void**)&b_max, 64, MAX_N * sizeof(float)) != 0 ||
                                                                                                                                                                                     posix_memalign((void**)&c_max, 64, MAX_N * sizeof(float)) != 0) {
                                                                                                                                                                                     std::cerr << "ERROR: Failed to allocate aligned memory" << std::endl;
                                                                                                                                                                                 return;
                                                                                                                                                                                     }

                                                                                                                                                                                     // Initialize vectors (seed=42 for deterministic init, like CPU/GEMM)
                                                                                                                                                                                     std::cout << "Initializing vectors (seed=42)..." << std::endl;
                                                                                                                                                                                     initializeVector(b_max, MAX_N, 42);
                                                                                                                                                                                     initializeVector(c_max, MAX_N, 43);
                                                                                                                                                                                     // Initialize a to zero (will be overwritten anyway)
                                                                                                                                                                                     std::fill(a_max, a_max + MAX_N, 0.0f);

                                                                                                                                                                                     // Open CSV file
                                                                                                                                                                                     ensureDirectoryExists(output_file);
                                                                                                                                                                                     bool write_header = !fileExists(output_file);
                                                                                                                                                                                     std::ofstream csv_file(output_file, std::ios::app);

                                                                                                                                                                                     if (write_header) {
                                                                                                                                                                                         writeCSVHeader(csv_file);
                                                                                                                                                                                     }

                                                                                                                                                                                     // Determine which configurations to test
                                                                                                                                                                                     std::vector<int> thread_configs_to_test;
                                                                                                                                                                                     std::vector<int> sizes_to_test;

                                                                                                                                                                                     if (test_mode) {
                                                                                                                                                                                         // Test mode: only first 2 thread configs and first 2 sizes
                                                                                                                                                                                         thread_configs_to_test.push_back(THREAD_COUNTS[0]);
                                                                                                                                                                                         thread_configs_to_test.push_back(THREAD_COUNTS[1]);
                                                                                                                                                                                         sizes_to_test.push_back(PROBLEM_SIZES[0]);
                                                                                                                                                                                         sizes_to_test.push_back(PROBLEM_SIZES[1]);

                                                                                                                                                                                         std::cout << "\n*** TEST MODE: Only " << thread_configs_to_test.size()
                                                                                                                                                                                         << " thread configs × " << sizes_to_test.size() << " sizes ***" << std::endl;
                                                                                                                                                                                     } else {
                                                                                                                                                                                         // Full benchmark
                                                                                                                                                                                         for (int i = 0; i < NUM_THREAD_CONFIGS; i++) {
                                                                                                                                                                                             thread_configs_to_test.push_back(THREAD_COUNTS[i]);
                                                                                                                                                                                         }
                                                                                                                                                                                         for (int i = 0; i < NUM_SIZES; i++) {
                                                                                                                                                                                             sizes_to_test.push_back(PROBLEM_SIZES[i]);
                                                                                                                                                                                         }
                                                                                                                                                                                     }

                                                                                                                                                                                     // Initialize perf counter group (cycles, instructions, cache-misses)
                                                                                                                                                                                     // Opened once; inherit=1 ensures child threads (OpenMP workers) are counted
                                                                                                                                                                                     PerfGroupCounter perf_counter;
                                                                                                                                                                                     bool perf_available = perf_counter.open();

                                                                                                                                                                                     if (perf_available) {
                                                                                                                                                                                         std::cout << "\nPerf counters: Available (cycles, instructions, cache-misses)" << std::endl;
                                                                                                                                                                                     } else {
                                                                                                                                                                                         std::cerr << "\nWARNING: Perf counters not accessible (permission denied or not supported).\n"
                                                                                                                                                                                         << "cpu_cycles, cpu_instructions, cpu_ipc, cpu_cache_misses will be -1.\n"
                                                                                                                                                                                         << "Run as root or set /proc/sys/kernel/perf_event_paranoid to 1 or lower.\n" << std::endl;
                                                                                                                                                                                     }

                                                                                                                                                                                     int run_id_global = 0;

                                                                                                                                                                                     // Main benchmark loop: outer loop over thread counts, inner loop over problem sizes
                                                                                                                                                                                     for (int num_threads : thread_configs_to_test) {
                                                                                                                                                                                         // Set OpenMP thread count
                                                                                                                                                                                         omp_set_num_threads(num_threads);

                                                                                                                                                                                         // Verify that OpenMP actually uses this many threads
                                                                                                                                                                                         int actual_threads = 0;
                                                                                                                                                                                         #pragma omp parallel
                                                                                                                                                                                         {
                                                                                                                                                                                             #pragma omp single
                                                                                                                                                                                             actual_threads = omp_get_num_threads();
                                                                                                                                                                                         }

                                                                                                                                                                                         std::cout << "\n========================================" << std::endl;
                                                                                                                                                                                         std::cout << "=== Testing with " << num_threads << " threads";
                                                                                                                                                                                         if (actual_threads != num_threads) {
                                                                                                                                                                                             std::cout << " (WARNING: OpenMP reports " << actual_threads << " threads!)";
                                                                                                                                                                                         }
                                                                                                                                                                                         std::cout << " ===" << std::endl;
                                                                                                                                                                                         std::cout << "========================================" << std::endl;

                                                                                                                                                                                         for (int n : sizes_to_test) {
                                                                                                                                                                                             std::cout << "\n--- n=" << n << " ---" << std::endl;

                                                                                                                                                                                             // Adaptive batch size based on TARGET_RUNTIME_S
                                                                                                                                                                                             int batches = determine_batch_size(a_max, b_max, c_max, n, num_threads, TARGET_RUNTIME_S);

                                                                                                                                                                                             std::cout << "Determined batch size: " << batches << " batches (targeting ~"
                                                                                                                                                                                             << TARGET_RUNTIME_S << "s runtime)" << std::endl;

                                                                                                                                                                                             int run_id_per_size = 1;  // Reset for each new problem size (like GEMM)

                                                                                                                                                                                             // Run REPEATS measurements with this batch size
                                                                                                                                                                                             for (int rep = 0; rep < REPEATS; rep++) {
                                                                                                                                                                                                 run_id_global++;

                                                                                                                                                                                                 MeasurementResult result = run_single_measurement(a_max, b_max, c_max, n,
                                                                                                                                                                                                                                                   batches, num_threads,
                                                                                                                                                                                                                                                   rapl_zones, perf_counter);

                                                                                                                                                                                                 // below_target: computed per-run based on actual measured time
                                                                                                                                                                                                 bool below_target = (result.time_s < TARGET_RUNTIME_S);

                                                                                                                                                                                                 // Write CSV row for this single run
                                                                                                                                                                                                 writeCSVRow(csv_file, run_id_global, run_id_per_size, cpu_model,
                                                                                                                                                                                                             num_threads, n, batches, result.time_s, result.energy_j,
                                                                                                                                                                                                             result.temp_c, below_target,
                                                                                                                                                                                                             result.cpu_cycles, result.cpu_instructions, result.cpu_cache_misses);
                                                                                                                                                                                                 csv_file.flush();

                                                                                                                                                                                                 // Console output for each measurement
                                                                                                                                                                                                 double gflops = (result.time_s > 0) ?
                                                                                                                                                                                                 (2.0 * static_cast<double>(n) * batches / result.time_s / 1e9) : 0.0;
                                                                                                                                                                                                 double power = (result.time_s > 0 && result.energy_j >= 0) ?
                                                                                                                                                                                                 (result.energy_j / result.time_s) : -1.0;

                                                                                                                                                                                                 char check = below_target ? '!' : '+';
                                                                                                                                                                                                 std::cout << "  " << check << " [" << (rep + 1) << "/" << REPEATS << "] "
                                                                                                                                                                                                 << std::fixed << std::setprecision(3) << result.time_s << "s";
                                                                                                                                                                                                 if (result.energy_j >= 0) {
                                                                                                                                                                                                     std::cout << ", " << std::setprecision(1) << result.energy_j << " J";
                                                                                                                                                                                                     if (power >= 0) {
                                                                                                                                                                                                         std::cout << ", " << std::setprecision(0) << power << " W";
                                                                                                                                                                                                     }
                                                                                                                                                                                                 }
                                                                                                                                                                                                 if (result.temp_c > 0) {
                                                                                                                                                                                                     std::cout << ", " << result.temp_c << "°C";
                                                                                                                                                                                                 }
                                                                                                                                                                                                 if (below_target) {
                                                                                                                                                                                                     std::cout << " (!)";
                                                                                                                                                                                                 }
                                                                                                                                                                                                 std::cout << std::endl;

                                                                                                                                                                                                 run_id_per_size++;  // Increment after each run (like GEMM)
                                                                                                                                                                                             }

                                                                                                                                                                                             // Cooldown after all 50 runs (except for last configuration)
                                                                                                                                                                                             bool is_last_size = (n == sizes_to_test.back());
                                                                                                                                                                                             bool is_last_thread = (num_threads == thread_configs_to_test.back());

                                                                                                                                                                                             if (!(is_last_size && is_last_thread)) {
                                                                                                                                                                                                 std::cout << "  Cooling down for 60 seconds..." << std::endl;
                                                                                                                                                                                                 std::this_thread::sleep_for(std::chrono::seconds(60));
                                                                                                                                                                                             }

                                                                                                                                                                                             std::cout << std::endl;
                                                                                                                                                                                         }
                                                                                                                                                                                     }

                                                                                                                                                                                     csv_file.close();

                                                                                                                                                                                     std::cout << "\n========================================" << std::endl;
                                                                                                                                                                                     std::cout << "Benchmark complete!" << std::endl;
                                                                                                                                                                                     std::cout << "Results saved to: " << output_file << std::endl;
                                                                                                                                                                                     std::cout << "Total CSV rows (individual runs): " << run_id_global << std::endl;
                                                                                                                                                                                     int total_configs = thread_configs_to_test.size() * sizes_to_test.size();
                                                                                                                                                                                     std::cout << "Configurations tested: " << total_configs
                                                                                                                                                                                     << " (" << thread_configs_to_test.size() << " threads × "
                                                                                                                                                                                     << sizes_to_test.size() << " sizes × " << REPEATS << " runs)" << std::endl;
                                                                                                                                                                                     std::cout << "========================================" << std::endl;

                                                                                                                                                                                     // Cleanup
                                                                                                                                                                                     free(a_max);
                                                                                                                                                                                     free(b_max);
                                                                                                                                                                                     free(c_max);
                                                                                                                                                                             }

                                                                                                                                                                             // ============================================================================
                                                                                                                                                                             // Main
                                                                                                                                                                             // ============================================================================

                                                                                                                                                                             int main(int argc, char** argv) {
                                                                                                                                                                                 bool test_mode = false;
                                                                                                                                                                                 const char* output_file = DEFAULT_OUTPUT_FILE;

                                                                                                                                                                                 // Parse command line arguments
                                                                                                                                                                                 for (int i = 1; i < argc; i++) {
                                                                                                                                                                                     std::string arg = argv[i];
                                                                                                                                                                                     if (arg == "--test" || arg == "-t") {
                                                                                                                                                                                         test_mode = true;
                                                                                                                                                                                     } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
                                                                                                                                                                                         output_file = argv[++i];
                                                                                                                                                                                     } else if (arg == "--help" || arg == "-h") {
                                                                                                                                                                                         std::cout << "Usage: " << argv[0] << " [options]\n"
                                                                                                                                                                                         << "Options:\n"
                                                                                                                                                                                         << "  -t, --test           Test mode (reduced configurations with console output)\n"
                                                                                                                                                                                         << "  -o, --output FILE    Output CSV file (default: " << DEFAULT_OUTPUT_FILE << ")\n"
                                                                                                                                                                                         << "  -h, --help           Show this help\n"
                                                                                                                                                                                         << "\nHardcoded configurations:\n"
                                                                                                                                                                                         << "  Thread counts: ";
                                                                                                                                                                                         for (int i = 0; i < NUM_THREAD_CONFIGS; i++) {
                                                                                                                                                                                             std::cout << THREAD_COUNTS[i];
                                                                                                                                                                                             if (i < NUM_THREAD_CONFIGS - 1) std::cout << ", ";
                                                                                                                                                                                         }
                                                                                                                                                                                         std::cout << "\n  Problem sizes: ";
                                                                                                                                                                                         for (int i = 0; i < NUM_SIZES; i++) {
                                                                                                                                                                                             std::cout << PROBLEM_SIZES[i];
                                                                                                                                                                                             if (i < NUM_SIZES - 1) std::cout << ", ";
                                                                                                                                                                                         }
                                                                                                                                                                                         std::cout << "\n  Repetitions per config: " << REPEATS << " (each run writes a CSV row)\n";
                                                                                                                                                                                         std::cout << "  Target runtime: " << TARGET_RUNTIME_S << "s\n";
                                                                                                                                                                                         std::cout << "  STREAM Triad: a[i] = b[i] + " << SCALAR << " * c[i]\n";
                                                                                                                                                                                         return 0;
                                                                                                                                                                                     }
                                                                                                                                                                                 }

                                                                                                                                                                                 std::cout << "Output file: " << output_file << std::endl;
                                                                                                                                                                                 std::cout << "Test mode: " << (test_mode ? "yes" : "no") << std::endl;

                                                                                                                                                                                 run_benchmark(output_file, test_mode);

                                                                                                                                                                                 return EXIT_SUCCESS;
                                                                                                                                                                             }


