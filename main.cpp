// ============================================================================
// CPU Reduction (DOT with ones) Energy Benchmark with Adaptive Batching
// Updated to match CSV_COLUMNS.md format and use adaptive batching logic
// ============================================================================
// Compile: g++ -O3 -march=native -std=c++17 -o reduction_cpu main.cpp -lopenblas -lpthread -lm
// Usage: ./reduction_cpu [--test] [--output <path>]

#include <cblas.h>
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
#include <dirent.h>
#include <cstdlib>
#include <cmath>

// ============================================================================
// Configuration - Hardcoded
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    REPEATS          = 50;  // 50 measurements per configuration
constexpr int    MAX_BATCH_SIZE   = 1000000;  // Upper limit for batches

// Hardcoded thread counts (as per CSV_COLUMNS.md)
static const int THREAD_COUNTS[] = {1, 2, 4, 8, 10, 16, 20};
static const int NUM_THREAD_CONFIGS = sizeof(THREAD_COUNTS) / sizeof(THREAD_COUNTS[0]);

// Problem sizes in number of elements
static const int PROBLEM_SIZES[] = {
    1000000, 2000000, 4000000, 8000000, 16000000, 32000000,
    64000000, 128000000, 256000000
};
static const int NUM_SIZES = sizeof(PROBLEM_SIZES) / sizeof(PROBLEM_SIZES[0]);
static const int MAX_N = *std::max_element(std::begin(PROBLEM_SIZES), std::end(PROBLEM_SIZES));

static const char* DEFAULT_OUTPUT_FILE = "reduction_cpu_amd.csv";

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

// Median helper for aggregation over 50 runs
template <class T>
T median(std::vector<T> v) {
    if (v.empty()) return T(-1);
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (v.size() % 2) return v[mid];
    auto lo = *std::max_element(v.begin(), v.begin() + mid);
    return (lo + v[mid]) / T(2);
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
// Batched Reduction Function
// ============================================================================

// Runs 'batches' reduction operations (each of size n)
void run_batched_reduction(const float* x, const float* ones, int n, int batches) {
    volatile float dummy = 0.0f; // Prevent dead-code elimination
    
    for (int b = 0; b < batches; b++) {
        float result = cblas_sdot(n, x, 1, ones, 1);
        dummy += result;
    }
}

// ============================================================================
// Adaptive Batch Size Determination (based on TARGET_RUNTIME_S)
// ============================================================================

int determine_batch_size(const float* x, const float* ones, int n, double target_runtime_s) {
    // Quick test run with small number of batches to estimate time
    int test_batches = 10;
    
    auto start = std::chrono::steady_clock::now();
    run_batched_reduction(x, ones, n, test_batches);
    auto end = std::chrono::steady_clock::now();
    
    std::chrono::duration<double> duration = end - start;
    double time_per_batch = duration.count() / test_batches;
    
    if (time_per_batch <= 0) {
        time_per_batch = 1e-9; // Safeguard
    }
    
    // Estimate batches needed for target runtime
    int estimated_batches = static_cast<int>(target_runtime_s / time_per_batch);
    
    // Clamp to reasonable range
    if (estimated_batches < 1) estimated_batches = 1;
    if (estimated_batches > MAX_BATCH_SIZE) estimated_batches = MAX_BATCH_SIZE;
    
    return estimated_batches;
}

// ============================================================================
// Single Measurement Run
// ============================================================================

struct MeasurementResult {
    double time_s;
    double energy_j;
    int temp_c;
};

MeasurementResult run_single_measurement(const float* x, const float* ones, int n, int batches,
                                        const std::vector<RAPLZone>& rapl_zones) {
    MeasurementResult result;
    
    // Read energy before
    auto energy_before = readRAPLEnergyPerZone(rapl_zones);
    
    // Time and execute batched reduction
    auto start = std::chrono::steady_clock::now();
    run_batched_reduction(x, ones, n, batches);
    auto end = std::chrono::steady_clock::now();
    
    // Read energy after
    auto energy_after = readRAPLEnergyPerZone(rapl_zones);
    
    // Compute time
    std::chrono::duration<double> duration = end - start;
    result.time_s = duration.count();
    
    // Compute energy
    result.energy_j = computeTotalEnergyDelta(energy_before, energy_after, rapl_zones);
    
    // Read temperature
    result.temp_c = readCPUTemperature();
    
    return result;
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
                const std::string& device_name, int num_threads, int problem_size,
                int batches, double time_s, double energy_j, int temp_c,
                bool below_target) {
    // Compute derived metrics
    // For reduction: 1 FLOP per element (addition)
    double flops_total = static_cast<double>(problem_size) * batches;
    double energy_per_batch_j = (batches > 0) ? (energy_j / batches) : 0.0;
    double energy_per_second_j = (time_s > 0) ? (energy_j / time_s) : 0.0;
    double energy_per_flop_j = (flops_total > 0) ? (energy_j / flops_total) : 0.0;
    double time_per_gemm_ms = (batches > 0) ? (1e3 * time_s / batches) : 0.0;
    double gflops_per_s = (time_s > 0) ? (flops_total / time_s / 1e9) : 0.0;
    double avg_power_w = (time_s > 0 && energy_j >= 0) ? (energy_j / time_s) : -1.0;
    
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
         << "\n"; // throttle_reasons (empty for CPU)
}

// ============================================================================
// Main Benchmark Function
// ============================================================================

void run_benchmark(const char* output_file, bool test_mode) {
    // Get CPU info
    std::string cpu_model = getCPUModel();
    std::cout << "=== CPU Reduction Benchmark (Adaptive Batching) ===" << std::endl;
    std::cout << "CPU: " << cpu_model << std::endl;
    std::cout << "Target runtime per measurement: " << TARGET_RUNTIME_S << "s" << std::endl;
    std::cout << "Repetitions per configuration: " << REPEATS << std::endl;
    
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
    std::cout << "\nAllocating vectors (max size: " << MAX_N << " elements = "
              << (MAX_N * sizeof(float) / (1024.0 * 1024.0)) << " MB)..." << std::endl;
    
    float* x_max = nullptr;
    float* ones_max = nullptr;
    
    if (posix_memalign((void**)&x_max, 64, MAX_N * sizeof(float)) != 0 ||
        posix_memalign((void**)&ones_max, 64, MAX_N * sizeof(float)) != 0) {
        std::cerr << "ERROR: Failed to allocate aligned memory" << std::endl;
        return;
    }
    
    // Initialize vectors
    initializeVector(x_max, MAX_N, 42);
    std::fill(ones_max, ones_max + MAX_N, 1.0f);
    
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
    
    int run_id_global = 0;
    
    // Benchmark loop: outer loop over problem sizes, inner loop over thread counts
    // This ensures run_id_per_size correctly resets for each new problem_size
    for (int n : sizes_to_test) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Problem size: " << n << " elements" << std::endl;
        std::cout << "========================================" << std::endl;
        
        int run_id_per_size = 0;
        
        for (int num_threads : thread_configs_to_test) {
            run_id_per_size++;  // Increment for each thread count within this problem size
            
            // Set OpenBLAS thread count
            openblas_set_num_threads(num_threads);
            
            std::cout << "\n--- Thread count: " << num_threads << " ---" << std::endl;
            
            // Adaptive batch size based on TARGET_RUNTIME_S
            int batches = determine_batch_size(x_max, ones_max, n, TARGET_RUNTIME_S);
            
            std::cout << "Determined batch size: " << batches << " (targeting ~" 
                     << TARGET_RUNTIME_S << "s runtime)" << std::endl;
            
            // Vectors to collect measurements
            std::vector<double> time_v;
            std::vector<double> energy_v;
            std::vector<int> temp_v;
            
            // Run REPEATS measurements with this batch size
            std::cout << "Running " << REPEATS << " measurements..." << std::endl;
            
            for (int rep = 0; rep < REPEATS; rep++) {
                MeasurementResult result = run_single_measurement(x_max, ones_max, n, 
                                                                  batches, rapl_zones);
                
                time_v.push_back(result.time_s);
                energy_v.push_back(result.energy_j);
                if (result.temp_c > 0) {
                    temp_v.push_back(result.temp_c);
                }
                
                // Console output for each measurement
                if (test_mode) {
                    double gflops = (result.time_s > 0) ? 
                        (static_cast<double>(n) * batches / result.time_s / 1e9) : 0.0;
                    double power = (result.time_s > 0 && result.energy_j >= 0) ? 
                        (result.energy_j / result.time_s) : -1.0;
                    
                    std::cout << "  [" << (rep + 1) << "/" << REPEATS << "] "
                             << std::fixed << std::setprecision(3) << result.time_s << "s, "
                             << std::setprecision(2) << gflops << " GFLOPS";
                    if (result.energy_j >= 0) {
                        std::cout << ", " << std::setprecision(1) << result.energy_j << " J";
                        if (power >= 0) {
                            std::cout << ", " << std::setprecision(0) << power << " W";
                        }
                    }
                    if (result.temp_c > 0) {
                        std::cout << ", " << result.temp_c << "°C";
                    }
                    std::cout << std::endl;
                }
            }
            
            // Compute medians over all REPEATS
            double median_time_s = median(time_v);
            double median_energy_j = median(energy_v);
            int median_temp_c = temp_v.empty() ? -1 : median(temp_v);
            
            // Check if we hit the target runtime
            bool below_target = (median_time_s < TARGET_RUNTIME_S) && (batches >= MAX_BATCH_SIZE);
            
            // Write single CSV row with median values
            run_id_global++;
            writeCSVRow(csv_file, run_id_global, run_id_per_size, cpu_model,
                       num_threads, n, batches, median_time_s, median_energy_j,
                       median_temp_c, below_target);
            csv_file.flush();
            
            // Summary output
            double gflops = (median_time_s > 0) ? 
                (static_cast<double>(n) * batches / median_time_s / 1e9) : 0.0;
            double power = (median_time_s > 0 && median_energy_j >= 0) ? 
                (median_energy_j / median_time_s) : -1.0;
            
            std::cout << "→ Median: " << std::fixed << std::setprecision(3) 
                     << median_time_s << "s, "
                     << std::setprecision(2) << gflops << " GFLOPS";
            if (median_energy_j >= 0) {
                std::cout << ", " << std::setprecision(1) << median_energy_j << " J";
                if (power >= 0) {
                    std::cout << ", " << std::setprecision(0) << power << " W";
                }
            }
            if (median_temp_c > 0) {
                std::cout << ", " << median_temp_c << "°C";
            }
            if (below_target) {
                std::cout << " [!!! below target]";
            }
            std::cout << std::endl;
        }
    }
    
    csv_file.close();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark complete!" << std::endl;
    std::cout << "Results saved to: " << output_file << std::endl;
    std::cout << "Total configurations: " << run_id_global << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Cleanup
    free(x_max);
    free(ones_max);
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
            std::cout << "\n  Repetitions per config: " << REPEATS << "\n";
            std::cout << "  Target runtime: " << TARGET_RUNTIME_S << "s\n";
            return 0;
        }
    }
    
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "Test mode: " << (test_mode ? "yes" : "no") << std::endl;
    
    run_benchmark(output_file, test_mode);
    
    return EXIT_SUCCESS;
}

// ============================================================================
// Key Changes from Original main.cpp
// ============================================================================
/*
 * [✓] Adaptive batch size based on TARGET_RUNTIME_S (like main2.cpp)
 * [✓] CSV output according to CSV_COLUMNS.md (26 columns)
 * [✓] 50 repetitions per configuration (REPEATS = 50)
 * [✓] Median aggregation: one CSV row per (problem_size, num_threads)
 * [✓] Hardcoded thread counts: 1, 2, 4, 8, 10, 16, 20
 * [✓] Hardcoded problem sizes (9 sizes from 1M to 256M elements)
 * [✓] Test mode: --test flag for reduced configurations with console output
 * [✓] RAPL logic preserved (discoverRAPLZones, per-zone overflow handling)
 * [✓] For CPU: gpu_e2e_time_s = gpu_kernel_time_s = wall_time_s (all identical)
 * [✓] run_id_global increments for each CSV row written
 * [✓] run_id_per_size resets to 1 for each new problem_size
 * [✓] below_target flag: 't' if runtime < target despite hitting MAX_BATCH_SIZE
 * [✓] FLOPs calculation: 1 FLOP per element (reduction = sum = additions only)
 */
