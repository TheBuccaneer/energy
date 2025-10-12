// main_reduction_cpu.cpp - CPU Reduction (DOT with ones) Energy Benchmark with OpenBLAS
// Compile: g++ -O3 -march=native -std=c++17 -o reduction_cpu main_reduction_cpu.cpp -lopenblas -lpthread -lm
// Usage: ./reduction_cpu --threads N

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
// Configuration - Hardcoded (except NUM_THREADS from command-line)
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    REPEATS          = 5;
constexpr bool   AGGREGATE_MEDIAN = true;   // Report median over REPEATS (like GPU)

// E2E bandwidth mode: "mixed" or "memcpy_only"
// - "mixed": bytes=memcpy, time=memcpy+DOT (current behavior, main CSV)
// - "memcpy_only": also measure pure memcpy time, write to sidecar CSV
static const char* E2E_BW_MODE = "mixed";  // Change to "memcpy_only" for pure memcpy BW

// N values in elements (19 sizes)
static const int N_SIZES[] = {
    1000000, 1500000, 2000000, 3000000, 4000000, 6000000,
    8000000, 12000000, 16000000, 24000000, 32000000, 48000000,
    64000000, 96000000, 128000000, 160000000, 192000000,
    256000000, 384000000
};
static const int NUM_SIZES = sizeof(N_SIZES) / sizeof(N_SIZES[0]);
static const int MAX_N     = *std::max_element(std::begin(N_SIZES), std::end(N_SIZES));

static const char* OUTPUT_FILE = "data/raw/reduction_cpu_openblas.csv";
static const char* OUTPUT_FILE_MEMCPY = "data/raw/reduction_cpu_openblas_memcpy.csv";

// E2E "transfer" emulation
static const bool  E2E_INJECT_HOST_MEMCPY = true; // memcpy(work←x) per pass

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

std::string getCPUModel() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string model = line.substr(pos + 2);
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

// Median helper for aggregation
template <class T>
T median(std::vector<T> v) {
    if (v.empty()) return T(-1);
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (v.size() % 2) return v[mid];
    auto lo = *std::max_element(v.begin(), v.begin() + mid);
    return (lo + v[mid]) / T(2);
}

// Command-line parsing
int parseThreadsArgument(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            int threads = std::atoi(argv[i + 1]);
            if (threads > 0) {
                return threads;
            }
        }
    }
    return -1;  // Not found or invalid
}

void printUsage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " --threads N\n";
    std::cerr << "  --threads N    Number of OpenBLAS threads (required, N > 0)\n";
    std::exit(EXIT_FAILURE);
}

// ============================================================================
// RAPL Energy Reading (powercap interface) - PER ZONE
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

double readCPUTemperature() {
    const char* hwmon_base = "/sys/class/hwmon";
    
    DIR* dir = opendir(hwmon_base);
    if (!dir) return -1.0;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string hwmon_name = entry->d_name;
        if (hwmon_name.find("hwmon") != 0) continue;
        
        std::string hwmon_path = std::string(hwmon_base) + "/" + hwmon_name;
        
        // Read name file
        std::ifstream name_file(hwmon_path + "/name");
        std::string sensor_name;
        if (name_file.is_open()) {
            std::getline(name_file, sensor_name);
        }
        
        // Look for coretemp (Intel) or k10temp (AMD)
        if (sensor_name != "coretemp" && sensor_name != "k10temp") {
            continue;
        }
        
        // Find temp*_label files
        DIR* temp_dir = opendir(hwmon_path.c_str());
        if (!temp_dir) continue;
        
        struct dirent* temp_entry;
        while ((temp_entry = readdir(temp_dir)) != nullptr) {
            std::string temp_file = temp_entry->d_name;
            if (temp_file.find("temp") != 0 || temp_file.find("_label") == std::string::npos) {
                continue;
            }
            
            // Read label
            std::ifstream label_file(hwmon_path + "/" + temp_file);
            std::string label;
            if (label_file.is_open()) {
                std::getline(label_file, label);
            }
            
            // Check for package/Tdie/Tctl
            bool is_package = (label.find("Package") != std::string::npos ||
                             label.find("Tdie") != std::string::npos ||
                             label.find("Tctl") != std::string::npos);
            
            if (is_package) {
                // Extract temp number (e.g., temp1_label → temp1_input)
                size_t pos = temp_file.find("_label");
                std::string temp_input_file = temp_file.substr(0, pos) + "_input";
                
                std::ifstream input_file(hwmon_path + "/" + temp_input_file);
                if (input_file.is_open()) {
                    int millidegrees = 0;
                    input_file >> millidegrees;
                    closedir(temp_dir);
                    closedir(dir);
                    return millidegrees / 1000.0;
                }
            }
        }
        closedir(temp_dir);
    }
    closedir(dir);
    
    return -1.0;
}

// ============================================================================
// Pass Determination (Calibration)
// ============================================================================

int determinePassesKernel(float* x, float* ones, int n, float target_seconds) {
    auto start = std::chrono::steady_clock::now();
    volatile float result = cblas_sdot(n, x, 1, ones, 1);
    auto end = std::chrono::steady_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    float t_one = elapsed.count();
    
    if (t_one <= 0) t_one = 1e-6f;
    
    int passes = (int)std::ceil(target_seconds / t_one * 1.05);
    return std::max(1, passes);
}

int determinePassesE2E(float* x, float* work, float* ones, int n, 
                       float target_seconds, bool inject_memcpy) {
    auto start = std::chrono::steady_clock::now();
    
    if (inject_memcpy) {
        std::memcpy(work, x, n * sizeof(float));
        volatile float result = cblas_sdot(n, work, 1, ones, 1);
    } else {
        volatile float result = cblas_sdot(n, x, 1, ones, 1);
    }
    
    auto end = std::chrono::steady_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    float t_one = elapsed.count();
    
    if (t_one <= 0) t_one = 1e-6f;
    
    int passes = (int)std::ceil(target_seconds / t_one * 1.05);
    return std::max(1, passes);
}

// ============================================================================
// CSV Output Functions
// ============================================================================

void writeCSVHeader(std::ofstream& file) {
    file << "timestamp,host,gpu_name,matrix_size,mode,batches,"
         << "seconds_target,seconds_gpu,seconds_wall,"
         << "energy_j,avg_power_w,below_target,"
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
                 const std::string& cpu_model, int n,
                 int passes_kernel, int passes_e2e,
                 const std::string& mode,
                 float seconds_kernel, float seconds_e2e,
                 double energy_kernel_j, double energy_e2e_j,
                 size_t bytes_total, double bw_gb_s,
                 double temp_c, int num_threads, bool inject_memcpy) {
    
    bool is_kernel_mode = (mode == "kernel");
    
    float seconds_gpu = seconds_kernel;
    float seconds_wall = is_kernel_mode ? seconds_kernel : seconds_e2e;
    
    double energy_j = is_kernel_mode ? energy_kernel_j : energy_e2e_j;
    
    double avg_power_w_kernel = -1.0;
    double avg_power_w_e2e = -1.0;
    
    if (energy_kernel_j >= 0 && seconds_kernel > 0) {
        avg_power_w_kernel = energy_kernel_j / seconds_kernel;
    }
    if (energy_e2e_j >= 0 && seconds_e2e > 0) {
        avg_power_w_e2e = energy_e2e_j / seconds_e2e;
    }
    
    double avg_power_w = -1.0;
    if (energy_j >= 0 && seconds_wall > 0) {
        avg_power_w = energy_j / seconds_wall;
    }
    
    bool below_target = (seconds_gpu < TARGET_RUNTIME_S);
    
    std::ostringstream notes;
    notes << "OPENBLAS_NUM_THREADS=" << num_threads << "; "
          << "e2e_host_memcpy=" << (inject_memcpy ? 1 : 0) << "; "
          << "e2e_bw=" << E2E_BW_MODE;
    
    file << getTimestamp() << ","
         << host << ","
         << cpu_model << ","
         << "0" << ","
         << mode << ","
         << (is_kernel_mode ? passes_kernel : passes_e2e) << ","
         << std::fixed << std::setprecision(2) << TARGET_RUNTIME_S << ","
         << std::setprecision(4) << seconds_gpu << ","
         << seconds_wall << ","
         << std::setprecision(3) << energy_j << ","
         << std::setprecision(1) << avg_power_w << ","
         << (below_target ? 1 : 0) << ","
         << "reduction" << ","
         << "openblas" << ","
         << "fp32" << ","
         << n << ","
         << passes_kernel << ","
         << passes_e2e << ","
         << std::setprecision(4) << seconds_kernel << ","
         << std::setprecision(3) << energy_kernel_j << ","
         << std::setprecision(1) << avg_power_w_kernel << ","
         << avg_power_w_e2e << ",";
    
    // bytes_total and bw_gb_s - output -1 if not defined (no memcpy in E2E)
    if (bytes_total != (size_t)-1) {
        file << bytes_total << ","
             << std::setprecision(2) << bw_gb_s << ",";
    } else {
        file << "-1" << ","
             << "-1" << ",";
    }
    
    file << mode << ","
         << mode << ","
         << "0" << ","
         << cpu_model << ","
         << "NA" << ","
         << "NA" << ","
         << "NA" << ","
         << "0" << ","
         << "0" << ","
         << "NA" << ","
         << "NA" << ",";
    
    // temp_c - empty if not available
    if (temp_c >= 0) {
        file << std::setprecision(1) << temp_c << ",";
    } else {
        file << ",";
    }
    
    file << "0" << ","
         << notes.str() << "\n";
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char** argv) {
    // Parse command-line arguments
    int num_threads = parseThreadsArgument(argc, argv);
    if (num_threads < 0) {
        std::cerr << "Error: --threads parameter is required\n\n";
        printUsage(argv[0]);
    }
    
    // Set OpenBLAS threads
    openblas_set_num_threads(num_threads);
    int actual_threads = openblas_get_num_threads();
    
    std::string hostname = getHostname();
    std::string cpu_model = getCPUModel();
    
    std::cout << "========================================\n";
    std::cout << "CPU Reduction Energy Benchmark (OpenBLAS)\n";
    std::cout << "========================================\n";
    std::cout << "System:            " << hostname << "\n";
    std::cout << "CPU:               " << cpu_model << "\n";
    std::cout << "OpenBLAS Threads:  " << actual_threads << " (requested: " << num_threads << ")\n";
    std::cout << "Target runtime:    " << TARGET_RUNTIME_S << "s\n";
    std::cout << "Repeats:           " << REPEATS << "\n";
    std::cout << "N sizes:           " << NUM_SIZES << " sizes (1M-384M)\n";
    std::cout << "E2E memcpy inject: " << (E2E_INJECT_HOST_MEMCPY ? "YES" : "NO") << "\n";
    std::cout << "E2E BW mode:       " << E2E_BW_MODE << "\n";
    std::cout << "Output:            " << OUTPUT_FILE << "\n";
    if (std::string(E2E_BW_MODE) == "memcpy_only") {
        std::cout << "Memcpy sidecar:    " << OUTPUT_FILE_MEMCPY << "\n";
    }
    std::cout << "========================================\n\n";
    
    // Discover RAPL zones
    std::vector<RAPLZone> rapl_zones = discoverRAPLZones();
    if (rapl_zones.empty()) {
        std::cout << "WARNING: No RAPL zones found. Energy measurements unavailable.\n";
        std::cout << "         (Run as root or configure /sys/class/powercap permissions)\n\n";
    } else {
        std::cout << "Found " << rapl_zones.size() << " RAPL package zone(s):\n";
        for (const auto& zone : rapl_zones) {
            std::cout << "  - " << zone.name << " (range: " 
                     << (zone.max_energy_range_uj / 1e6) << " J)\n";
        }
        std::cout << "\n";
    }
    
    // Allocate 64-byte aligned buffers using posix_memalign
    const size_t max_bytes = MAX_N * sizeof(float);
    
    float* x_max = nullptr;
    float* ones_max = nullptr;
    float* work_max = nullptr;
    
    if (posix_memalign((void**)&x_max, 64, max_bytes) != 0 ||
        posix_memalign((void**)&ones_max, 64, max_bytes) != 0 ||
        posix_memalign((void**)&work_max, 64, max_bytes) != 0) {
        std::cerr << "Error: Failed to allocate aligned memory\n";
        return EXIT_FAILURE;
    }
    
    std::cout << "Allocated 64-byte aligned buffers: "
              << (3 * max_bytes / (1024*1024)) << " MB total\n\n";
    
    // Initialize x_max with fixed seed
    std::cout << "Initializing x vector...\n";
    initializeVector(x_max, MAX_N, 42);
    
    // Initialize ones_max (persistent)
    std::cout << "Initializing ones vector...\n";
    std::fill(ones_max, ones_max + MAX_N, 1.0f);
    
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
    
    // ========================================================================
    // Main measurement loop
    // ========================================================================
    
    std::cout << "\nStarting measurements...\n";
    std::cout << "========================================\n\n";
    
    for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
        int n = N_SIZES[size_idx];
        size_t n_bytes = n * sizeof(float);
        
        std::cout << "N = " << n << " elements (" 
                  << (n_bytes / (1024*1024)) << " MB)\n";
        
        // Mini warmups
        for (int w = 0; w < 2; w++) {
            volatile float dummy = cblas_sdot(n, x_max, 1, ones_max, 1);
        }
        
        // Determine passes
        std::cout << "  Determining passes... " << std::flush;
        int passes_kernel = determinePassesKernel(x_max, ones_max, n, TARGET_RUNTIME_S);
        int passes_e2e = determinePassesE2E(x_max, work_max, ones_max, n, 
                                           TARGET_RUNTIME_S, E2E_INJECT_HOST_MEMCPY);
        std::cout << "kernel=" << passes_kernel << ", e2e=" << passes_e2e << "\n";
        
        // Bytes for each mode
        size_t bytes_total_kernel = 2ULL * n * sizeof(float) * passes_kernel;
        size_t bytes_total_e2e = E2E_INJECT_HOST_MEMCPY ? 
                                 (1ULL * n * sizeof(float) * passes_e2e) : (size_t)-1;
        
        // Vectors to collect measurements over REPEATS
        std::vector<float>  secs_kernel_v, secs_e2e_v;
        std::vector<double> energy_kernel_v, energy_e2e_v;
        std::vector<double> temp_v;
        
        // For memcpy_only mode
        std::vector<float>  secs_memcpy_only_v;
        std::vector<double> energy_memcpy_only_v;
        
        // Run REPEATS measurements
        for (int rep = 0; rep < REPEATS; rep++) {
            
            // ================================================================
            // KERNEL-ONLY Measurement
            // ================================================================
            
            auto energy_before_kernel = readRAPLEnergyPerZone(rapl_zones);
            auto start_kernel = std::chrono::steady_clock::now();
            
            volatile float sum_kernel = 0.0f;
            for (int p = 0; p < passes_kernel; p++) {
                sum_kernel += cblas_sdot(n, x_max, 1, ones_max, 1);
            }
            
            auto end_kernel = std::chrono::steady_clock::now();
            auto energy_after_kernel = readRAPLEnergyPerZone(rapl_zones);
            
            std::chrono::duration<double> duration_kernel = end_kernel - start_kernel;
            float seconds_kernel = duration_kernel.count();
            
            double energy_kernel_j = computeTotalEnergyDelta(energy_before_kernel, 
                                                             energy_after_kernel, 
                                                             rapl_zones);
            
            // ================================================================
            // E2E Measurement
            // ================================================================
            
            auto energy_before_e2e = readRAPLEnergyPerZone(rapl_zones);
            auto start_e2e = std::chrono::steady_clock::now();
            
            volatile float sum_e2e = 0.0f;
            for (int p = 0; p < passes_e2e; p++) {
                if (E2E_INJECT_HOST_MEMCPY) {
                    std::memcpy(work_max, x_max, n_bytes);
                    sum_e2e += cblas_sdot(n, work_max, 1, ones_max, 1);
                } else {
                    sum_e2e += cblas_sdot(n, x_max, 1, ones_max, 1);
                }
            }
            
            auto end_e2e = std::chrono::steady_clock::now();
            auto energy_after_e2e = readRAPLEnergyPerZone(rapl_zones);
            
            std::chrono::duration<double> duration_e2e = end_e2e - start_e2e;
            float seconds_e2e = duration_e2e.count();
            
            double energy_e2e_j = computeTotalEnergyDelta(energy_before_e2e, 
                                                          energy_after_e2e, 
                                                          rapl_zones);
            
            // ================================================================
            // Optional: MEMCPY-ONLY Measurement (for memcpy_only mode)
            // ================================================================
            
            float seconds_memcpy_only = -1.0f;
            double energy_memcpy_only_j = -1.0;
            
            if (std::string(E2E_BW_MODE) == "memcpy_only" && E2E_INJECT_HOST_MEMCPY) {
                auto energy_before_memcpy = readRAPLEnergyPerZone(rapl_zones);
                auto start_memcpy = std::chrono::steady_clock::now();
                
                // Pure memcpy loop (same passes as e2e)
                for (int p = 0; p < passes_e2e; p++) {
                    std::memcpy(work_max, x_max, n_bytes);
                }
                
                auto end_memcpy = std::chrono::steady_clock::now();
                auto energy_after_memcpy = readRAPLEnergyPerZone(rapl_zones);
                
                std::chrono::duration<double> duration_memcpy = end_memcpy - start_memcpy;
                seconds_memcpy_only = duration_memcpy.count();
                
                energy_memcpy_only_j = computeTotalEnergyDelta(energy_before_memcpy,
                                                               energy_after_memcpy,
                                                               rapl_zones);
                
                secs_memcpy_only_v.push_back(seconds_memcpy_only);
                energy_memcpy_only_v.push_back(energy_memcpy_only_j);
            }
            
            // Read temperature
            double temp_c = readCPUTemperature();
            
            // Collect measurements
            secs_kernel_v.push_back(seconds_kernel);
            secs_e2e_v.push_back(seconds_e2e);
            energy_kernel_v.push_back(energy_kernel_j);
            energy_e2e_v.push_back(energy_e2e_j);
            if (temp_c >= 0) temp_v.push_back(temp_c);
            
            // Compute power
            double power_kernel = (energy_kernel_j >= 0 && seconds_kernel > 0) 
                                  ? (energy_kernel_j / seconds_kernel) : -1.0;
            double power_e2e = (energy_e2e_j >= 0 && seconds_e2e > 0) 
                              ? (energy_e2e_j / seconds_e2e) : -1.0;
            
            // Console progress
            std::cout << "  + Run " << (rep + 1) << "/" << REPEATS << ": "
                     << "kernel=" << std::fixed << std::setprecision(3) << seconds_kernel << "s";
            if (energy_kernel_j >= 0) {
                std::cout << " E=" << std::setprecision(1) << energy_kernel_j << "J";
                if (power_kernel >= 0) {
                    std::cout << " P=" << std::setprecision(0) << power_kernel << "W";
                }
            }
            std::cout << " | e2e=" << std::setprecision(3) << seconds_e2e << "s";
            if (energy_e2e_j >= 0) {
                std::cout << " E=" << std::setprecision(1) << energy_e2e_j << "J";
                if (power_e2e >= 0) {
                    std::cout << " P=" << std::setprecision(0) << power_e2e << "W";
                }
            }
            if (temp_c >= 0) {
                std::cout << " T=" << std::setprecision(0) << temp_c << "°C";
            }
            if (seconds_kernel < TARGET_RUNTIME_S) {
                std::cout << " !";
            }
            std::cout << "\n";
        }
        
        // Aggregate and write CSV rows
        if (AGGREGATE_MEDIAN) {
            // Compute medians
            float  secK = median(secs_kernel_v);
            float  secE = median(secs_e2e_v);
            double enK  = median(energy_kernel_v);
            double enE  = median(energy_e2e_v);
            double tempC = temp_v.empty() ? -1.0 : median(temp_v);
            
            // Compute bandwidth from medians
            double bwK = (bytes_total_kernel > 0 && secK > 0) ? 
                         (bytes_total_kernel / (1e9 * secK)) : -1.0;
            double bwE = (bytes_total_e2e != (size_t)-1 && secE > 0) ? 
                         (bytes_total_e2e / (1e9 * secE)) : -1.0;
            
            // Write kernel row
            writeCSVRow(csv_file, hostname, cpu_model, n,
                       passes_kernel, passes_e2e,
                       "kernel",
                       secK, secE,  // seconds_kernel, seconds_e2e
                       enK, enE,    // energy_kernel_j, energy_e2e_j
                       bytes_total_kernel, bwK,
                       tempC, actual_threads, E2E_INJECT_HOST_MEMCPY);
            
            // Write e2e row
            writeCSVRow(csv_file, hostname, cpu_model, n,
                       passes_kernel, passes_e2e,
                       "e2e",
                       secK, secE,  // seconds_kernel, seconds_e2e
                       enK, enE,    // energy_kernel_j, energy_e2e_j
                       bytes_total_e2e, bwE,
                       tempC, actual_threads, E2E_INJECT_HOST_MEMCPY);
            
            csv_file.flush();
            
            std::cout << "  → Median: kernel=" << std::setprecision(3) << secK << "s, "
                     << "e2e=" << std::setprecision(3) << secE << "s\n"
                     << "     BW_kernel=" << std::setprecision(1) << bwK << " GB/s, "
                     << "BW_e2e=" << bwE << " GB/s\n";
            
            // If memcpy_only mode, write sidecar CSV
            if (std::string(E2E_BW_MODE) == "memcpy_only" && !secs_memcpy_only_v.empty()) {
                // Open/create sidecar CSV
                static bool sidecar_header_written = false;
                if (!sidecar_header_written) {
                    ensureDirectoryExists(OUTPUT_FILE_MEMCPY);
                    bool write_header = !fileExists(OUTPUT_FILE_MEMCPY);
                    
                    std::ofstream sidecar_file(OUTPUT_FILE_MEMCPY, std::ios::app);
                    if (write_header) {
                        writeCSVHeader(sidecar_file);
                    }
                    sidecar_file.close();
                    sidecar_header_written = true;
                }
                
                std::ofstream sidecar_file(OUTPUT_FILE_MEMCPY, std::ios::app);
                
                float  secM = median(secs_memcpy_only_v);
                double enM  = median(energy_memcpy_only_v);
                double bwM  = (bytes_total_e2e != (size_t)-1 && secM > 0) ?
                              (bytes_total_e2e / (1e9 * secM)) : -1.0;
                
                // Write memcpy_only row (as "e2e" mode with pure memcpy time)
                writeCSVRow(sidecar_file, hostname, cpu_model, n,
                           passes_kernel, passes_e2e,
                           "e2e",
                           secK, secM,  // seconds_kernel, seconds_memcpy_only
                           enK, enM,    // energy_kernel_j, energy_memcpy_only_j
                           bytes_total_e2e, bwM,
                           tempC, actual_threads, E2E_INJECT_HOST_MEMCPY);
                
                sidecar_file.close();
                
                std::cout << "     BW_memcpy_only=" << bwM << " GB/s (sidecar)\n";
            }
        }
        
        std::cout << "\n";
    }
    
    std::cout << "========================================\n";
    std::cout << "Benchmark complete!\n";
    std::cout << "Results saved to: " << OUTPUT_FILE << "\n";
    if (AGGREGATE_MEDIAN) {
        std::cout << "Total measurements: " << (NUM_SIZES * 2) << " rows (median over " 
                 << REPEATS << " repeats)\n";
    } else {
        std::cout << "Total measurements: " << (NUM_SIZES * REPEATS * 2) << " rows\n";
    }
    std::cout << "========================================\n";
    
    // Cleanup
    csv_file.close();
    
    free(x_max);
    free(ones_max);
    free(work_max);
    
    return EXIT_SUCCESS;
}

// ============================================================================
// Abnahmekriterien & Checkliste
// ============================================================================
/*
 * [✓] Threads via --threads Parameter (Pflicht, kein Hardcode)
 * [✓] Zwei Modi pro N (kernel & e2e), getrennt kalibriert auf ~1s
 * [✓] Einsen-Vektor persistent; DOT via cblas_sdot (inc=1)
 * [✓] Timing: steady_clock um genau das Fenster (energie-kongruent)
 * [✓] RAPL: ΔE pro Zone mit max_energy_range_uj, Overflow-Guard, fehlerhafte Zonen überspringen
 * [✓] CSV-Header exakt wie GPU-Version; alle Felder korrekt befüllt
 * [✓] Bytes/BW: Kernel 2*N*4*passes; E2E 1*N*4*passes nur wenn inject=true, sonst -1
 * [✓] includes_transfer=0 (CPU), PCIe/Clocks NA/0, Temp best effort (hwmon)
 * [✓] Append-Modus, Header nur einmal; notes mit threads + e2e_host_memcpy + e2e_bw mode
 * [✓] Speicher 64-B aligned via posix_memalign (kein UB)
 * [✓] volatile für Dead-Code-Protection
 * [✓] Link-Flags: -lopenblas -lpthread -lm
 * [✓] Median-Aggregation: Pro N genau 2 CSV-Zeilen (kernel/e2e) mit Median über REPEATS
 * [✓] E2E_BW_MODE: "mixed" (default) oder "memcpy_only" (schreibt Sidecar-CSV)
 */