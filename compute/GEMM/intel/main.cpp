// main.cpp - CPU GEMM Energy Benchmark with OpenBLAS
// 
// Build: g++ -O3 -march=native -std=c++17 -o cpu_bench main.cpp -lopenblas
// Usage: ./cpu_bench
//
// Requirements:
// - OpenBLAS (CBLAS API)
// - CPU stabilization via enable_CPU_Intel.sh (run before this script)
// - RAPL access for energy measurements (optional, falls back to NA)

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
#include <random>
#include <algorithm>
#include <vector>
#include <glob.h>
#include <unistd.h>
#include <sys/stat.h>

// ============================================================================
// Configuration - hardcoded (same as GPU variant)
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    MAX_BATCH_SIZE   = 200000;
constexpr int    MACRO_REPEATS    = 5;
constexpr int    WARMUP_SIZE      = 512;  // Size for warmup SGEMM
constexpr int    NUM_THREADS      = 20;   // OpenBLAS threads

// GEMM sizes - same as GPU variant
static const int GEMM_SIZES[] = {
    64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896,
    1024, 1152, 1280, 1408, 1536
};
static const int NUM_SIZES = sizeof(GEMM_SIZES) / sizeof(GEMM_SIZES[0]);
static const int MAX_SIZE = *std::max_element(std::begin(GEMM_SIZES), std::end(GEMM_SIZES));

static const char* OUTPUT_FILE = "data/raw/energy_benchmark_cpu.csv";

// ============================================================================
// OpenBLAS Threading
// ============================================================================

extern "C" {
    void openblas_set_num_threads(int num_threads);
    int  openblas_get_num_threads();
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

void ensureDirectoryExists(const char* filepath) {
    std::string path(filepath);
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        std::string dir = path.substr(0, pos);
        std::string cmd = "mkdir -p " + dir;
        system(cmd.c_str());
    }
}

bool fileExists(const char* filepath) {
    struct stat buffer;
    return (stat(filepath, &buffer) == 0);
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
    
    double total_j = 0.0;
    bool found = false;
    
    for (const auto& zone : candidates) {
        // Check if it's a directory
        struct stat st;
        if (stat(zone.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            continue;
        }
        
        // Read zone name
        std::string name_path = zone + "/name";
        std::string name = readFile(name_path);
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        
        // Only consider package zones
        if (name.find("package") == std::string::npos) {
            continue;
        }
        
        // Read energy_uj
        std::string energy_path = zone + "/energy_uj";
        std::string energy_str = readFile(energy_path);
        
        if (!energy_str.empty()) {
            try {
                unsigned long long uj = std::stoull(energy_str);
                total_j += uj / 1e6;  // µJ → J
                found = true;
            } catch (...) {
                // Ignore parse errors
            }
        }
    }
    
    return found ? total_j : -1.0;
}

// ============================================================================
// CPU Temperature Measurement
// ============================================================================

double readCPUTemperature() {
    // Read CPU package temperature from hwmon
    // Returns temperature in °C, or -1.0 if not accessible
    
    auto hwmon_dirs = globPaths("/sys/class/hwmon/hwmon*");
    
    for (const auto& hwmon_dir : hwmon_dirs) {
        std::string name_path = hwmon_dir + "/name";
        std::string name = readFile(name_path);
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        
        bool is_cpu_sensor = (name.find("coretemp") != std::string::npos ||  // Intel
                             name.find("k10temp") != std::string::npos);      // AMD
        
        if (!is_cpu_sensor) {
            continue;
        }
        
        // Find package temperature
        auto temp_labels = globPaths(hwmon_dir + "/temp*_label");
        
        for (const auto& label_path : temp_labels) {
            std::string label = readFile(label_path);
            std::transform(label.begin(), label.end(), label.begin(), ::tolower);
            
            // Intel: "package id 0", AMD: "tdie" or "tctl"
            bool is_package = (label.find("package") != std::string::npos ||
                              label.find("tdie") != std::string::npos ||
                              label.find("tctl") != std::string::npos);
            
            if (is_package) {
                // Extract temp number from path (e.g., temp1_label → temp1_input)
                size_t pos = label_path.find("_label");
                if (pos != std::string::npos) {
                    std::string input_path = label_path.substr(0, pos) + "_input";
                    std::string temp_str = readFile(input_path);
                    
                    if (!temp_str.empty()) {
                        try {
                            int temp_milli = std::stoi(temp_str);
                            return temp_milli / 1000.0;  // milli°C → °C
                        } catch (...) {
                            // Ignore parse errors
                        }
                    }
                }
            }
        }
    }
    
    return -1.0;  // Not found
}

// ============================================================================
// Matrix Initialization
// ============================================================================

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
// Batch Size Determination
// ============================================================================

int determineBatchSize(float* A, float* B, float* C, int n, int lda, double target_seconds) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int batch = 1;
    
    while (batch <= MAX_BATCH_SIZE) {
        auto start = std::chrono::steady_clock::now();
        
        for (int b = 0; b < batch; b++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       n, n, n, alpha, A, lda, B, lda, beta, C, lda);
        }
        
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        if (elapsed.count() >= target_seconds) {
            return batch;
        }
        
        if (batch >= MAX_BATCH_SIZE) {
            return batch;  // Hit max limit
        }
        
        batch = std::min(batch * 2, MAX_BATCH_SIZE);
    }
    
    return batch;
}

// ============================================================================
// CSV Output Functions
// ============================================================================

void writeCSVHeader(std::ofstream& file) {
    file << "timestamp,host,gpu_name,matrix_size,mode,batches,"
         << "seconds_target,seconds_gpu,seconds_wall,"
         << "energy_j,avg_power_w,below_target,"
         << "pcie_gen_current,pcie_width_current,"
         << "clocks_sm_mhz,clocks_mem_mhz,temp_c,throttle_reasons\n";
}

void writeCSVRow(std::ofstream& file, const std::string& host,
                const std::string& cpu_model, int n, int batches,
                double compute_time_s, double energy_j, double avg_power_w,
                bool below_target, double temp_c) {
    file << getTimestamp() << ","
         << host << ","
         << cpu_model << ","
         << n << ","
         << "e2e" << ","
         << batches << ","
         << TARGET_RUNTIME_S << ","
         << std::fixed << std::setprecision(4) << compute_time_s << ","
         << compute_time_s << ","  // wall time = compute time for CPU
         << std::setprecision(3);
    
    // Energy (NA if not available)
    if (energy_j >= 0) {
        file << energy_j;
    } else {
        file << "NA";
    }
    file << ",";
    
    // Power (NA if not available)
    if (avg_power_w >= 0) {
        file << std::setprecision(1) << avg_power_w;
    } else {
        file << "NA";
    }
    file << ",";
    
    file << (below_target ? 1 : 0) << ","
         << "NA" << ","  // pcie_gen_current
         << "NA" << ","  // pcie_width_current
         << "NA" << ","  // clocks_sm_mhz
         << "NA" << ","; // clocks_mem_mhz
    
    // Temperature (NA if not available)
    if (temp_c >= 0) {
        file << std::setprecision(1) << temp_c;
    } else {
        file << "NA";
    }
    file << ",";
    
    file << "NA" << "\n";  // throttle_reasons
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main() {
    // Set OpenBLAS threads
    openblas_set_num_threads(NUM_THREADS);
    
    // Get system info
    std::string cpu_model = getCPUModel();
    std::string hostname = getHostname();
    
    // Test RAPL availability
    double rapl_test = readRAPLEnergy();
    bool rapl_available = (rapl_test >= 0);
    
    if (!rapl_available) {
        std::cerr << "WARN: RAPL not accessible (Permission denied). "
                  << "Energy/power will be NA.\n";
    }
    
    // Print configuration
    std::cout << "========================================\n";
    std::cout << "CPU GEMM Energy Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "System:         " << hostname << "\n";
    std::cout << "CPU:            " << cpu_model << "\n";
    std::cout << "Threads:        " << openblas_get_num_threads() << "\n";
    std::cout << "Target runtime: " << TARGET_RUNTIME_S << "s\n";
    std::cout << "Macro repeats:  " << MACRO_REPEATS << "\n";
    std::cout << "Matrix sizes:   " << NUM_SIZES << " sizes (64-1536)\n";
    std::cout << "RAPL available: " << (rapl_available ? "Yes" : "No") << "\n";
    std::cout << "Output:         " << OUTPUT_FILE << "\n";
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
    
    // Warm-up: single SGEMM call
    std::cout << "Running warm-up (512x512)...\n";
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               WARMUP_SIZE, WARMUP_SIZE, WARMUP_SIZE,
               alpha, h_A, MAX_SIZE, h_B, MAX_SIZE, beta, h_C, MAX_SIZE);
    std::cout << "Warm-up complete.\n\n";
    
    // Prepare CSV output - append mode (never overwrite)
    ensureDirectoryExists(OUTPUT_FILE);
    bool write_header = !fileExists(OUTPUT_FILE);
    
    std::ofstream csv_file(OUTPUT_FILE, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Cannot open output file: " << OUTPUT_FILE << "\n";
        free(h_A);
        free(h_B);
        free(h_C);
        return EXIT_FAILURE;
    }
    
    if (write_header) {
        writeCSVHeader(csv_file);
    }
    
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
        int batches = determineBatchSize(h_A, h_B, h_C, n, MAX_SIZE, TARGET_RUNTIME_S);
        std::cout << "using " << batches << " batches\n";
        
        // Run MACRO_REPEATS measurements
        for (int rep = 0; rep < MACRO_REPEATS; rep++) {
            // ================================================================
            // Measurement Start
            // ================================================================
            
            double energy_before = readRAPLEnergy();
            
            auto compute_start = std::chrono::steady_clock::now();
            
            // Execute batch
            for (int b = 0; b < batches; b++) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                           n, n, n, alpha, h_A, MAX_SIZE, h_B, MAX_SIZE, 
                           beta, h_C, MAX_SIZE);
            }
            
            auto compute_end = std::chrono::steady_clock::now();
            
            double energy_after = readRAPLEnergy();
            
            // ================================================================
            // Measurement End
            // ================================================================
            
            // Calculate timing
            std::chrono::duration<double> compute_duration = compute_end - compute_start;
            double compute_time_s = compute_duration.count();
            
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
            
            // Read CPU temperature
            double temp_c = readCPUTemperature();
            
            // Check if below target
            bool below_target = (compute_time_s < TARGET_RUNTIME_S);
            
            // Write to CSV
            writeCSVRow(csv_file, hostname, cpu_model, n, batches,
                       compute_time_s, energy_j, avg_power_w,
                       below_target, temp_c);
            csv_file.flush();
            
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
            if (temp_c >= 0) {
                std::cout << " T=" << std::setprecision(1) << temp_c << "°C";
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
    free(h_A);
    free(h_B);
    free(h_C);
    
    return EXIT_SUCCESS;
}