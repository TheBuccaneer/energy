// stream_cpu.cpp - OpenMP STREAM Triad CPU Benchmark with RAPL Energy
// Compile: g++ -O3 -march=native -fopenmp -std=c++17 -o stream_cpu stream_cpu.cpp
// Optional: -DNT_STORE for non-temporal stores (12/24 B/iter instead of 32)
// Optional: -DUSE_DOUBLE for FP64
// Requires: GCC 9+ or Clang 10+ for OpenMP 5.0 nontemporal support

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
#include <set>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <cstdlib>
#include <cmath>
#include <omp.h>

// ============================================================================
// Data Type Configuration
// ============================================================================

#ifdef USE_DOUBLE
using real = double;
constexpr const char* DTYPE_STR = "float64";
#ifdef NT_STORE
constexpr size_t BYTES_PER_ITER = 24ULL;  // NT: 2 reads + 1 NT write
#else
constexpr size_t BYTES_PER_ITER = 32ULL;  // RFO: RFO(A) + R(B) + R(C) + W(A)
#endif
#else
using real = float;
constexpr const char* DTYPE_STR = "float32";
#ifdef NT_STORE
constexpr size_t BYTES_PER_ITER = 12ULL;  // NT: 2 reads + 1 NT write
#else
constexpr size_t BYTES_PER_ITER = 32ULL;  // RFO: RFO(A) + R(B) + R(C) + W(A)
#endif
#endif

// ============================================================================
// Configuration
// ============================================================================

// GPU sizes from main.cu - we'll check which are DRAM-regime on CPU
#ifdef USE_DOUBLE
static constexpr size_t GPU_SIZES[] = {
    1ULL << 20,  // 1M
    1ULL << 22,  // 4M
    1ULL << 24,  // 16M
    1ULL << 26,  // 64M
    1ULL << 27,  // 128M
    1ULL << 28   // 256M
};
#else
static constexpr size_t GPU_SIZES[] = {
    1ULL << 20,  // 1M
    1ULL << 22,  // 4M
    1ULL << 24,  // 16M
    1ULL << 26,  // 64M
    1ULL << 27,  // 128M
    1ULL << 28,  // 256M
    1ULL << 29   // 512M
};
#endif

// CPU-extra sizes (×2 steps beyond GPU, if needed)
static constexpr size_t CPU_EXTRA_SIZES[] = {
#ifdef USE_DOUBLE
    1ULL << 29,  // 512M
    1ULL << 30   // 1G
#else
    1ULL << 30,  // 1G
    1ULL << 31   // 2G (careful with RAM)
#endif
};

static constexpr int REPEATS = 50;
static constexpr double TARGET_S = 1.1;
static constexpr double SAFETY_FACTOR = 1.02;
static constexpr real SCALAR_Q = 3.0;

const char* CSV_PATH = "data/raw/stream_triad_cpu.csv";

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

// Get number of physical packages (sockets)
int getNumSockets() {
    std::set<int> packages;
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("physical id") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                int pkg_id = std::stoi(line.substr(pos + 1));
                packages.insert(pkg_id);
            }
        }
    }
    
    return packages.empty() ? 1 : packages.size();
}

// Get LLC size per socket (largest found)
size_t getLLCPerSocket() {
    const char* base = "/sys/devices/system/cpu";
    size_t max_llc = 0;
    
    DIR* dir = opendir(base);
    if (!dir) return 0;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.find("cpu") != 0 || !isdigit(name[3])) continue;
        
        std::string cache_path = std::string(base) + "/" + name + "/cache/index3/size";
        std::ifstream cache_file(cache_path);
        if (!cache_file.is_open()) continue;
        
        std::string size_str;
        std::getline(cache_file, size_str);
        
        size_t value = 0;
        char unit = 0;
        std::istringstream iss(size_str);
        iss >> value >> unit;
        
        size_t bytes = value;
        if (unit == 'K' || unit == 'k') bytes = value * 1024;
        else if (unit == 'M' || unit == 'm') bytes = value * 1024 * 1024;
        
        if (bytes > max_llc) max_llc = bytes;
    }
    closedir(dir);
    
    return max_llc;
}

// FIX #2: Sum LLC across all sockets
size_t getTotalCPULLCSize() {
    int num_sockets = getNumSockets();
    size_t llc_per_socket = getLLCPerSocket();
    
    if (llc_per_socket == 0) {
        // Fallback: assume 32 MB LLC per socket
        llc_per_socket = 32 * 1024 * 1024;
    }
    
    return num_sockets * llc_per_socket;
}

bool is_dram_regime(size_t N) {
    // Rule: Each of 3 arrays >= 4× LLC OR >= 1M elements (whichever is larger)
    size_t llc_total = getTotalCPULLCSize();
    
    size_t array_bytes = N * sizeof(real);
    size_t min_dram_bytes = 4 * llc_total;
    size_t min_1m_bytes = 1000000 * sizeof(real);
    
    size_t threshold = std::max(min_dram_bytes, min_1m_bytes);
    
    return array_bytes >= threshold;
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

// ============================================================================
// RAPL Energy Reading (from main.cpp)
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
    if (!dir) return zones;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.find("rapl") != std::string::npos) {
            std::string zone_path = std::string(base_path) + "/" + name;
            
            std::ifstream name_file(zone_path + "/name");
            std::string zone_name;
            if (name_file.is_open()) {
                std::getline(name_file, zone_name);
            }
            
            bool is_pkg = (zone_name.find("package") != std::string::npos);
            
            if (is_pkg) {
                RAPLZone zone;
                zone.path = zone_path;
                zone.name = zone_name;
                
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

std::vector<unsigned long long> readRAPLEnergyPerZone(const std::vector<RAPLZone>& zones) {
    std::vector<unsigned long long> readings;
    
    for (const auto& zone : zones) {
        std::ifstream energy_file(zone.path + "/energy_uj");
        if (energy_file.is_open()) {
            unsigned long long uj = 0;
            energy_file >> uj;
            readings.push_back(uj);
        } else {
            readings.push_back(0);
        }
    }
    
    return readings;
}

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
        
        if (b == 0 || a == 0 || r == 0) continue;
        
        unsigned long long delta_uj = (a >= b) ? (a - b) : (a + r - b);
        total_j += delta_uj / 1e6;
        any_valid = true;
    }
    
    return any_valid ? total_j : -1.0;
}

// ============================================================================
// STREAM Triad Kernel (OpenMP)
// FIX #1: Use OpenMP 5.0 nontemporal clause instead of broken intrinsics
// ============================================================================

void triad_kernel(real* __restrict__ a, 
                 const real* __restrict__ b, 
                 const real* __restrict__ c,
                 real q, 
                 size_t N) {
    
#ifdef NT_STORE
    // OpenMP 5.0 nontemporal - requires GCC 9+ / Clang 10+
    #if defined(__GNUC__) && __GNUC__ >= 9
        #pragma omp parallel for simd nontemporal(a) schedule(static)
        for (size_t i = 0; i < N; i++) {
            a[i] = b[i] + q * c[i];
        }
    #else
        // Fallback: standard implementation (compiler too old for nontemporal)
        #pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < N; i++) {
            a[i] = b[i] + q * c[i];
        }
        #warning "NT_STORE defined but compiler does not support OpenMP 5.0 nontemporal - using standard stores"
    #endif
#else
    // Standard version with write-allocate (32 B/iter)
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < N; i++) {
        a[i] = b[i] + q * c[i];
    }
#endif
}

// Parallel initialization (first-touch)
void parallel_init(real* arr, size_t N, real value) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; i++) {
        arr[i] = value;
    }
}

// ============================================================================
// CSV Output
// FIX #3: Handle missing/bad energy properly with empty fields and notes
// FIX #4: Set batches=1 (consistent with reduction), put repeat in notes
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
                const std::string& cpu_model,
                size_t N,
                size_t passes,
                int repeat_idx,
                double seconds_kernel,
                double energy_j,
                double avg_power_w,
                size_t bytes_total,
                double bw_gb_s,
                const std::string& cohort,
                int num_threads,
                const std::string& omp_places,
                const std::string& omp_proc_bind,
                bool has_rapl) {
    
    int below_target = (seconds_kernel >= TARGET_S) ? 0 : 1;
    
    // FIX #3: Build notes with energy status
    std::ostringstream notes;
    notes << "sweep;repeats=" << REPEATS 
          << ";repeat_idx=" << repeat_idx  // FIX #4: repeat in notes
          << ";cohort=" << cohort
          << ";OMP_NUM_THREADS=" << num_threads
          << ";places=" << omp_places
          << ";proc_bind=" << omp_proc_bind
          << ";bytes_per_iter=" << BYTES_PER_ITER
#ifdef NT_STORE
          << ";nt_store=1";
#else
          << ";nt_store=0";
#endif
    
    // FIX #3: Add energy quality markers
    if (!has_rapl) {
        notes << ";no_energy";
    } else if (energy_j < 0) {
        notes << ";bad_energy_delta";
    }
    
    // Nice-to-have: pJ/Byte and GB/s/W (size-independent metrics)
    if (energy_j > 0 && bytes_total > 0) {
        double pj_per_byte = (energy_j * 1e12) / bytes_total;
        notes << ";pJ_per_B=" << std::fixed << std::setprecision(3) << pj_per_byte;
    }
    if (avg_power_w > 0 && bw_gb_s > 0) {
        double gbps_per_w = bw_gb_s / avg_power_w;
        notes << ";GBps_per_W=" << std::fixed << std::setprecision(4) << gbps_per_w;
    }
    
    // FIX #3: Empty fields for missing/bad energy instead of -1
    std::string energy_str = (has_rapl && energy_j >= 0) ? 
        (std::ostringstream() << std::fixed << std::setprecision(3) << energy_j).str() : "";
    std::string power_str = (has_rapl && avg_power_w >= 0) ? 
        (std::ostringstream() << std::fixed << std::setprecision(1) << avg_power_w).str() : "";
    
    file << timestamp << ","
         << hostname << ","
         << "NA" << ","  // gpu_name
         << "0" << ","   // matrix_size
         << "kernel" << ","
         << "1" << ","   // FIX #4: batches=1 (consistent)
         << std::fixed << std::setprecision(2) << TARGET_S << ","
         << std::setprecision(4) << seconds_kernel << ","
         << seconds_kernel << ","  // seconds_wall = seconds_kernel
         << energy_str << ","
         << power_str << ","
         << below_target << ","
         << "stream_triad" << ","
         << "openmp" << ","
         << DTYPE_STR << ","
         << N << ","
         << passes << ","  // passes_kernel
         << passes << ","  // passes_e2e (same)
         << std::setprecision(4) << seconds_kernel << ","
         << energy_str << ","
         << power_str << ","
         << power_str << ","  // avg_power_w_e2e
         << bytes_total << ","
         << std::setprecision(2) << bw_gb_s << ","
         << "steady_clock" << ","  // time_mode
         << "rapl_energy_uj_delta" << ","  // energy_mode
         << "0" << ","  // includes_transfer
         << "cpu:" << cpu_model << ","
         << "NA" << ","  // driver_version
         << "0,0,0,0,"  // pcie stats
         << "NA,NA,"    // clocks
         << ","         // temp_c (empty)
         << "0" << ","  // throttle_reasons
         << notes.str() << "\n";
}

// ============================================================================
// Calibration
// ============================================================================

size_t calibrateWithWarmup(real* a, const real* b, const real* c, 
                          real q, size_t N) {
    // Untimed warm-up pass
    triad_kernel(a, b, c, q, N);
    
    // Timed calibration pass
    auto start = std::chrono::steady_clock::now();
    triad_kernel(a, b, c, q, N);
    auto end = std::chrono::steady_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double t_pass = elapsed.count();
    
    size_t passes = std::max(static_cast<size_t>(1), 
                            static_cast<size_t>(std::ceil((TARGET_S / t_pass) * SAFETY_FACTOR)));
    return passes;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Get environment variables
    const char* omp_threads_env = std::getenv("OMP_NUM_THREADS");
    const char* omp_places_env = std::getenv("OMP_PLACES");
    const char* omp_proc_bind_env = std::getenv("OMP_PROC_BIND");
    
    std::string omp_places = omp_places_env ? omp_places_env : "unset";
    std::string omp_proc_bind = omp_proc_bind_env ? omp_proc_bind_env : "unset";
    
    int num_threads = omp_get_max_threads();
    std::string hostname = getHostname();
    std::string cpu_model = getCPUModel();
    
    // FIX #2: Report socket count and total LLC
    int num_sockets = getNumSockets();
    size_t llc_per_socket = getLLCPerSocket();
    size_t llc_total = getTotalCPULLCSize();
    
    std::cout << "========================================\n";
    std::cout << "STREAM Triad - CPU (OpenMP)\n";
    std::cout << "========================================\n";
    std::cout << "Host:           " << hostname << "\n";
    std::cout << "CPU:            " << cpu_model << "\n";
    std::cout << "Sockets:        " << num_sockets << "\n";
    std::cout << "LLC/socket:     " << readableBytes(llc_per_socket) << "\n";
    std::cout << "LLC Total:      " << readableBytes(llc_total) << "\n";
    std::cout << "OpenMP Threads: " << num_threads << "\n";
    std::cout << "OMP_PLACES:     " << omp_places << "\n";
    std::cout << "OMP_PROC_BIND:  " << omp_proc_bind << "\n";
    std::cout << "Data type:      " << DTYPE_STR << "\n";
    std::cout << "Bytes/iter:     " << BYTES_PER_ITER;
#ifdef NT_STORE
    std::cout << " (NT stores, OpenMP 5.0)";
#else
    std::cout << " (write-allocate)";
#endif
    std::cout << "\n";
    std::cout << "Target runtime: " << TARGET_S << "s\n";
    std::cout << "Repeats/size:   " << REPEATS << "\n";
    std::cout << "Output:         " << CSV_PATH << "\n";
    std::cout << "========================================\n\n";
    
    // Discover RAPL zones
    std::vector<RAPLZone> rapl_zones = discoverRAPLZones();
    bool has_rapl = !rapl_zones.empty();
    
    if (!has_rapl) {
        std::cout << "⚠️  No RAPL zones found. Energy measurements unavailable.\n";
        std::cout << "    (Run as root or configure /sys/class/powercap permissions)\n\n";
    } else {
        std::cout << "Found " << rapl_zones.size() << " RAPL package zone(s):\n";
        for (const auto& zone : rapl_zones) {
            std::cout << "  - " << zone.name << " (range: " 
                     << (zone.max_energy_range_uj / 1e6) << " J)\n";
        }
        std::cout << "\n";
    }
    
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
    
    // Build size list with cohorts
    struct SizeInfo {
        size_t N;
        std::string cohort;
    };
    
    std::vector<SizeInfo> size_list;
    
    // Add GPU sizes that meet DRAM regime on CPU
    constexpr size_t num_gpu_sizes = sizeof(GPU_SIZES) / sizeof(GPU_SIZES[0]);
    for (size_t i = 0; i < num_gpu_sizes; i++) {
        if (is_dram_regime(GPU_SIZES[i])) {
            size_list.push_back({GPU_SIZES[i], "common"});
        }
    }
    
    // Add CPU-extra sizes
    constexpr size_t num_extra_sizes = sizeof(CPU_EXTRA_SIZES) / sizeof(CPU_EXTRA_SIZES[0]);
    for (size_t i = 0; i < num_extra_sizes; i++) {
        size_list.push_back({CPU_EXTRA_SIZES[i], "cpu_extra"});
    }
    
    std::cout << "Size list (" << size_list.size() << " sizes):\n";
    for (const auto& info : size_list) {
        std::cout << "  N=" << info.N 
                  << " (" << readableBytes(3 * info.N * sizeof(real)) << " total)"
                  << " [" << info.cohort << "]\n";
    }
    std::cout << "\n";
    
    // Find max size for allocation
    size_t max_N = 0;
    for (const auto& info : size_list) {
        max_N = std::max(max_N, info.N);
    }
    
    // Allocate arrays (aligned)
    real* a = nullptr;
    real* b = nullptr;
    real* c = nullptr;
    
    if (posix_memalign((void**)&a, 64, max_N * sizeof(real)) != 0 ||
        posix_memalign((void**)&b, 64, max_N * sizeof(real)) != 0 ||
        posix_memalign((void**)&c, 64, max_N * sizeof(real)) != 0) {
        std::cerr << "Error: Failed to allocate aligned memory\n";
        return EXIT_FAILURE;
    }
    
    std::cout << "Allocated 64-byte aligned buffers: "
              << readableBytes(3 * max_N * sizeof(real)) << " total\n\n";
    
    // Initialize arrays (parallel first-touch)
    std::cout << "Initializing arrays (parallel first-touch)...\n";
    parallel_init(a, max_N, 0.0);
    parallel_init(b, max_N, 1.0);
    parallel_init(c, max_N, 2.0);
    std::cout << "✓ Initialization complete\n\n";
    
    // ========================================================================
    // SIZE SWEEP
    // ========================================================================
    
    std::cout << "Starting measurements...\n";
    std::cout << "========================================\n\n";
    
    for (const auto& size_info : size_list) {
        size_t N = size_info.N;
        std::string cohort = size_info.cohort;
        
        std::cout << "┌────────────────────────────────────┐\n";
        std::cout << "│ N = " << N 
                  << " (2^" << static_cast<int>(std::log2(N)) << ")"
                  << " [" << cohort << "]\n";
        std::cout << "└────────────────────────────────────┘\n";
        
        // Calibrate
        std::cout << "Calibrating (with warm-up)... " << std::flush;
        size_t passes = calibrateWithWarmup(a, b, c, SCALAR_Q, N);
        std::cout << passes << " passes\n";
        
        size_t bytes_total = BYTES_PER_ITER * N * passes;
        
        // Measurement loop
        std::cout << "Measuring " << REPEATS << " runs...\n";
        
        for (int run = 0; run < REPEATS; run++) {
            // Reset array a (outside measurement)
            parallel_init(a, N, 0.0);
            
            // Energy start
            auto energy_before = readRAPLEnergyPerZone(rapl_zones);
            
            // Timing start
            auto start = std::chrono::steady_clock::now();
            
            // Kernel loop
            for (size_t p = 0; p < passes; p++) {
                triad_kernel(a, b, c, SCALAR_Q, N);
            }
            
            // Timing stop
            auto end = std::chrono::steady_clock::now();
            
            // Energy stop
            auto energy_after = readRAPLEnergyPerZone(rapl_zones);
            
            // Calculate metrics
            std::chrono::duration<double> elapsed = end - start;
            double seconds_kernel = elapsed.count();
            
            // FIX #3: Handle missing/bad energy properly
            double energy_j = -1.0;
            double avg_power_w = -1.0;
            
            if (has_rapl) {
                energy_j = computeTotalEnergyDelta(energy_before, energy_after, rapl_zones);
                if (energy_j >= 0 && seconds_kernel > 0) {
                    avg_power_w = energy_j / seconds_kernel;
                }
            }
            
            double bw_gb_s = (bytes_total / 1e9) / seconds_kernel;
            
            // Write CSV
            writeCSVRow(csv_file, getTimestamp(), hostname, cpu_model, N, passes,
                       run, seconds_kernel, energy_j, avg_power_w, bytes_total,
                       bw_gb_s, cohort, num_threads, omp_places, omp_proc_bind,
                       has_rapl);
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
        
        std::cout << "✓ Complete\n\n";
    }
    
    // Cleanup
    csv_file.close();
    free(a);
    free(b);
    free(c);
    
    std::cout << "========================================\n";
    std::cout << "✓ Benchmark complete!\n";
    std::cout << "Results: " << CSV_PATH << "\n";
    std::cout << "========================================\n";
    
    return EXIT_SUCCESS;
}

// ============================================================================
// Build Instructions
// ============================================================================
/*
 * Standard (FP32, write-allocate, 32 B/iter):
 *   g++ -O3 -march=native -fopenmp -std=c++17 -o stream_cpu stream_cpu.cpp
 *
 * FP64:
 *   g++ -O3 -march=native -fopenmp -std=c++17 -DUSE_DOUBLE -o stream_cpu_fp64 stream_cpu.cpp
 *
 * Non-temporal stores (12/24 B/iter) - REQUIRES GCC 9+ or Clang 10+ for OpenMP 5.0:
 *   g++ -O3 -march=native -fopenmp -std=c++17 -DNT_STORE -o stream_cpu_nt stream_cpu.cpp
 *
 * Usage:
 *   export OMP_NUM_THREADS=<phys_cores>
 *   export OMP_PLACES=cores
 *   export OMP_PROC_BIND=spread
 *   ./stream_cpu
 *
 * Requirements:
 *   - Linux with /sys/class/powercap (RAPL) - run as root or configure permissions
 *   - GCC 9+ or Clang 10+ for OpenMP 5.0 nontemporal support (if using -DNT_STORE)
 *   - GCC 7+ or Clang 9+ without -DNT_STORE
 *
 * RAPL Permissions (if not running as root):
 *   sudo chmod -R a+r /sys/class/powercap/intel-rapl*
 */

// ============================================================================
// Acceptance Checklist
// ============================================================================
/*
 * [✓] FIX #1: NT-Store mit OpenMP 5.0 nontemporal (korrekt, portabel)
 * [✓] FIX #2: LLC-Summe über alle Sockets (getNumSockets × getLLCPerSocket)
 * [✓] FIX #3: Fehlende/schlechte Energie → leere Felder + notes markers
 * [✓] FIX #4: batches=1 (wie reduction), repeat_idx in notes
 * [✓] Nice-to-have: pJ/Byte und GB/s/W in notes (größenunabhängig)
 * [✓] DRAM-Regime Check mit korrekter LLC-Summe
 * [✓] Common/cpu_extra Cohorts
 * [✓] Parallele First-Touch Initialisierung
 * [✓] RAPL Overflow-Guard pro Zone
 * [✓] CSV-Schema exakt wie reduction
 * [✓] Timing: steady_clock um genau das Kernel-Fenster
 * [✓] OMP-Settings in notes dokumentiert
 */