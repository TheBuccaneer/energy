// ============================================================================
// CPU Conv2D Energy Benchmark with Adaptive Batching
// Operation: 2D Convolution (NCHW layout)
// Uses oneDNN (DNNL) for optimized convolution
// ============================================================================
// Compile (with oneDNN): 
//   g++ -O3 -march=native -std=c++17 -fopenmp -o conv2d_cpu main.cpp -ldnnl -lpthread -lm
// Compile (fallback without oneDNN):
//   g++ -O3 -march=native -std=c++17 -fopenmp -DUSE_NAIVE_CONV -o conv2d_cpu main.cpp -lpthread -lm
// Usage: ./conv2d_cpu [--test] [--output <path>]

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

#ifndef USE_NAIVE_CONV
#include <dnnl.hpp>
#endif

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

    ~PerfGroupCounter() { close_fds(); }

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
        if (fd_leader_ < 0) { available_ = false; return false; }

        pe.disabled = 0;
        pe.config = PERF_COUNT_HW_INSTRUCTIONS;
        fd_instructions_ = perf_event_open(&pe, 0, -1, fd_leader_, 0);
        if (fd_instructions_ < 0) { close_fds(); available_ = false; return false; }

        pe.config = PERF_COUNT_HW_CACHE_MISSES;
        fd_cache_misses_ = perf_event_open(&pe, 0, -1, fd_leader_, 0);
        if (fd_cache_misses_ < 0) { close_fds(); available_ = false; return false; }

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
        if (!available_) { cycles = -1; instructions = -1; cache_misses = -1; return; }
        PerfReadFormat data;
        memset(&data, 0, sizeof(data));
        ssize_t ret = ::read(fd_leader_, &data, sizeof(data));
        if (ret != sizeof(data) || data.nr != 3) {
            cycles = -1; instructions = -1; cache_misses = -1; return;
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
    int fd_leader_, fd_instructions_, fd_cache_misses_;
    bool available_;
};

// ============================================================================
// Configuration
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    REPEATS          = 50;
constexpr int    MAX_BATCH_SIZE   = 100000;

static const int THREAD_COUNTS[] = {1, 2, 4, 8, 10, 16, 20, 32, 64};
static const int NUM_THREAD_CONFIGS = sizeof(THREAD_COUNTS) / sizeof(THREAD_COUNTS[0]);

static const char* DEFAULT_OUTPUT_FILE = "conv2d_cpu_amd.csv";

// ============================================================================
// Conv2D Shape Definition
// ============================================================================
// shape_id -> N, C, H, W, K, R, S, stride, pad
// FLOPs per conv = 2 * N * K * C * R * S * H_out * W_out
// H_out = (H + 2*pad - R) / stride + 1

struct Conv2DShape {
    int shape_id;
    int N;      // Batch size
    int C;      // Input channels
    int H;      // Input height
    int W;      // Input width
    int K;      // Output channels (filters)
    int R;      // Kernel height
    int S;      // Kernel width
    int stride;
    int pad;
    int H_out;
    int W_out;
    double flops_per_batch;
    
    void compute_derived() {
        H_out = (H + 2 * pad - R) / stride + 1;
        W_out = (W + 2 * pad - S) / stride + 1;
        // FLOPs = 2 * N * K * C * R * S * H_out * W_out (multiply-add per output element)
        flops_per_batch = 2.0 * N * K * C * R * S * H_out * W_out;
    }
};

// ResNet-like shapes + Stem + Pointwise
static Conv2DShape CONV_SHAPES[] = {
    // shape_id, N,  C,   H,   W,   K,  R, S, stride, pad
    {1,         32, 64,  56,  56,  64,  3, 3, 1,      1, 0, 0, 0},  // ResNet conv3_x style
    {2,         32, 64,  56,  56,  128, 3, 3, 2,      1, 0, 0, 0},  // Downsample
    {3,         32, 128, 28,  28,  256, 3, 3, 2,      1, 0, 0, 0},  // conv4_x entry
    {4,         32, 256, 14,  14,  512, 3, 3, 2,      1, 0, 0, 0},  // conv5_x entry
    {5,         32, 3,   224, 224, 64,  7, 7, 2,      3, 0, 0, 0},  // Stem (large spatial)
    {6,         32, 256, 56,  56,  256, 1, 1, 1,      0, 0, 0, 0},  // Pointwise 1x1
};
static const int NUM_SHAPES = sizeof(CONV_SHAPES) / sizeof(CONV_SHAPES[0]);

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
                size_t amd_pos = model.find("AMD");
                if (amd_pos != std::string::npos) model = model.substr(amd_pos + 4);
                size_t intel_pos = model.find("Intel(R)");
                if (intel_pos != std::string::npos) model = model.substr(intel_pos + 9);
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
    if (file_path.has_parent_path()) fs::create_directories(file_path.parent_path());
}

bool fileExists(const char* filepath) {
    struct stat buffer;
    return (stat(filepath, &buffer) == 0);
}

void initializeBuffer(float* buf, size_t size, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) buf[i] = dist(gen);
}

// ============================================================================
// RAPL Energy Reading
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
            if (name_file.is_open()) std::getline(name_file, zone_name);
            if (zone_name.find("package") != std::string::npos) {
                RAPLZone zone;
                zone.path = zone_path;
                zone.name = zone_name;
                std::ifstream range_file(zone_path + "/max_energy_range_uj");
                if (range_file.is_open()) range_file >> zone.max_energy_range_uj;
                else zone.max_energy_range_uj = 0;
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
        unsigned long long uj = 0;
        if (energy_file.is_open()) energy_file >> uj;
        readings.push_back(uj);
    }
    return readings;
}

double computeTotalEnergyDelta(const std::vector<unsigned long long>& before,
                               const std::vector<unsigned long long>& after,
                               const std::vector<RAPLZone>& zones) {
    if (before.size() != after.size() || before.size() != zones.size()) return -1.0;
    double total_j = 0.0;
    bool any_valid = false;
    for (size_t i = 0; i < zones.size(); i++) {
        unsigned long long b = before[i], a = after[i], r = zones[i].max_energy_range_uj;
        if (b == 0 || a == 0 || r == 0) continue;
        unsigned long long delta_uj = (a >= b) ? (a - b) : (a + r - b);
        total_j += delta_uj / 1e6;
        any_valid = true;
    }
    return any_valid ? total_j : -1.0;
}

int readCPUTemperature() {
    std::vector<std::string> paths = {
        "/sys/class/hwmon/hwmon0/temp1_input",
        "/sys/class/hwmon/hwmon1/temp1_input",
        "/sys/class/hwmon/hwmon2/temp1_input",
        "/sys/class/hwmon/hwmon3/temp1_input"
    };
    for (const auto& path : paths) {
        std::ifstream f(path);
        if (f.is_open()) { int t; f >> t; return t / 1000; }
    }
    return -1;
}

int readCPUFrequencyMHz() {
    std::ifstream f("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    if (f.is_open()) { int khz; f >> khz; return khz / 1000; }
    return -1;
}

// ============================================================================
// Conv2D Implementation
// ============================================================================

#ifdef USE_NAIVE_CONV

// Naive OpenMP implementation (fallback)
void conv2d_naive(const float* input, const float* weights, float* output,
                  int N, int C, int H, int W, int K, int R, int S,
                  int stride, int pad, int H_out, int W_out, int num_threads) {
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float sum = 0.0f;
                    for (int c = 0; c < C; c++) {
                        for (int r = 0; r < R; r++) {
                            for (int s = 0; s < S; s++) {
                                int ih = oh * stride - pad + r;
                                int iw = ow * stride - pad + s;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int input_idx = ((n * C + c) * H + ih) * W + iw;
                                    int weight_idx = ((k * C + c) * R + r) * S + s;
                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }
                    int output_idx = ((n * K + k) * H_out + oh) * W_out + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

float run_batched_conv2d(const float* input, const float* weights, float* output,
                         const Conv2DShape& shape, int batches, int num_threads) {
    for (int b = 0; b < batches; b++) {
        conv2d_naive(input, weights, output,
                     shape.N, shape.C, shape.H, shape.W,
                     shape.K, shape.R, shape.S,
                     shape.stride, shape.pad, shape.H_out, shape.W_out,
                     num_threads);
    }
    
    // Strong DCE prevention: checksum over output
    size_t output_size = (size_t)shape.N * shape.K * shape.H_out * shape.W_out;
    float checksum = 0.0f;
    #pragma omp parallel for reduction(+:checksum) schedule(static)
    for (size_t i = 0; i < output_size; i++) {
        checksum += output[i];
    }
    return checksum;
}

#else

// oneDNN implementation
class Conv2DEngine {
public:
    Conv2DEngine() : engine_(dnnl::engine::kind::cpu, 0), stream_(engine_) {}
    
    void setup(const Conv2DShape& shape, int num_threads) {
        // Thread control via OpenMP (dnnl uses OMP internally)
        omp_set_num_threads(num_threads);
        
        // Memory dimensions (NCHW)
        dnnl::memory::dims src_dims = {shape.N, shape.C, shape.H, shape.W};
        dnnl::memory::dims weights_dims = {shape.K, shape.C, shape.R, shape.S};
        dnnl::memory::dims dst_dims = {shape.N, shape.K, shape.H_out, shape.W_out};
        dnnl::memory::dims strides = {shape.stride, shape.stride};
        dnnl::memory::dims padding = {shape.pad, shape.pad};
        
        // Memory descriptors
        auto src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
        auto weights_md = dnnl::memory::desc(weights_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw);
        auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
        
        // Convolution descriptor
        auto conv_desc = dnnl::convolution_forward::primitive_desc(
            engine_,
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            src_md, weights_md, dst_md,
            strides, padding, padding
        );
        
        conv_prim_ = dnnl::convolution_forward(conv_desc);
        
        // Store sizes for memory allocation
        src_size_ = shape.N * shape.C * shape.H * shape.W;
        weights_size_ = shape.K * shape.C * shape.R * shape.S;
        dst_size_ = shape.N * shape.K * shape.H_out * shape.W_out;
        
        src_md_ = src_md;
        weights_md_ = weights_md;
        dst_md_ = dst_md;
    }
    
    float execute(const float* input, const float* weights, float* output, int batches) {
        auto src_mem = dnnl::memory(src_md_, engine_, (void*)input);
        auto weights_mem = dnnl::memory(weights_md_, engine_, (void*)weights);
        auto dst_mem = dnnl::memory(dst_md_, engine_, output);
        
        for (int b = 0; b < batches; b++) {
            conv_prim_.execute(stream_, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_WEIGHTS, weights_mem},
                {DNNL_ARG_DST, dst_mem}
            });
        }
        stream_.wait();
        
        // Strong DCE prevention
        float checksum = 0.0f;
        #pragma omp parallel for reduction(+:checksum) schedule(static)
        for (size_t i = 0; i < dst_size_; i++) {
            checksum += output[i];
        }
        return checksum;
    }
    
    size_t src_size() const { return src_size_; }
    size_t weights_size() const { return weights_size_; }
    size_t dst_size() const { return dst_size_; }
    
private:
    dnnl::engine engine_;
    dnnl::stream stream_;
    dnnl::convolution_forward conv_prim_;
    dnnl::memory::desc src_md_, weights_md_, dst_md_;
    size_t src_size_, weights_size_, dst_size_;
};

static Conv2DEngine g_conv_engine;

float run_batched_conv2d(const float* input, const float* weights, float* output,
                         const Conv2DShape& shape, int batches, int num_threads) {
    g_conv_engine.setup(shape, num_threads);
    return g_conv_engine.execute(input, weights, output, batches);
}

#endif

// ============================================================================
// Adaptive Batch Size Determination
// ============================================================================

int determine_batch_size(float* input, float* weights, float* output,
                         const Conv2DShape& shape, int num_threads, double target_runtime_s) {
    int batches = 1;
    volatile float sink = 0.0f;

    while (batches <= MAX_BATCH_SIZE) {
        auto start = std::chrono::steady_clock::now();
        sink += run_batched_conv2d(input, weights, output, shape, batches, num_threads);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double measured_time = elapsed.count();

        if (measured_time <= 1e-9) {
            batches = std::min(batches * 2, MAX_BATCH_SIZE);
            continue;
        }
        if (measured_time >= target_runtime_s) return batches;

        double time_per_batch = measured_time / batches;
        int needed_batches = static_cast<int>(std::ceil(target_runtime_s / time_per_batch));
        if (needed_batches <= batches) needed_batches = batches * 2;
        batches = std::min(needed_batches, MAX_BATCH_SIZE);
    }
    return MAX_BATCH_SIZE;
}

// ============================================================================
// Measurement
// ============================================================================

struct MeasurementResult {
    double time_s;
    double energy_j;
    int temp_c;
    int64_t cpu_cycles;
    int64_t cpu_instructions;
    int64_t cpu_cache_misses;
};

MeasurementResult run_single_measurement(float* input, float* weights, float* output,
                                         const Conv2DShape& shape, int batches, int num_threads,
                                         const std::vector<RAPLZone>& rapl_zones,
                                         PerfGroupCounter& perf_counter) {
    MeasurementResult result;
    volatile float sink = 0.0f;

    auto wall_start = std::chrono::steady_clock::now();
    auto energy_before = readRAPLEnergyPerZone(rapl_zones);
    perf_counter.start();

    sink += run_batched_conv2d(input, weights, output, shape, batches, num_threads);

    perf_counter.stop();
    auto energy_after = readRAPLEnergyPerZone(rapl_zones);
    auto wall_end = std::chrono::steady_clock::now();

    perf_counter.read(result.cpu_cycles, result.cpu_instructions, result.cpu_cache_misses);
    std::chrono::duration<double> duration = wall_end - wall_start;
    result.time_s = duration.count();
    result.energy_j = computeTotalEnergyDelta(energy_before, energy_after, rapl_zones);
    result.temp_c = readCPUTemperature();

    return result;
}

// ============================================================================
// CSV Output
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
                 const std::string& device_name, int num_threads, int shape_id,
                 int batches, double time_s, double energy_j, int temp_c,
                 bool below_target, double flops_per_batch,
                 int64_t cpu_cycles, int64_t cpu_instructions, int64_t cpu_cache_misses) {
    
    double flops_total = flops_per_batch * batches;
    double energy_per_batch_j = (batches > 0 && energy_j >= 0) ? (energy_j / batches) : -1.0;
    double energy_per_second_j = (time_s > 0 && energy_j >= 0) ? (energy_j / time_s) : -1.0;
    double energy_per_flop_j = (flops_total > 0 && energy_j >= 0) ? (energy_j / flops_total) : -1.0;
    double time_per_gemm_ms = (batches > 0) ? (1e3 * time_s / batches) : 0.0;
    double gflops_per_s = (time_s > 0) ? (flops_total / time_s / 1e9) : 0.0;
    double avg_power_w = (time_s > 0 && energy_j >= 0) ? (energy_j / time_s) : -1.0;

    double cpu_ipc = -1.0;
    if (cpu_cycles > 0 && cpu_instructions >= 0) {
        cpu_ipc = static_cast<double>(cpu_instructions) / static_cast<double>(cpu_cycles);
    }
    int sm_clock_mhz = readCPUFrequencyMHz();

    file << getTimestamp() << ","
         << run_id_global << ","
         << run_id_per_size << ","
         << device_name << ","
         << num_threads << ","
         << shape_id << ","  // problem_size = shape_id
         << batches << ","
         << std::fixed << std::setprecision(6)
         << time_s << "," << time_s << "," << time_s << ","
         << energy_j << ","
         << std::scientific << std::setprecision(6) << energy_per_batch_j << ","
         << std::fixed << std::setprecision(6) << energy_per_second_j << ","
         << std::scientific << std::setprecision(6) << energy_per_flop_j << ","
         << std::fixed << std::setprecision(6) << time_per_gemm_ms << "," << time_per_gemm_ms << ","
         << std::scientific << std::setprecision(6) << flops_total << ","
         << std::fixed << std::setprecision(2) << gflops_per_s << "," << avg_power_w << ","
         << (below_target ? 't' : 'f') << ",,,";
    file << ((sm_clock_mhz > 0) ? std::to_string(sm_clock_mhz) : "") << ",,";
    file << ((temp_c > 0) ? std::to_string(temp_c) : "") << ",,"
         << cpu_cycles << "," << cpu_instructions << ","
         << std::fixed << std::setprecision(6) << cpu_ipc << "," << cpu_cache_misses << "\n";
}

// ============================================================================
// Main Benchmark
// ============================================================================

void run_benchmark(const char* output_file, bool test_mode) {
    std::string cpu_model = getCPUModel();
    std::cout << "=== CPU Conv2D Benchmark (Adaptive Batching) ===" << std::endl;
    std::cout << "CPU: " << cpu_model << std::endl;
#ifdef USE_NAIVE_CONV
    std::cout << "Backend: Naive OpenMP (fallback)" << std::endl;
#else
    std::cout << "Backend: oneDNN" << std::endl;
#endif
    std::cout << "Target runtime: " << TARGET_RUNTIME_S << "s, Repeats: " << REPEATS << std::endl;

    // Compute derived values for shapes
    for (int i = 0; i < NUM_SHAPES; i++) {
        CONV_SHAPES[i].compute_derived();
        std::cout << "Shape " << CONV_SHAPES[i].shape_id << ": "
                  << "N=" << CONV_SHAPES[i].N << " C=" << CONV_SHAPES[i].C
                  << " H=" << CONV_SHAPES[i].H << " W=" << CONV_SHAPES[i].W
                  << " K=" << CONV_SHAPES[i].K << " R=" << CONV_SHAPES[i].R
                  << " S=" << CONV_SHAPES[i].S << " stride=" << CONV_SHAPES[i].stride
                  << " pad=" << CONV_SHAPES[i].pad
                  << " -> H_out=" << CONV_SHAPES[i].H_out << " W_out=" << CONV_SHAPES[i].W_out
                  << " FLOPs/batch=" << std::scientific << CONV_SHAPES[i].flops_per_batch << std::endl;
    }

    // Find max buffer sizes
    size_t max_input_size = 0, max_weights_size = 0, max_output_size = 0;
    for (int i = 0; i < NUM_SHAPES; i++) {
        size_t in_sz = (size_t)CONV_SHAPES[i].N * CONV_SHAPES[i].C * CONV_SHAPES[i].H * CONV_SHAPES[i].W;
        size_t wt_sz = (size_t)CONV_SHAPES[i].K * CONV_SHAPES[i].C * CONV_SHAPES[i].R * CONV_SHAPES[i].S;
        size_t out_sz = (size_t)CONV_SHAPES[i].N * CONV_SHAPES[i].K * CONV_SHAPES[i].H_out * CONV_SHAPES[i].W_out;
        max_input_size = std::max(max_input_size, in_sz);
        max_weights_size = std::max(max_weights_size, wt_sz);
        max_output_size = std::max(max_output_size, out_sz);
    }

    std::cout << "\nAllocating buffers: input=" << (max_input_size * 4 / 1024 / 1024) << "MB, "
              << "weights=" << (max_weights_size * 4 / 1024 / 1024) << "MB, "
              << "output=" << (max_output_size * 4 / 1024 / 1024) << "MB" << std::endl;

    float *input = nullptr, *weights = nullptr, *output = nullptr;
    if (posix_memalign((void**)&input, 64, max_input_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&weights, 64, max_weights_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&output, 64, max_output_size * sizeof(float)) != 0) {
        std::cerr << "ERROR: Memory allocation failed" << std::endl;
        return;
    }

    initializeBuffer(input, max_input_size, 42);
    initializeBuffer(weights, max_weights_size, 43);
    std::fill(output, output + max_output_size, 0.0f);

    std::vector<RAPLZone> rapl_zones = discoverRAPLZones();
    if (rapl_zones.empty()) {
        std::cerr << "WARNING: RAPL not available. Energy = -1.0\n";
    } else {
        std::cout << "RAPL: Found " << rapl_zones.size() << " package zone(s)" << std::endl;
    }

    ensureDirectoryExists(output_file);
    bool write_header = !fileExists(output_file);
    std::ofstream csv_file(output_file, std::ios::app);
    if (write_header) writeCSVHeader(csv_file);

    std::vector<int> thread_configs, shapes_to_test;
    if (test_mode) {
        thread_configs = {THREAD_COUNTS[0], THREAD_COUNTS[1]};
        shapes_to_test = {CONV_SHAPES[0].shape_id, CONV_SHAPES[1].shape_id};
        std::cout << "\n*** TEST MODE: " << thread_configs.size() << " threads × "
                  << shapes_to_test.size() << " shapes ***" << std::endl;
    } else {
        for (int i = 0; i < NUM_THREAD_CONFIGS; i++) thread_configs.push_back(THREAD_COUNTS[i]);
        for (int i = 0; i < NUM_SHAPES; i++) shapes_to_test.push_back(CONV_SHAPES[i].shape_id);
    }

    PerfGroupCounter perf_counter;
    bool perf_available = perf_counter.open();
    std::cout << "Perf counters: " << (perf_available ? "Available" : "Not available") << std::endl;

    int run_id_global = 0;

    for (int num_threads : thread_configs) {
        omp_set_num_threads(num_threads);
        std::cout << "\n======== Threads: " << num_threads << " ========" << std::endl;

        for (int shape_id : shapes_to_test) {
            Conv2DShape& shape = CONV_SHAPES[shape_id - 1];
            std::cout << "\n--- Shape " << shape_id << " ---" << std::endl;

            int batches = determine_batch_size(input, weights, output, shape, num_threads, TARGET_RUNTIME_S);
            std::cout << "Batch size: " << batches << std::endl;

            int run_id_per_size = 1;
            for (int rep = 0; rep < REPEATS; rep++) {
                run_id_global++;

                MeasurementResult result = run_single_measurement(input, weights, output, shape,
                                                                  batches, num_threads, rapl_zones, perf_counter);
                bool below_target = (result.time_s < TARGET_RUNTIME_S);

                writeCSVRow(csv_file, run_id_global, run_id_per_size, cpu_model, num_threads,
                            shape_id, batches, result.time_s, result.energy_j, result.temp_c,
                            below_target, shape.flops_per_batch,
                            result.cpu_cycles, result.cpu_instructions, result.cpu_cache_misses);
                csv_file.flush();

                double gflops = (result.time_s > 0) ? (shape.flops_per_batch * batches / result.time_s / 1e9) : 0.0;
                char check = below_target ? '!' : '+';
                std::cout << "  " << check << " [" << (rep + 1) << "/" << REPEATS << "] "
                          << std::fixed << std::setprecision(3) << result.time_s << "s, "
                          << std::setprecision(1) << gflops << " GFLOPS";
                if (result.energy_j >= 0) std::cout << ", " << result.energy_j << " J";
                if (result.temp_c > 0) std::cout << ", " << result.temp_c << "°C";
                std::cout << std::endl;

                run_id_per_size++;
            }

            bool is_last = (shape_id == shapes_to_test.back() && num_threads == thread_configs.back());
            if (!is_last) {
                std::cout << "  Cooling down 60s..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(60));
            }
        }
    }

    csv_file.close();
    std::cout << "\n=== Benchmark complete! ===" << std::endl;
    std::cout << "Results: " << output_file << " (" << run_id_global << " rows)" << std::endl;

    free(input); free(weights); free(output);
}

int main(int argc, char** argv) {
    bool test_mode = false;
    const char* output_file = DEFAULT_OUTPUT_FILE;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test" || arg == "-t") test_mode = true;
        else if ((arg == "--output" || arg == "-o") && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--test] [--output <path>]\n";
            return 0;
        }
    }

    std::cout << "Output: " << output_file << ", Test mode: " << (test_mode ? "yes" : "no") << std::endl;
    run_benchmark(output_file, test_mode);
    return 0;
}
