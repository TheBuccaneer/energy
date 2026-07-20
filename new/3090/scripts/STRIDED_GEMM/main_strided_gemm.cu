#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* SCHEMA_VERSION = "cpu-gpu-v2";
constexpr double TARGET_RUNTIME_S = 1.0;
constexpr double MIN_RUNTIME_S = 0.75;
constexpr double MAX_RUNTIME_S = 1.25;
constexpr int DEFAULT_REPETITIONS = 10;
constexpr int MAX_CALIBRATION_STEPS = 14;
constexpr int MAX_BATCHES = 10000000;
constexpr int CUDA_DEVICE = 0;  // CUDA_VISIBLE_DEVICES maps the selected physical GPU here.
constexpr float PADDING_A = 0.0f;
constexpr float PADDING_B = 0.0f;
constexpr float PADDING_C = -12345.0f;
constexpr size_t VRAM_MIN_SAFETY_MARGIN_BYTES = static_cast<size_t>(512) * 1024 * 1024;

const std::vector<int> SIZES{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

#define CUDA_CHECK(call) do { \
    const cudaError_t status__ = (call); \
    if (status__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA failure: ") + cudaGetErrorString(status__) + \
                                 " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    const cublasStatus_t status__ = (call); \
    if (status__ != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuBLAS failure code ") + std::to_string(static_cast<int>(status__)) + \
                                 " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while (0)

#define NVML_CHECK(call) do { \
    const nvmlReturn_t status__ = (call); \
    if (status__ != NVML_SUCCESS) { \
        throw std::runtime_error(std::string("NVML failure: ") + nvmlErrorString(status__) + \
                                 " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while (0)

struct Options {
    std::string output_file;
    int repetitions{DEFAULT_REPETITIONS};
    std::string session_id{"manual"};
    uint32_t seed{1};
};

struct Telemetry {
    unsigned int pcie_gen{0};
    unsigned int pcie_width{0};
    unsigned int sm_clock_mhz{0};
    unsigned int mem_clock_mhz{0};
    unsigned int temp_c{0};
    unsigned long long throttle_reasons{0};
};

struct ResultRow {
    std::string session_id;
    int sequence_index{};
    int repetition{};
    std::string device_name;
    int problem_size{};
    int batches{};
    double e2e_time_s{};
    double kernel_time_s{};
    double wall_time_s{};
    double device_energy_j{};
    double energy_per_op_j{};
    double energy_per_second_j{};
    double energy_per_flop_j{};
    double time_per_op_ms_kernel{};
    double time_per_op_ms_e2e{};
    double flops_total{};
    double gflops_per_s{};
    double logical_bytes_per_op{};
    double avg_power_w{};
    std::string runtime_status;
    int pcie_gen{-1};
    int pcie_width{-1};
    int sm_clock_mhz{-1};
    int clock_before_mhz{-1};
    int clock_after_mhz{-1};
    int mem_clock_mhz{-1};
    int temp_c{-1};
    int temp_before_c{-1};
    int temp_after_c{-1};
    std::string throttle_reasons;
    bool checksum_ok{false};
};

__host__ __device__ inline float value_a(int row, int col) {
    return 0.5f + static_cast<float>((row * 3 + col * 5) % 17) * 0.03125f;
}

__host__ __device__ inline float value_b(int row, int col) {
    return 0.25f + static_cast<float>((row * 7 + col * 11) % 19) * 0.0234375f;
}

__global__ void initialize_matrices(float* a, float* b, float* c, int n, int ld) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(ld);
    if (index >= count) return;

    const int row = static_cast<int>(index / static_cast<size_t>(ld));
    const int col = static_cast<int>(index % static_cast<size_t>(ld));
    if (col < n) {
        a[index] = value_a(row, col);
        b[index] = value_b(row, col);
        c[index] = 0.0f;
    } else {
        a[index] = PADDING_A;
        b[index] = PADDING_B;
        c[index] = PADDING_C;
    }
}

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t raw = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_r(&raw, &local);
    std::ostringstream out;
    out << std::put_time(&local, "%Y-%m-%dT%H:%M:%S");
    return out.str();
}

std::string csv_escape(const std::string& value) {
    std::string escaped = value;
    size_t pos = 0;
    while ((pos = escaped.find('"', pos)) != std::string::npos) {
        escaped.insert(pos, 1, '"');
        pos += 2;
    }
    return '"' + escaped + '"';
}

Options parse_options(int argc, char** argv) {
    Options options;
    options.output_file = argc > 1 ? argv[1] : "strided_gemm_gpu.csv";
    if (argc > 2) options.repetitions = std::max(1, std::stoi(argv[2]));
    if (argc > 3) options.session_id = argv[3];
    if (argc > 4) options.seed = static_cast<uint32_t>(std::stoul(argv[4]));
    return options;
}

std::vector<int> parse_filter(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return {};
    std::vector<int> values;
    std::stringstream stream(raw);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) values.push_back(std::stoi(token));
    }
    return values;
}

bool selected(int value, const std::vector<int>& filter) {
    return filter.empty() || std::find(filter.begin(), filter.end(), value) != filter.end();
}

std::string runtime_status(double seconds) {
    if (seconds < MIN_RUNTIME_S) return "below";
    if (seconds > MAX_RUNTIME_S) return "above";
    return "in_range";
}

int scale_batches(double measured_seconds, int current) {
    if (measured_seconds >= TARGET_RUNTIME_S) return current;
    const double safe_seconds = std::max(measured_seconds, 1.0e-9);
    const long long estimate = static_cast<long long>(
        std::ceil(TARGET_RUNTIME_S * static_cast<double>(current) / safe_seconds));
    const long long next = std::max<long long>(
        current + 1,
        std::min<long long>(estimate, static_cast<long long>(current) * 10));
    return static_cast<int>(std::min<long long>(MAX_BATCHES, next));
}

std::string throttle_hex(unsigned long long reasons) {
    std::ostringstream out;
    out << "0x" << std::hex << std::uppercase << reasons;
    return out.str();
}

Telemetry read_telemetry(nvmlDevice_t device) {
    Telemetry telemetry;
    NVML_CHECK(nvmlDeviceGetCurrPcieLinkGeneration(device, &telemetry.pcie_gen));
    NVML_CHECK(nvmlDeviceGetCurrPcieLinkWidth(device, &telemetry.pcie_width));
    NVML_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &telemetry.sm_clock_mhz));
    NVML_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &telemetry.mem_clock_mhz));
    NVML_CHECK(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &telemetry.temp_c));
    const nvmlReturn_t throttle_status =
        nvmlDeviceGetCurrentClocksThrottleReasons(device, &telemetry.throttle_reasons);
    if (throttle_status != NVML_SUCCESS) telemetry.throttle_reasons = 0;
    return telemetry;
}

unsigned long long read_energy_mj(nvmlDevice_t device) {
    unsigned long long energy = 0;
    const nvmlReturn_t status = nvmlDeviceGetTotalEnergyConsumption(device, &energy);
    if (status == NVML_ERROR_NOT_SUPPORTED) {
        throw std::runtime_error(
            "NVML total-energy counter is not supported on this GPU. "
            "Do not mix this run with a power-sampling fallback without redesigning both GPU pipelines.");
    }
    if (status != NVML_SUCCESS) {
        throw std::runtime_error(std::string("NVML energy read failed: ") + nvmlErrorString(status));
    }
    return energy;
}

std::string gpu_name(nvmlDevice_t device) {
    char buffer[NVML_DEVICE_NAME_BUFFER_SIZE]{};
    NVML_CHECK(nvmlDeviceGetName(device, buffer, sizeof(buffer)));
    return buffer;
}

nvmlDevice_t nvml_handle_for_cuda_device() {
    char pci_bus_id[32]{};
    CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), CUDA_DEVICE));
    nvmlDevice_t device{};
    NVML_CHECK(nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &device));
    return device;
}

void require_expected_gpu(const std::string& name) {
    const char* expected = std::getenv("BENCH_EXPECTED_GPU");
    if (!expected || !*expected) return;
    if (name.find(expected) == std::string::npos) {
        throw std::runtime_error(
            "Selected GPU mismatch: expected name containing '" + std::string(expected) +
            "', found '" + name + "'");
    }
}

void gemm(cublasHandle_t handle, const float* a, const float* b, float* c, int n, int ld) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major. Row-major A and B are seen as A^T and B^T.
    // Computing B_col * A_col stores (A_row * B_row)^T in column-major memory,
    // which has the same byte layout as row-major C = A * B.
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        b, CUDA_R_32F, ld,
        a, CUDA_R_32F, ld,
        &beta,
        c, CUDA_R_32F, ld,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT));
}

double measure_kernel_seconds(cublasHandle_t handle, const float* a, const float* b,
                              float* c, int n, int ld, int batches, cudaStream_t stream,
                              cudaEvent_t start, cudaEvent_t stop) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int batch = 0; batch < batches; ++batch) gemm(handle, a, b, c, n, ld);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return static_cast<double>(milliseconds) / 1000.0;
}

int calibrate(cublasHandle_t handle, const float* a, const float* b, float* c,
              int n, int ld, cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
    gemm(handle, a, b, c, n, ld);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int batches = 1;
    for (int step = 0; step < MAX_CALIBRATION_STEPS; ++step) {
        const double seconds =
            measure_kernel_seconds(handle, a, b, c, n, ld, batches, stream, start, stop);
        if (seconds >= TARGET_RUNTIME_S || batches == MAX_BATCHES) return batches;
        batches = scale_batches(seconds, batches);
    }
    return batches;
}

double expected_value(int row, int col, int n) {
    long double sum = 0.0L;
    for (int k = 0; k < n; ++k) {
        sum += static_cast<long double>(value_a(row, k)) * value_b(k, col);
    }
    return static_cast<double>(sum);
}

float read_device_value(const float* device_data, size_t index) {
    float value = 0.0f;
    CUDA_CHECK(cudaMemcpy(&value, device_data + index, sizeof(float), cudaMemcpyDeviceToHost));
    return value;
}

bool correct(const float* device_c, int n, int ld) {
    const std::vector<std::pair<int, int>> samples{
        {0, 0}, {0, n - 1}, {n / 3, n / 2},
        {n / 2, n / 3}, {n - 1, 0}, {n - 1, n - 1}
    };
    for (const auto& [row, col] : samples) {
        const size_t index = static_cast<size_t>(row) * static_cast<size_t>(ld) + col;
        const float actual = read_device_value(device_c, index);
        const double expected = expected_value(row, col, n);
        const double relative =
            std::abs(static_cast<double>(actual) - expected) / std::max(1.0, std::abs(expected));
        if (relative > 2.0e-3) return false;
    }

    const std::vector<std::pair<int, int>> padding_samples{
        {0, n}, {0, ld - 1},
        {n / 2, n}, {n / 2, ld - 1},
        {n - 1, n}, {n - 1, ld - 1}
    };
    for (const auto& [row, col] : padding_samples) {
        const size_t index = static_cast<size_t>(row) * static_cast<size_t>(ld) + col;
        if (read_device_value(device_c, index) != PADDING_C) return false;
    }
    return true;
}

void write_header(std::ofstream& output) {
    output
        << "schema_version,timestamp,session_id,sequence_index,run_id_global,repetition,"
           "workload,implementation,execution_mode,device_name,num_threads,problem_size,problem_spec,batches,"
           "e2e_time_s,kernel_time_s,wall_time_s,device_energy_j,total_energy_j,dram_energy_j,"
           "energy_per_op_j,energy_per_second_j,energy_per_flop_j,"
           "time_per_op_ms_kernel,time_per_op_ms_e2e,flops_total,gflops_per_s,"
           "logical_bytes_per_op,avg_power_w,runtime_status,pcie_gen,pcie_width,"
           "sm_clock_mhz,clock_before_mhz,clock_after_mhz,mem_clock_mhz,temp_c,"
           "temp_before_c,temp_after_c,throttle_reasons,cpu_cycles,"
           "cpu_instructions,cpu_ipc,cpu_cache_misses,checksum_ok\n";
}

void write_row(std::ofstream& output, const ResultRow& row) {
    output << SCHEMA_VERSION << ',' << timestamp() << ',' << csv_escape(row.session_id) << ','
           << row.sequence_index << ',' << row.sequence_index << ',' << row.repetition << ','
           << csv_escape("STRIDED_GEMM") << ','
           << csv_escape("cublas_gemm_ex_fp32_pedantic_ld2n") << ','
           << "gpu_resident" << ',' << csv_escape(row.device_name) << ',' << -1 << ','
           << row.problem_size << ','
           << csv_escape("N=" + std::to_string(row.problem_size) +
                         ";ld=" + std::to_string(2 * row.problem_size)) << ','
           << row.batches << ','
           << std::fixed << std::setprecision(6)
           << row.e2e_time_s << ',' << row.kernel_time_s << ',' << row.wall_time_s << ','
           << row.device_energy_j << ',' << row.device_energy_j << ',' << -1.0 << ','
           << std::scientific << std::setprecision(9)
           << row.energy_per_op_j << ',' << row.energy_per_second_j << ',' << row.energy_per_flop_j << ','
           << row.time_per_op_ms_kernel << ',' << row.time_per_op_ms_e2e << ','
           << row.flops_total << ','
           << std::fixed << std::setprecision(6)
           << row.gflops_per_s << ',' << row.logical_bytes_per_op << ',' << row.avg_power_w << ','
           << row.runtime_status << ',' << row.pcie_gen << ',' << row.pcie_width << ','
           << row.sm_clock_mhz << ',' << row.clock_before_mhz << ',' << row.clock_after_mhz << ','
           << row.mem_clock_mhz << ',' << row.temp_c << ',' << row.temp_before_c << ','
           << row.temp_after_c << ',' << csv_escape(row.throttle_reasons) << ','
           << -1 << ',' << -1 << ',' << -1.0 << ',' << -1 << ','
           << (row.checksum_ok ? 't' : 'f') << '\n';
}

void print_result(const ResultRow& row) {
    std::cout << "[STRIDED_GEMM] N=" << row.problem_size
              << " ld=" << 2 * row.problem_size
              << " rep=" << row.repetition
              << " batches=" << row.batches
              << " e2e=" << std::fixed << std::setprecision(3) << row.e2e_time_s << " s"
              << " | kernel=" << row.kernel_time_s << " s"
              << " | energy=" << row.device_energy_j << " J"
              << " | power=" << std::setprecision(1) << row.avg_power_w << " W"
              << " | temp=" << row.temp_c << " C"
              << " | runtime=" << row.runtime_status
              << " | checksum=" << (row.checksum_ok ? "OK" : "FAIL") << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    cublasHandle_t handle{};
    cudaStream_t stream{};
    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    bool nvml_initialized = false;

    try {
        const Options options = parse_options(argc, argv);
        const auto parent = std::filesystem::path(options.output_file).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);

        CUDA_CHECK(cudaSetDevice(CUDA_DEVICE));
        NVML_CHECK(nvmlInit_v2());
        nvml_initialized = true;
        nvmlDevice_t nvml_device = nvml_handle_for_cuda_device();
        const std::string device_name = gpu_name(nvml_device);
        require_expected_gpu(device_name);

        // Fail before creating a campaign file if the direct cumulative energy counter is unavailable.
        (void)read_energy_mj(nvml_device);

        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));

        int cuda_runtime_version = 0;
        int cuda_driver_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));
        CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));

        std::cout << "STRIDED_GEMM(ld=2N) | " << device_name
                  << " | session=" << options.session_id
                  << " | reps=" << options.repetitions
                  << " | mode=gpu_resident"
                  << " | compute=CUBLAS_COMPUTE_32F_PEDANTIC"
                  << " | CUDA runtime=" << cuda_runtime_version
                  << " | CUDA driver=" << cuda_driver_version << '\n';

        const std::vector<int> size_filter = parse_filter("BENCH_SIZE_FILTER");
        std::vector<int> sizes;
        for (int n : SIZES) {
            if (selected(n, size_filter)) sizes.push_back(n);
        }
        if (sizes.empty()) throw std::runtime_error("No sizes remain after BENCH_SIZE_FILTER");
        std::mt19937 generator(options.seed);
        std::shuffle(sizes.begin(), sizes.end(), generator);

        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot open output file: " + options.output_file);
        write_header(output);

        int sequence = 0;
        for (int n : sizes) {
            if (n > std::numeric_limits<int>::max() / 2) {
                throw std::runtime_error("Leading-dimension overflow for N=" + std::to_string(n));
            }
            const int ld = 2 * n;

            if (static_cast<size_t>(n) >
                std::numeric_limits<size_t>::max() / static_cast<size_t>(ld)) {
                throw std::runtime_error("Matrix element-count overflow for N=" + std::to_string(n) +
                                         ", ld=" + std::to_string(ld));
            }
            const size_t count = static_cast<size_t>(n) * static_cast<size_t>(ld);
            if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
                throw std::runtime_error("Matrix byte-size overflow for N=" + std::to_string(n) +
                                         ", ld=" + std::to_string(ld));
            }
            const size_t bytes = count * sizeof(float);
            if (bytes > std::numeric_limits<size_t>::max() / 3) {
                throw std::runtime_error("Three-matrix byte-size overflow for N=" + std::to_string(n) +
                                         ", ld=" + std::to_string(ld));
            }
            const size_t required_bytes = 3 * bytes;

            size_t free_bytes = 0;
            size_t total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const size_t safety_margin =
                std::max(VRAM_MIN_SAFETY_MARGIN_BYTES, free_bytes / 10);
            if (free_bytes < safety_margin || required_bytes > free_bytes - safety_margin) {
                const double gib = 1024.0 * 1024.0 * 1024.0;
                throw std::runtime_error(
                    "Insufficient free GPU memory for STRIDED_GEMM N=" + std::to_string(n) +
                    ", ld=" + std::to_string(ld) +
                    ": matrices require " + std::to_string(required_bytes / gib) +
                    " GiB, safety margin " + std::to_string(safety_margin / gib) +
                    " GiB, free " + std::to_string(free_bytes / gib) + " GiB");
            }

            float* a = nullptr;
            float* b = nullptr;
            float* c = nullptr;

            // Best-effort cleanup for all exception paths. This must never throw,
            // otherwise a cleanup error could hide the original failure.
            auto free_buffers = [&]() noexcept {
                if (c) {
                    cudaFree(c);
                    c = nullptr;
                }
                if (b) {
                    cudaFree(b);
                    b = nullptr;
                }
                if (a) {
                    cudaFree(a);
                    a = nullptr;
                }
            };

            try {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a), bytes));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b), bytes));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c), bytes));

                const int threads = 256;
                const int blocks = static_cast<int>((count + threads - 1) / threads);
                initialize_matrices<<<blocks, threads, 0, stream>>>(a, b, c, n, ld);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));

                const int batches = calibrate(handle, a, b, c, n, ld, stream, start_event, stop_event);
                std::cout << "[CALIBRATION] N=" << n << " ld=" << ld
                          << " batches=" << batches << '\n';

                for (int repetition = 1; repetition <= options.repetitions; ++repetition) {
                    const Telemetry before = read_telemetry(nvml_device);
                    const unsigned long long energy_before = read_energy_mj(nvml_device);
                    const auto wall_start = std::chrono::steady_clock::now();
    
                    CUDA_CHECK(cudaEventRecord(start_event, stream));
                    for (int batch = 0; batch < batches; ++batch) gemm(handle, a, b, c, n, ld);
                    CUDA_CHECK(cudaEventRecord(stop_event, stream));
                    CUDA_CHECK(cudaEventSynchronize(stop_event));
    
                    const auto wall_end = std::chrono::steady_clock::now();
                    const unsigned long long energy_after = read_energy_mj(nvml_device);
                    const Telemetry after = read_telemetry(nvml_device);
    
                    if (energy_after < energy_before) {
                        throw std::runtime_error("NVML total-energy counter moved backwards");
                    }
    
                    float kernel_ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_event, stop_event));
                    const double kernel_seconds = static_cast<double>(kernel_ms) / 1000.0;
                    const double wall_seconds =
                        std::chrono::duration<double>(wall_end - wall_start).count();
                    const double energy_j = static_cast<double>(energy_after - energy_before) / 1000.0;
                    const bool checksum_ok = correct(c, n, ld);
    
                    const double flops_per_op = 2.0 * n * static_cast<double>(n) * n;
                    const double flops_total = flops_per_op * batches;
                    const double logical_bytes_per_op =
                        3.0 * n * static_cast<double>(n) * sizeof(float);
    
                    ResultRow row;
                    row.session_id = options.session_id;
                    row.sequence_index = ++sequence;
                    row.repetition = repetition;
                    row.device_name = device_name;
                    row.problem_size = n;
                    row.batches = batches;
                    row.e2e_time_s = wall_seconds;
                    row.kernel_time_s = kernel_seconds;
                    row.wall_time_s = wall_seconds;
                    row.device_energy_j = energy_j;
                    row.energy_per_op_j = energy_j / batches;
                    row.energy_per_second_j = energy_j / wall_seconds;
                    row.energy_per_flop_j = energy_j / flops_total;
                    row.time_per_op_ms_kernel = 1000.0 * kernel_seconds / batches;
                    row.time_per_op_ms_e2e = 1000.0 * wall_seconds / batches;
                    row.flops_total = flops_total;
                    row.gflops_per_s = flops_total / kernel_seconds / 1.0e9;
                    row.logical_bytes_per_op = logical_bytes_per_op;
                    row.avg_power_w = energy_j / wall_seconds;
                    row.runtime_status = runtime_status(wall_seconds);
                    row.pcie_gen = static_cast<int>(after.pcie_gen);
                    row.pcie_width = static_cast<int>(after.pcie_width);
                    row.clock_before_mhz = static_cast<int>(before.sm_clock_mhz);
                    row.clock_after_mhz = static_cast<int>(after.sm_clock_mhz);
                    row.sm_clock_mhz = static_cast<int>((before.sm_clock_mhz + after.sm_clock_mhz) / 2);
                    row.mem_clock_mhz = static_cast<int>((before.mem_clock_mhz + after.mem_clock_mhz) / 2);
                    row.temp_before_c = static_cast<int>(before.temp_c);
                    row.temp_after_c = static_cast<int>(after.temp_c);
                    row.temp_c = static_cast<int>(std::max(before.temp_c, after.temp_c));
                    row.throttle_reasons =
                        throttle_hex(before.throttle_reasons | after.throttle_reasons);
                    row.checksum_ok = checksum_ok;
    
                    write_row(output, row);
                    output.flush();
                    print_result(row);

                    if (!checksum_ok) {
                        throw std::runtime_error("Checksum failed for N=" + std::to_string(n));
                    }
                }

                free_buffers();
            } catch (...) {
                free_buffers();
                throw;
            }
        }

        CUDA_CHECK(cudaEventDestroy(start_event));
        start_event = nullptr;
        CUDA_CHECK(cudaEventDestroy(stop_event));
        stop_event = nullptr;
        CUDA_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;
        CUBLAS_CHECK(cublasDestroy(handle));
        handle = nullptr;
        NVML_CHECK(nvmlShutdown());
        nvml_initialized = false;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
        if (stream) cudaStreamDestroy(stream);
        if (handle) cublasDestroy(handle);
        if (nvml_initialized) nvmlShutdown();
        return 2;
    }
}
