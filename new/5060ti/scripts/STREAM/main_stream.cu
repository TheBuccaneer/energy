#include <cuda_runtime.h>
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
constexpr int BLOCK_SIZE = 256;
constexpr float SCALAR = 3.0f;
constexpr size_t VRAM_MIN_SAFETY_MARGIN_BYTES =
    static_cast<size_t>(512) * 1024 * 1024;

const std::vector<size_t> SIZES{
    1000000, 2000000, 4000000, 8000000, 16000000,
    32000000, 64000000, 128000000, 256000000
};

#define CUDA_CHECK(call) do { \
    const cudaError_t status__ = (call); \
    if (status__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA failure: ") + cudaGetErrorString(status__) + \
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
    size_t problem_size{};
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

__host__ __device__ inline float value_b(size_t index) {
    return 1.0f + static_cast<float>(index % 17) * 0.01f;
}

__host__ __device__ inline float value_c(size_t index) {
    return 0.5f + static_cast<float>(index % 13) * 0.02f;
}

__global__ void initialize_stream_vectors(
    float* a,
    float* b,
    float* c,
    size_t n
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) return;

    a[index] = 0.0f;
    b[index] = value_b(index);
    c[index] = value_c(index);
}

__global__ void stream_triad_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    size_t n
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) {
        a[index] = b[index] + SCALAR * c[index];
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
    options.output_file = argc > 1 ? argv[1] : "stream_gpu.csv";
    if (argc > 2) options.repetitions = std::max(1, std::stoi(argv[2]));
    if (argc > 3) options.session_id = argv[3];
    if (argc > 4) options.seed = static_cast<uint32_t>(std::stoul(argv[4]));
    return options;
}

std::vector<size_t> parse_size_filter(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return {};

    std::vector<size_t> values;
    std::stringstream stream(raw);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            values.push_back(static_cast<size_t>(std::stoull(token)));
        }
    }
    return values;
}

bool selected(size_t value, const std::vector<size_t>& filter) {
    return filter.empty()
        || std::find(filter.begin(), filter.end(), value) != filter.end();
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
        std::ceil(
            TARGET_RUNTIME_S * static_cast<double>(current) / safe_seconds
        )
    );
    const long long next = std::max<long long>(
        current + 1,
        std::min<long long>(estimate, static_cast<long long>(current) * 10)
    );
    return static_cast<int>(
        std::min<long long>(MAX_BATCHES, next)
    );
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
        nvmlDeviceGetCurrentClocksThrottleReasons(
            device,
            &telemetry.throttle_reasons
        );
    if (throttle_status != NVML_SUCCESS) {
        telemetry.throttle_reasons = 0;
    }
    return telemetry;
}

unsigned long long read_energy_mj(nvmlDevice_t device) {
    unsigned long long energy = 0;
    const nvmlReturn_t status =
        nvmlDeviceGetTotalEnergyConsumption(device, &energy);

    if (status == NVML_ERROR_NOT_SUPPORTED) {
        throw std::runtime_error(
            "NVML total-energy counter is not supported on this GPU. "
            "Do not mix this run with a power-sampling fallback."
        );
    }
    if (status != NVML_SUCCESS) {
        throw std::runtime_error(
            std::string("NVML energy read failed: ")
            + nvmlErrorString(status)
        );
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
    CUDA_CHECK(
        cudaDeviceGetPCIBusId(
            pci_bus_id,
            static_cast<int>(sizeof(pci_bus_id)),
            CUDA_DEVICE
        )
    );

    nvmlDevice_t device{};
    NVML_CHECK(nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &device));
    return device;
}

void require_expected_gpu(const std::string& name) {
    const char* expected = std::getenv("BENCH_EXPECTED_GPU");
    if (!expected || !*expected) return;

    if (name.find(expected) == std::string::npos) {
        throw std::runtime_error(
            "Selected GPU mismatch: expected name containing '"
            + std::string(expected)
            + "', found '"
            + name
            + "'"
        );
    }
}

unsigned int grid_blocks(size_t n, const cudaDeviceProp& properties) {
    if (n > std::numeric_limits<size_t>::max() - (BLOCK_SIZE - 1)) {
        throw std::runtime_error(
            "Grid-size arithmetic overflow for N=" + std::to_string(n)
        );
    }

    const size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks == 0) {
        throw std::runtime_error("Grid contains zero blocks");
    }
    if (blocks > static_cast<size_t>(properties.maxGridSize[0])) {
        throw std::runtime_error(
            "Grid exceeds CUDA device limit for N=" + std::to_string(n)
            + ": need " + std::to_string(blocks)
            + " blocks, limit "
            + std::to_string(properties.maxGridSize[0])
        );
    }
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error(
            "Grid exceeds unsigned-int launch range for N="
            + std::to_string(n)
        );
    }
    return static_cast<unsigned int>(blocks);
}

void enqueue_stream_triad(
    float* a,
    const float* b,
    const float* c,
    size_t n,
    unsigned int blocks,
    int batches,
    cudaStream_t stream
) {
    // One operation is one complete N-element Triad pass. Therefore
    // batches is exactly the number of full-array operations in the window.
    for (int batch = 0; batch < batches; ++batch) {
        stream_triad_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

double measure_kernel_seconds(
    float* a,
    const float* b,
    const float* c,
    size_t n,
    unsigned int blocks,
    int batches,
    cudaStream_t stream,
    cudaEvent_t start,
    cudaEvent_t stop
) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    enqueue_stream_triad(a, b, c, n, blocks, batches, stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return static_cast<double>(milliseconds) / 1000.0;
}

int calibrate(
    float* a,
    const float* b,
    const float* c,
    size_t n,
    unsigned int blocks,
    cudaStream_t stream,
    cudaEvent_t start,
    cudaEvent_t stop
) {
    enqueue_stream_triad(a, b, c, n, blocks, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int batches = 1;
    for (int step = 0; step < MAX_CALIBRATION_STEPS; ++step) {
        const double seconds = measure_kernel_seconds(
            a,
            b,
            c,
            n,
            blocks,
            batches,
            stream,
            start,
            stop
        );
        if (seconds >= TARGET_RUNTIME_S || batches == MAX_BATCHES) {
            return batches;
        }
        batches = scale_batches(seconds, batches);
    }
    return batches;
}

float read_device_value(const float* device_data, size_t index) {
    float value = 0.0f;
    CUDA_CHECK(
        cudaMemcpy(
            &value,
            device_data + index,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );
    return value;
}

bool correct(const float* device_a, size_t n) {
    const std::vector<size_t> samples{0, n / 7, n / 2, n - 1};

    for (const size_t index : samples) {
        const double actual =
            static_cast<double>(read_device_value(device_a, index));
        const double expected =
            static_cast<double>(value_b(index))
            + static_cast<double>(SCALAR)
            * static_cast<double>(value_c(index));

        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            return false;
        }

        const double relative =
            std::abs(actual - expected)
            / std::max(1.0, std::abs(expected));

        if (!std::isfinite(relative) || relative > 1.0e-6) {
            return false;
        }
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
    output << SCHEMA_VERSION << ',' << timestamp() << ','
           << csv_escape(row.session_id) << ','
           << row.sequence_index << ',' << row.sequence_index << ','
           << row.repetition << ','
           << csv_escape("STREAM") << ','
           << csv_escape("cuda_stream_triad_fp32") << ','
           << "gpu_resident" << ','
           << csv_escape(row.device_name) << ',' << -1 << ','
           << row.problem_size << ','
           << csv_escape("elements=" + std::to_string(row.problem_size)) << ','
           << row.batches << ','
           << std::fixed << std::setprecision(6)
           << row.e2e_time_s << ',' << row.kernel_time_s << ','
           << row.wall_time_s << ','
           << row.device_energy_j << ',' << row.device_energy_j << ','
           << -1.0 << ','
           << std::scientific << std::setprecision(9)
           << row.energy_per_op_j << ','
           << row.energy_per_second_j << ','
           << row.energy_per_flop_j << ','
           << row.time_per_op_ms_kernel << ','
           << row.time_per_op_ms_e2e << ','
           << row.flops_total << ','
           << std::fixed << std::setprecision(6)
           << row.gflops_per_s << ','
           << row.logical_bytes_per_op << ','
           << row.avg_power_w << ','
           << row.runtime_status << ','
           << row.pcie_gen << ',' << row.pcie_width << ','
           << row.sm_clock_mhz << ','
           << row.clock_before_mhz << ','
           << row.clock_after_mhz << ','
           << row.mem_clock_mhz << ','
           << row.temp_c << ','
           << row.temp_before_c << ','
           << row.temp_after_c << ','
           << csv_escape(row.throttle_reasons) << ','
           << -1 << ',' << -1 << ',' << -1.0 << ',' << -1 << ','
           << (row.checksum_ok ? 't' : 'f') << '\n';
}

void print_result(const ResultRow& row) {
    const double logical_bandwidth_gb_s =
        row.logical_bytes_per_op
        * static_cast<double>(row.batches)
        / row.kernel_time_s
        / 1.0e9;

    std::cout << "[STREAM] N=" << row.problem_size
              << " rep=" << row.repetition
              << " batches=" << row.batches
              << " e2e=" << std::fixed << std::setprecision(3)
              << row.e2e_time_s << " s"
              << " | kernel=" << row.kernel_time_s << " s"
              << " | logical_BW=" << std::setprecision(1)
              << logical_bandwidth_gb_s << " GB/s"
              << " | energy=" << std::setprecision(3)
              << row.device_energy_j << " J"
              << " | power=" << std::setprecision(1)
              << row.avg_power_w << " W"
              << " | temp=" << row.temp_c << " C"
              << " | runtime=" << row.runtime_status
              << " | checksum="
              << (row.checksum_ok ? "OK" : "FAIL") << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    cudaStream_t stream{};
    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    bool nvml_initialized = false;

    try {
        const Options options = parse_options(argc, argv);
        const auto parent =
            std::filesystem::path(options.output_file).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        CUDA_CHECK(cudaSetDevice(CUDA_DEVICE));

        cudaDeviceProp device_properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&device_properties, CUDA_DEVICE));

        NVML_CHECK(nvmlInit_v2());
        nvml_initialized = true;
        nvmlDevice_t nvml_device = nvml_handle_for_cuda_device();
        const std::string device_name = gpu_name(nvml_device);
        require_expected_gpu(device_name);

        // Fail before creating a campaign file if the direct cumulative
        // energy counter is unavailable.
        (void)read_energy_mj(nvml_device);

        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));

        int cuda_runtime_version = 0;
        int cuda_driver_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));
        CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));

        std::cout << "STREAM | " << device_name
                  << " | session=" << options.session_id
                  << " | reps=" << options.repetitions
                  << " | mode=gpu_resident"
                  << " | scalar=" << SCALAR
                  << " | block_size=" << BLOCK_SIZE
                  << " | CUDA runtime=" << cuda_runtime_version
                  << " | CUDA driver=" << cuda_driver_version
                  << '\n';

        const std::vector<size_t> size_filter =
            parse_size_filter("BENCH_SIZE_FILTER");
        std::vector<size_t> sizes;
        for (const size_t n : SIZES) {
            if (selected(n, size_filter)) sizes.push_back(n);
        }
        if (sizes.empty()) {
            throw std::runtime_error(
                "No sizes remain after BENCH_SIZE_FILTER"
            );
        }

        std::mt19937 generator(options.seed);
        std::shuffle(sizes.begin(), sizes.end(), generator);

        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "Cannot open output file: " + options.output_file
            );
        }
        write_header(output);

        int sequence = 0;
        for (const size_t n : sizes) {
            if (n == 0) {
                throw std::runtime_error("STREAM size must be positive");
            }
            if (n > std::numeric_limits<size_t>::max() / sizeof(float)) {
                throw std::runtime_error(
                    "Array byte-size overflow for N=" + std::to_string(n)
                );
            }
            const size_t bytes = n * sizeof(float);

            if (bytes > std::numeric_limits<size_t>::max() / 3) {
                throw std::runtime_error(
                    "Three-array byte-size overflow for N="
                    + std::to_string(n)
                );
            }
            const size_t required_bytes = 3 * bytes;

            size_t free_bytes = 0;
            size_t total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const size_t safety_margin =
                std::max(VRAM_MIN_SAFETY_MARGIN_BYTES, free_bytes / 10);

            if (
                free_bytes < safety_margin
                || required_bytes > free_bytes - safety_margin
            ) {
                const double gib = 1024.0 * 1024.0 * 1024.0;
                throw std::runtime_error(
                    "Insufficient free GPU memory for STREAM N="
                    + std::to_string(n)
                    + ": arrays require "
                    + std::to_string(required_bytes / gib)
                    + " GiB, safety margin "
                    + std::to_string(safety_margin / gib)
                    + " GiB, free "
                    + std::to_string(free_bytes / gib)
                    + " GiB"
                );
            }

            const unsigned int blocks =
                grid_blocks(n, device_properties);

            float* a = nullptr;
            float* b = nullptr;
            float* c = nullptr;

            // Best-effort cleanup for every exception path. This must never
            // throw, otherwise cleanup could hide the original failure.
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

                initialize_stream_vectors<<<
                    blocks,
                    BLOCK_SIZE,
                    0,
                    stream
                >>>(a, b, c, n);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));

                const int batches = calibrate(
                    a,
                    b,
                    c,
                    n,
                    blocks,
                    stream,
                    start_event,
                    stop_event
                );

                std::cout << "[CALIBRATION] N=" << n
                          << " batches=" << batches << '\n';

                for (
                    int repetition = 1;
                    repetition <= options.repetitions;
                    ++repetition
                ) {
                    const Telemetry before = read_telemetry(nvml_device);
                    const unsigned long long energy_before =
                        read_energy_mj(nvml_device);
                    const auto wall_start =
                        std::chrono::steady_clock::now();

                    CUDA_CHECK(cudaEventRecord(start_event, stream));
                    enqueue_stream_triad(
                        a,
                        b,
                        c,
                        n,
                        blocks,
                        batches,
                        stream
                    );
                    CUDA_CHECK(cudaEventRecord(stop_event, stream));
                    CUDA_CHECK(cudaEventSynchronize(stop_event));

                    const auto wall_end =
                        std::chrono::steady_clock::now();
                    const unsigned long long energy_after =
                        read_energy_mj(nvml_device);
                    const Telemetry after = read_telemetry(nvml_device);

                    if (energy_after < energy_before) {
                        throw std::runtime_error(
                            "NVML total-energy counter moved backwards"
                        );
                    }

                    float kernel_ms = 0.0f;
                    CUDA_CHECK(
                        cudaEventElapsedTime(
                            &kernel_ms,
                            start_event,
                            stop_event
                        )
                    );

                    const double kernel_seconds =
                        static_cast<double>(kernel_ms) / 1000.0;
                    const double wall_seconds =
                        std::chrono::duration<double>(
                            wall_end - wall_start
                        ).count();
                    const double energy_j =
                        static_cast<double>(energy_after - energy_before)
                        / 1000.0;

                    if (
                        !std::isfinite(kernel_seconds)
                        || !std::isfinite(wall_seconds)
                        || !std::isfinite(energy_j)
                        || kernel_seconds <= 0.0
                        || wall_seconds <= 0.0
                        || energy_j <= 0.0
                    ) {
                        throw std::runtime_error(
                            "Non-positive or non-finite measurement for N="
                            + std::to_string(n)
                        );
                    }

                    const bool checksum_ok = correct(a, n);

                    // One operation is one complete N-element Triad pass,
                    // exactly matching the CPU STREAM batch definition.
                    const double flops_per_op =
                        2.0 * static_cast<double>(n);
                    const double flops_total =
                        flops_per_op * static_cast<double>(batches);
                    const double logical_bytes_per_op =
                        3.0
                        * static_cast<double>(n)
                        * sizeof(float);

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
                    row.energy_per_op_j =
                        energy_j / static_cast<double>(batches);
                    row.energy_per_second_j = energy_j / wall_seconds;
                    row.energy_per_flop_j = energy_j / flops_total;
                    row.time_per_op_ms_kernel =
                        1000.0 * kernel_seconds
                        / static_cast<double>(batches);
                    row.time_per_op_ms_e2e =
                        1000.0 * wall_seconds
                        / static_cast<double>(batches);
                    row.flops_total = flops_total;
                    row.gflops_per_s =
                        flops_total / kernel_seconds / 1.0e9;
                    row.logical_bytes_per_op = logical_bytes_per_op;
                    row.avg_power_w = energy_j / wall_seconds;
                    row.runtime_status = runtime_status(wall_seconds);
                    row.pcie_gen = static_cast<int>(after.pcie_gen);
                    row.pcie_width = static_cast<int>(after.pcie_width);
                    row.clock_before_mhz =
                        static_cast<int>(before.sm_clock_mhz);
                    row.clock_after_mhz =
                        static_cast<int>(after.sm_clock_mhz);
                    row.sm_clock_mhz = static_cast<int>(
                        (before.sm_clock_mhz + after.sm_clock_mhz) / 2
                    );
                    row.mem_clock_mhz = static_cast<int>(
                        (before.mem_clock_mhz + after.mem_clock_mhz) / 2
                    );
                    row.temp_before_c = static_cast<int>(before.temp_c);
                    row.temp_after_c = static_cast<int>(after.temp_c);
                    row.temp_c = static_cast<int>(
                        std::max(before.temp_c, after.temp_c)
                    );
                    row.throttle_reasons = throttle_hex(
                        before.throttle_reasons
                        | after.throttle_reasons
                    );
                    row.checksum_ok = checksum_ok;

                    write_row(output, row);
                    output.flush();
                    print_result(row);

                    if (!checksum_ok) {
                        throw std::runtime_error(
                            "Checksum failed for N=" + std::to_string(n)
                        );
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
        NVML_CHECK(nvmlShutdown());
        nvml_initialized = false;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
        if (stream) cudaStreamDestroy(stream);
        if (nvml_initialized) nvmlShutdown();
        return 2;
    }
}
