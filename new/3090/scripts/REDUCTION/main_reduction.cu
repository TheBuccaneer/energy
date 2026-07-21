#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr int MAX_BATCHES = 10000000;
constexpr int MAX_CALIBRATION_STEPS = 12;
constexpr double TARGET_RUNTIME_S = 1.0;
constexpr long double MAX_RELATIVE_ERROR = 1.0e-4L;
constexpr int INIT_THREADS = 256;

const std::vector<std::size_t> SIZES{
    1000000, 2000000, 4000000, 8000000, 16000000,
    32000000, 64000000, 128000000, 256000000
};

struct Options {
    std::string output_file = "reduction_gpu.csv";
    std::string session_id = "manual";
    int repetitions = 10;
    unsigned int seed = 1;
    int device = 0;
};

struct Telemetry {
    int pcie_gen;
    int pcie_width;
    int sm_clock_mhz;
    int mem_clock_mhz;
    int temp_c;
    unsigned long long throttle_reasons;
};

struct CheckResult {
    bool ok;
    long double relative_error;
};

void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void nvml_check(nvmlReturn_t status, const char* what) {
    if (status != NVML_SUCCESS) {
        throw std::runtime_error(
            std::string(what) + ": " + nvmlErrorString(status));
    }
}

class NvmlSession {
public:
    NvmlSession() { nvml_check(nvmlInit_v2(), "nvmlInit_v2"); }
    ~NvmlSession() { nvmlShutdown(); }
    NvmlSession(const NvmlSession&) = delete;
    NvmlSession& operator=(const NvmlSession&) = delete;
};

std::string option_value(int& index, int argc, char** argv,
                         const std::string& argument) {
    const auto equals = argument.find('=');
    if (equals != std::string::npos) {
        return argument.substr(equals + 1);
    }
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + argument);
    }
    return argv[++index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    if (const char* value = std::getenv("BENCH_OUTPUT")) {
        options.output_file = value;
    }
    if (const char* value = std::getenv("SESSION_ID")) {
        options.session_id = value;
    }
    if (const char* value = std::getenv("REPS")) {
        options.repetitions = std::stoi(value);
    }
    if (const char* value = std::getenv("SEED")) {
        options.seed = static_cast<unsigned int>(std::stoul(value));
    }
    if (const char* value = std::getenv("BENCH_CUDA_DEVICE")) {
        options.device = std::stoi(value);
    }

    int first_flag = 1;
    int positional = 0;
    while (first_flag < argc && argv[first_flag][0] != '-') {
        const std::string value = argv[first_flag++];
        switch (positional++) {
            case 0: options.output_file = value; break;
            case 1: options.repetitions = std::stoi(value); break;
            case 2: options.session_id = value; break;
            case 3:
                options.seed = static_cast<unsigned int>(std::stoul(value));
                break;
            case 4: options.device = std::stoi(value); break;
            default:
                throw std::runtime_error("Too many positional arguments");
        }
    }

    for (int i = first_flag; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--output" || arg == "--output-file" ||
            arg.rfind("--output=", 0) == 0 ||
            arg.rfind("--output-file=", 0) == 0) {
            options.output_file = option_value(i, argc, argv, arg);
        } else if (arg == "--session-id" || arg == "--session_id" ||
                   arg.rfind("--session-id=", 0) == 0 ||
                   arg.rfind("--session_id=", 0) == 0) {
            options.session_id = option_value(i, argc, argv, arg);
        } else if (arg == "--repetitions" || arg == "--reps" ||
                   arg.rfind("--repetitions=", 0) == 0 ||
                   arg.rfind("--reps=", 0) == 0) {
            options.repetitions = std::stoi(option_value(i, argc, argv, arg));
        } else if (arg == "--seed" || arg.rfind("--seed=", 0) == 0) {
            options.seed = static_cast<unsigned int>(
                std::stoul(option_value(i, argc, argv, arg)));
        } else if (arg == "--device" || arg.rfind("--device=", 0) == 0) {
            options.device = std::stoi(option_value(i, argc, argv, arg));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: main_reduction [output reps session_id seed [device]]\n"
                << "   or: main_reduction [--output FILE] [--session-id ID] "
                << "[--repetitions N] [--seed N] [--device N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.repetitions < 1 || options.device < 0) {
        throw std::runtime_error("Invalid repetitions or CUDA device");
    }
    return options;
}

std::optional<std::unordered_set<std::size_t>> read_size_filter() {
    const char* raw = std::getenv("BENCH_SIZE_FILTER");
    if (!raw || !*raw) {
        return std::nullopt;
    }

    std::unordered_set<std::size_t> values;
    std::stringstream stream(raw);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            values.insert(static_cast<std::size_t>(std::stoull(token)));
        }
    }
    if (values.empty()) {
        throw std::runtime_error("BENCH_SIZE_FILTER contains no values");
    }
    return values;
}

std::vector<std::size_t> selected_sizes(unsigned int seed) {
    const auto filter = read_size_filter();
    std::vector<std::size_t> sizes;
    for (std::size_t n : SIZES) {
        if (!filter || filter->count(n)) {
            sizes.push_back(n);
        }
    }
    if (sizes.empty()) {
        throw std::runtime_error("REDUCTION size filter produced no configurations");
    }
    std::mt19937 random(seed);
    std::shuffle(sizes.begin(), sizes.end(), random);
    return sizes;
}

__global__ void initialize_input(float* x, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (i < n) {
        x[i] = 0.5f + static_cast<float>(i % 29) * 0.0078125f;
    }
}

long double expected_result(std::size_t n) {
    const std::size_t q = n / 29;
    const std::size_t r = n % 29;
    const long double rr = static_cast<long double>(r);
    const long double remainder =
        rr * 0.5L + (rr * (rr - 1.0L) * 0.5L) / 128.0L;
    return static_cast<long double>(q) * 17.671875L + remainder;
}

CheckResult check_result(float actual, long double expected) {
    if (!std::isfinite(actual)) {
        return {false, std::numeric_limits<long double>::infinity()};
    }
    const long double relative =
        std::abs(static_cast<long double>(actual) - expected) /
        std::max(1.0L, std::abs(expected));
    return {relative <= MAX_RELATIVE_ERROR, relative};
}

Telemetry read_telemetry(nvmlDevice_t device) {
    unsigned int pcie_gen = 0;
    unsigned int pcie_width = 0;
    unsigned int sm_clock = 0;
    unsigned int mem_clock = 0;
    unsigned int temp = 0;
    unsigned long long throttle = 0;

    nvml_check(nvmlDeviceGetCurrPcieLinkGeneration(device, &pcie_gen),
               "nvmlDeviceGetCurrPcieLinkGeneration");
    nvml_check(nvmlDeviceGetCurrPcieLinkWidth(device, &pcie_width),
               "nvmlDeviceGetCurrPcieLinkWidth");
    nvml_check(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock),
               "nvmlDeviceGetClockInfo(SM)");
    nvml_check(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock),
               "nvmlDeviceGetClockInfo(MEM)");
    nvml_check(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp),
               "nvmlDeviceGetTemperature");
    nvml_check(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle),
               "nvmlDeviceGetCurrentClocksThrottleReasons");

    return {
        static_cast<int>(pcie_gen), static_cast<int>(pcie_width),
        static_cast<int>(sm_clock), static_cast<int>(mem_clock),
        static_cast<int>(temp), throttle
    };
}

unsigned long long read_energy_mj(nvmlDevice_t device) {
    unsigned long long energy_mj = 0;
    const nvmlReturn_t status =
        nvmlDeviceGetTotalEnergyConsumption(device, &energy_mj);
    if (status == NVML_ERROR_NOT_SUPPORTED) {
        throw std::runtime_error(
            "NVML total-energy counter is not supported; audit stop required");
    }
    nvml_check(status, "nvmlDeviceGetTotalEnergyConsumption");
    return energy_mj;
}

void enqueue_reductions(void* workspace, std::size_t workspace_bytes,
                        const float* input, float* result, int num_items,
                        int batches, cudaStream_t stream) {
    for (int batch = 0; batch < batches; ++batch) {
        cuda_check(
            cub::DeviceReduce::Sum(
                workspace, workspace_bytes, input, result, num_items, stream),
            "cub::DeviceReduce::Sum");
    }
}

double event_seconds(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0.0f;
    cuda_check(cudaEventElapsedTime(&milliseconds, start, stop),
               "cudaEventElapsedTime");
    return static_cast<double>(milliseconds) / 1000.0;
}

int scale_batches(double seconds, int batches) {
    if (!(seconds > 0.0)) {
        return std::min(MAX_BATCHES, batches * 10);
    }
    const double estimate =
        std::ceil(static_cast<double>(batches) * TARGET_RUNTIME_S / seconds);
    const long long next = std::max<long long>(batches + 1,
                                               static_cast<long long>(estimate));
    return static_cast<int>(std::min<long long>(MAX_BATCHES, next));
}

int calibrate(void* workspace, std::size_t workspace_bytes,
              const float* input, float* result, int num_items,
              cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
    enqueue_reductions(
        workspace, workspace_bytes, input, result, num_items, 1, stream);
    cuda_check(cudaStreamSynchronize(stream), "warm-up synchronize");

    int batches = 1;
    for (int step = 0; step < MAX_CALIBRATION_STEPS; ++step) {
        cuda_check(cudaEventRecord(start, stream), "calibration start event");
        enqueue_reductions(
            workspace, workspace_bytes, input, result, num_items, batches, stream);
        cuda_check(cudaEventRecord(stop, stream), "calibration stop event");
        cuda_check(cudaEventSynchronize(stop), "calibration stop synchronize");
        const double seconds = event_seconds(start, stop);
        if (seconds >= TARGET_RUNTIME_S || batches == MAX_BATCHES) {
            return batches;
        }
        batches = scale_batches(seconds, batches);
    }
    return batches;
}

std::string timestamp_now() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    std::ostringstream out;
    out << std::put_time(&local, "%Y-%m-%dT%H:%M:%S");
    return out.str();
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped += '"';
        }
        escaped += ch;
    }
    escaped += '"';
    return escaped;
}

std::string runtime_status(double seconds) {
    if (seconds < 0.75) {
        return "below";
    }
    if (seconds <= 1.25) {
        return "in_range";
    }
    return "above";
}

void write_header(std::ofstream& output) {
    output
        << "schema_version,timestamp,session_id,sequence_index,run_id_global,"
        << "repetition,workload,implementation,execution_mode,device_name,"
        << "num_threads,problem_size,problem_spec,batches,e2e_time_s,"
        << "kernel_time_s,wall_time_s,device_energy_j,total_energy_j,"
        << "dram_energy_j,energy_per_op_j,energy_per_second_j,"
        << "energy_per_flop_j,time_per_op_ms_kernel,time_per_op_ms_e2e,"
        << "flops_total,gflops_per_s,logical_bytes_per_op,avg_power_w,"
        << "runtime_status,pcie_gen,pcie_width,sm_clock_mhz,clock_before_mhz,"
        << "clock_after_mhz,mem_clock_mhz,temp_c,temp_before_c,temp_after_c,"
        << "throttle_reasons,cpu_cycles,cpu_instructions,cpu_ipc,"
        << "cpu_cache_misses,checksum_ok\n";
}

void write_row(std::ofstream& output, const Options& options, int sequence,
               int repetition, const std::string& device_name, std::size_t n,
               int batches, double e2e_time_s, double kernel_time_s,
               double energy_j, const Telemetry& before,
               const Telemetry& after, bool checksum_ok) {
    const double flops_total =
        static_cast<double>(n - 1) * static_cast<double>(batches);
    const double logical_bytes_per_op =
        static_cast<double>(n) * sizeof(float) + sizeof(float);
    const double energy_per_op = energy_j / batches;
    const double energy_per_second = energy_j / e2e_time_s;
    const double energy_per_flop = energy_j / flops_total;
    const double time_per_op_ms_kernel = 1000.0 * kernel_time_s / batches;
    const double time_per_op_ms_e2e = 1000.0 * e2e_time_s / batches;
    const double gflops_per_s = flops_total / kernel_time_s / 1.0e9;
    const int sm_clock = (before.sm_clock_mhz + after.sm_clock_mhz) / 2;
    const int mem_clock = (before.mem_clock_mhz + after.mem_clock_mhz) / 2;
    const int temp = std::max(before.temp_c, after.temp_c);
    const unsigned long long throttle =
        before.throttle_reasons | after.throttle_reasons;
    std::ostringstream throttle_text;
    throttle_text << "0x" << std::hex << throttle;

    output << std::setprecision(12)
           << "cpu-gpu-v2," << timestamp_now() << ','
           << csv_escape(options.session_id) << ','
           << sequence << ',' << sequence << ',' << repetition << ','
           << "REDUCTION,cub_device_reduce_sum_fp32,gpu_resident,"
           << csv_escape(device_name) << ','
           << -1 << ',' << n << ',' << "elements=" << n << ',' << batches << ','
           << e2e_time_s << ',' << kernel_time_s << ',' << e2e_time_s << ','
           << energy_j << ',' << energy_j << ',' << -1 << ','
           << energy_per_op << ',' << energy_per_second << ','
           << energy_per_flop << ',' << time_per_op_ms_kernel << ','
           << time_per_op_ms_e2e << ',' << flops_total << ','
           << gflops_per_s << ',' << logical_bytes_per_op << ','
           << energy_per_second << ',' << runtime_status(e2e_time_s) << ','
           << after.pcie_gen << ',' << after.pcie_width << ',' << sm_clock << ','
           << before.sm_clock_mhz << ',' << after.sm_clock_mhz << ','
           << mem_clock << ',' << temp << ',' << before.temp_c << ','
           << after.temp_c << ',' << throttle_text.str() << ','
           << -1 << ',' << -1 << ',' << -1 << ',' << -1 << ','
           << (checksum_ok ? "true" : "false") << '\n';
}

void verify_expected_gpu(const std::string& device_name) {
    const char* expected = std::getenv("BENCH_EXPECTED_GPU");
    if (!expected || !*expected) {
        return;
    }
    std::string actual_lower = device_name;
    std::string expected_lower = expected;
    std::transform(actual_lower.begin(), actual_lower.end(), actual_lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(expected_lower.begin(), expected_lower.end(), expected_lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (actual_lower.find(expected_lower) == std::string::npos) {
        throw std::runtime_error(
            "Unexpected GPU: expected substring '" + expected_lower +
            "', found '" + device_name + "'");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto parent = std::filesystem::path(options.output_file).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Cannot open output file: " + options.output_file);
        }
        write_header(output);

        cuda_check(cudaSetDevice(options.device), "cudaSetDevice");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, options.device),
                   "cudaGetDeviceProperties");
        const std::string device_name = properties.name;
        verify_expected_gpu(device_name);

        NvmlSession nvml_session;
        char pci_bus_id[32]{};
        cuda_check(cudaDeviceGetPCIBusId(
                       pci_bus_id, static_cast<int>(sizeof(pci_bus_id)),
                       options.device),
                   "cudaDeviceGetPCIBusId");
        nvmlDevice_t nvml_device{};
        nvml_check(nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &nvml_device),
                   "nvmlDeviceGetHandleByPciBusId_v2");
        (void)read_energy_mj(nvml_device);

        std::vector<std::size_t> sizes = selected_sizes(options.seed);
        std::cout << "REDUCTION(sum) | " << device_name
                  << " | session=" << options.session_id
                  << " | reps=" << options.repetitions
                  << " | configs=" << sizes.size() << '\n';

        int sequence = 0;
        for (std::size_t n : sizes) {
            if (n > static_cast<std::size_t>(INT_MAX)) {
                throw std::runtime_error("REDUCTION N exceeds CUB int range");
            }
            const int num_items = static_cast<int>(n);
            const std::size_t input_bytes = n * sizeof(float);

            float* d_x = nullptr;
            float* d_result = nullptr;
            void* d_workspace = nullptr;
            cudaStream_t stream{};
            cudaEvent_t start_event{};
            cudaEvent_t stop_event{};

            cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
            cuda_check(cudaEventCreate(&start_event), "cudaEventCreate(start)");
            cuda_check(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_x), input_bytes),
                       "cudaMalloc(d_x)");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_result), sizeof(float)),
                       "cudaMalloc(d_result)");
            const std::uintptr_t input_begin =
                reinterpret_cast<std::uintptr_t>(d_x);
            const std::uintptr_t input_end = input_begin + input_bytes;
            const std::uintptr_t output_address =
                reinterpret_cast<std::uintptr_t>(d_result);
            if (output_address >= input_begin && output_address < input_end) {
                throw std::runtime_error("CUB input and output ranges overlap");
            }

            const int init_blocks = static_cast<int>(
                (n + INIT_THREADS - 1) / INIT_THREADS);
            initialize_input<<<init_blocks, INIT_THREADS, 0, stream>>>(d_x, n);
            cuda_check(cudaGetLastError(), "initialize_input launch");
            cuda_check(cudaStreamSynchronize(stream), "initialize_input synchronize");

            std::size_t workspace_bytes = 0;
            cuda_check(cub::DeviceReduce::Sum(
                           nullptr, workspace_bytes, d_x, d_result,
                           num_items, stream),
                       "CUB workspace query");
            if (workspace_bytes > 0) {
                cuda_check(cudaMalloc(&d_workspace, workspace_bytes),
                           "cudaMalloc(CUB workspace)");
            }

            const int batches = calibrate(
                d_workspace, workspace_bytes, d_x, d_result, num_items,
                stream, start_event, stop_event);
            const long double expected = expected_result(n);

            for (int repetition = 1;
                 repetition <= options.repetitions; ++repetition) {
                const Telemetry before = read_telemetry(nvml_device);
                const unsigned long long energy_before_mj =
                    read_energy_mj(nvml_device);
                const auto host_start = std::chrono::steady_clock::now();

                cuda_check(cudaEventRecord(start_event, stream),
                           "measurement start event");
                enqueue_reductions(
                    d_workspace, workspace_bytes, d_x, d_result,
                    num_items, batches, stream);
                cuda_check(cudaEventRecord(stop_event, stream),
                           "measurement stop event");
                cuda_check(cudaEventSynchronize(stop_event),
                           "measurement synchronize");
                cuda_check(cudaGetLastError(), "measurement CUDA status");

                const auto host_end = std::chrono::steady_clock::now();
                const unsigned long long energy_after_mj =
                    read_energy_mj(nvml_device);
                const Telemetry after = read_telemetry(nvml_device);

                if (energy_after_mj < energy_before_mj) {
                    throw std::runtime_error("Non-monotonic NVML energy counter");
                }
                const double energy_j = static_cast<double>(
                    energy_after_mj - energy_before_mj) / 1000.0;
                const double e2e_time_s = std::chrono::duration<double>(
                    host_end - host_start).count();
                const double kernel_time_s =
                    event_seconds(start_event, stop_event);
                const double timing_excess = kernel_time_s - e2e_time_s;
                const double timing_limit =
                    std::max(0.0005, 0.005 * e2e_time_s);
                if (timing_excess > timing_limit) {
                    throw std::runtime_error(
                        "CUDA event time materially exceeds host E2E time");
                }
                if (timing_excess > 0.0) {
                    std::cerr << "WARN: kernel_time_s exceeds e2e_time_s by "
                              << timing_excess << " s\n";
                }
                if (!(energy_j > 0.0) || !(e2e_time_s > 0.0) ||
                    !(kernel_time_s > 0.0)) {
                    throw std::runtime_error("Non-positive REDUCTION measurement");
                }

                float result = 0.0f;
                cuda_check(cudaMemcpyAsync(
                               &result, d_result, sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
                           "cudaMemcpyAsync(result D2H)");
                cuda_check(cudaStreamSynchronize(stream),
                           "result D2H synchronize");
                const CheckResult check = check_result(result, expected);

                write_row(
                    output, options, ++sequence, repetition, device_name,
                    n, batches, e2e_time_s, kernel_time_s, energy_j,
                    before, after, check.ok);
                output.flush();

                std::cout << "N=" << n
                          << " batches=" << batches
                          << " kernel_s=" << kernel_time_s
                          << " e2e_s=" << e2e_time_s
                          << " energy_j=" << energy_j
                          << " checksum=" << (check.ok ? "OK" : "FAIL")
                          << " relative_error=" << std::scientific
                          << static_cast<double>(check.relative_error)
                          << std::defaultfloat << '\n';

                if (!check.ok) {
                    throw std::runtime_error(
                        "REDUCTION checksum failed for N=" + std::to_string(n));
                }
            }

            cuda_check(cudaFree(d_workspace), "cudaFree(CUB workspace)");
            cuda_check(cudaFree(d_result), "cudaFree(d_result)");
            cuda_check(cudaFree(d_x), "cudaFree(d_x)");
            cuda_check(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");
            cuda_check(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
            cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        return 2;
    }
}
