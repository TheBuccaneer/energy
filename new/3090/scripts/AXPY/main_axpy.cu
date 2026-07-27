// AXPY v1.0 FROZEN -- GPU implementation (CUDA, RTX 3090 / RTX 5060 Ti)
//
// Authoritative source: AXPY_MEASUREMENT_CONTRACT_v1_0_FROZEN_2026-07-25.md
// Patch/audit closure:  AXPY_CONTRACT_v1_0_PATCH_AND_AUDIT_CLOSURE.md
//
// implementation = cuda_axpy_inplace_fp32
// execution_mode = gpu_resident
//
// Contract 6.5: this file must be BYTE-IDENTICAL between the 3090 and
// 5060ti platform trees. There is no per-platform token substitution here
// (unlike the CPU sources, where platform label/output filename are
// permitted to differ) -- device identity is resolved entirely at runtime
// via NVML + the optional BENCH_EXPECTED_GPU environment variable, and the
// output path is supplied on the command line by the (platform-specific)
// runner, not compiled in.
//
// Structural conventions (device binding via PCI bus ID, NVML telemetry,
// CUDA-event kernel timing, VRAM safety margin, exception-safe cleanup)
// are reused from the real STREAM GPU reference (main_stream.cu) as
// instructed. The CSV row type/writer and the calibration/checksum/
// anti-collapse logic are NOT reused from that reference verbatim: the
// reference's write_row() aggregates energy in rounded scientific
// notation and its scale_batches() imposes an extra 10x-per-step growth
// cap that the AXPY contract's exact formula (8.2) does not specify. The
// AXPY contract takes precedence, so this file defines its own row
// writer and scaling function, matching the CPU AXPY sources' approach
// exactly for the same reasons.
//
// This file also bakes in, from the start, the three hardening fixes that
// were found necessary for the CPU sources after a real quickcheck run
// (see PATCH_SUMMARY.md / CLAUDE_PATCH_PROMPT_AXPY_CPU_SOURCES_v1.md):
//   F3 - runtime_status=="below" hard-fails before a row is written.
//   F4 - the anti-collapse probe's B_probe growth is bounded: hard error
//        at the cap, strict-progress assertion on every growth step.
//   F5 - device energy must be finite AND > 0.0 before use (not merely
//        ">= 0.0"), so 0.0 J and NaN are rejected, not just negative values.

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
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------
// Frozen AXPY semantics (contract sections 1, 2, 4) -- do not modify.
// ---------------------------------------------------------------------

constexpr const char* SCHEMA_VERSION = "cpu-gpu-v2";
constexpr double TARGET_RUNTIME_S = 1.0;
constexpr double MIN_RUNTIME_S = 0.75;
constexpr double MAX_RUNTIME_S = 1.25;
constexpr int DEFAULT_REPETITIONS = 10;
constexpr int MAX_CALIBRATION_STEPS_AXPY = 12;   // contract 7: max 12 steps
constexpr long long MAX_BATCHES = 250000;         // contract 4.2
constexpr int CUDA_DEVICE = 0;  // CUDA_VISIBLE_DEVICES maps the selected physical GPU here.
constexpr int BLOCK_SIZE = 256; // contract 6.1
constexpr float ALPHA = 3.0f;
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

// ---------------------------------------------------------------------
// Deterministic inputs (contract 4.1) -- identical construction to the
// CPU AXPY sources, host- and device-side.
// ---------------------------------------------------------------------

__host__ __device__ inline int kx_of(size_t i) {
    return static_cast<int>(i % 29) - 14;
}

__host__ __device__ inline int ky_of(size_t i) {
    return static_cast<int>(i % 31) - 15;
}

__host__ __device__ inline float x_value(size_t i) {
    return ldexpf(static_cast<float>(kx_of(i)), -16);
}

__host__ __device__ inline float y0_value(size_t i) {
    return ldexpf(static_cast<float>(ky_of(i)), -8);
}

// ---------------------------------------------------------------------
// Contract-mandated host-side self-test (section 8): the integer
// coefficient bound must stay < 2^24 for every batches in [0, MAX_BATCHES]
// across all 899 (=29*31) periodic (kx,ky) states. Pure host arithmetic,
// runs once at process start before touching the GPU.
// ---------------------------------------------------------------------

void run_contract_selftest() {
    constexpr long long kMaxAbsCoefficient = 10503840LL;
    constexpr long long kTwoPow24 = 1LL << 24;
    if (kMaxAbsCoefficient >= kTwoPow24) {
        throw std::runtime_error(
            "AXPY contract self-test failed: integer coefficient bound "
            "10503840 is not < 2^24");
    }
    for (int ky = -15; ky <= 15; ++ky) {
        for (int kx = -14; kx <= 14; ++kx) {
            const long long coeff_at_max = 256LL * ky + 3LL * MAX_BATCHES * kx;
            if (coeff_at_max > kMaxAbsCoefficient || coeff_at_max < -kMaxAbsCoefficient) {
                throw std::runtime_error(
                    "AXPY contract self-test failed: coefficient out of the "
                    "exact-FP32-representable bound at MAX_BATCHES");
            }
            const float as_float = static_cast<float>(coeff_at_max);
            if (static_cast<long long>(as_float) != coeff_at_max) {
                throw std::runtime_error(
                    "AXPY contract self-test failed: coefficient not exactly "
                    "representable as FP32 integer");
            }
        }
    }
}

// ---------------------------------------------------------------------
// Kernels (contract 6.1/6.2) -- exactly one kernel type for the measured
// batch loop; separate, unmeasured kernels for first-touch init and the
// mandatory outside-window reset. No fused multi-pass kernel, no CUDA
// Graphs, no Unified Memory.
// ---------------------------------------------------------------------

__global__ void init_x_kernel(float* __restrict__ x, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x_value(i);
}

__global__ void reset_y_kernel(float* __restrict__ y, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y0_value(i);
}

__global__ void axpy_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    size_t n
) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = ALPHA * x[i] + y[i];
    }
}

// ---------------------------------------------------------------------
// Host-side utilities
// ---------------------------------------------------------------------

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
    options.output_file = argc > 1 ? argv[1] : "axpy_gpu.csv";
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
        if (!token.empty()) values.push_back(static_cast<size_t>(std::stoull(token)));
    }
    return values;
}

bool selected(size_t value, const std::vector<size_t>& filter) {
    return filter.empty() || std::find(filter.begin(), filter.end(), value) != filter.end();
}

bool env_flag_set(const char* name) {
    const char* raw = std::getenv(name);
    return raw && *raw && std::string(raw) != "0";
}

// contract 7.4/8.3: below/in_range/above; evaluated on e2e_time_s for GPU.
std::string runtime_status(double seconds) {
    if (seconds < MIN_RUNTIME_S) return "below";
    if (seconds > MAX_RUNTIME_S) return "above";
    return "in_range";
}

// PATCH-lesson (equivalent to CPU F1-adjacent hardening, applied here from
// the start): exact contract formula (8.2), no extra per-step growth cap.
long long axpy_scale_batches(double calibration_time_s, long long current) {
    const double safe_seconds = std::max(calibration_time_s, 1.0e-12);
    const long long estimate = static_cast<long long>(
        std::ceil(TARGET_RUNTIME_S * static_cast<double>(current) / safe_seconds));
    const long long next = std::max<long long>(current + 1, estimate);
    return std::min<long long>(MAX_BATCHES, next);
}

std::string throttle_hex(unsigned long long reasons) {
    std::ostringstream out;
    out << "0x" << std::hex << std::uppercase << reasons;
    return out.str();
}

// ---------------------------------------------------------------------
// NVML / device binding (reused pattern from the real STREAM GPU
// reference: PCI-bus-ID handoff from CUDA to NVML, BENCH_EXPECTED_GPU
// substring check, cumulative-energy-only policy).
// ---------------------------------------------------------------------

Telemetry read_telemetry(nvmlDevice_t device) {
    Telemetry telemetry;
    NVML_CHECK(nvmlDeviceGetCurrPcieLinkGeneration(device, &telemetry.pcie_gen));
    NVML_CHECK(nvmlDeviceGetCurrPcieLinkWidth(device, &telemetry.pcie_width));
    NVML_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &telemetry.sm_clock_mhz));
    NVML_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &telemetry.mem_clock_mhz));
    NVML_CHECK(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &telemetry.temp_c));

    const nvmlReturn_t throttle_status =
        nvmlDeviceGetCurrentClocksThrottleReasons(device, &telemetry.throttle_reasons);
    if (throttle_status != NVML_SUCCESS) {
        telemetry.throttle_reasons = 0;
    }
    return telemetry;
}

unsigned long long read_energy_mj(nvmlDevice_t device) {
    unsigned long long energy = 0;
    const nvmlReturn_t status = nvmlDeviceGetTotalEnergyConsumption(device, &energy);
    if (status == NVML_ERROR_NOT_SUPPORTED) {
        throw std::runtime_error(
            "NVML total-energy counter is not supported on this GPU. "
            "Do not mix this run with a power-sampling fallback (contract 6.6).");
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
    CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, static_cast<int>(sizeof(pci_bus_id)), CUDA_DEVICE));
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

unsigned int grid_blocks(size_t n, const cudaDeviceProp& properties) {
    if (n > std::numeric_limits<size_t>::max() - (BLOCK_SIZE - 1)) {
        throw std::runtime_error("Grid-size arithmetic overflow for N=" + std::to_string(n));
    }
    const size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks == 0) {
        throw std::runtime_error("Grid contains zero blocks");
    }
    if (blocks > static_cast<size_t>(properties.maxGridSize[0])) {
        throw std::runtime_error(
            "Grid exceeds CUDA device limit for N=" + std::to_string(n) +
            ": need " + std::to_string(blocks) + " blocks, limit " +
            std::to_string(properties.maxGridSize[0]));
    }
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("Grid exceeds unsigned-int launch range for N=" + std::to_string(n));
    }
    return static_cast<unsigned int>(blocks);
}

// ---------------------------------------------------------------------
// Batch enqueue (contract 6.2/6.3): exactly one kernel launch per batch,
// in the same stream, no per-kernel sync, no library AXPY substitution.
// ---------------------------------------------------------------------

void enqueue_axpy_batches(
    const float* x,
    float* y,
    size_t n,
    unsigned int blocks,
    long long batches,
    cudaStream_t stream
) {
    for (long long batch = 0; batch < batches; ++batch) {
        axpy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(x, y, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

// kernel_time_s: CUDA events wrapping the entire batch sequence (contract 6.3).
double measure_kernel_seconds(
    const float* x,
    float* y,
    size_t n,
    unsigned int blocks,
    long long batches,
    cudaStream_t stream,
    cudaEvent_t start,
    cudaEvent_t stop
) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    enqueue_axpy_batches(x, y, n, blocks, batches, stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return static_cast<double>(milliseconds) / 1000.0;
}

void reset_y(float* y, size_t n, unsigned int blocks, cudaStream_t stream) {
    reset_y_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(y, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------------
// Calibration (contract 7/8.2): GPU calibration_time_s == kernel_time_s.
// ---------------------------------------------------------------------

long long calibrate(
    const float* x,
    float* y,
    size_t n,
    unsigned int blocks,
    cudaStream_t stream,
    cudaEvent_t start,
    cudaEvent_t stop
) {
    // Warm-up (contract 8.1): x already initialized, y already = y0.
    enqueue_axpy_batches(x, y, n, blocks, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    reset_y(y, n, blocks, stream);

    long long batches = 1;
    for (int step = 0; step < MAX_CALIBRATION_STEPS_AXPY; ++step) {
        reset_y(y, n, blocks, stream);  // reset outside the timing window, every step
        const double seconds = measure_kernel_seconds(x, y, n, blocks, batches, stream, start, stop);
        if (!std::isfinite(seconds) || seconds <= 0.0) {
            throw std::runtime_error("Non-finite or non-positive calibration time for N=" + std::to_string(n));
        }
        if (seconds >= TARGET_RUNTIME_S || batches == MAX_BATCHES) {
            return batches;
        }
        batches = axpy_scale_batches(seconds, batches);
    }
    return batches;
}

// ---------------------------------------------------------------------
// Exact FP32 checksum (contract 5) -- single-float D2H reads at the eight
// mandated sample indices, outside the timing/energy window.
// ---------------------------------------------------------------------

struct ChecksumResult {
    bool ok{false};
    double max_abs_error{0.0};
    double max_rel_error{0.0};
};

float read_device_float(const float* device_data, size_t index) {
    float value = 0.0f;
    CUDA_CHECK(cudaMemcpy(&value, device_data + index, sizeof(float), cudaMemcpyDeviceToHost));
    return value;
}

std::vector<size_t> checksum_sample_indices(size_t n) {
    return {0, 1, n / 7, n / 3, n / 2, (2 * n) / 3, n - 2, n - 1};
}

ChecksumResult check_axpy(const float* x, const float* y, size_t n, long long batches) {
    ChecksumResult result;
    result.ok = true;

    for (const size_t i : checksum_sample_indices(n)) {
        const int kx = kx_of(i);
        const int ky = ky_of(i);
        const long long expected_coefficient = 256LL * ky + 3LL * batches * kx;
        const float expected_y = std::ldexp(static_cast<float>(expected_coefficient), -16);
        const float expected_x = x_value(i);
        const float actual_x = read_device_float(x, i);
        const float actual_y = read_device_float(y, i);

        const bool finite_ok =
            std::isfinite(actual_x) && std::isfinite(actual_y) &&
            std::isfinite(expected_x) && std::isfinite(expected_y);

        const double abs_err_x = std::abs(static_cast<double>(actual_x) - static_cast<double>(expected_x));
        const double abs_err_y = std::abs(static_cast<double>(actual_y) - static_cast<double>(expected_y));
        result.max_abs_error = std::max({result.max_abs_error, abs_err_x, abs_err_y});

        const double rel_err_x = abs_err_x / std::max(1.0, std::abs(static_cast<double>(expected_x)));
        const double rel_err_y = abs_err_y / std::max(1.0, std::abs(static_cast<double>(expected_y)));
        result.max_rel_error = std::max({result.max_rel_error, rel_err_x, rel_err_y});

        const bool exact_ok = finite_ok && (actual_x == expected_x) && (actual_y == expected_y);
        if (!exact_ok) result.ok = false;
    }
    return result;
}

// ---------------------------------------------------------------------
// CSV row (contract 9.1/9.3/9.5) -- bespoke writer, not reused from the
// STREAM/REDUCTION reference (see header comment for why).
// ---------------------------------------------------------------------

struct AxpyRow {
    std::string session_id;
    int sequence_index{};
    int run_id_global{};
    int repetition{};
    std::string device_name;
    long long problem_size{};
    std::string problem_spec;
    long long batches{};
    double e2e_time_s{};
    double kernel_time_s{};
    double wall_time_s{};
    double device_energy_j{};
    double total_energy_j{};
    double dram_energy_j{-1.0};
    double energy_per_op_j{};
    double energy_per_second_j{};
    double energy_per_flop_j{};
    double time_per_op_ms_kernel{};
    double time_per_op_ms_e2e{};
    long long flops_total{};
    double gflops_per_s{};
    long long logical_bytes_per_op{};
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

void write_axpy_header(std::ofstream& output) {
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

// Lossless (max_digits10), non-scientific-rounded double serialization --
// contract 9.5, identical convention to the CPU AXPY sources.
void write_lossless_double(std::ofstream& output, double value) {
    output << std::defaultfloat << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
}

void write_axpy_row(std::ofstream& output, const AxpyRow& row) {
    output << SCHEMA_VERSION << ',' << timestamp() << ','
           << csv_escape(row.session_id) << ','
           << row.sequence_index << ',' << row.run_id_global << ','
           << row.repetition << ','
           << csv_escape("AXPY") << ','
           << csv_escape("cuda_axpy_inplace_fp32") << ','
           << "gpu_resident" << ','
           << csv_escape(row.device_name) << ',' << -1 << ','   // num_threads: CPU-only sentinel
           << row.problem_size << ',' << csv_escape(row.problem_spec) << ','
           << row.batches << ',';

    write_lossless_double(output, row.e2e_time_s); output << ',';
    write_lossless_double(output, row.kernel_time_s); output << ',';
    write_lossless_double(output, row.wall_time_s); output << ',';
    write_lossless_double(output, row.device_energy_j); output << ',';
    write_lossless_double(output, row.total_energy_j); output << ',';
    write_lossless_double(output, row.dram_energy_j); output << ',';
    write_lossless_double(output, row.energy_per_op_j); output << ',';
    write_lossless_double(output, row.energy_per_second_j); output << ',';
    write_lossless_double(output, row.energy_per_flop_j); output << ',';
    write_lossless_double(output, row.time_per_op_ms_kernel); output << ',';
    write_lossless_double(output, row.time_per_op_ms_e2e); output << ',';

    // Exact decimal integers (contract 9.5) -- never floating formatted.
    output << row.flops_total << ',';
    write_lossless_double(output, row.gflops_per_s); output << ',';
    output << row.logical_bytes_per_op << ',';
    write_lossless_double(output, row.avg_power_w); output << ',';

    output << row.runtime_status << ','
           << row.pcie_gen << ',' << row.pcie_width << ','
           << row.sm_clock_mhz << ','
           << row.clock_before_mhz << ',' << row.clock_after_mhz << ','
           << row.mem_clock_mhz << ','
           << row.temp_c << ',' << row.temp_before_c << ',' << row.temp_after_c << ','
           << csv_escape(row.throttle_reasons) << ','
           << -1 << ',' << -1 << ',' << std::fixed << std::setprecision(6) << -1.0
           << std::defaultfloat << ',' << -1 << ','   // cpu_cycles/instructions/ipc/cache_misses: CPU-only sentinels
           << (row.checksum_ok ? 't' : 'f') << '\n';
}

void print_axpy_result(const AxpyRow& row) {
    std::cout << "[AXPY] N=" << row.problem_size
              << " rep=" << row.repetition
              << " batches=" << row.batches
              << " e2e_time_s=" << std::fixed << std::setprecision(6) << row.e2e_time_s
              << " kernel_time_s=" << row.kernel_time_s
              << " device_energy_j=" << std::setprecision(6) << row.device_energy_j
              << " avg_power_w=" << std::setprecision(3) << row.avg_power_w
              << " runtime_status=" << row.runtime_status
              << " checksum=" << (row.checksum_ok ? "OK" : "FAIL")
              << std::defaultfloat << '\n';
}

// ---------------------------------------------------------------------
// Anti-collapse scaling probe (contract 12.4), GPU time basis =
// kernel_time_s. Opt-in via AXPY_ANTI_COLLAPSE_PROBE=1; runs once at
// N=1,000,000. Bounded growth from the start (the F4 lesson from the CPU
// patch): hard error at the cap, strict-progress assertion, no path that
// can recompute the same B_probe forever.
// ---------------------------------------------------------------------

void run_anti_collapse_probe(
    nvmlDevice_t nvml_device,
    cudaDeviceProp& device_properties,
    cudaStream_t stream,
    cudaEvent_t start,
    cudaEvent_t stop
) {
    const size_t n = 1000000;
    const unsigned int blocks = grid_blocks(n, device_properties);
    const size_t bytes = n * sizeof(float);

    float* x = nullptr;
    float* y = nullptr;
    auto free_buffers = [&]() noexcept {
        if (y) { cudaFree(y); y = nullptr; }
        if (x) { cudaFree(x); x = nullptr; }
    };

    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&x), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&y), bytes));

        init_x_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(x, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        reset_y(y, n, blocks, stream);

        const long long b_cal = calibrate(x, y, n, blocks, stream, start, stop);

        long long b_probe = std::min<long long>(std::max<long long>(100, b_cal / 4), MAX_BATCHES / 2);
        const long long max_probe = MAX_BATCHES / 2;

        auto measure = [&](long long batches) {
            reset_y(y, n, blocks, stream);
            const double seconds = measure_kernel_seconds(x, y, n, blocks, batches, stream, start, stop);
            const ChecksumResult checksum = check_axpy(x, y, n, batches);
            return std::make_pair(seconds, checksum);
        };

        double t1 = 0.0;
        ChecksumResult c1;
        for (;;) {
            std::tie(t1, c1) = measure(b_probe);
            if (t1 >= 0.020) break;

            if (b_probe >= max_probe) {
                throw std::runtime_error(
                    "Anti-collapse probe cannot reach minimum duration before "
                    "batch cap (B_probe=" + std::to_string(b_probe) +
                    ", max_probe=" + std::to_string(max_probe) + ")");
            }
            const long long next = std::min<long long>(b_probe * 2, max_probe);
            if (next <= b_probe) {
                throw std::runtime_error(
                    "Anti-collapse probe made no forward progress (B_probe=" +
                    std::to_string(b_probe) + ")");
            }
            b_probe = next;
        }

        const long long two_b_probe = 2 * b_probe;
        const auto [t2, c2] = measure(two_b_probe);

        const bool duration_ok = (t1 >= 0.020) && (t2 >= 0.020);
        const bool cap_ok = (two_b_probe <= MAX_BATCHES);
        const bool checksum_ok = c1.ok && c2.ok;
        const double ratio = t1 > 0.0 ? t2 / t1 : std::numeric_limits<double>::infinity();
        const bool ratio_ok = std::isfinite(ratio) && ratio >= 1.7 && ratio <= 2.3;
        const bool gate_pass = duration_ok && cap_ok && checksum_ok && ratio_ok;

        const Telemetry telemetry = read_telemetry(nvml_device);

        std::cout << std::setprecision(9)
                  << "[ANTI_COLLAPSE] N=" << n
                  << " device=gpu"
                  << " B_cal=" << b_cal
                  << " B_probe=" << b_probe
                  << " two_B_probe=" << two_b_probe
                  << " t1=" << t1
                  << " t2=" << t2
                  << " ratio=" << ratio
                  << " time_basis=kernel_time_s"
                  << " sm_clock_mhz=" << telemetry.sm_clock_mhz
                  << " temp_c=" << telemetry.temp_c
                  << " throttle_reasons=" << throttle_hex(telemetry.throttle_reasons)
                  << " checksum1=" << (c1.ok ? "OK" : "FAIL")
                  << " checksum2=" << (c2.ok ? "OK" : "FAIL")
                  << " gate=" << (gate_pass ? "PASS" : "FAIL")
                  << '\n';

        if (!gate_pass) {
            throw std::runtime_error("Anti-collapse gate FAILED (contract 12.4)");
        }
        free_buffers();
    } catch (...) {
        free_buffers();
        throw;
    }
}

}  // namespace

int main(int argc, char** argv) {
    cudaStream_t stream{};
    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    bool nvml_initialized = false;

    try {
        run_contract_selftest();

        const Options options = parse_options(argc, argv);
        const auto parent = std::filesystem::path(options.output_file).parent_path();
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
        require_expected_gpu(device_name);  // hard abort on wrong GPU (contract 6.4)

        // Fail before creating a campaign file if the direct cumulative
        // energy counter is unavailable (contract 6.6: no sampled-power fallback).
        (void)read_energy_mj(nvml_device);

        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));

        int cuda_runtime_version = 0;
        int cuda_driver_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));
        CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));

        char pci_bus_id[32]{};
        CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, static_cast<int>(sizeof(pci_bus_id)), CUDA_DEVICE));

        std::cout << "AXPY | " << device_name
                  << " | pci_bus_id=" << pci_bus_id
                  << " | implementation=cuda_axpy_inplace_fp32"
                  << " | session=" << options.session_id
                  << " | reps=" << options.repetitions
                  << " | mode=gpu_resident"
                  << " | alpha=" << ALPHA
                  << " | block_size=" << BLOCK_SIZE
                  << " | CUDA runtime=" << cuda_runtime_version
                  << " | CUDA driver=" << cuda_driver_version
                  << '\n';

        const std::vector<size_t> size_filter = parse_size_filter("BENCH_SIZE_FILTER");
        std::vector<size_t> sizes;
        for (const size_t n : SIZES) {
            if (selected(n, size_filter)) sizes.push_back(n);
        }
        if (sizes.empty()) {
            throw std::runtime_error("No sizes remain after BENCH_SIZE_FILTER");
        }

        std::mt19937 generator(options.seed);
        std::shuffle(sizes.begin(), sizes.end(), generator);

        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Cannot open output file: " + options.output_file);
        }
        write_axpy_header(output);

        int sequence = 0;
        for (const size_t n : sizes) {
            if (n == 0) {
                throw std::runtime_error("AXPY size must be positive");
            }
            if (n > std::numeric_limits<size_t>::max() / sizeof(float)) {
                throw std::runtime_error("Array byte-size overflow for N=" + std::to_string(n));
            }
            const size_t bytes = n * sizeof(float);

            // Two resident arrays (x, y): 8*N bytes total (contract 4).
            if (bytes > std::numeric_limits<size_t>::max() / 2) {
                throw std::runtime_error("Two-array byte-size overflow for N=" + std::to_string(n));
            }
            const size_t required_bytes = 2 * bytes;

            size_t free_bytes = 0;
            size_t total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const size_t safety_margin = std::max(VRAM_MIN_SAFETY_MARGIN_BYTES, free_bytes / 10);

            if (free_bytes < safety_margin || required_bytes > free_bytes - safety_margin) {
                const double gib = 1024.0 * 1024.0 * 1024.0;
                throw std::runtime_error(
                    "Insufficient free GPU memory for AXPY N=" + std::to_string(n) +
                    ": arrays require " + std::to_string(required_bytes / gib) + " GiB, safety margin " +
                    std::to_string(safety_margin / gib) + " GiB, free " + std::to_string(free_bytes / gib) + " GiB");
            }

            const unsigned int blocks = grid_blocks(n, device_properties);

            float* x = nullptr;
            float* y = nullptr;
            auto free_buffers = [&]() noexcept {
                if (y) { cudaFree(y); y = nullptr; }
                if (x) { cudaFree(x); x = nullptr; }
            };

            try {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&x), bytes));
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&y), bytes));

                init_x_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(x, n);   // x initialized once, never modified again
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));
                reset_y(y, n, blocks, stream);                            // first-touch y = y0

                const long long batches = calibrate(x, y, n, blocks, stream, start_event, stop_event);
                std::cout << "[CALIBRATION] N=" << n << " batches=" << batches << '\n';

                if (n == 1000000 && batches == MAX_BATCHES) {
                    // contract 7.6 / CODING-AUFTRAG section 7: hard quickcheck
                    // failure if N=1M stays below 0.75s at the batch cap, on
                    // the GPU calibration time basis (kernel_time_s).
                    reset_y(y, n, blocks, stream);
                    const double seconds = measure_kernel_seconds(x, y, n, blocks, batches, stream, start_event, stop_event);
                    if (seconds < MIN_RUNTIME_S) {
                        throw std::runtime_error(
                            "Hard quickcheck failure: N=1M at MAX_BATCHES stays below "
                            "0.75s on the calibration time basis (contract 7.6)");
                    }
                }

                for (int repetition = 1; repetition <= options.repetitions; ++repetition) {
                    reset_y(y, n, blocks, stream);  // reset outside window before every rep

                    const Telemetry before = read_telemetry(nvml_device);
                    const unsigned long long energy_before = read_energy_mj(nvml_device);
                    const auto wall_start = std::chrono::steady_clock::now();

                    CUDA_CHECK(cudaEventRecord(start_event, stream));
                    enqueue_axpy_batches(x, y, n, blocks, batches, stream);
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
                    const double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
                    const double energy_j = static_cast<double>(energy_after - energy_before) / 1000.0;

                    // F5-equivalent (baked in from the start): finite AND
                    // strictly > 0.0 for every measured quantity -- 0.0 J,
                    // NaN, and negative values are all rejected, not merely
                    // negative ones.
                    if (!std::isfinite(kernel_seconds) || !std::isfinite(wall_seconds) ||
                        !std::isfinite(energy_j) || kernel_seconds <= 0.0 ||
                        wall_seconds <= 0.0 || energy_j <= 0.0) {
                        throw std::runtime_error(
                            "Non-finite or non-positive measurement for N=" + std::to_string(n) +
                            ", repetition=" + std::to_string(repetition) +
                            ": kernel_time_s=" + std::to_string(kernel_seconds) +
                            ", e2e_time_s=" + std::to_string(wall_seconds) +
                            ", device_energy_j=" + std::to_string(energy_j));
                    }

                    const ChecksumResult checksum = check_axpy(x, y, n, batches);

                    const long long flops_total = 2LL * static_cast<long long>(n) * batches;
                    const long long logical_bytes_per_op = 12LL * static_cast<long long>(n);

                    AxpyRow row;
                    row.session_id = options.session_id;
                    row.sequence_index = ++sequence;
                    row.run_id_global = row.sequence_index;
                    row.repetition = repetition;
                    row.device_name = device_name;
                    row.problem_size = static_cast<long long>(n);
                    row.problem_spec =
                        "elements=" + std::to_string(n) +
                        ";alpha=3.0;x=period29*2^-16;y0=period31*2^-8;"
                        "reset=outside_window;max_batches=250000";
                    row.batches = batches;
                    row.e2e_time_s = wall_seconds;
                    row.kernel_time_s = kernel_seconds;
                    row.wall_time_s = wall_seconds;
                    row.device_energy_j = energy_j;              // contract 6.6 / 9.3
                    row.total_energy_j = energy_j;                // GPU: no separate DRAM domain
                    row.dram_energy_j = -1.0;                     // GPU sentinel (contract 9.4)
                    row.energy_per_op_j = energy_j / static_cast<double>(batches);
                    row.energy_per_second_j = energy_j / wall_seconds;
                    row.energy_per_flop_j = energy_j / static_cast<double>(flops_total);
                    row.time_per_op_ms_kernel = 1000.0 * kernel_seconds / static_cast<double>(batches);
                    row.time_per_op_ms_e2e = 1000.0 * wall_seconds / static_cast<double>(batches);
                    row.flops_total = flops_total;
                    row.gflops_per_s = static_cast<double>(flops_total) / kernel_seconds / 1.0e9;
                    row.logical_bytes_per_op = logical_bytes_per_op;
                    row.avg_power_w = energy_j / wall_seconds;
                    row.runtime_status = runtime_status(wall_seconds);  // bound to e2e_time_s (contract 6.4)
                    row.pcie_gen = static_cast<int>(after.pcie_gen);
                    row.pcie_width = static_cast<int>(after.pcie_width);
                    row.clock_before_mhz = static_cast<int>(before.sm_clock_mhz);
                    row.clock_after_mhz = static_cast<int>(after.sm_clock_mhz);
                    row.sm_clock_mhz = static_cast<int>((before.sm_clock_mhz + after.sm_clock_mhz) / 2);
                    row.mem_clock_mhz = static_cast<int>((before.mem_clock_mhz + after.mem_clock_mhz) / 2);
                    row.temp_before_c = static_cast<int>(before.temp_c);
                    row.temp_after_c = static_cast<int>(after.temp_c);
                    row.temp_c = static_cast<int>(std::max(before.temp_c, after.temp_c));
                    row.throttle_reasons = throttle_hex(before.throttle_reasons | after.throttle_reasons);
                    row.checksum_ok = checksum.ok;

                    // F3-equivalent (baked in from the start): a 'below' row
                    // must never be written as a valid campaign line.
                    if (row.runtime_status == "below") {
                        throw std::runtime_error(
                            "Hard failure: runtime_status=below for N=" + std::to_string(n) +
                            ", repetition=" + std::to_string(repetition) +
                            ", batches=" + std::to_string(batches) +
                            ", e2e_time_s=" + std::to_string(row.e2e_time_s));
                    }

                    write_axpy_row(output, row);
                    output.flush();
                    print_axpy_result(row);
                    std::cout << "  max_abs_error=" << std::scientific << std::setprecision(3)
                              << checksum.max_abs_error << " max_rel_error=" << checksum.max_rel_error
                              << std::defaultfloat << '\n';

                    if (!checksum.ok) {
                        throw std::runtime_error("AXPY checksum failed for N=" + std::to_string(n));
                    }
                }

                free_buffers();
            } catch (...) {
                free_buffers();
                throw;
            }
        }

        if (env_flag_set("AXPY_ANTI_COLLAPSE_PROBE")) {
            run_anti_collapse_probe(nvml_device, device_properties, stream, start_event, stop_event);
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
