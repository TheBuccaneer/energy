// CONV2D GPU benchmark (CUDA + cuDNN, RTX 3090 / RTX 5060 Ti)
//
// implementation = cudnn_convolution_fwd_fp32
// execution_mode = gpu_resident
//
// This file must be BYTE-IDENTICAL between the 3090 and 5060ti platform
// trees (no per-platform token substitution -- device identity is resolved
// entirely at runtime via NVML + the optional BENCH_EXPECTED_GPU
// environment variable, exactly like the GPU AXPY reference).
//
// Reference split, as instructed:
//   - GPU infrastructure (program structure, CLI, CUDA device selection,
//     PCI-ID/NVML device mapping, BENCH_EXPECTED_GPU, NVML energy, GPU
//     telemetry, timing order, 45-column CSV schema/formulas/sentinels/
//     serialization, session/run metadata, console format, error handling)
//     is taken from the current GPU AXPY implementation (main_axpy.cu),
//     which this file's macros, Telemetry/Options types, timestamp/
//     csv_escape helpers, and measurement-order skeleton are ported from
//     verbatim where the CONV2D contract does not require a difference.
//   - CONV2D semantics (six shapes, deterministic input/weight generation,
//     logical FLOP/byte formulas, the 32 geometrically-determined checksum
//     positions, the independent CPU-side reference and its tolerance
//     logic, and the anti-collapse probe's bounded-growth math) are taken
//     from the current CPU CONV2D implementation (main_conv2d_intel.cpp),
//     ported to host-callable, oneDNN-independent form.
//
// Explicit, intentional deviations from the AXPY GPU reference (per this
// task's CONV2D-specific instructions):
//   - Calibration, batch scaling, and runtime_status use e2e_time_s, not
//     kernel_time_s (AXPY uses kernel_time_s for calibration/scaling).
//     gflops_per_s and time_per_op_ms_kernel still use kernel_time_s,
//     unchanged from the AXPY CSV formula set.
//   - "below" is NOT a hard campaign-ending failure here. It triggers a
//     bounded below-retry (batch increase, remeasure the same repetition,
//     max 3 retries) instead of AXPY's hard fail. This replaces AXPY's
//     F3-style hard fail for this workload only, per explicit instruction.
//   - avg_power_w is defined here as device_energy_j / e2e_time_s
//     (numerically identical to AXPY's device_energy_j / wall_time_s,
//     since wall_time_s == e2e_time_s in both files -- restated explicitly
//     because this task's text calls it out by name).
//
// Algorithm/engine selection uses the cuDNN legacy convolution API
// (cudnnGetConvolutionForwardAlgorithm_v7 + cudnnConvolutionForward).
// NVIDIA's cuDNN compatibility documentation lists cudnnConvolutionForward
// as forward-compatible across 9.x releases; cudnnGetConvolutionForwardAlgorithm_v7
// is marked deprecated as of cuDNN 9.0 but remains present (confirmed
// against the published 9.1-9.5 API references at the time this file was
// written) -- no specific *older* selection function (the pre-v7,
// cuDNN-8-removed cudnnGetConvolutionForwardAlgorithm) is used, matching
// the instruction not to hard-code a specific old cuDNN selection function.
// A silent fallback between API generations is deliberately NOT
// implemented: if a future cuDNN major version removes this symbol, the
// build must fail loudly (a compile error), not silently degrade.

#include <cuda_runtime.h>
#include <cudnn.h>
#include <nvml.h>

#include <algorithm>
#include <array>
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
// Shared GPU-infrastructure constants (ported from main_axpy.cu).
// ---------------------------------------------------------------------

constexpr const char* SCHEMA_VERSION = "cpu-gpu-v2";
constexpr double TARGET_RUNTIME_S = 1.0;
constexpr double MIN_RUNTIME_S = 0.75;
constexpr double MAX_RUNTIME_S = 1.25;
constexpr int DEFAULT_REPETITIONS = 10;
constexpr int CUDA_DEVICE = 0;  // CUDA_VISIBLE_DEVICES maps the selected physical GPU here.
constexpr int BLOCK_SIZE = 256;
constexpr size_t VRAM_MIN_SAFETY_MARGIN_BYTES =
    static_cast<size_t>(512) * 1024 * 1024;

// ---------------------------------------------------------------------
// CONV2D-specific constants (ported from the CPU CONV2D reference, so
// that CPU and GPU CONV2D share identical batch-cap/anti-collapse
// semantics for the same workload).
// ---------------------------------------------------------------------

constexpr long long MAX_BATCHES = 100000;
constexpr int MAX_CALIBRATION_STEPS_CONV2D = 12;
constexpr int MAX_BELOW_RETRIES = 3;

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

#define CUDNN_CHECK(call) do { \
    const cudnnStatus_t status__ = (call); \
    if (status__ != CUDNN_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuDNN failure: ") + cudnnGetErrorString(status__) + \
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
// Overflow-safe 64-bit arithmetic (ported from CPU CONV2D; throws instead
// of std::exit so GPU resources unwind through the top-level catch/cleanup
// in main(), matching the AXPY GPU exception-based error-handling
// convention).
// ---------------------------------------------------------------------

std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b, const char* what) {
    std::uint64_t result = 0;
    if (__builtin_mul_overflow(a, b, &result)) {
        throw std::runtime_error(std::string("Integer overflow while computing ") + what);
    }
    return result;
}

std::uint64_t checked_add_u64(std::uint64_t a, std::uint64_t b, const char* what) {
    std::uint64_t result = 0;
    if (__builtin_add_overflow(a, b, &result)) {
        throw std::runtime_error(std::string("Integer overflow while computing ") + what);
    }
    return result;
}

long long checked_u64_to_long_long(std::uint64_t value, const char* what) {
    const auto max_value = static_cast<std::uint64_t>(std::numeric_limits<long long>::max());
    if (value > max_value) {
        throw std::runtime_error(std::string("Value exceeds signed 64-bit range while computing ") + what);
    }
    return static_cast<long long>(value);
}

// ---------------------------------------------------------------------
// CONV2D shapes (ported verbatim from CPU CONV2D -- same six shapes, same
// shape_ids, same geometry, so the GPU checksum positions and FLOP/byte
// metrics stay bit-for-bit comparable to the CPU campaign).
// ---------------------------------------------------------------------

struct Shape {
    int id;
    int n, c, h, w, k, r, s, stride, pad;

    [[nodiscard]] int h_out() const { return (h + 2 * pad - r) / stride + 1; }
    [[nodiscard]] int w_out() const { return (w + 2 * pad - s) / stride + 1; }

    void validate_geometry() const {
        if (h_out() <= 0 || w_out() <= 0) {
            throw std::runtime_error(
                "Invalid output geometry for shape_id=" + std::to_string(id) +
                " (Hout=" + std::to_string(h_out()) + ", Wout=" + std::to_string(w_out()) + ")");
        }
    }

    [[nodiscard]] std::uint64_t input_elements() const {
        std::uint64_t v = checked_mul_u64(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(c), "input_elements");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(h), "input_elements");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(w), "input_elements");
        return v;
    }
    [[nodiscard]] std::uint64_t weight_elements() const {
        std::uint64_t v = checked_mul_u64(static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(c), "weight_elements");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(r), "weight_elements");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(s), "weight_elements");
        return v;
    }
    [[nodiscard]] std::uint64_t output_elements() const {
        std::uint64_t v = checked_mul_u64(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k), "output_elements");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(h_out()), "output_elements");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(w_out()), "output_elements");
        return v;
    }
    [[nodiscard]] std::uint64_t flops_per_op() const {
        // 2 * N * K * C * R * S * Hout * Wout -- logical direct-convolution
        // equivalent, identical definition to CPU CONV2D.
        std::uint64_t v = checked_mul_u64(2ULL, static_cast<std::uint64_t>(n), "flops_per_op");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(k), "flops_per_op");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(c), "flops_per_op");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(r), "flops_per_op");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(s), "flops_per_op");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(h_out()), "flops_per_op");
        v = checked_mul_u64(v, static_cast<std::uint64_t>(w_out()), "flops_per_op");
        return v;
    }
    [[nodiscard]] std::uint64_t logical_bytes_per_op() const {
        std::uint64_t elems = checked_add_u64(input_elements(), weight_elements(), "logical_bytes_per_op");
        elems = checked_add_u64(elems, output_elements(), "logical_bytes_per_op");
        return checked_mul_u64(elems, static_cast<std::uint64_t>(sizeof(float)), "logical_bytes_per_op");
    }
};

const std::vector<Shape> SHAPES{
    {1, 32, 64, 56, 56, 64, 3, 3, 1, 1},
    {2, 32, 64, 56, 56, 128, 3, 3, 2, 1},
    {3, 32, 128, 28, 28, 256, 3, 3, 2, 1},
    {4, 32, 256, 14, 14, 512, 3, 3, 2, 1},
    {5, 32, 3, 224, 224, 64, 7, 7, 2, 3},
    {6, 32, 256, 56, 56, 256, 1, 1, 1, 0}
};

// ---------------------------------------------------------------------
// Deterministic FP32 inputs (identical formulas to CPU CONV2D, section
// 11), generated directly on the device via a dedicated kernel per
// buffer -- mirroring the AXPY GPU reference's init_x_kernel/
// reset_y_kernel pattern rather than the CPU version's host-side
// initialize_inputs() + H2D transfer, since host-to-device transfers are
// explicitly listed as unnecessary overhead to avoid outside the
// measurement window when device-side generation is just as simple.
// Host-callable equivalents (input_value/weight_value) are used ONLY by
// the independent CPU-side checksum reference below; they are never
// executed on the GPU compute path.
// ---------------------------------------------------------------------

__host__ __device__ inline float input_value(std::uint64_t i) {
    return -0.5f + static_cast<float>(i % 31) * 0.03125f;
}
__host__ __device__ inline float weight_value(std::uint64_t i) {
    return -0.25f + static_cast<float>(i % 23) * 0.015625f;
}

__global__ void init_input_kernel(float* __restrict__ input, std::uint64_t count) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) input[i] = input_value(i);
}
__global__ void init_weight_kernel(float* __restrict__ weights, std::uint64_t count) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) weights[i] = weight_value(i);
}
__global__ void zero_output_kernel(float* __restrict__ output, std::uint64_t count) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = 0.0f;
}

unsigned int grid_blocks_1d(std::uint64_t n) {
    const std::uint64_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks == 0 || blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("Invalid 1D grid size for N=" + std::to_string(n));
    }
    return static_cast<unsigned int>(blocks);
}

// ---------------------------------------------------------------------
// Independent CPU-side reference and the 32 geometrically-determined
// checksum positions (CODING-AUFTRAG CONV2D CPU sections 22-25, ported
// verbatim). Operates purely on the deterministic index->value formulas
// above -- no GPU compute, no oneDNN, no cuDNN warm-up result reused.
// ---------------------------------------------------------------------

struct SamplePosition {
    std::uint64_t flat_index;
    int n, k, oh, ow;
};

std::vector<SamplePosition> checksum_sample_positions(const Shape& shape) {
    constexpr int SAMPLE_COUNT = 32;
    const std::uint64_t output_elements = shape.output_elements();
    if (output_elements == 0) {
        throw std::runtime_error("Zero output elements for shape_id=" + std::to_string(shape.id));
    }

    std::vector<SamplePosition> positions;
    positions.reserve(SAMPLE_COUNT);
    const std::uint64_t hout = static_cast<std::uint64_t>(shape.h_out());
    const std::uint64_t wout = static_cast<std::uint64_t>(shape.w_out());
    const std::uint64_t kdim = static_cast<std::uint64_t>(shape.k);

    for (int j = 0; j < SAMPLE_COUNT; ++j) {
        const std::uint64_t numerator = checked_mul_u64(
            static_cast<std::uint64_t>(j), output_elements - 1, "checksum sample numerator");
        const std::uint64_t flat_index = numerator / 31;

        std::uint64_t remainder = flat_index;
        const std::uint64_t ow = remainder % wout;
        remainder /= wout;
        const std::uint64_t oh = remainder % hout;
        remainder /= hout;
        const std::uint64_t kk = remainder % kdim;
        remainder /= kdim;
        const std::uint64_t nn = remainder;

        positions.push_back(SamplePosition{
            flat_index,
            static_cast<int>(nn), static_cast<int>(kk),
            static_cast<int>(oh), static_cast<int>(ow)});
    }
    return positions;
}

long double reference_value(const Shape& shape, int n, int k, int oh, int ow) {
    long double sum = 0.0L;
    for (int ci = 0; ci < shape.c; ++ci) {
        for (int rr = 0; rr < shape.r; ++rr) {
            const int ih = oh * shape.stride - shape.pad + rr;
            if (ih < 0 || ih >= shape.h) continue;
            for (int ss = 0; ss < shape.s; ++ss) {
                const int iw = ow * shape.stride - shape.pad + ss;
                if (iw < 0 || iw >= shape.w) continue;

                const std::uint64_t input_index =
                    ((static_cast<std::uint64_t>(n) * shape.c + ci) * shape.h + ih) * shape.w + iw;
                const std::uint64_t weight_index =
                    ((static_cast<std::uint64_t>(k) * shape.c + ci) * shape.r + rr) * shape.s + ss;

                // Pure host-side deterministic formula -- NOT a read from any
                // GPU buffer, NOT a reuse of a cuDNN result.
                sum += static_cast<long double>(input_value(input_index)) *
                       static_cast<long double>(weight_value(weight_index));
            }
        }
    }
    return sum;
}

constexpr long double MAX_ABS_ERROR = 1.0e-3L;
constexpr long double MAX_REL_ERROR = 1.0e-4L;
constexpr long double RELATIVE_ERROR_FLOOR = 1.0e-30L;

struct ChecksumDiagnostics {
    bool ok{false};
    long double max_abs_error{0.0L};
    long double max_rel_error{0.0L};
    long double max_normalized_error{0.0L};
    long double worst_reference{0.0L};
    float worst_actual{0.0f};
    int nonfinite_count{0};
    std::uint64_t worst_flat_index{0};
    int worst_n{0}, worst_k{0}, worst_oh{0}, worst_ow{0};
};

void print_checksum_diagnostics(const Shape& shape, int repetition, const ChecksumDiagnostics& diag) {
    std::cout << "[CHECKSUM] shape=" << shape.id
              << " rep=" << repetition
              << " samples=32"
              << std::scientific << std::setprecision(6)
              << " max_abs_error=" << diag.max_abs_error
              << " max_rel_error=" << diag.max_rel_error
              << " max_normalized_error=" << diag.max_normalized_error
              << " reference=" << diag.worst_reference
              << " actual=" << static_cast<long double>(diag.worst_actual)
              << std::defaultfloat
              << " nonfinite=" << diag.nonfinite_count
              << " worst_flat_index=" << diag.worst_flat_index
              << " worst_n=" << diag.worst_n
              << " worst_k=" << diag.worst_k
              << " worst_oh=" << diag.worst_oh
              << " worst_ow=" << diag.worst_ow
              << " gate=" << (diag.ok ? "PASS" : "FAIL")
              << '\n';
}

// ---------------------------------------------------------------------
// cuDNN forward-convolution runner. Algorithm/engine selection happens
// exactly once per shape, inside prepare(), outside every timing/energy
// window; run_batches() reuses that selection unconditionally for
// warm-up, calibration, below-retries, official measurement, and the
// anti-collapse probe alike (CODING-AUFTRAG algorithm-selection rules
// 1-5).
// ---------------------------------------------------------------------

class ConvRunner {
public:
    explicit ConvRunner(const Shape& shape) : shape_(shape) {}

    ~ConvRunner() {
        if (workspace_) cudaFree(workspace_);
        if (output_) cudaFree(output_);
        if (weights_) cudaFree(weights_);
        if (input_) cudaFree(input_);
        if (y_desc_) cudnnDestroyTensorDescriptor(y_desc_);
        if (conv_desc_) cudnnDestroyConvolutionDescriptor(conv_desc_);
        if (w_desc_) cudnnDestroyFilterDescriptor(w_desc_);
        if (x_desc_) cudnnDestroyTensorDescriptor(x_desc_);
        if (handle_) cudnnDestroy(handle_);
    }

    ConvRunner(const ConvRunner&) = delete;
    ConvRunner& operator=(const ConvRunner&) = delete;

    void prepare(cudaStream_t stream) {
        stream_ = stream;
        shape_.validate_geometry();

        CUDNN_CHECK(cudnnCreate(&handle_));
        CUDNN_CHECK(cudnnSetStream(handle_, stream_));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            x_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            shape_.n, shape_.c, shape_.h, shape_.w));

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc_));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            w_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            shape_.k, shape_.c, shape_.r, shape_.s));

        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc_, shape_.pad, shape_.pad, shape_.stride, shape_.stride,
            /*dilation_h=*/1, /*dilation_w=*/1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        // No TF32, no FP16/BF16, no Tensor-Core reduced-precision paths:
        // restricted to FMA-instruction kernels only.
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc_, CUDNN_FMA_MATH));

        int out_n = 0, out_c = 0, out_h = 0, out_w = 0;
        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
            conv_desc_, x_desc_, w_desc_, &out_n, &out_c, &out_h, &out_w));
        if (out_n != shape_.n || out_c != shape_.k ||
            out_h != shape_.h_out() || out_w != shape_.w_out()) {
            throw std::runtime_error(
                "cuDNN output geometry mismatch for shape_id=" + std::to_string(shape_.id));
        }

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

        const std::uint64_t input_count = shape_.input_elements();
        const std::uint64_t weight_count = shape_.weight_elements();
        const std::uint64_t output_count = shape_.output_elements();

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&input_), input_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&weights_), weight_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output_), output_count * sizeof(float)));

        init_input_kernel<<<grid_blocks_1d(input_count), BLOCK_SIZE, 0, stream_>>>(input_, input_count);
        CUDA_CHECK(cudaGetLastError());
        init_weight_kernel<<<grid_blocks_1d(weight_count), BLOCK_SIZE, 0, stream_>>>(weights_, weight_count);
        CUDA_CHECK(cudaGetLastError());
        zero_output_kernel<<<grid_blocks_1d(output_count), BLOCK_SIZE, 0, stream_>>>(output_, output_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream_));  // first-touch/init fully outside any timing window

        select_algorithm();  // exactly once per shape, outside all timing/energy windows

        if (workspace_size_bytes_ > 0) {
            CUDA_CHECK(cudaMalloc(&workspace_, workspace_size_bytes_));
        } else {
            workspace_ = nullptr;  // zero-byte workspace handled explicitly, no allocation
        }
    }

    // The single, unconditional execute path: identical for warm-up,
    // calibration, below-retries, official measurement, and the
    // anti-collapse probe. One cuDNN forward call == one logical batch.
    void run_batches(std::uint64_t batches) const {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        for (std::uint64_t b = 0; b < batches; ++b) {
            CUDNN_CHECK(cudnnConvolutionForward(
                handle_, &alpha, x_desc_, input_, w_desc_, weights_, conv_desc_,
                algo_, workspace_, workspace_size_bytes_, &beta, y_desc_, output_));
        }
    }

    float read_output(std::uint64_t flat_index) const {
        float value = 0.0f;
        CUDA_CHECK(cudaMemcpy(&value, output_ + flat_index, sizeof(float), cudaMemcpyDeviceToHost));
        return value;
    }

    [[nodiscard]] const std::string& algo_name() const { return algo_name_; }
    [[nodiscard]] size_t workspace_size_bytes() const { return workspace_size_bytes_; }

private:
    void select_algorithm() {
        constexpr int kRequested = 8;
        int returned = 0;
        std::array<cudnnConvolutionFwdAlgoPerf_t, kRequested> results{};
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            handle_, x_desc_, w_desc_, conv_desc_, y_desc_,
            kRequested, &returned, results.data()));

        bool found = false;
        for (int i = 0; i < returned; ++i) {
            const cudnnConvolutionFwdAlgoPerf_t& candidate = results[i];
            if (candidate.status != CUDNN_STATUS_SUCCESS) continue;
            // Only accept FMA-restricted candidates: no TF32, no FP16/BF16,
            // no Tensor-Core reduced-precision math -- rule 9/10.
            if (candidate.mathType != CUDNN_FMA_MATH) continue;
            algo_ = candidate.algo;
            workspace_size_bytes_ = candidate.memory;
            found = true;
            break;
        }
        if (!found) {
            throw std::runtime_error(
                "No successful FP32/FMA-compatible cuDNN forward algorithm for shape_id=" +
                std::to_string(shape_.id));
        }

        // Defense in depth: cross-check against the dedicated workspace
        // query and take the larger of the two reported sizes.
        size_t queried_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            handle_, x_desc_, w_desc_, conv_desc_, y_desc_, algo_, &queried_size));
        workspace_size_bytes_ = std::max(workspace_size_bytes_, queried_size);

        algo_name_ = algo_to_string(algo_);
    }

    static std::string algo_to_string(cudnnConvolutionFwdAlgo_t algo) {
        switch (algo) {
            case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: return "implicit_gemm";
            case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return "implicit_precomp_gemm";
            case CUDNN_CONVOLUTION_FWD_ALGO_GEMM: return "gemm";
            case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: return "direct";
            case CUDNN_CONVOLUTION_FWD_ALGO_FFT: return "fft";
            case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: return "fft_tiling";
            case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: return "winograd";
            case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: return "winograd_nonfused";
            default: return "algo_" + std::to_string(static_cast<int>(algo));
        }
    }

    Shape shape_;
    cudaStream_t stream_{};
    cudnnHandle_t handle_{};
    cudnnTensorDescriptor_t x_desc_{};
    cudnnFilterDescriptor_t w_desc_{};
    cudnnTensorDescriptor_t y_desc_{};
    cudnnConvolutionDescriptor_t conv_desc_{};
    float* input_{nullptr};
    float* weights_{nullptr};
    float* output_{nullptr};
    void* workspace_{nullptr};
    size_t workspace_size_bytes_{0};
    cudnnConvolutionFwdAlgo_t algo_{};
    std::string algo_name_;
};

ChecksumDiagnostics verify_checksum(const Shape& shape, const ConvRunner& runner) {
    ChecksumDiagnostics diag;
    diag.ok = true;

    for (const SamplePosition& pos : checksum_sample_positions(shape)) {
        const float actual = runner.read_output(pos.flat_index);
        const long double reference = reference_value(shape, pos.n, pos.k, pos.oh, pos.ow);

        const bool finite = std::isfinite(actual) && std::isfinite(static_cast<double>(reference));
        if (!finite) {
            ++diag.nonfinite_count;
            diag.ok = false;
            if (diag.nonfinite_count == 1) {
                const long double infinity = std::numeric_limits<long double>::infinity();
                diag.max_abs_error = infinity;
                diag.max_rel_error = infinity;
                diag.max_normalized_error = infinity;
                diag.worst_reference = reference;
                diag.worst_actual = actual;
                diag.worst_flat_index = pos.flat_index;
                diag.worst_n = pos.n; diag.worst_k = pos.k;
                diag.worst_oh = pos.oh; diag.worst_ow = pos.ow;
            }
            continue;
        }

        const long double abs_error = std::abs(static_cast<long double>(actual) - reference);
        const long double relative_error = abs_error / std::max(RELATIVE_ERROR_FLOOR, std::abs(reference));
        const long double allowed_error = std::max(MAX_ABS_ERROR, MAX_REL_ERROR * std::max(1.0L, std::abs(reference)));
        const long double normalized_error = abs_error / allowed_error;
        const bool sample_ok = abs_error <= allowed_error;

        if (abs_error > diag.max_abs_error) diag.max_abs_error = abs_error;
        if (relative_error > diag.max_rel_error) diag.max_rel_error = relative_error;
        if (normalized_error > diag.max_normalized_error) {
            diag.max_normalized_error = normalized_error;
            diag.worst_reference = reference;
            diag.worst_actual = actual;
            diag.worst_flat_index = pos.flat_index;
            diag.worst_n = pos.n; diag.worst_k = pos.k;
            diag.worst_oh = pos.oh; diag.worst_ow = pos.ow;
        }
        if (!sample_ok) diag.ok = false;
    }
    return diag;
}

// ---------------------------------------------------------------------
// Host utilities (ported from main_axpy.cu).
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
    options.output_file = argc > 1 ? argv[1] : "conv2d_gpu.csv";
    if (argc > 2) options.repetitions = std::max(1, std::stoi(argv[2]));
    if (argc > 3) options.session_id = argv[3];
    if (argc > 4) options.seed = static_cast<uint32_t>(std::stoul(argv[4]));
    return options;
}

std::vector<int> parse_int_filter(const char* name) {
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

bool env_flag_set(const char* name) {
    const char* raw = std::getenv(name);
    return raw && std::string(raw) == "1";
}

// runtime_status is bound to e2e_time_s for CONV2D (explicit deviation --
// see header comment), using the identical below/in_range/above bounds as
// the GPU AXPY reference.
std::string runtime_status(double seconds) {
    if (seconds < MIN_RUNTIME_S) return "below";
    if (seconds > MAX_RUNTIME_S) return "above";
    return "in_range";
}

// Exact contract-style batch scaling formula (no artificial per-step growth
// cap), reused for both calibration and below-retry batch increases.
long long conv2d_scale_batches(double measured_seconds, long long current, long long max_batches) {
    const double safe_seconds = std::max(measured_seconds, 1.0e-9);
    const long long estimate = static_cast<long long>(
        std::ceil(TARGET_RUNTIME_S * static_cast<double>(current) / safe_seconds));
    const long long next = std::max<long long>(current + 1, estimate);
    return std::min<long long>(max_batches, next);
}

// ---------------------------------------------------------------------
// NVML / device binding (ported verbatim from main_axpy.cu).
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
            "Do not mix this run with a power-sampling fallback.");
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

std::string throttle_hex(unsigned long long reasons) {
    std::ostringstream out;
    out << "0x" << std::hex << std::uppercase << reasons;
    return out.str();
}

// ---------------------------------------------------------------------
// CSV row (schema_version cpu-gpu-v2, exact 45-column schema/formulas/
// sentinels/serialization ported from main_axpy.cu).
// ---------------------------------------------------------------------

struct Conv2dRow {
    std::string session_id;
    long long sequence_index{};
    long long run_id_global{};
    int repetition{};
    std::string device_name;
    int problem_size{};  // shape_id
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

void write_conv2d_header(std::ofstream& output) {
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

void write_lossless_double(std::ofstream& output, double value) {
    output << std::defaultfloat << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
}

void write_conv2d_row(std::ofstream& output, const Conv2dRow& row) {
    output << SCHEMA_VERSION << ',' << timestamp() << ','
           << csv_escape(row.session_id) << ','
           << row.sequence_index << ',' << row.run_id_global << ','
           << row.repetition << ','
           << csv_escape("CONV2D") << ','
           << csv_escape("cudnn_convolution_fwd_fp32") << ','
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
           << std::defaultfloat << ',' << -1 << ','   // cpu_cycles/instructions/ipc/cache_misses
           << (row.checksum_ok ? 't' : 'f') << '\n';
}

void print_conv2d_result(const Conv2dRow& row) {
    std::cout << "[CONV2D] shape=" << row.problem_size
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

std::string build_problem_spec(const Shape& shape, const ConvRunner& runner) {
    std::ostringstream out;
    out << "shape_id=" << shape.id
        << ";N=" << shape.n << ";C=" << shape.c << ";H=" << shape.h << ";W=" << shape.w
        << ";K=" << shape.k << ";R=" << shape.r << ";S=" << shape.s
        << ";stride=" << shape.stride << ";pad=" << shape.pad
        << ";Hout=" << shape.h_out() << ";Wout=" << shape.w_out()
        << ";layout=NCHW;conv=cross_correlation;dtype=f32;math=FMA"
        << ";algo=" << runner.algo_name()
        << ";workspace_bytes=" << runner.workspace_size_bytes();
    return out.str();
}

// ---------------------------------------------------------------------
// Calibration (e2e_time_s basis -- explicit CONV2D deviation from AXPY).
// ---------------------------------------------------------------------

struct CalibrationResult {
    long long batches;
    double seconds;
};

CalibrationResult calibrate(ConvRunner& runner, cudaStream_t stream) {
    runner.run_batches(1);  // warm-up, unmeasured, same execute path
    CUDA_CHECK(cudaStreamSynchronize(stream));

    long long batches = 1;
    for (int step = 0; step < MAX_CALIBRATION_STEPS_CONV2D; ++step) {
        const auto start = std::chrono::steady_clock::now();
        runner.run_batches(static_cast<std::uint64_t>(batches));
        CUDA_CHECK(cudaStreamSynchronize(stream));  // e2e basis: wait for completion before stopping the clock
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();

        if (!std::isfinite(seconds) || seconds <= 0.0) {
            throw std::runtime_error("Non-finite or non-positive calibration time");
        }
        if (batches == MAX_BATCHES) {
            if (seconds < MIN_RUNTIME_S) {
                throw std::runtime_error(
                    "Unable to reach minimum runtime at maximum batch count during calibration "
                    "(batches=" + std::to_string(batches) + ", seconds=" + std::to_string(seconds) + ")");
            }
            return {batches, seconds};
        }
        if (seconds >= TARGET_RUNTIME_S) {
            return {batches, seconds};
        }
        batches = conv2d_scale_batches(seconds, batches, MAX_BATCHES);
    }
    throw std::runtime_error(
        "Calibration did not converge within " + std::to_string(MAX_CALIBRATION_STEPS_CONV2D) + " steps");
}

// ---------------------------------------------------------------------
// Anti-collapse probe (opt-in via CONV2D_ANTI_COLLAPSE_PROBE=1). shape_id
// == 1 only. Never touches the CSV file in any way -- the caller in
// main() branches before Options/output-file handling exist at all.
// e2e_time_s basis, matching this file's calibration deviation. Bounded
// growth (hard error at the cap, strict-progress assertion), reusing the
// same hardened pattern as the CPU CONV2D anti-collapse probe.
// ---------------------------------------------------------------------

void run_anti_collapse_probe(nvmlDevice_t nvml_device, cudaStream_t stream) {
    const Shape& shape = SHAPES.front();  // shape_id == 1

    ConvRunner runner(shape);
    runner.prepare(stream);  // algorithm selected exactly once, reused below

    runner.run_batches(1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const ChecksumDiagnostics warmup_diag = verify_checksum(shape, runner);
    if (!warmup_diag.ok) {
        print_checksum_diagnostics(shape, 0, warmup_diag);
        throw std::runtime_error("Anti-collapse probe: warm-up checksum failed");
    }

    const CalibrationResult calibration = calibrate(runner, stream);
    const long long b_cal = calibration.batches;

    long long b_probe = std::min<long long>(std::max<long long>(100, b_cal / 4), MAX_BATCHES / 2);
    const long long max_probe = MAX_BATCHES / 2;

    auto measure = [&](long long batches) {
        const auto start = std::chrono::steady_clock::now();
        runner.run_batches(static_cast<std::uint64_t>(batches));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        const ChecksumDiagnostics diag = verify_checksum(shape, runner);
        return std::make_pair(seconds, diag);
    };

    double t1 = 0.0;
    ChecksumDiagnostics d1;
    for (;;) {
        std::tie(t1, d1) = measure(b_probe);
        if (t1 >= 0.020) break;
        if (b_probe >= max_probe) {
            throw std::runtime_error("Anti-collapse probe cannot reach minimum duration before batch cap");
        }
        const long long next = std::min<long long>(b_probe * 2, max_probe);
        if (next <= b_probe) {
            throw std::runtime_error("Anti-collapse probe made no forward progress");
        }
        b_probe = next;
    }

    const long long two_b_probe = 2 * b_probe;
    const auto [t2, d2] = measure(two_b_probe);

    const bool duration_ok = (t1 >= 0.020) && (t2 >= 0.020);
    const bool checksum_ok = d1.ok && d2.ok;
    const double ratio = t1 > 0.0 ? t2 / t1 : std::numeric_limits<double>::infinity();
    const bool ratio_ok = std::isfinite(ratio) && ratio >= 1.7 && ratio <= 2.3;
    const bool gate_pass = duration_ok && checksum_ok && ratio_ok;

    (void)nvml_device;
    std::cout << std::setprecision(9)
              << "[ANTI_COLLAPSE] shape=1"
              << " B=" << b_probe
              << " two_B=" << two_b_probe
              << " t1=" << t1
              << " t2=" << t2
              << " ratio=" << ratio
              << " time_basis=e2e_time_s"
              << " checksum_B=" << (d1.ok ? "PASS" : "FAIL")
              << " checksum_2B=" << (d2.ok ? "PASS" : "FAIL")
              << " gate=" << (gate_pass ? "PASS" : "FAIL")
              << '\n';

    if (!gate_pass) {
        throw std::runtime_error("Anti-collapse gate FAILED");
    }
}

}  // namespace

int main(int argc, char** argv) {
    cudaStream_t stream{};
    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    bool nvml_initialized = false;

    try {
        CUDA_CHECK(cudaSetDevice(CUDA_DEVICE));

        NVML_CHECK(nvmlInit_v2());
        nvml_initialized = true;
        nvmlDevice_t nvml_device = nvml_handle_for_cuda_device();
        const std::string device_name = gpu_name(nvml_device);
        require_expected_gpu(device_name);  // hard abort on wrong GPU

        (void)read_energy_mj(nvml_device);  // fail fast if the energy counter is unavailable

        CUDA_CHECK(cudaStreamCreate(&stream));

        // -----------------------------------------------------------------
        // Anti-collapse probe branch: exclusive, opt-in, and must never
        // create, open, truncate, or write the CSV file. Options/argv[1]
        // are never touched in this branch.
        // -----------------------------------------------------------------
        if (env_flag_set("CONV2D_ANTI_COLLAPSE_PROBE")) {
            run_anti_collapse_probe(nvml_device, stream);
            CUDA_CHECK(cudaStreamDestroy(stream));
            stream = nullptr;
            NVML_CHECK(nvmlShutdown());
            nvml_initialized = false;
            return 0;
        }

        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));

        const Options options = parse_options(argc, argv);
        const auto parent = std::filesystem::path(options.output_file).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        int cuda_runtime_version = 0, cuda_driver_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));
        CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));
        size_t cudnn_version = cudnnGetVersion();

        char pci_bus_id[32]{};
        CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, static_cast<int>(sizeof(pci_bus_id)), CUDA_DEVICE));

        std::cout << "CONV2D | " << device_name
                  << " | pci_bus_id=" << pci_bus_id
                  << " | implementation=cudnn_convolution_fwd_fp32"
                  << " | session=" << options.session_id
                  << " | reps=" << options.repetitions
                  << " | mode=gpu_resident"
                  << " | cudnn_version=" << cudnn_version
                  << " | CUDA runtime=" << cuda_runtime_version
                  << " | CUDA driver=" << cuda_driver_version
                  << '\n';

        const std::vector<int> shape_filter = parse_int_filter("BENCH_SIZE_FILTER");
        std::vector<int> shape_ids;
        for (const Shape& shape : SHAPES) {
            if (selected(shape.id, shape_filter)) shape_ids.push_back(shape.id);
        }
        if (shape_ids.empty()) {
            throw std::runtime_error("No shapes remain after BENCH_SIZE_FILTER");
        }
        std::mt19937 generator(options.seed);
        std::shuffle(shape_ids.begin(), shape_ids.end(), generator);

        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Cannot open output file: " + options.output_file);
        }
        write_conv2d_header(output);

        long long sequence = 0;

        for (const int shape_id : shape_ids) {
            const Shape& shape = SHAPES[static_cast<std::size_t>(shape_id - 1)];

            ConvRunner runner(shape);
            runner.prepare(stream);  // algorithm selected exactly once for this shape

            // Unmeasured validation call before calibration (algorithm-selection rule 8).
            runner.run_batches(1);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const ChecksumDiagnostics warmup_diag = verify_checksum(shape, runner);
            if (!warmup_diag.ok) {
                print_checksum_diagnostics(shape, 0, warmup_diag);
                throw std::runtime_error(
                    "CONV2D warm-up checksum failed for shape_id=" + std::to_string(shape.id) +
                    " (algo=" + runner.algo_name() + ")");
            }

            const CalibrationResult calibration = calibrate(runner, stream);
            long long batches = calibration.batches;
            std::cout << "[CALIBRATION] shape=" << shape.id << " batches=" << batches
                      << " runtime_s=" << calibration.seconds
                      << " status=" << runtime_status(calibration.seconds) << '\n';

            int completed = 0;
            while (completed < options.repetitions) {
                int below_retries = 0;
                for (;;) {
                    const Telemetry before = read_telemetry(nvml_device);
                    const unsigned long long energy_before = read_energy_mj(nvml_device);
                    const auto wall_start = std::chrono::steady_clock::now();

                    CUDA_CHECK(cudaEventRecord(start_event, stream));
                    runner.run_batches(static_cast<std::uint64_t>(batches));
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
                    const double e2e_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
                    const double energy_j = static_cast<double>(energy_after - energy_before) / 1000.0;

                    if (!std::isfinite(kernel_seconds) || !std::isfinite(e2e_seconds) ||
                        !std::isfinite(energy_j) || kernel_seconds <= 0.0 ||
                        e2e_seconds <= 0.0 || energy_j <= 0.0) {
                        throw std::runtime_error(
                            "Non-finite or non-positive measurement for shape_id=" +
                            std::to_string(shape.id) + ", repetition=" + std::to_string(completed + 1) +
                            ": kernel_time_s=" + std::to_string(kernel_seconds) +
                            ", e2e_time_s=" + std::to_string(e2e_seconds) +
                            ", device_energy_j=" + std::to_string(energy_j));
                    }

                    const std::string status = runtime_status(e2e_seconds);  // e2e basis (CONV2D deviation)

                    if (status == "below") {
                        if (below_retries >= MAX_BELOW_RETRIES) {
                            throw std::runtime_error(
                                "runtime_status=below after " + std::to_string(MAX_BELOW_RETRIES) +
                                " retries for shape_id=" + std::to_string(shape.id) +
                                ", batches=" + std::to_string(batches) +
                                ", e2e_time_s=" + std::to_string(e2e_seconds));
                        }
                        const long long increased = conv2d_scale_batches(e2e_seconds, batches, MAX_BATCHES);
                        if (increased <= batches) {
                            throw std::runtime_error(
                                "runtime_status=below and batches cannot be increased further "
                                "for shape_id=" + std::to_string(shape.id) +
                                " (batches=" + std::to_string(batches) + ")");
                        }
                        std::cout << "[CALIBRATION] shape=" << shape.id
                                  << " below_retry=" << (below_retries + 1)
                                  << " old_batches=" << batches
                                  << " new_batches=" << increased
                                  << " observed_e2e_time_s=" << e2e_seconds << '\n';
                        batches = increased;  // persists for this and all following reps of this shape
                        ++below_retries;
                        continue;  // remeasure the same repetition
                    }

                    if (status == "above") {
                        std::cout << "[WARNING] runtime_status=above shape=" << shape.id
                                   << " e2e_time_s=" << e2e_seconds << '\n';
                    }

                    const ChecksumDiagnostics diag = verify_checksum(shape, runner);
                    if (!diag.ok) {
                        print_checksum_diagnostics(shape, completed + 1, diag);
                        throw std::runtime_error(
                            "CONV2D checksum failed for shape_id=" + std::to_string(shape.id) +
                            ", repetition=" + std::to_string(completed + 1) +
                            " (algo=" + runner.algo_name() + ")");
                    }

                    const std::uint64_t flops_total_u64 = checked_mul_u64(
                        shape.flops_per_op(), static_cast<std::uint64_t>(batches), "flops_total");

                    Conv2dRow row;
                    row.session_id = options.session_id;
                    row.sequence_index = ++sequence;
                    row.run_id_global = row.sequence_index;
                    row.repetition = completed + 1;
                    row.device_name = device_name;
                    row.problem_size = shape.id;
                    row.problem_spec = build_problem_spec(shape, runner);
                    row.batches = batches;
                    row.e2e_time_s = e2e_seconds;
                    row.kernel_time_s = kernel_seconds;
                    row.wall_time_s = e2e_seconds;
                    row.device_energy_j = energy_j;
                    row.total_energy_j = energy_j;   // same NVML delta, no separate DRAM domain
                    row.dram_energy_j = -1.0;
                    row.energy_per_op_j = energy_j / static_cast<double>(batches);
                    row.energy_per_second_j = energy_j / row.wall_time_s;
                    row.energy_per_flop_j = energy_j / static_cast<double>(flops_total_u64);
                    row.time_per_op_ms_kernel = 1000.0 * kernel_seconds / static_cast<double>(batches);
                    row.time_per_op_ms_e2e = 1000.0 * e2e_seconds / static_cast<double>(batches);
                    row.flops_total = checked_u64_to_long_long(flops_total_u64, "flops_total");
                    row.gflops_per_s = static_cast<double>(flops_total_u64) / kernel_seconds / 1.0e9;
                    row.logical_bytes_per_op = checked_u64_to_long_long(
                        shape.logical_bytes_per_op(), "logical_bytes_per_op");
                    row.avg_power_w = energy_j / e2e_seconds;  // explicit CONV2D formula (== wall_time_s basis)
                    row.runtime_status = status;
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
                    row.checksum_ok = diag.ok;

                    write_conv2d_row(output, row);
                    output.flush();
                    if (!output.good()) {
                        throw std::runtime_error(
                            "CSV write failed for shape_id=" + std::to_string(shape.id) +
                            ", repetition=" + std::to_string(row.repetition) +
                            ", sequence=" + std::to_string(row.sequence_index));
                    }
                    print_conv2d_result(row);

                    ++completed;
                    break;  // proceed to the next repetition
                }
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
