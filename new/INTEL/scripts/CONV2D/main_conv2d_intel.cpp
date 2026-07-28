// CONV2D CPU benchmark (INTEL) -- oneDNN convolution_auto, FP32 forward
// inference, NCHW/OIHW, user-managed scratchpad, direct dnnl_primitive_execute
// C-API hot path.
//
// Scope of this file: measurement program only (per FINALER CODING-AUFTRAG
// v4). No runner, no quickcheck, no GPU code is created or implied here.
//
// This file must remain semantically identical to the AMD sibling after
// normalizing: platform label, default output filename, and the platform's
// benchmark_common.hpp include path (both use bench::THREAD_COUNTS from
// their own already-existing common header; no local thread list is
// defined here).

#include "benchmark_common.hpp"

#include <omp.h>

// ---------------------------------------------------------------------
// Mandatory public oneDNN configuration header + OpenMP threading-runtime
// compile-time gate (CODING-AUFTRAG section 6). No internal oneDNN headers,
// no private symbols.
// ---------------------------------------------------------------------
#if __has_include(<oneapi/dnnl/dnnl_config.h>)
#include <oneapi/dnnl/dnnl_config.h>
#elif __has_include(<dnnl_config.h>)
#include <dnnl_config.h>
#else
#error "Public oneDNN configuration header not found."
#endif

#ifndef DNNL_CPU_THREADING_RUNTIME
#error "Cannot determine oneDNN CPU threading runtime."
#endif

#if DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_OMP
#error "Official CONV2D benchmark requires oneDNN OpenMP threading runtime."
#endif

// oneDNN is mandatory; there is no naive fallback in the official benchmark
// (CODING-AUFTRAG section 5).
#ifdef USE_NAIVE_CONV
#error "Official CONV2D benchmark requires oneDNN; USE_NAIVE_CONV is forbidden."
#endif

#if __has_include(<dnnl.hpp>)
#include <dnnl.hpp>
#elif __has_include(<oneapi/dnnl/dnnl.hpp>)
#include <oneapi/dnnl/dnnl.hpp>
#else
#error "oneDNN headers not found. Install libdnnl-dev."
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

// =======================================================================
// Overflow-safe 64-bit arithmetic (CODING-AUFTRAG section 8.3)
// =======================================================================
std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b, const char* what) {
    std::uint64_t result = 0;
    if (__builtin_mul_overflow(a, b, &result)) {
        std::cerr << "FATAL: integer overflow while computing " << what << '\n';
        std::exit(1);
    }
    return result;
}

std::uint64_t checked_add_u64(std::uint64_t a, std::uint64_t b, const char* what) {
    std::uint64_t result = 0;
    if (__builtin_add_overflow(a, b, &result)) {
        std::cerr << "FATAL: integer overflow while computing " << what << '\n';
        std::exit(1);
    }
    return result;
}


long long checked_u64_to_long_long(std::uint64_t value, const char* what) {
    const auto max_value =
        static_cast<std::uint64_t>(std::numeric_limits<long long>::max());
    if (value > max_value) {
        std::cerr << "FATAL: value exceeds signed 64-bit range while computing "
                  << what << '\n';
        std::exit(1);
    }
    return static_cast<long long>(value);
}

template <typename Sequence>
std::string format_integer_sequence(const Sequence& values) {
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) out << ':';
        out << values[index];
    }
    out << ']';
    return out.str();
}

std::string describe_memory_desc(const dnnl::memory::desc& desc) {
    std::ostringstream out;
    out << "dims=" << format_integer_sequence(desc.get_dims())
        << ";dtype=" << static_cast<int>(desc.get_data_type())
        << ";format_kind=" << static_cast<int>(desc.get_format_kind())
        << ";size_bytes=" << desc.get_size();

    if (desc.get_format_kind() == dnnl::memory::format_kind::blocked) {
        out << ";strides=" << format_integer_sequence(desc.get_strides())
            << ";inner_nblks=" << desc.get_inner_nblks()
            << ";inner_blks=" << format_integer_sequence(desc.get_inner_blks())
            << ";inner_idxs=" << format_integer_sequence(desc.get_inner_idxs());
    }
    return out.str();
}

// =======================================================================
// Frozen shapes (CODING-AUFTRAG section 7)
// =======================================================================
struct Shape {
    int id;
    int n, c, h, w, k, r, s, stride, pad;

    [[nodiscard]] int h_out() const { return (h + 2 * pad - r) / stride + 1; }
    [[nodiscard]] int w_out() const { return (w + 2 * pad - s) / stride + 1; }

    void validate_geometry() const {
        if (h_out() <= 0 || w_out() <= 0) {
            std::cerr << "FATAL: invalid output geometry for shape_id=" << id
                       << " (Hout=" << h_out() << ", Wout=" << w_out() << ")\n";
            std::exit(1);
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
        // 2 * N * K * C * R * S * Hout * Wout
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

    [[nodiscard]] std::string problem_spec() const {
        std::ostringstream out;
        out << "N=" << n << ";C=" << c << ";H=" << h << ";W=" << w
            << ";K=" << k << ";R=" << r << ";S=" << s
            << ";stride=" << stride << ";pad=" << pad
            << ";Hout=" << h_out() << ";Wout=" << w_out()
            << ";dtype=f32;input_layout=NCHW;weight_layout=OIHW;output_layout=NCHW"
            << ";bias=none;activation=none;groups=1;dilation=1"
            << ";algorithm_policy=convolution_auto;output=overwrite"
            << ";reuse_regime=warm_resident;scratchpad=user"
            << ";onednn_cpu_runtime=OpenMP";
        return out.str();
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

constexpr int MAX_BATCHES = 100000;

struct Config {
    std::size_t shape_index;
    int threads;
};

// =======================================================================
// OpenMP thread guard (CODING-AUFTRAG section 12)
// =======================================================================
int active_openmp_threads() {
    int active = 0;
#pragma omp parallel
    {
#pragma omp single
        active = omp_get_num_threads();
    }
    return active;
}

void require_thread_team(int requested, const char* phase) {
    const int observed = active_openmp_threads();
    if (observed != requested) {
        std::cerr << "FATAL: requested OpenMP threads=" << requested
                   << ", observed=" << observed << ", phase=" << phase << '\n';
        std::exit(1);
    }
}

// =======================================================================
// Deterministic FP32 inputs (CODING-AUFTRAG section 11), fully outside the
// measurement window.
// =======================================================================
void initialize_inputs(float* input, std::uint64_t input_count,
                        float* weights, std::uint64_t weight_count,
                        float* output, std::uint64_t output_count) {
#pragma omp parallel for schedule(static)
    for (std::uint64_t i = 0; i < input_count; ++i) {
        input[i] = -0.5f + static_cast<float>(i % 31) * 0.03125f;
    }
#pragma omp parallel for schedule(static)
    for (std::uint64_t i = 0; i < weight_count; ++i) {
        weights[i] = -0.25f + static_cast<float>(i % 23) * 0.015625f;
    }
#pragma omp parallel for schedule(static)
    for (std::uint64_t i = 0; i < output_count; ++i) {
        output[i] = 0.0f;
    }
}

// =======================================================================
// Independent mathematical reference (CODING-AUFTRAG section 22-24). Does
// NOT use oneDNN, a oneDNN warm-up result, or any other convolution
// library. Operates directly on the logical NCHW/OIHW host buffers.
// =======================================================================
struct SamplePosition {
    std::uint64_t flat_index;
    int n, k, oh, ow;
};

std::vector<SamplePosition> checksum_sample_positions(const Shape& shape) {
    constexpr int SAMPLE_COUNT = 32;
    const std::uint64_t output_elements = shape.output_elements();
    if (output_elements == 0) {
        std::cerr << "FATAL: zero output elements for shape_id=" << shape.id << '\n';
        std::exit(1);
    }

    std::vector<SamplePosition> positions;
    positions.reserve(SAMPLE_COUNT);
    const std::uint64_t hout = static_cast<std::uint64_t>(shape.h_out());
    const std::uint64_t wout = static_cast<std::uint64_t>(shape.w_out());
    const std::uint64_t kdim = static_cast<std::uint64_t>(shape.k);

    for (int j = 0; j < SAMPLE_COUNT; ++j) {
        const std::uint64_t numerator = checked_mul_u64(
            static_cast<std::uint64_t>(j), output_elements - 1,
            "checksum sample numerator");
        const std::uint64_t flat_index = numerator / 31;

        // Decompose flat_index (logical NCHW output layout) into n,k,oh,ow.
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

long double reference_value(
    const Shape& shape, const float* input, const float* weights,
    int n, int k, int oh, int ow
) {
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

                sum += static_cast<long double>(input[input_index]) *
                       static_cast<long double>(weights[weight_index]);
            }
        }
    }
    return sum;
}

// =======================================================================
// Checksum tolerance (CODING-AUFTRAG section 25)
// =======================================================================
constexpr long double MAX_ABS_ERROR = 1.0e-3L;
constexpr long double MAX_REL_ERROR = 1.0e-4L;
constexpr long double RELATIVE_ERROR_FLOOR = 1.0e-30L;

struct ChecksumDiagnostics {
    bool ok{false};
    long double max_abs_error{0.0L};
    long double max_rel_error{0.0L};
    long double max_normalized_error{0.0L};
    int nonfinite_count{0};
    std::uint64_t worst_flat_index{0};
    int worst_n{0}, worst_k{0}, worst_oh{0}, worst_ow{0};
};

ChecksumDiagnostics verify_checksum(
    const Shape& shape, const float* input, const float* weights, const float* output
) {
    ChecksumDiagnostics diag;
    diag.ok = true;
    const std::uint64_t wout = static_cast<std::uint64_t>(shape.w_out());
    const std::uint64_t hout = static_cast<std::uint64_t>(shape.h_out());
    const std::uint64_t kdim = static_cast<std::uint64_t>(shape.k);

    for (const SamplePosition& pos : checksum_sample_positions(shape)) {
        const float actual = output[pos.flat_index];
        const long double reference =
            reference_value(shape, input, weights, pos.n, pos.k, pos.oh, pos.ow);

        const bool finite = std::isfinite(actual) && std::isfinite(static_cast<double>(reference));
        if (!finite) {
            ++diag.nonfinite_count;
            diag.ok = false;
            if (diag.nonfinite_count == 1) {
                const long double infinity =
                    std::numeric_limits<long double>::infinity();
                diag.max_abs_error = infinity;
                diag.max_rel_error = infinity;
                diag.max_normalized_error = infinity;
                diag.worst_flat_index = pos.flat_index;
                diag.worst_n = pos.n;
                diag.worst_k = pos.k;
                diag.worst_oh = pos.oh;
                diag.worst_ow = pos.ow;
            }
            continue;
        }

        const long double abs_error = std::abs(static_cast<long double>(actual) - reference);
        const long double relative_error =
            abs_error / std::max(RELATIVE_ERROR_FLOOR, std::abs(reference));
        const long double allowed_error =
            std::max(MAX_ABS_ERROR, MAX_REL_ERROR * std::max(1.0L, std::abs(reference)));
        const long double normalized_error = abs_error / allowed_error;
        const bool sample_ok = abs_error <= allowed_error;

        if (abs_error > diag.max_abs_error) diag.max_abs_error = abs_error;
        if (relative_error > diag.max_rel_error) diag.max_rel_error = relative_error;
        if (normalized_error > diag.max_normalized_error) {
            diag.max_normalized_error = normalized_error;
            diag.worst_flat_index = pos.flat_index;
            diag.worst_n = pos.n;
            diag.worst_k = pos.k;
            diag.worst_oh = pos.oh;
            diag.worst_ow = pos.ow;
        }
        if (!sample_ok) diag.ok = false;
    }
    (void)wout; (void)hout; (void)kdim;
    return diag;
}

void print_checksum_diagnostics(
    int shape_id, int threads, int repetition, const ChecksumDiagnostics& diag
) {
    std::cout << "[CHECKSUM]"
              << " shape=" << shape_id
              << " threads=" << threads
              << " rep=" << repetition
              << " samples=32"
              << std::scientific << std::setprecision(6)
              << " max_abs_error=" << static_cast<double>(diag.max_abs_error)
              << " max_rel_error=" << static_cast<double>(diag.max_rel_error)
              << " max_normalized_error=" << static_cast<double>(diag.max_normalized_error)
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

// =======================================================================
// oneDNN convolution runner: convolution_auto, user-managed scratchpad,
// fixed C exec-argument list, direct dnnl_primitive_execute/dnnl_stream_wait
// (CODING-AUFTRAG sections 13, 15, 16).
// =======================================================================
class ConvRunner {
public:
    ConvRunner(const Shape& shape, float* input, float* weights, float* output)
        : shape_(shape),
          engine_(dnnl::engine::kind::cpu, 0),
          stream_(engine_) {
        using dt = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;

        const dnnl::memory::dims src_dims{shape.n, shape.c, shape.h, shape.w};
        const dnnl::memory::dims weight_dims{shape.k, shape.c, shape.r, shape.s};
        const dnnl::memory::dims dst_dims{shape.n, shape.k, shape.h_out(), shape.w_out()};
        const dnnl::memory::dims strides{shape.stride, shape.stride};
        const dnnl::memory::dims padding{shape.pad, shape.pad};

        const auto src_user_desc = dnnl::memory::desc(src_dims, dt::f32, tag::nchw);
        const auto weight_user_desc = dnnl::memory::desc(weight_dims, dt::f32, tag::oihw);
        const auto dst_user_desc = dnnl::memory::desc(dst_dims, dt::f32, tag::nchw);
        const auto src_any_desc = dnnl::memory::desc(src_dims, dt::f32, tag::any);
        const auto weight_any_desc = dnnl::memory::desc(weight_dims, dt::f32, tag::any);
        const auto dst_any_desc = dnnl::memory::desc(dst_dims, dt::f32, tag::any);

        // Section 13.1: primitive_attr with scratchpad_mode::user is
        // mandatory; the final primitive_desc is created WITH these
        // attributes (not retrofitted afterwards).
        dnnl::primitive_attr attributes;
        attributes.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        primitive_desc_.emplace(
            engine_, dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_auto,
            src_any_desc, weight_any_desc, dst_any_desc,
            strides, padding, padding, attributes);
        primitive_.emplace(*primitive_desc_);

        src_user_.emplace(src_user_desc, engine_, input);
        weight_user_.emplace(weight_user_desc, engine_, weights);
        dst_user_.emplace(dst_user_desc, engine_, output);

        reorder_src_ = primitive_desc_->src_desc() != src_user_->get_desc();
        reorder_weights_ = primitive_desc_->weights_desc() != weight_user_->get_desc();
        reorder_dst_ = primitive_desc_->dst_desc() != dst_user_->get_desc();

        if (reorder_src_) src_internal_.emplace(primitive_desc_->src_desc(), engine_);
        else src_internal_.emplace(*src_user_);
        if (reorder_weights_) weight_internal_.emplace(primitive_desc_->weights_desc(), engine_);
        else weight_internal_.emplace(*weight_user_);
        if (reorder_dst_) dst_internal_.emplace(primitive_desc_->dst_desc(), engine_);
        else dst_internal_.emplace(*dst_user_);

        // Section 13.2/13.3: scratchpad descriptor + allocation, still
        // outside the measurement window (constructor time).
        const dnnl::memory::desc scratchpad_desc = primitive_desc_->scratchpad_desc();
        scratchpad_size_bytes_ = scratchpad_desc.get_size();
        scratchpad_required_ = scratchpad_size_bytes_ > 0;
        if (scratchpad_required_) {
            scratchpad_.emplace(scratchpad_desc, engine_);
        }
    }

    // Reorders (C++ convenience API is explicitly permitted here: fully
    // outside the measurement window, completed + waited on before
    // returning) plus scratchpad first-touch plus fixed C argument-list
    // assembly (section 13.4, 14, 15.2).
    void prepare(int requested_threads) {
        if (reorder_src_) {
            dnnl::reorder(*src_user_, *src_internal_)
                .execute(stream_, *src_user_, *src_internal_);
        }
        if (reorder_weights_) {
            dnnl::reorder(*weight_user_, *weight_internal_)
                .execute(stream_, *weight_user_, *weight_internal_);
        }
        stream_.wait();

        if (scratchpad_required_) {
            void* handle = scratchpad_->get_data_handle();
            if (handle == nullptr) {
                std::cerr << "FATAL: scratchpad data handle is null\n";
                std::exit(1);
            }
            require_thread_team(requested_threads, "scratchpad first-touch");
            auto* bytes = static_cast<unsigned char*>(handle);
#pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < scratchpad_size_bytes_; ++i) {
                bytes[i] = 0;
            }
        }

        // Internal destination buffer must be initialized before the
        // official measurement (section 14); a full warm-up pass below
        // fully overwrites it, satisfying this requirement.

        prepare_execute_arguments();
        validate_execute_arguments();
    }

    // Section 16.1/16.2: identical direct C-API path for warm-up,
    // calibration, anti-collapse, and official measurement.
    void run_batches(std::uint64_t batches) const {
        for (std::uint64_t batch = 0; batch < batches; ++batch) {
            const dnnl_status_t status = dnnl_primitive_execute(
                primitive_->get(), stream_.get(),
                execute_arg_count_, execute_args_c_.data());
            if (status != dnnl_success) {
                std::cerr << "FATAL: oneDNN primitive execution failed with status="
                           << static_cast<int>(status) << '\n';
                std::exit(1);
            }
        }
        const dnnl_status_t wait_status = dnnl_stream_wait(stream_.get());
        if (wait_status != dnnl_success) {
            std::cerr << "FATAL: oneDNN stream wait failed with status="
                       << static_cast<int>(wait_status) << '\n';
            std::exit(1);
        }
    }

    // C++ convenience reorder is explicitly permitted here: fully outside
    // the measurement window (called only before/after the timed section).
    void copy_output() {
        if (reorder_dst_) {
            dnnl::reorder(*dst_internal_, *dst_user_)
                .execute(stream_, *dst_internal_, *dst_user_);
            stream_.wait();
        }
    }

    [[nodiscard]] std::string implementation_string() const {
        std::ostringstream out;
        out << "onednn_convolution_auto:" << impl_info_str()
            << ";scratchpad=user;execute_api=c";
        return out.str();
    }
    [[nodiscard]] std::string impl_info_str() const {
        return primitive_desc_ ? primitive_desc_->impl_info_str() : "oneDNN_unknown";
    }
    [[nodiscard]] bool reorder_src() const { return reorder_src_; }
    [[nodiscard]] bool reorder_weights() const { return reorder_weights_; }
    [[nodiscard]] bool reorder_dst() const { return reorder_dst_; }
    [[nodiscard]] std::string src_layout_string() const {
        return describe_memory_desc(primitive_desc_->src_desc());
    }
    [[nodiscard]] std::string weight_layout_string() const {
        return describe_memory_desc(primitive_desc_->weights_desc());
    }
    [[nodiscard]] std::string dst_layout_string() const {
        return describe_memory_desc(primitive_desc_->dst_desc());
    }
    [[nodiscard]] std::size_t scratchpad_size_bytes() const { return scratchpad_size_bytes_; }
    [[nodiscard]] bool scratchpad_required() const { return scratchpad_required_; }
    [[nodiscard]] int execute_arg_count() const { return execute_arg_count_; }

private:
    void prepare_execute_arguments() {
        execute_arg_count_ = 0;
        execute_args_c_[static_cast<std::size_t>(execute_arg_count_++)] =
            {DNNL_ARG_SRC, src_internal_->get()};
        execute_args_c_[static_cast<std::size_t>(execute_arg_count_++)] =
            {DNNL_ARG_WEIGHTS, weight_internal_->get()};
        execute_args_c_[static_cast<std::size_t>(execute_arg_count_++)] =
            {DNNL_ARG_DST, dst_internal_->get()};
        if (scratchpad_required_) {
            execute_args_c_[static_cast<std::size_t>(execute_arg_count_++)] =
                {DNNL_ARG_SCRATCHPAD, scratchpad_->get()};
        }
    }

    void validate_execute_arguments() const {
        const int expected = scratchpad_required_ ? 4 : 3;
        bool ok = (execute_arg_count_ == expected) &&
                  (execute_arg_count_ >= 3 && execute_arg_count_ <= 4);
        std::array<int, 4> seen_keys{};
        for (int i = 0; i < execute_arg_count_; ++i) {
            const dnnl_exec_arg_t& arg = execute_args_c_[static_cast<std::size_t>(i)];
            if (arg.memory == nullptr) ok = false;
            seen_keys[static_cast<std::size_t>(i)] = arg.arg;
            for (int j = 0; j < i; ++j) {
                if (seen_keys[static_cast<std::size_t>(j)] == arg.arg) ok = false;
            }
        }
        bool has_scratchpad_key = false;
        for (int i = 0; i < execute_arg_count_; ++i) {
            if (execute_args_c_[static_cast<std::size_t>(i)].arg == DNNL_ARG_SCRATCHPAD) {
                has_scratchpad_key = true;
            }
        }
        if (scratchpad_required_ != has_scratchpad_key) ok = false;

        if (!ok) {
            std::cerr << "FATAL: invalid prepared oneDNN execute argument list\n";
            std::exit(1);
        }
    }

    Shape shape_;
    dnnl::engine engine_;
    dnnl::stream stream_;
    std::optional<dnnl::convolution_forward::primitive_desc> primitive_desc_;
    std::optional<dnnl::convolution_forward> primitive_;
    std::optional<dnnl::memory> src_user_;
    std::optional<dnnl::memory> weight_user_;
    std::optional<dnnl::memory> dst_user_;
    std::optional<dnnl::memory> src_internal_;
    std::optional<dnnl::memory> weight_internal_;
    std::optional<dnnl::memory> dst_internal_;
    std::optional<dnnl::memory> scratchpad_;
    bool reorder_src_{false};
    bool reorder_weights_{false};
    bool reorder_dst_{false};
    bool scratchpad_required_{false};
    std::size_t scratchpad_size_bytes_{0};

    static constexpr std::size_t MAX_EXEC_ARGS = 4;
    std::array<dnnl_exec_arg_t, MAX_EXEC_ARGS> execute_args_c_{};
    int execute_arg_count_{0};
};

// =======================================================================
// Adaptive calibration (CODING-AUFTRAG section 20) -- same direct C-API
// execute path as everything else.
// =======================================================================
struct CalibrationResult {
    long long batches;
    double seconds;  // the seconds value actually measured for `batches`
};

CalibrationResult calibrate(ConvRunner& runner, int requested_threads) {
    require_thread_team(requested_threads, "warm-up");
    runner.run_batches(1);

    long long batches = 1;
    for (int step = 0; step < bench::MAX_CALIBRATION_STEPS; ++step) {
        require_thread_team(requested_threads, "calibration");
        const auto start = std::chrono::steady_clock::now();
        runner.run_batches(static_cast<std::uint64_t>(batches));
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();

        if (!std::isfinite(seconds) || seconds <= 0.0) {
            std::cerr << "FATAL: non-finite or non-positive calibration time\n";
            std::exit(1);
        }

        // PATCH (Fehler A): the previous version returned here whenever
        // batches==MAX_BATCHES, *without* checking whether the minimum
        // runtime (section 20) was actually reached -- silently accepting
        // a cap-hit measurement that was still below 0.75s. That made the
        // FATAL below unreachable (batches==MAX_BATCHES always returns
        // first). Cap-hit now explicitly enforces MIN_RUNTIME_S.
        if (batches == MAX_BATCHES) {
            if (seconds < bench::MIN_RUNTIME_S) {
                std::cerr << "FATAL: unable to reach minimum runtime at maximum batch count"
                             " (batches=" << batches << ", seconds=" << seconds << ")\n";
                std::exit(1);
            }
            return {batches, seconds};
        }
        if (seconds >= bench::TARGET_RUNTIME_S) {
            return {batches, seconds};
        }

        const long long scaled = static_cast<long long>(
            bench::scale_batches(seconds, static_cast<int>(batches), MAX_BATCHES));
        batches = std::max<long long>(batches + 1, scaled);
        batches = std::min<long long>(batches, MAX_BATCHES);
    }

    // PATCH (Fehler B, defensive hardening): exhausting all calibration
    // steps without ever reaching TARGET_RUNTIME_S or a validated
    // MAX_BATCHES plateau means there is no measured, trustworthy batches
    // value to return. Under a correctly functioning scale_batches this
    // path should be unreachable in practice (its 10x-per-step growth cap
    // reaches MAX_BATCHES=100000 from batches=1 in ~5-6 steps, well within
    // bench::MAX_CALIBRATION_STEPS=14) -- but the previous version still
    // fell through to a silent, unmeasured `return batches;` here instead
    // of treating loop exhaustion as the calibration failure it is.
    std::cerr << "FATAL: calibration did not converge within "
               << bench::MAX_CALIBRATION_STEPS
               << " steps (last batches=" << batches << ")\n";
    std::exit(1);
}

// =======================================================================
// Anti-collapse probe (CODING-AUFTRAG section 27). Opt-in via
// CONV2D_ANTI_COLLAPSE_PROBE=1. shape_id=1, platform-maximum threads.
// Writes no CSV rows; reuses the same primitive/stream/scratchpad/fixed
// argument list and the same direct dnnl_primitive_execute path.
// =======================================================================
void run_anti_collapse_probe(const std::string& model) {
    (void)model;
    const Shape& shape = SHAPES.front();  // shape_id == 1
    shape.validate_geometry();
    const int threads = *std::max_element(bench::THREAD_COUNTS.begin(), bench::THREAD_COUNTS.end());

    omp_set_num_threads(threads);
    require_thread_team(threads, "anti-collapse setup");

    const std::uint64_t input_count = shape.input_elements();
    const std::uint64_t weight_count = shape.weight_elements();
    const std::uint64_t output_count = shape.output_elements();

    // PATCH (First-Touch): std::vector<float>(count) value-initializes
    // every element to 0.0f in its constructor -- a serial, single-thread
    // write that is the *true* NUMA first touch, executed before the
    // parallel initialize_inputs() below ever runs. That left every page
    // resident on whichever NUMA node ran this constructor, regardless of
    // the configured thread count. bench::allocate_aligned() (posix_memalign)
    // leaves memory uninitialized, so the parallel loop in
    // initialize_inputs() becomes the genuine first touch, matching the
    // AXPY reference pattern.
    std::unique_ptr<float, decltype(&std::free)> input(
        bench::allocate_aligned(input_count), &std::free);
    std::unique_ptr<float, decltype(&std::free)> weights(
        bench::allocate_aligned(weight_count), &std::free);
    std::unique_ptr<float, decltype(&std::free)> output(
        bench::allocate_aligned(output_count), &std::free);
    if (!input || !weights || !output) {
        std::cerr << "FATAL: allocation failed for CONV2D anti-collapse probe shape_id="
                   << shape.id << '\n';
        std::exit(1);
    }
    initialize_inputs(input.get(), input_count, weights.get(), weight_count,
                       output.get(), output_count);

    ConvRunner runner(shape, input.get(), weights.get(), output.get());
    runner.prepare(threads);

    const dnnl::version_t* version = dnnl::version();
    const int observed_threads = active_openmp_threads();
    const dnnl::cpu_isa effective_isa = dnnl::get_effective_cpu_isa();
    std::cout << "[CONFIG] shape=" << shape.id
              << " threads_requested=" << threads
              << " threads_observed=" << observed_threads
              << " mode=anti_collapse\n";
    std::cout << "[ONEDNN]"
              << " version=" << version->major << "." << version->minor << "." << version->patch
              << " cpu_threading_runtime=OpenMP"
              << " implementation=" << runner.impl_info_str()
              << " src_layout=" << bench::csv_escape(runner.src_layout_string())
              << " weight_layout=" << bench::csv_escape(runner.weight_layout_string())
              << " dst_layout=" << bench::csv_escape(runner.dst_layout_string())
              << " src_reorder=" << (runner.reorder_src() ? "yes" : "no")
              << " weight_reorder=" << (runner.reorder_weights() ? "yes" : "no")
              << " dst_reorder=" << (runner.reorder_dst() ? "yes" : "no")
              << " scratchpad_mode=user"
              << " scratchpad_size_bytes=" << runner.scratchpad_size_bytes()
              << " execute_api=dnnl_primitive_execute"
              << " execute_arg_count=" << runner.execute_arg_count()
              << '\n';
    std::cout << "[ENV]"
              << " requested_threads=" << threads
              << " observed_threads=" << observed_threads
              << " OMP_NUM_THREADS_env="
              << (std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "unset")
              << " OMP_PROC_BIND="
              << (std::getenv("OMP_PROC_BIND") ? std::getenv("OMP_PROC_BIND") : "unset")
              << " OMP_PLACES="
              << (std::getenv("OMP_PLACES") ? std::getenv("OMP_PLACES") : "unset")
              << " ONEDNN_VERBOSE="
              << (std::getenv("ONEDNN_VERBOSE") ? std::getenv("ONEDNN_VERBOSE") : "unset")
              << " DNNL_VERBOSE="
              << (std::getenv("DNNL_VERBOSE") ? std::getenv("DNNL_VERBOSE") : "unset")
              << " effective_isa=" << static_cast<int>(effective_isa)
              << '\n';

    runner.run_batches(1);

    const CalibrationResult calibration = calibrate(runner, threads);
    const long long b_cal = calibration.batches;
    long long b_probe = std::min<long long>(
        std::max<long long>(100, b_cal / 4), MAX_BATCHES / 2);

    auto measure = [&](long long batches) {
        require_thread_team(threads, "anti-collapse measurement");
        const auto start = std::chrono::steady_clock::now();
        runner.run_batches(static_cast<std::uint64_t>(batches));
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        runner.copy_output();
        const ChecksumDiagnostics diag = verify_checksum(shape, input.get(), weights.get(), output.get());
        return std::make_pair(seconds, diag);
    };

    const long long max_probe = MAX_BATCHES / 2;
    double t1 = 0.0;
    ChecksumDiagnostics d1;
    for (;;) {
        std::tie(t1, d1) = measure(b_probe);
        if (t1 >= 0.020) break;
        if (b_probe >= max_probe) {
            std::cerr << "FATAL: anti-collapse probe cannot reach minimum duration "
                         "before batch cap\n";
            std::exit(1);
        }
        const long long next = std::min<long long>(b_probe * 2, max_probe);
        if (next <= b_probe) {
            std::cerr << "FATAL: anti-collapse probe made no forward progress\n";
            std::exit(1);
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

    std::cout << std::setprecision(9)
              << "[ANTI_COLLAPSE] shape=1"
              << " threads=" << threads
              << " B=" << b_probe
              << " two_B=" << two_b_probe
              << " t1=" << t1
              << " t2=" << t2
              << " ratio=" << ratio
              << " checksum_B=" << (d1.ok ? "PASS" : "FAIL")
              << " checksum_2B=" << (d2.ok ? "PASS" : "FAIL")
              << " gate=" << (gate_pass ? "PASS" : "FAIL")
              << '\n';

    if (!gate_pass) {
        std::cerr << "FATAL: anti-collapse gate FAILED\n";
        std::exit(1);
    }
}

// =======================================================================
// CSV row (schema_version cpu-gpu-v2, exact 45-column schema). Deliberately
// NOT reusing bench::make_cpu_row()/bench::write_row(): those helpers bind
// the primary energy/power formulas to total_energy_j (package+DRAM) and
// serialize flops_total/logical_bytes_per_op/energy fields in rounded
// scientific notation. This benchmark instead follows the same pattern
// already established by the hardened AXPY reference (main_axpy_intel.cpp):
// device_energy_j-only formulas and lossless (max_digits10) serialization.
// =======================================================================
struct Conv2dRow {
    std::string session_id;
    long long sequence_index{};
    long long run_id_global{};
    int repetition{};
    std::string implementation;
    std::string device_name;
    int num_threads{};
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
    int clock_before_mhz{-1};
    int clock_after_mhz{-1};
    int temp_c{-1};
    int temp_before_c{-1};
    int temp_after_c{-1};
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

// Lossless double serialization (no scientific-6 rounding for formula
// anchors), matching the hardened AXPY convention exactly.
void write_lossless_double(std::ofstream& output, double value) {
    output << std::defaultfloat
           << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
}

void write_conv2d_row(std::ofstream& output, const Conv2dRow& row) {
    output << bench::SCHEMA_VERSION << ',' << bench::timestamp() << ','
           << bench::csv_escape(row.session_id) << ','
           << row.sequence_index << ',' << row.run_id_global << ','
           << row.repetition << ','
           << bench::csv_escape("CONV2D") << ','
           << bench::csv_escape(row.implementation) << ','
           << "cpu_native" << ','
           << bench::csv_escape(row.device_name) << ',' << row.num_threads << ','
           << row.problem_size << ',' << bench::csv_escape(row.problem_spec) << ','
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

    // Exact decimal integers -- never floating-point formatted.
    output << row.flops_total << ',';
    write_lossless_double(output, row.gflops_per_s); output << ',';
    output << row.logical_bytes_per_op << ',';
    write_lossless_double(output, row.avg_power_w); output << ',';

    // CPU sentinels written literally (contract convention shared with
    // AXPY): pcie_gen/pcie_width/sm_clock_mhz/mem_clock_mhz = -1,
    // throttle_reasons blank, cpu_cycles/cpu_instructions/cpu_cache_misses
    // = -1, cpu_ipc = -1.000000.
    output << row.runtime_status << ','
           << -1 << ',' << -1 << ','           // pcie_gen, pcie_width
           << -1 << ','                        // sm_clock_mhz
           << row.clock_before_mhz << ','
           << row.clock_after_mhz << ','
           << -1 << ','                        // mem_clock_mhz
           << row.temp_c << ',' << row.temp_before_c << ',' << row.temp_after_c << ','
           << ','                               // throttle_reasons (blank)
           << -1 << ',' << -1 << ','
           << std::fixed << std::setprecision(6) << -1.0 << std::defaultfloat << ','
           << -1 << ','
           << (row.checksum_ok ? 't' : 'f') << '\n';
}

void print_conv2d_result(const Conv2dRow& row) {
    std::cout << "[CONV2D] shape=" << row.problem_size
              << " threads=" << row.num_threads
              << " rep=" << row.repetition
              << " batches=" << row.batches
              << " runtime_s=" << std::fixed << std::setprecision(6) << row.e2e_time_s
              << " device_energy_j=" << std::setprecision(6) << row.device_energy_j
              << " status=" << row.runtime_status
              << std::defaultfloat << '\n';
}

// =======================================================================
// Env-var filters (BENCH_SIZE_FILTER = comma-separated shape_ids,
// BENCH_THREAD_FILTER), following the established AXPY/REDUCTION/STREAM
// convention exactly (CODING-AUFTRAG section 31).
// =======================================================================
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

template <typename T>
bool selected(const T value, const std::vector<T>& filter) {
    return filter.empty() ||
        std::find(filter.begin(), filter.end(), value) != filter.end();
}

bool env_flag_set(const char* name) {
    const char* raw = std::getenv(name);
    return raw && *raw && std::string(raw) != "0";
}

std::vector<Config> build_configs() {
    const std::vector<int> shape_filter = parse_int_filter("BENCH_SIZE_FILTER");
    const std::vector<int> thread_filter = parse_int_filter("BENCH_THREAD_FILTER");

    for (const int shape_id : shape_filter) {
        bool found = false;
        for (const Shape& shape : SHAPES) {
            if (shape.id == shape_id) { found = true; break; }
        }
        if (!found) {
            std::cerr << "FATAL: BENCH_SIZE_FILTER references unknown shape_id=" << shape_id << '\n';
            std::exit(1);
        }
    }
    for (const int threads : thread_filter) {
        if (std::find(bench::THREAD_COUNTS.begin(), bench::THREAD_COUNTS.end(), threads)
                == bench::THREAD_COUNTS.end()) {
            std::cerr << "FATAL: BENCH_THREAD_FILTER references unknown thread count=" << threads << '\n';
            std::exit(1);
        }
    }

    std::vector<Config> configs;
    for (std::size_t index = 0; index < SHAPES.size(); ++index) {
        if (!selected(SHAPES[index].id, shape_filter)) continue;
        for (const int threads : bench::THREAD_COUNTS) {
            if (selected(threads, thread_filter)) {
                configs.push_back({index, threads});
            }
        }
    }
    if (configs.empty()) {
        std::cerr << "FATAL: CONV2D filters produced no configurations\n";
        std::exit(1);
    }
    return configs;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bench::Options options =
            bench::parse_options(argc, argv, "conv2d_intel.csv");
        const bool anti_collapse_mode =
            env_flag_set("CONV2D_ANTI_COLLAPSE_PROBE");

        omp_set_dynamic(0);
        if (omp_get_dynamic() != 0) {
            throw std::runtime_error("OpenMP dynamic teams could not be disabled");
        }

        for (const Shape& shape : SHAPES) shape.validate_geometry();

        const std::string model = bench::cpu_model();
        const dnnl::version_t* version = dnnl::version();

        if (anti_collapse_mode) {
            std::cout << "[BENCHMARK]"
                      << " workload=CONV2D"
                      << " platform=INTEL"
                      << " backend=oneDNN"
                      << " schema=cpu-gpu-v2"
                      << " mode=anti_collapse"
                      << " session_id=" << options.session_id
                      << " repetitions=0"
                      << " seed=" << options.seed
                      << " configurations=1"
                      << " device_name=" << model
                      << " dram_rapl_available=not_sampled"
                      << " reuse_regime=warm_resident"
                      << " algorithm_policy=convolution_auto"
                      << " scratchpad_mode=user"
                      << " execute_api=dnnl_primitive_execute"
                      << " onednn_cpu_threading_runtime=OpenMP"
                      << " onednn_version=" << version->major << "."
                      << version->minor << "." << version->patch
                      << '\n';

            std::cout << "onednn_cpu_threading_runtime=OpenMP"
                      << " DNNL_CPU_THREADING_RUNTIME="
                      << DNNL_CPU_THREADING_RUNTIME
                      << " DNNL_RUNTIME_OMP=" << DNNL_RUNTIME_OMP
#ifdef _OPENMP
                      << " openmp_spec_date=" << _OPENMP
#else
                      << " openmp_spec_date=unavailable"
#endif
                      << '\n';

            // Exclusive probe mode: no regular configuration loop, no RAPL
            // requirement, and no CSV file creation or truncation.
            run_anti_collapse_probe(model);
            return 0;
        }

        std::vector<Config> configs = build_configs();
        bench::shuffle_configs(configs, options.seed);

        bench::Rapl rapl;
        bench::require_rapl(rapl);

        const auto parent =
            std::filesystem::path(options.output_file).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);

        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "Cannot open output file: " + options.output_file);
        }
        write_conv2d_header(output);
        output.flush();
        if (!output.good()) {
            throw std::runtime_error(
                "Failed to write CONV2D CSV header: " + options.output_file);
        }

        std::cout << "[BENCHMARK]"
                  << " workload=CONV2D"
                  << " platform=INTEL"
                  << " backend=oneDNN"
                  << " schema=cpu-gpu-v2"
                  << " mode=measurement"
                  << " session_id=" << options.session_id
                  << " repetitions=" << options.repetitions
                  << " seed=" << options.seed
                  << " configurations=" << configs.size()
                  << " device_name=" << model
                  << " dram_rapl_available="
                  << (rapl.dram_available() ? "yes" : "no")
                  << " reuse_regime=warm_resident"
                  << " algorithm_policy=convolution_auto"
                  << " scratchpad_mode=user"
                  << " execute_api=dnnl_primitive_execute"
                  << " onednn_cpu_threading_runtime=OpenMP"
                  << " onednn_version=" << version->major << "."
                  << version->minor << "." << version->patch
                  << '\n';

        std::cout << "onednn_cpu_threading_runtime=OpenMP"
                  << " DNNL_CPU_THREADING_RUNTIME="
                  << DNNL_CPU_THREADING_RUNTIME
                  << " DNNL_RUNTIME_OMP=" << DNNL_RUNTIME_OMP
#ifdef _OPENMP
                  << " openmp_spec_date=" << _OPENMP
#else
                  << " openmp_spec_date=unavailable"
#endif
                  << '\n';

        long long sequence = 0;

        for (const Config& config : configs) {
            const Shape& shape = SHAPES[config.shape_index];

            omp_set_num_threads(config.threads);
            require_thread_team(config.threads, "before first-touch");
            const int observed_threads = active_openmp_threads();

            std::cout << "[CONFIG] shape=" << shape.id
                      << " threads_requested=" << config.threads
                      << " threads_observed=" << observed_threads << '\n';

            const std::uint64_t input_count = shape.input_elements();
            const std::uint64_t weight_count = shape.weight_elements();
            const std::uint64_t output_count = shape.output_elements();

            // PATCH (First-Touch): see identical rationale at the
            // anti-collapse probe's buffer setup above.
            std::unique_ptr<float, decltype(&std::free)> input(
                bench::allocate_aligned(input_count), &std::free);
            std::unique_ptr<float, decltype(&std::free)> weights(
                bench::allocate_aligned(weight_count), &std::free);
            std::unique_ptr<float, decltype(&std::free)> output_buffer(
                bench::allocate_aligned(output_count), &std::free);
            if (!input || !weights || !output_buffer) {
                std::cerr << "FATAL: allocation failed for CONV2D shape_id="
                           << shape.id << '\n';
                std::exit(1);
            }

            initialize_inputs(input.get(), input_count, weights.get(), weight_count,
                               output_buffer.get(), output_count);

            require_thread_team(config.threads, "before primitive creation");
            ConvRunner runner(shape, input.get(), weights.get(), output_buffer.get());

            require_thread_team(config.threads, "before reorders");
            runner.prepare(config.threads);

            // execute_arg_count is only populated by prepare() (fixed C
            // argument list assembly happens there); log after prepare()
            // so the reported count reflects reality instead of the
            // pre-prepare initial value of 0.
            std::cout << "[ONEDNN]"
                      << " version=" << version->major << "." << version->minor << "." << version->patch
                      << " cpu_threading_runtime=OpenMP"
                      << " implementation=" << runner.impl_info_str()
                      << " src_layout=" << bench::csv_escape(runner.src_layout_string())
                      << " weight_layout=" << bench::csv_escape(runner.weight_layout_string())
                      << " dst_layout=" << bench::csv_escape(runner.dst_layout_string())
                      << " src_reorder=" << (runner.reorder_src() ? "yes" : "no")
                      << " weight_reorder=" << (runner.reorder_weights() ? "yes" : "no")
                      << " dst_reorder=" << (runner.reorder_dst() ? "yes" : "no")
                      << " scratchpad_mode=user"
                      << " scratchpad_size_bytes=" << runner.scratchpad_size_bytes()
                      << " execute_api=dnnl_primitive_execute"
                      << " execute_arg_count=" << runner.execute_arg_count()
                      << '\n';

            const dnnl::cpu_isa effective_isa = dnnl::get_effective_cpu_isa();
            std::cout << "[ENV]"
                      << " requested_threads=" << config.threads
                      << " observed_threads=" << observed_threads
                      << " OMP_NUM_THREADS_env="
                      << (std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "unset")
                      << " OMP_PROC_BIND=" << (std::getenv("OMP_PROC_BIND") ? std::getenv("OMP_PROC_BIND") : "unset")
                      << " OMP_PLACES=" << (std::getenv("OMP_PLACES") ? std::getenv("OMP_PLACES") : "unset")
                      << " ONEDNN_VERBOSE=" << (std::getenv("ONEDNN_VERBOSE") ? std::getenv("ONEDNN_VERBOSE") : "unset")
                      << " DNNL_VERBOSE=" << (std::getenv("DNNL_VERBOSE") ? std::getenv("DNNL_VERBOSE") : "unset")
                      << " effective_isa=" << static_cast<int>(effective_isa)
                      << '\n';

            require_thread_team(config.threads, "before warm-up");
            runner.run_batches(1);
            runner.copy_output();
            const ChecksumDiagnostics warmup_diag =
                verify_checksum(shape, input.get(), weights.get(), output_buffer.get());
            print_checksum_diagnostics(shape.id, config.threads, 0, warmup_diag);
            if (!warmup_diag.ok) {
                throw std::runtime_error(
                    "CONV2D warm-up checksum failed for shape_id=" + std::to_string(shape.id) +
                    " (impl=" + runner.impl_info_str() + ")");
            }

            require_thread_team(config.threads, "before calibration");
            const CalibrationResult calibration = calibrate(runner, config.threads);
            const long long batches = calibration.batches;
            // PATCH: runtime_s/status were missing from this line
            // (CODING-AUFTRAG section 20/33 both call for them).
            std::cout << "[CALIBRATION] shape=" << shape.id
                      << " threads=" << config.threads
                      << " batches=" << batches
                      << " runtime_s=" << calibration.seconds
                      << " status=" << bench::runtime_status(calibration.seconds)
                      << '\n';

            for (int repetition = 1; repetition <= options.repetitions; ++repetition) {
                require_thread_team(config.threads, "before official measurement");

                const int clock_before = bench::average_online_cpu_frequency_mhz();
                const int temp_before = bench::cpu_temperature_c();
                const auto energy_before = rapl.read();
                const auto start = std::chrono::steady_clock::now();

                runner.run_batches(static_cast<std::uint64_t>(batches));

                const auto end = std::chrono::steady_clock::now();
                const auto energy_after = rapl.read();
                const int clock_after = bench::average_online_cpu_frequency_mhz();
                const int temp_after = bench::cpu_temperature_c();

                const double seconds = std::chrono::duration<double>(end - start).count();
                const auto energy = rapl.delta(energy_before, energy_after);

                if (!std::isfinite(seconds) || seconds <= 0.0) {
                    throw std::runtime_error(
                        "Non-finite or non-positive measurement for shape_id=" + std::to_string(shape.id));
                }
                if (!std::isfinite(energy.package_j) || energy.package_j <= 0.0) {
                    throw std::runtime_error(
                        "Invalid RAPL package energy (must be finite and > 0.0) for shape_id=" +
                        std::to_string(shape.id) + ", threads=" + std::to_string(config.threads) +
                        ", repetition=" + std::to_string(repetition));
                }
                if (rapl.dram_available() && (!std::isfinite(energy.dram_j) || energy.dram_j < 0.0)) {
                    throw std::runtime_error(
                        "Invalid RAPL DRAM energy (must be finite and >= 0.0) for shape_id=" +
                        std::to_string(shape.id));
                }

                const std::string runtime_status = bench::runtime_status(seconds);
                if (runtime_status == "below") {
                    // Section 30/32: below is a hard failure. No CSV row is
                    // written and the sequence counter is not advanced.
                    throw std::runtime_error(
                        "Hard failure: runtime_status=below for shape_id=" + std::to_string(shape.id) +
                        ", threads=" + std::to_string(config.threads) +
                        ", repetition=" + std::to_string(repetition) +
                        ", batches=" + std::to_string(batches) +
                        ", e2e_time_s=" + std::to_string(seconds));
                }
                if (runtime_status == "above") {
                    std::cout << "[WARNING] runtime_status=above shape=" << shape.id
                              << " threads=" << config.threads
                              << " rep=" << repetition
                              << " e2e_time_s=" << seconds << '\n';
                }

                // Reorder result back to the logical NCHW output layout only
                // after the measurement window; the independent checksum
                // reads that logical buffer, never oneDNN's internal layout.
                runner.copy_output();
                const ChecksumDiagnostics diag =
                    verify_checksum(shape, input.get(), weights.get(), output_buffer.get());
                print_checksum_diagnostics(shape.id, config.threads, repetition, diag);

                if (!diag.ok) {
                    // Section 25/32: a checksum failure never produces a
                    // valid CSV row and checksum_ok is never set true.
                    throw std::runtime_error(
                        "CONV2D checksum failed for shape_id=" + std::to_string(shape.id) +
                        ", threads=" + std::to_string(config.threads) +
                        ", repetition=" + std::to_string(repetition) +
                        " (impl=" + runner.impl_info_str() + ")");
                }

                // All gates passed: flops/bytes, row construction, and the
                // sequence counter advance only from this point on
                // (section 29).
                const std::uint64_t flops_total_u64 =
                    checked_mul_u64(shape.flops_per_op(), static_cast<std::uint64_t>(batches), "flops_total");
                const long long flops_total =
                    checked_u64_to_long_long(flops_total_u64, "flops_total");
                const long long logical_bytes_per_op =
                    checked_u64_to_long_long(
                        shape.logical_bytes_per_op(),
                        "logical_bytes_per_op");
                const double total_energy_j =
                    energy.package_j + (energy.dram_j >= 0.0 ? energy.dram_j : 0.0);

                // PATCH (Blocker 3): sequence_index must be a CSV column
                // value, so a candidate has to be decided before writing --
                // but the *shared* `sequence` counter itself must only
                // advance once the write is confirmed to have succeeded
                // (section 29). The previous version incremented `sequence`
                // directly at row-construction time, before the write was
                // even attempted, and never checked the stream state after
                // flush().
                const long long candidate_sequence = sequence + 1;

                Conv2dRow row;
                row.session_id = options.session_id;
                row.sequence_index = candidate_sequence;
                row.run_id_global = row.sequence_index;
                row.repetition = repetition;
                row.implementation = runner.implementation_string();
                row.device_name = model;
                row.num_threads = config.threads;
                row.problem_size = shape.id;
                row.problem_spec = shape.problem_spec();
                row.batches = batches;
                row.e2e_time_s = seconds;
                row.kernel_time_s = seconds;
                row.wall_time_s = seconds;
                row.device_energy_j = energy.package_j;
                row.total_energy_j = total_energy_j;
                row.dram_energy_j = energy.dram_j;
                row.energy_per_op_j = row.device_energy_j / static_cast<double>(batches);
                row.energy_per_second_j = row.device_energy_j / row.wall_time_s;
                row.energy_per_flop_j = row.device_energy_j / static_cast<double>(flops_total);
                row.time_per_op_ms_kernel = 1000.0 * row.kernel_time_s / static_cast<double>(batches);
                row.time_per_op_ms_e2e = 1000.0 * row.e2e_time_s / static_cast<double>(batches);
                row.flops_total = flops_total;
                row.gflops_per_s = static_cast<double>(flops_total) / row.kernel_time_s / 1.0e9;
                row.logical_bytes_per_op = logical_bytes_per_op;
                row.avg_power_w = row.device_energy_j / row.wall_time_s;
                row.runtime_status = runtime_status;
                row.clock_before_mhz = clock_before;
                row.clock_after_mhz = clock_after;
                row.temp_c = std::max(temp_before, temp_after);
                row.temp_before_c = temp_before;
                row.temp_after_c = temp_after;
                row.checksum_ok = diag.ok;

                write_conv2d_row(output, row);
                output.flush();
                if (!output.good()) {
                    std::cerr << "FATAL: CSV write failed for shape_id=" << shape.id
                               << ", threads=" << config.threads
                               << ", repetition=" << repetition
                               << ", sequence=" << candidate_sequence << '\n';
                    std::exit(1);
                }
                sequence = candidate_sequence;  // commit only after a verified write
                print_conv2d_result(row);
            }
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        return 2;
    }
}
