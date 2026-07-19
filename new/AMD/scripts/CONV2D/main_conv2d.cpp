#include "benchmark_common.hpp"

#include <omp.h>
#include <optional>
#include <unordered_map>

#if !defined(USE_NAIVE_CONV)
#  if __has_include(<dnnl.hpp>)
#    include <dnnl.hpp>
#  elif __has_include(<oneapi/dnnl/dnnl.hpp>)
#    include <oneapi/dnnl/dnnl.hpp>
#  else
#    error "oneDNN headers not found. Install libdnnl-dev or compile with -DUSE_NAIVE_CONV."
#  endif
#  if __has_include(<dnnl_config.h>)
#    include <dnnl_config.h>
#  elif __has_include(<oneapi/dnnl/dnnl_config.h>)
#    include <oneapi/dnnl/dnnl_config.h>
#  endif
#  if defined(DNNL_CPU_THREADING_RUNTIME) && DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_OMP
#    error "This benchmark requires an OpenMP-built oneDNN so per-configuration thread counts are controlled."
#  endif
#endif

namespace {

constexpr int MAX_BATCHES = 100000;

struct Shape {
    int id;
    int n;
    int c;
    int h;
    int w;
    int k;
    int r;
    int s;
    int stride;
    int pad;

    [[nodiscard]] int h_out() const { return (h + 2 * pad - r) / stride + 1; }
    [[nodiscard]] int w_out() const { return (w + 2 * pad - s) / stride + 1; }
    [[nodiscard]] size_t input_elements() const {
        return static_cast<size_t>(n) * c * h * w;
    }
    [[nodiscard]] size_t weight_elements() const {
        return static_cast<size_t>(k) * c * r * s;
    }
    [[nodiscard]] size_t output_elements() const {
        return static_cast<size_t>(n) * k * h_out() * w_out();
    }
    [[nodiscard]] double flops_per_op() const {
        return 2.0 * n * static_cast<double>(k) * c * r * s * h_out() * w_out();
    }
    [[nodiscard]] double logical_bytes_per_op() const {
        return static_cast<double>(input_elements() + weight_elements() + output_elements()) * sizeof(float);
    }
    [[nodiscard]] std::string problem_spec() const {
        return "N=" + std::to_string(n) + ";C=" + std::to_string(c)
            + ";H=" + std::to_string(h) + ";W=" + std::to_string(w)
            + ";K=" + std::to_string(k) + ";R=" + std::to_string(r)
            + ";S=" + std::to_string(s) + ";stride=" + std::to_string(stride)
            + ";pad=" + std::to_string(pad);
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

struct Config {
    size_t shape_index;
    int threads;
};

void initialize(float* input, size_t input_count,
                float* weights, size_t weight_count,
                float* output, size_t output_count) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < input_count; ++i) {
        input[i] = -0.5f + static_cast<float>(i % 31) * 0.03125f;
    }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < weight_count; ++i) {
        weights[i] = -0.25f + static_cast<float>(i % 23) * 0.015625f;
    }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < output_count; ++i) output[i] = 0.0f;
}

double sampled_checksum(const float* output, size_t count) {
    long double sum = 0.0L;
    constexpr size_t SAMPLE_COUNT = 32;
    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        const size_t index = (sample * (count - 1)) / (SAMPLE_COUNT - 1);
        sum += static_cast<long double>(output[index]) * static_cast<long double>(sample + 1);
    }
    return static_cast<double>(sum);
}

bool checksum_matches(double actual, double reference) {
    const double relative = std::abs(actual - reference) / std::max(1.0, std::abs(reference));
    return relative <= 2.0e-5;
}

#ifdef USE_NAIVE_CONV

class ConvRunner {
public:
    ConvRunner(const Shape& shape, const float* input, const float* weights, float* output)
        : shape_(shape), input_(input), weights_(weights), output_(output) {}

    void prepare() {}

    void run(int batches) {
        const Shape& q = shape_;
#pragma omp parallel
        {
            for (int batch = 0; batch < batches; ++batch) {
#pragma omp for collapse(4) schedule(static)
                for (int ni = 0; ni < q.n; ++ni) {
                    for (int ko = 0; ko < q.k; ++ko) {
                        for (int oh = 0; oh < q.h_out(); ++oh) {
                            for (int ow = 0; ow < q.w_out(); ++ow) {
                                float sum = 0.0f;
                                for (int ci = 0; ci < q.c; ++ci) {
                                    for (int rr = 0; rr < q.r; ++rr) {
                                        for (int ss = 0; ss < q.s; ++ss) {
                                            const int ih = oh * q.stride - q.pad + rr;
                                            const int iw = ow * q.stride - q.pad + ss;
                                            if (ih < 0 || ih >= q.h || iw < 0 || iw >= q.w) continue;
                                            const size_t input_index =
                                                ((static_cast<size_t>(ni) * q.c + ci) * q.h + ih) * q.w + iw;
                                            const size_t weight_index =
                                                ((static_cast<size_t>(ko) * q.c + ci) * q.r + rr) * q.s + ss;
                                            sum += input_[input_index] * weights_[weight_index];
                                        }
                                    }
                                }
                                const size_t output_index =
                                    ((static_cast<size_t>(ni) * q.k + ko) * q.h_out() + oh) * q.w_out() + ow;
                                output_[output_index] = sum;
                            }
                        }
                    }
                }
            }
        }
    }

    void copy_output() {}

    [[nodiscard]] std::string implementation() const { return "naive_openmp"; }

private:
    Shape shape_;
    const float* input_;
    const float* weights_;
    float* output_;
};

#else

class ConvRunner {
public:
    ConvRunner(const Shape& shape, float* input, float* weights, float* output)
        : engine_(dnnl::engine::kind::cpu, 0), stream_(engine_) {
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

        primitive_desc_.emplace(
            engine_, dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_auto,
            src_any_desc, weight_any_desc, dst_any_desc,
            strides, padding, padding);
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

        arguments_.emplace(DNNL_ARG_SRC, *src_internal_);
        arguments_.emplace(DNNL_ARG_WEIGHTS, *weight_internal_);
        arguments_.emplace(DNNL_ARG_DST, *dst_internal_);
    }

    void prepare() {
        if (reorder_src_) {
            dnnl::reorder(*src_user_, *src_internal_)
                .execute(stream_, *src_user_, *src_internal_);
        }
        if (reorder_weights_) {
            dnnl::reorder(*weight_user_, *weight_internal_)
                .execute(stream_, *weight_user_, *weight_internal_);
        }
        stream_.wait();
    }

    void run(int batches) {
        for (int batch = 0; batch < batches; ++batch) {
            primitive_->execute(stream_, arguments_);
        }
        stream_.wait();
    }

    void copy_output() {
        if (reorder_dst_) {
            dnnl::reorder(*dst_internal_, *dst_user_)
                .execute(stream_, *dst_internal_, *dst_user_);
        }
        stream_.wait();
    }

    [[nodiscard]] std::string implementation() const {
        return primitive_desc_ ? primitive_desc_->impl_info_str() : "oneDNN_unknown";
    }

private:
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
    std::unordered_map<int, dnnl::memory> arguments_;
    bool reorder_src_{false};
    bool reorder_weights_{false};
    bool reorder_dst_{false};
};
#endif

int calibrate(ConvRunner& runner) {
    runner.run(1);
    int batches = 1;
    for (int step = 0; step < bench::MAX_CALIBRATION_STEPS; ++step) {
        const auto start = std::chrono::steady_clock::now();
        runner.run(batches);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (seconds >= bench::TARGET_RUNTIME_S || batches == MAX_BATCHES) return batches;
        batches = bench::scale_batches(seconds, batches, MAX_BATCHES);
    }
    return batches;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bench::Options options = bench::parse_options(argc, argv, "conv2d_amd.csv");
        const auto parent = std::filesystem::path(options.output_file).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot open output file: " + options.output_file);
        bench::write_header(output);

        bench::Rapl rapl;
        bench::require_rapl(rapl);
        const std::string model = bench::cpu_model();
        omp_set_dynamic(0);

        std::vector<Config> configs;
        for (size_t index = 0; index < SHAPES.size(); ++index) {
            for (int threads : bench::THREAD_COUNTS) configs.push_back({index, threads});
        }
        bench::shuffle_configs(configs, options.seed);

#ifdef USE_NAIVE_CONV
        const char* backend = "naive_openmp";
#else
        const char* backend = "oneDNN";
#endif
        std::cout << "CONV2D | " << model
                  << " | backend=" << backend
                  << " | session=" << options.session_id
                  << " | reps=" << options.repetitions
                  << " | configs=" << configs.size()
                  << " | DRAM-RAPL=" << (rapl.dram_available() ? "yes" : "no") << '\n';

        int sequence = 0;
        size_t config_number = 0;
        for (const Config config : configs) {
            ++config_number;
            const Shape& shape = SHAPES[config.shape_index];
            std::cout << "[CONV2D] preparing config " << config_number << '/' << configs.size()
                      << ": shape=" << shape.id << " threads=" << config.threads << std::endl;
            float* input = bench::allocate_aligned(shape.input_elements());
            float* weights = bench::allocate_aligned(shape.weight_elements());
            float* result = bench::allocate_aligned(shape.output_elements());
            if (!input || !weights || !result) {
                free(input); free(weights); free(result);
                throw std::runtime_error("Allocation failed for CONV2D shape=" + std::to_string(shape.id));
            }

            omp_set_num_threads(config.threads);
            initialize(input, shape.input_elements(), weights, shape.weight_elements(),
                       result, shape.output_elements());
            ConvRunner runner(shape, input, weights, result);
            runner.prepare();
            runner.run(1);
            runner.copy_output();
            const double reference_checksum = sampled_checksum(result, shape.output_elements());
            const int batches = calibrate(runner);
            std::cout << "[CONV2D] calibrated config " << config_number << '/' << configs.size()
                      << ": batches=" << batches
                      << " implementation=" << runner.implementation() << std::endl;

            for (int repetition = 1; repetition <= options.repetitions; ++repetition) {
                const int clock_before = bench::average_online_cpu_frequency_mhz();
                const int temp_before = bench::cpu_temperature_c();
                const auto energy_before = rapl.read();
                const auto start = std::chrono::steady_clock::now();
                runner.run(batches);
                const auto end = std::chrono::steady_clock::now();
                const auto energy_after = rapl.read();
                const int clock_after = bench::average_online_cpu_frequency_mhz();
                const int temp_after = bench::cpu_temperature_c();

                runner.copy_output();
                const double checksum = sampled_checksum(result, shape.output_elements());
                const bool checksum_ok = checksum_matches(checksum, reference_checksum);
                const double seconds = std::chrono::duration<double>(end - start).count();
                const auto energy = rapl.delta(energy_before, energy_after);

                const auto row = bench::make_cpu_row(
                    options, ++sequence, repetition, "CONV2D", runner.implementation(), model,
                    config.threads, shape.id, shape.problem_spec(), batches, seconds, energy,
                    shape.flops_per_op(), shape.logical_bytes_per_op(), checksum_ok,
                    clock_before, clock_after, temp_before, temp_after);
                bench::write_row(output, row);
                output.flush();
                bench::print_result(row);
            }

            free(input); free(weights); free(result);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        return 2;
    }
}
