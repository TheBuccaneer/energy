#include "benchmark_common.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace {

constexpr int MAX_BATCHES = 10000000;
constexpr std::size_t BLOCK_SIZE = 4096;
constexpr long double MAX_RELATIVE_ERROR = 1.0e-4L;

const std::vector<std::size_t> SIZES{
    1000000, 2000000, 4000000, 8000000, 16000000,
    32000000, 64000000, 128000000, 256000000
};

struct Config {
    std::size_t n;
    int threads;
};

struct ReductionResult {
    float value;
    int observed_threads;
};

struct CheckResult {
    bool ok;
    long double relative_error;
};

inline float x_value(std::size_t i) {
    return 0.5f + static_cast<float>(i % 29) * 0.0078125f;
}

void initialize(float* x, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = x_value(i);
    }
}

int active_openmp_threads() {
    int active = 0;
#pragma omp parallel
    {
#pragma omp single
        active = omp_get_num_threads();
    }
    return active;
}

void require_thread_team(int requested, int observed, const char* phase) {
    if (observed != requested) {
        throw std::runtime_error(
            std::string("OpenMP team mismatch during ") + phase +
            ": requested=" + std::to_string(requested) +
            ", observed=" + std::to_string(observed));
    }
}

ReductionResult reduction_sum(const float* x, float* partials,
                              std::size_t n, std::size_t block_count,
                              int batches) {
    float final_result = 0.0f;
    int observed_threads = 0;

#pragma omp parallel shared(final_result, partials, observed_threads)
    {
#pragma omp single
        observed_threads = omp_get_num_threads();

        for (int batch = 0; batch < batches; ++batch) {
#pragma omp for schedule(static)
            for (std::size_t block = 0; block < block_count; ++block) {
                const std::size_t begin = block * BLOCK_SIZE;
                const std::size_t end = std::min(n, begin + BLOCK_SIZE);
                float local = 0.0f;

#pragma omp simd reduction(+:local)
                for (std::size_t i = begin; i < end; ++i) {
                    local += x[i];
                }
                partials[block] = local;
            }

#pragma omp single
            {
                float sum = 0.0f;
                for (std::size_t block = 0; block < block_count; ++block) {
                    sum += partials[block];
                }
                final_result = sum;
            }
        }
    }

    return {final_result, observed_threads};
}

int calibrate(const float* x, float* partials, std::size_t n,
              std::size_t block_count, int requested_threads) {
    ReductionResult warmup = reduction_sum(x, partials, n, block_count, 1);
    require_thread_team(requested_threads, warmup.observed_threads, "warm-up");

    int batches = 1;
    for (int step = 0; step < bench::MAX_CALIBRATION_STEPS; ++step) {
        const auto start = std::chrono::steady_clock::now();
        const ReductionResult result = reduction_sum(x, partials, n, block_count, batches);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        require_thread_team(requested_threads, result.observed_threads, "calibration");

        if (seconds >= bench::TARGET_RUNTIME_S || batches == MAX_BATCHES) {
            return batches;
        }
        batches = bench::scale_batches(seconds, batches, MAX_BATCHES);
    }
    return batches;
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

template <typename T>
std::optional<std::unordered_set<T>> read_filter(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return std::nullopt;
    }

    std::unordered_set<T> values;
    std::stringstream stream(raw);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            continue;
        }
        if constexpr (std::is_same_v<T, int>) {
            values.insert(std::stoi(token));
        } else {
            values.insert(static_cast<T>(std::stoull(token)));
        }
    }
    if (values.empty()) {
        throw std::runtime_error(std::string(name) + " contains no values");
    }
    return values;
}

std::vector<Config> build_configs() {
    const auto size_filter = read_filter<std::size_t>("BENCH_SIZE_FILTER");
    const auto thread_filter = read_filter<int>("BENCH_THREAD_FILTER");

    std::vector<Config> configs;
    for (std::size_t n : SIZES) {
        if (size_filter && !size_filter->count(n)) {
            continue;
        }
        for (int threads : bench::THREAD_COUNTS) {
            if (thread_filter && !thread_filter->count(threads)) {
                continue;
            }
            configs.push_back({n, threads});
        }
    }
    if (configs.empty()) {
        throw std::runtime_error("REDUCTION filters produced no configurations");
    }
    return configs;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bench::Options options =
            bench::parse_options(argc, argv, "reduction_intel.csv");
        const auto parent = std::filesystem::path(options.output_file).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Cannot open output file: " + options.output_file);
        }
        bench::write_header(output);

        bench::Rapl rapl;
        bench::require_rapl(rapl);
        const std::string model = bench::cpu_model();

        omp_set_dynamic(0);
        if (omp_get_dynamic() != 0) {
            throw std::runtime_error("OpenMP dynamic teams could not be disabled");
        }

        std::vector<Config> configs = build_configs();
        bench::shuffle_configs(configs, options.seed);

        std::cout << "REDUCTION(sum) | Intel | " << model
                  << " | session=" << options.session_id
                  << " | reps=" << options.repetitions
                  << " | configs=" << configs.size()
                  << " | DRAM-RAPL=" << (rapl.dram_available() ? "yes" : "no")
                  << '\n';

        int sequence = 0;
        for (const Config config : configs) {
            omp_set_num_threads(config.threads);
            require_thread_team(config.threads, active_openmp_threads(), "thread guard");

            const std::size_t block_count =
                (config.n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::unique_ptr<float, decltype(&std::free)> x(
                bench::allocate_aligned(config.n), &std::free);
            std::unique_ptr<float, decltype(&std::free)> partials(
                bench::allocate_aligned(block_count), &std::free);
            if (!x || !partials) {
                throw std::runtime_error(
                    "Allocation failed for REDUCTION N=" + std::to_string(config.n));
            }

            initialize(x.get(), config.n);
            const long double expected = expected_result(config.n);
            const int batches = calibrate(
                x.get(), partials.get(), config.n, block_count, config.threads);

            for (int repetition = 1; repetition <= options.repetitions; ++repetition) {
                const int clock_before = bench::average_online_cpu_frequency_mhz();
                const int temp_before = bench::cpu_temperature_c();
                const auto energy_before = rapl.read();
                const auto start = std::chrono::steady_clock::now();
                const ReductionResult result = reduction_sum(
                    x.get(), partials.get(), config.n, block_count, batches);
                const auto end = std::chrono::steady_clock::now();
                const auto energy_after = rapl.read();
                const int clock_after = bench::average_online_cpu_frequency_mhz();
                const int temp_after = bench::cpu_temperature_c();

                require_thread_team(
                    config.threads, result.observed_threads, "measurement");
                const CheckResult check = check_result(result.value, expected);
                const double seconds =
                    std::chrono::duration<double>(end - start).count();
                const auto energy = rapl.delta(energy_before, energy_after);
                const double flops_per_op = static_cast<double>(config.n - 1);
                const double logical_bytes_per_op =
                    static_cast<double>(config.n) * sizeof(float) + sizeof(float);

                const auto row = bench::make_cpu_row(
                    options, ++sequence, repetition,
                    "REDUCTION", "openmp_blocked_sum_fp32", model,
                    result.observed_threads, static_cast<long long>(config.n),
                    "elements=" + std::to_string(config.n), batches,
                    seconds, energy, flops_per_op, logical_bytes_per_op,
                    check.ok, clock_before, clock_after, temp_before, temp_after);
                bench::write_row(output, row);
                output.flush();
                bench::print_result(row);
                std::cout << "  relative_error=" << std::scientific
                          << static_cast<double>(check.relative_error)
                          << std::defaultfloat << '\n';

                if (!check.ok) {
                    throw std::runtime_error(
                        "REDUCTION checksum failed for N=" +
                        std::to_string(config.n) + ", threads=" +
                        std::to_string(config.threads));
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        return 2;
    }
}
