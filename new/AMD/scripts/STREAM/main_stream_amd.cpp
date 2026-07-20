#include "benchmark_common.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int MAX_BATCHES = 10000000;
constexpr float SCALAR = 3.0f;

const std::vector<size_t> SIZES{
    1000000, 2000000, 4000000, 8000000, 16000000,
    32000000, 64000000, 128000000, 256000000
};

struct Config {
    size_t n;
    int threads;
};

struct FreeDeleter {
    void operator()(float* pointer) const noexcept {
        std::free(pointer);
    }
};

using FloatBuffer = std::unique_ptr<float, FreeDeleter>;

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

std::vector<int> parse_thread_filter(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return {};

    std::vector<int> values;
    std::stringstream stream(raw);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
    }
    return values;
}

template <typename T>
bool selected(const T value, const std::vector<T>& filter) {
    return filter.empty()
        || std::find(filter.begin(), filter.end(), value) != filter.end();
}

int active_openmp_threads() {
    int active_threads = 0;
#pragma omp parallel
    {
#pragma omp single
        active_threads = omp_get_num_threads();
    }
    return active_threads;
}

void initialize(
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ c,
    const size_t n
) {
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        a[i] = 0.0f;
        b[i] = 1.0f + static_cast<float>(i % 17) * 0.01f;
        c[i] = 0.5f + static_cast<float>(i % 13) * 0.02f;
    }
}

void stream_triad(
    float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    const size_t n,
    const int batches
) {
#pragma omp parallel
    {
        for (int batch = 0; batch < batches; ++batch) {
#pragma omp for simd schedule(static)
            for (size_t i = 0; i < n; ++i) {
                a[i] = b[i] + SCALAR * c[i];
            }
        }
    }
}

int calibrate(
    float* a,
    const float* b,
    const float* c,
    const size_t n
) {
    stream_triad(a, b, c, n, 1);

    int batches = 1;
    for (int step = 0; step < bench::MAX_CALIBRATION_STEPS; ++step) {
        const auto start = std::chrono::steady_clock::now();
        stream_triad(a, b, c, n, batches);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start
        ).count();

        if (seconds >= bench::TARGET_RUNTIME_S || batches == MAX_BATCHES) {
            return batches;
        }
        batches = bench::scale_batches(seconds, batches, MAX_BATCHES);
    }
    return batches;
}

bool correct(
    const float* a,
    const float* b,
    const float* c,
    const size_t n
) {
    const std::vector<size_t> samples{0, n / 7, n / 2, n - 1};

    for (const size_t i : samples) {
        const double expected =
            static_cast<double>(b[i])
            + static_cast<double>(SCALAR) * static_cast<double>(c[i]);
        const double actual = static_cast<double>(a[i]);

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

}  // namespace

int main(int argc, char** argv) {
    try {
        const bench::Options options =
            bench::parse_options(argc, argv, "stream_amd.csv");

        const auto parent =
            std::filesystem::path(options.output_file).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "Cannot open output file: " + options.output_file
            );
        }
        bench::write_header(output);

        bench::Rapl rapl;
        bench::require_rapl(rapl);

        const std::string model = bench::cpu_model();
        const std::vector<size_t> size_filter =
            parse_size_filter("BENCH_SIZE_FILTER");
        const std::vector<int> thread_filter =
            parse_thread_filter("BENCH_THREAD_FILTER");

        omp_set_dynamic(0);

        std::vector<Config> configs;
        for (const size_t n : SIZES) {
            if (!selected(n, size_filter)) continue;

            for (const int threads : bench::THREAD_COUNTS) {
                if (selected(threads, thread_filter)) {
                    configs.push_back({n, threads});
                }
            }
        }

        if (configs.empty()) {
            throw std::runtime_error(
                "No configurations remain after BENCH_*_FILTER"
            );
        }

        bench::shuffle_configs(configs, options.seed);

        std::cout
            << "STREAM | " << model
            << " | platform=AMD"
            << " | session=" << options.session_id
            << " | reps=" << options.repetitions
            << " | configs=" << configs.size()
            << " | DRAM-RAPL="
            << (rapl.dram_available() ? "yes" : "no")
            << '\n';

        int sequence = 0;

        for (const Config config : configs) {
            FloatBuffer a(bench::allocate_aligned(config.n));
            FloatBuffer b(bench::allocate_aligned(config.n));
            FloatBuffer c(bench::allocate_aligned(config.n));

            if (!a || !b || !c) {
                throw std::runtime_error(
                    "Allocation failed for STREAM N="
                    + std::to_string(config.n)
                );
            }

            omp_set_num_threads(config.threads);
            const int active_threads = active_openmp_threads();

            if (active_threads != config.threads) {
                throw std::runtime_error(
                    "OpenMP thread request not honored: requested="
                    + std::to_string(config.threads)
                    + ", active="
                    + std::to_string(active_threads)
                );
            }

            std::cout
                << "[OpenMP] requested=" << config.threads
                << " active=" << active_threads
                << '\n';

            initialize(a.get(), b.get(), c.get(), config.n);
            const int batches =
                calibrate(a.get(), b.get(), c.get(), config.n);

            for (
                int repetition = 1;
                repetition <= options.repetitions;
                ++repetition
            ) {
                const int clock_before =
                    bench::average_online_cpu_frequency_mhz();
                const int temp_before =
                    bench::cpu_temperature_c();

                const auto energy_before = rapl.read();
                const auto start = std::chrono::steady_clock::now();

                stream_triad(
                    a.get(),
                    b.get(),
                    c.get(),
                    config.n,
                    batches
                );

                const auto end = std::chrono::steady_clock::now();
                const auto energy_after = rapl.read();

                const int clock_after =
                    bench::average_online_cpu_frequency_mhz();
                const int temp_after =
                    bench::cpu_temperature_c();

                const double seconds =
                    std::chrono::duration<double>(end - start).count();
                const auto energy =
                    rapl.delta(energy_before, energy_after);

                const bool checksum_ok =
                    correct(a.get(), b.get(), c.get(), config.n);

                const double flops_per_op =
                    2.0 * static_cast<double>(config.n);

                const double logical_bytes_per_op =
                    3.0
                    * static_cast<double>(config.n)
                    * sizeof(float);

                const auto row = bench::make_cpu_row(
                    options,
                    ++sequence,
                    repetition,
                    "STREAM",
                    "openmp_triad",
                    model,
                    active_threads,
                    static_cast<long long>(config.n),
                    "elements=" + std::to_string(config.n),
                    batches,
                    seconds,
                    energy,
                    flops_per_op,
                    logical_bytes_per_op,
                    checksum_ok,
                    clock_before,
                    clock_after,
                    temp_before,
                    temp_after
                );

                bench::write_row(output, row);
                output.flush();
                bench::print_result(row);
            }
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        return 2;
    }
}
