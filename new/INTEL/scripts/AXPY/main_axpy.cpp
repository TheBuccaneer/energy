#include "benchmark_common.hpp"

#include <omp.h>

namespace {

constexpr int MAX_BATCHES = 10000000;
constexpr float ALPHA = 3.0f;
const std::vector<size_t> SIZES{
    1000000, 2000000, 4000000, 8000000, 16000000,
    32000000, 64000000, 128000000, 256000000
};

struct Config {
    size_t n;
    int threads;
};

inline float x_value(size_t i) {
    return 0.25f + static_cast<float>(i % 23) * 0.015625f;
}

inline float y_initial(size_t i) {
    return 0.5f + static_cast<float>(i % 19) * 0.0078125f;
}

void initialize_x(float* x, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) x[i] = x_value(i);
}

void reset_y(float* y, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) y[i] = y_initial(i);
}

void axpy(const float* x, float* y, size_t n, int batches) {
#pragma omp parallel
    {
        for (int batch = 0; batch < batches; ++batch) {
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                y[i] = std::fma(ALPHA, x[i], y[i]);
            }
        }
    }
}

int calibrate(const float* x, float* y, size_t n) {
    int batches = 1;
    for (int step = 0; step < bench::MAX_CALIBRATION_STEPS; ++step) {
        reset_y(y, n);
        const auto start = std::chrono::steady_clock::now();
        axpy(x, y, n, batches);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (seconds >= bench::TARGET_RUNTIME_S || batches == MAX_BATCHES) return batches;
        batches = bench::scale_batches(seconds, batches, MAX_BATCHES);
    }
    return batches;
}

bool correct(const float* x, const float* y, size_t n, int batches) {
    const std::vector<size_t> samples{0, n / 7, n / 2, n - 1};
    for (size_t i : samples) {
        float expected = y_initial(i);
        for (int batch = 0; batch < batches; ++batch) {
            expected = std::fma(ALPHA, x[i], expected);
        }
        const double actual = y[i];
        const double relative = std::abs(actual - expected) / std::max(1.0, std::abs(static_cast<double>(expected)));
        if (relative > 1.0e-6) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bench::Options options = bench::parse_options(argc, argv, "axpy_intel.csv");
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
        for (size_t n : SIZES) {
            for (int threads : bench::THREAD_COUNTS) configs.push_back({n, threads});
        }
        bench::shuffle_configs(configs, options.seed);

        std::cout << "AXPY | " << model
                  << " | session=" << options.session_id
                  << " | reps=" << options.repetitions
                  << " | configs=" << configs.size()
                  << " | DRAM-RAPL=" << (rapl.dram_available() ? "yes" : "no") << '\n';

        int sequence = 0;
        for (const Config config : configs) {
            float* x = bench::allocate_aligned(config.n);
            float* y = bench::allocate_aligned(config.n);
            if (!x || !y) {
                free(x); free(y);
                throw std::runtime_error("Allocation failed for AXPY N=" + std::to_string(config.n));
            }

            omp_set_num_threads(config.threads);
            initialize_x(x, config.n);
            reset_y(y, config.n);
            const int batches = calibrate(x, y, config.n);

            for (int repetition = 1; repetition <= options.repetitions; ++repetition) {
                reset_y(y, config.n);
                const int clock_before = bench::average_online_cpu_frequency_mhz();
                const int temp_before = bench::cpu_temperature_c();
                const auto energy_before = rapl.read();
                const auto start = std::chrono::steady_clock::now();
                axpy(x, y, config.n, batches);
                const auto end = std::chrono::steady_clock::now();
                const auto energy_after = rapl.read();
                const int clock_after = bench::average_online_cpu_frequency_mhz();
                const int temp_after = bench::cpu_temperature_c();

                const double seconds = std::chrono::duration<double>(end - start).count();
                const auto energy = rapl.delta(energy_before, energy_after);
                const bool checksum_ok = correct(x, y, config.n, batches);
                const double flops_per_op = 2.0 * static_cast<double>(config.n);
                const double logical_bytes_per_op = 3.0 * static_cast<double>(config.n) * sizeof(float);

                const auto row = bench::make_cpu_row(
                    options, ++sequence, repetition, "AXPY", "openmp_axpy", model,
                    config.threads, static_cast<long long>(config.n), "elements=" + std::to_string(config.n), batches,
                    seconds, energy, flops_per_op, logical_bytes_per_op,
                    checksum_ok, clock_before, clock_after, temp_before, temp_after);
                bench::write_row(output, row);
                output.flush();
                bench::print_result(row);
            }

            free(x); free(y);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        return 2;
    }
}
