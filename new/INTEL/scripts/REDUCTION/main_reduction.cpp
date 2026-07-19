#include "benchmark_common.hpp"

#include <omp.h>

namespace {

constexpr int MAX_BATCHES = 10000000;
constexpr size_t BLOCK_SIZE = 4096;
const std::vector<size_t> SIZES{
    1000000, 2000000, 4000000, 8000000, 16000000,
    32000000, 64000000, 128000000, 256000000
};

struct Config {
    size_t n;
    int threads;
};

inline float x_value(size_t i) {
    return 0.5f + static_cast<float>(i % 29) * 0.0078125f;
}

void initialize(float* x, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) x[i] = x_value(i);
}

float reduction_sum(const float* x, float* partials,
                    size_t n, size_t block_count, int batches) {
    float batch_result = 0.0f;
    float final_result = 0.0f;
#pragma omp parallel shared(batch_result, final_result, partials)
    {
        for (int batch = 0; batch < batches; ++batch) {
#pragma omp for schedule(static)
            for (size_t block = 0; block < block_count; ++block) {
                const size_t begin = block * BLOCK_SIZE;
                const size_t end = std::min(n, begin + BLOCK_SIZE);
                float local = 0.0f;
                for (size_t i = begin; i < end; ++i) local += x[i];
                partials[block] = local;
            }
#pragma omp single
            batch_result = 0.0f;
#pragma omp for reduction(+:batch_result) schedule(static)
            for (size_t block = 0; block < block_count; ++block) {
                batch_result += partials[block];
            }
#pragma omp single
            final_result = batch_result;
        }
    }
    return final_result;
}

int calibrate(const float* x, float* partials, size_t n, size_t block_count) {
    volatile float sink = reduction_sum(x, partials, n, block_count, 1);
    (void)sink;
    int batches = 1;
    for (int step = 0; step < bench::MAX_CALIBRATION_STEPS; ++step) {
        const auto start = std::chrono::steady_clock::now();
        sink = reduction_sum(x, partials, n, block_count, batches);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (seconds >= bench::TARGET_RUNTIME_S || batches == MAX_BATCHES) return batches;
        batches = bench::scale_batches(seconds, batches, MAX_BATCHES);
    }
    return batches;
}

double expected_result(size_t n) {
    long double sum = 0.0L;
    for (size_t i = 0; i < n; ++i) sum += x_value(i);
    return static_cast<double>(sum);
}

bool correct(float actual, double expected) {
    const double relative = std::abs(static_cast<double>(actual) - expected) /
                            std::max(1.0, std::abs(expected));
    return relative <= 2.0e-3;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bench::Options options = bench::parse_options(argc, argv, "reduction_intel.csv");
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

        std::cout << "REDUCTION(sum) | " << model
                  << " | session=" << options.session_id
                  << " | reps=" << options.repetitions
                  << " | configs=" << configs.size()
                  << " | DRAM-RAPL=" << (rapl.dram_available() ? "yes" : "no") << '\n';

        int sequence = 0;
        for (const Config config : configs) {
            const size_t block_count = (config.n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            float* x = bench::allocate_aligned(config.n);
            float* partials = bench::allocate_aligned(block_count);
            if (!x || !partials) {
                free(x); free(partials);
                throw std::runtime_error("Allocation failed for REDUCTION N=" + std::to_string(config.n));
            }

            omp_set_num_threads(config.threads);
            initialize(x, config.n);
            const double expected = expected_result(config.n);
            const int batches = calibrate(x, partials, config.n, block_count);

            for (int repetition = 1; repetition <= options.repetitions; ++repetition) {
                const int clock_before = bench::average_online_cpu_frequency_mhz();
                const int temp_before = bench::cpu_temperature_c();
                const auto energy_before = rapl.read();
                const auto start = std::chrono::steady_clock::now();
                const float result = reduction_sum(x, partials, config.n, block_count, batches);
                const auto end = std::chrono::steady_clock::now();
                const auto energy_after = rapl.read();
                const int clock_after = bench::average_online_cpu_frequency_mhz();
                const int temp_after = bench::cpu_temperature_c();

                const double seconds = std::chrono::duration<double>(end - start).count();
                const auto energy = rapl.delta(energy_before, energy_after);
                const bool checksum_ok = correct(result, expected);
                const double flops_per_op = config.n > 0
                    ? static_cast<double>(config.n - 1)
                    : 0.0;
                const double logical_bytes_per_op =
                    static_cast<double>(config.n) * sizeof(float) + sizeof(float);

                const auto row = bench::make_cpu_row(
                    options, ++sequence, repetition, "REDUCTION", "openmp_sum_reduction", model,
                    config.threads, static_cast<long long>(config.n),
                    "elements=" + std::to_string(config.n), batches,
                    seconds, energy, flops_per_op, logical_bytes_per_op,
                    checksum_ok, clock_before, clock_after, temp_before, temp_after);
                bench::write_row(output, row);
                output.flush();
                bench::print_result(row);
            }

            free(x); free(partials);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        return 2;
    }
}
