#include "benchmark_common.hpp"

#include <cblas.h>

extern "C" {
void openblas_set_num_threads(int);
int openblas_get_num_threads();
int openblas_get_parallel();
char* openblas_get_config();
}

namespace {

constexpr int MAX_BATCHES = 10000000;
const std::vector<int> SIZES{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

struct Config {
    int n;
    int threads;
};

std::vector<int> parse_filter(const char* name) {
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

const char* openblas_backend_name(int mode) {
    switch (mode) {
        case 0: return "sequential";
        case 1: return "pthreads";
        case 2: return "openmp";
        default: return "unknown";
    }
}

inline float value_a(int row, int col) {
    return 0.5f + static_cast<float>((row * 3 + col * 5) % 17) * 0.03125f;
}

inline float value_b(int row, int col) {
    return 0.25f + static_cast<float>((row * 7 + col * 11) % 19) * 0.0234375f;
}

void initialize(float* a, float* b, float* c, int n, int ld) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < ld; ++col) {
            const size_t index = static_cast<size_t>(row) * static_cast<size_t>(ld)
                               + static_cast<size_t>(col);
            if (col < n) {
                a[index] = value_a(row, col);
                b[index] = value_b(row, col);
            } else {
                a[index] = 0.0f;
                b[index] = 0.0f;
            }
            c[index] = 0.0f;
        }
    }
}

void gemm(const float* a, const float* b, float* c, int n, int ld) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0f, a, ld, b, ld, 0.0f, c, ld);
}

int calibrate(const float* a, const float* b, float* c, int n, int ld) {
    gemm(a, b, c, n, ld);
    int batches = 1;
    for (int step = 0; step < bench::MAX_CALIBRATION_STEPS; ++step) {
        const auto start = std::chrono::steady_clock::now();
        for (int batch = 0; batch < batches; ++batch) gemm(a, b, c, n, ld);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (seconds >= bench::TARGET_RUNTIME_S || batches == MAX_BATCHES) return batches;
        batches = bench::scale_batches(seconds, batches, MAX_BATCHES);
    }
    return batches;
}

double expected_value(int row, int col, int n) {
    long double sum = 0.0L;
    for (int k = 0; k < n; ++k) {
        sum += static_cast<long double>(value_a(row, k)) * value_b(k, col);
    }
    return static_cast<double>(sum);
}

bool correct(const float* c, int n, int ld) {
    const std::vector<std::pair<int, int>> samples{
        {0, 0}, {0, n - 1}, {n / 3, n / 2},
        {n / 2, n / 3}, {n - 1, 0}, {n - 1, n - 1}
    };
    for (const auto& [row, col] : samples) {
        const double expected = expected_value(row, col, n);
        const double actual = c[static_cast<size_t>(row) * static_cast<size_t>(ld) + col];
        const double relative = std::abs(actual - expected) / std::max(1.0, std::abs(expected));
        if (relative > 2.0e-3) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bench::Options options = bench::parse_options(argc, argv, "strided_gemm_amd.csv");
        const auto parent = std::filesystem::path(options.output_file).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot open output file: " + options.output_file);
        bench::write_header(output);

        bench::Rapl rapl;
        bench::require_rapl(rapl);
        const std::string model = bench::cpu_model();
        const std::vector<int> size_filter = parse_filter("BENCH_SIZE_FILTER");
        const std::vector<int> thread_filter = parse_filter("BENCH_THREAD_FILTER");
        const int openblas_parallel = openblas_get_parallel();
        if (openblas_parallel == 0) {
            throw std::runtime_error("OpenBLAS library is sequential; a threaded build is required");
        }
        const char* openblas_config = openblas_get_config();
        std::cout << "OpenBLAS backend=" << openblas_backend_name(openblas_parallel)
                  << " | config=" << (openblas_config ? openblas_config : "unknown") << '\n';

        std::vector<Config> configs;
        for (int n : SIZES) {
            if (!selected(n, size_filter)) continue;
            for (int threads : bench::THREAD_COUNTS) {
                if (selected(threads, thread_filter)) configs.push_back({n, threads});
            }
        }
        if (configs.empty()) throw std::runtime_error("No configurations remain after BENCH_*_FILTER");
        bench::shuffle_configs(configs, options.seed);

        std::cout << "STRIDED_GEMM(ld=2N) | " << model
                  << " | session=" << options.session_id
                  << " | reps=" << options.repetitions
                  << " | configs=" << configs.size()
                  << " | DRAM-RAPL=" << (rapl.dram_available() ? "yes" : "no") << '\n';

        int sequence = 0;
        for (const Config config : configs) {
            const int n = config.n;
            const int ld = 2 * n;
            const size_t count = static_cast<size_t>(n) * static_cast<size_t>(ld);
            float* a = bench::allocate_aligned(count);
            float* b = bench::allocate_aligned(count);
            float* c = bench::allocate_aligned(count);
            if (!a || !b || !c) {
                free(a); free(b); free(c);
                throw std::runtime_error("Allocation failed for STRIDED_GEMM N=" + std::to_string(n));
            }

            openblas_set_num_threads(config.threads);
            const int active_threads = openblas_get_num_threads();
            if (active_threads != config.threads) {
                free(a); free(b); free(c);
                throw std::runtime_error(
                    "OpenBLAS thread request not honored: requested=" + std::to_string(config.threads) +
                    ", active=" + std::to_string(active_threads));
            }
            std::cout << "[OpenBLAS] requested=" << config.threads
                      << " active=" << active_threads << '\n';
            initialize(a, b, c, n, ld);
            const int batches = calibrate(a, b, c, n, ld);

            for (int repetition = 1; repetition <= options.repetitions; ++repetition) {
                const int clock_before = bench::average_online_cpu_frequency_mhz();
                const int temp_before = bench::cpu_temperature_c();
                const auto energy_before = rapl.read();
                const auto start = std::chrono::steady_clock::now();
                for (int batch = 0; batch < batches; ++batch) gemm(a, b, c, n, ld);
                const auto end = std::chrono::steady_clock::now();
                const auto energy_after = rapl.read();
                const int clock_after = bench::average_online_cpu_frequency_mhz();
                const int temp_after = bench::cpu_temperature_c();

                const double seconds = std::chrono::duration<double>(end - start).count();
                const auto energy = rapl.delta(energy_before, energy_after);
                const bool checksum_ok = correct(c, n, ld);
                const double flops_per_op = 2.0 * n * static_cast<double>(n) * n;
                const double logical_bytes_per_op = 3.0 * n * static_cast<double>(n) * sizeof(float);

                // bench::make_cpu_row sets execution_mode to cpu_native for CPU rows.
                const auto row = bench::make_cpu_row(
                    options, ++sequence, repetition, "STRIDED_GEMM", "openblas_sgemm_ld2n", model,
                    config.threads, n, "N=" + std::to_string(n) + ";ld=" + std::to_string(ld), batches, seconds, energy,
                    flops_per_op, logical_bytes_per_op, checksum_ok,
                    clock_before, clock_after, temp_before, temp_after);
                bench::write_row(output, row);
                output.flush();
                bench::print_result(row);
            }

            free(a); free(b); free(c);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        return 2;
    }
}
