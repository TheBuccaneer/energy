// AXPY v1.0 FROZEN -- CPU implementation (AMD)
//
// Authoritative source: AXPY_MEASUREMENT_CONTRACT_v1_0_FROZEN_2026-07-25.md
// Patch/audit closure:  AXPY_CONTRACT_v1_0_PATCH_AND_AUDIT_CLOSURE.md
//
// implementation = openmp_axpy_inplace_fp32
// execution_mode = cpu_native
//
// This file must remain semantically identical to the sibling platform file
// after normalizing: platform label, default output filename, and the
// hardware-mandated thread grid (contract 3.2). No other divergence is
// permitted (contract 13.1 / CODING-AUFTRAG section 5).

#include "benchmark_common.hpp"

#include <omp.h>

#include <algorithm>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------
// Frozen AXPY semantics (contract sections 1, 2, 4) -- do not modify.
// ---------------------------------------------------------------------

constexpr float ALPHA = 3.0f;
constexpr long long MAX_BATCHES = 250000;
constexpr int MAX_CALIBRATION_STEPS_AXPY = 12;

const std::vector<size_t> SIZES{
    1000000, 2000000, 4000000, 8000000, 16000000,
    32000000, 64000000, 128000000, 256000000
};

// PATCH F1: the thread grid is taken exclusively from the shared
// bench::THREAD_COUNTS definition in the real benchmark_common.hpp. No
// locally invented, platform-specific thread table is used here, so that
// after normalizing only the platform label and default output filename,
// the AMD and Intel sources are otherwise identical.
int platform_max_threads() {
    return *std::max_element(bench::THREAD_COUNTS.begin(), bench::THREAD_COUNTS.end());
}

struct Config {
    size_t n;
    int threads;
};

// ---------------------------------------------------------------------
// Deterministic inputs (contract 4.1)
// ---------------------------------------------------------------------

inline int kx_of(size_t i) {
    return static_cast<int>(i % 29) - 14;
}

inline int ky_of(size_t i) {
    return static_cast<int>(i % 31) - 15;
}

inline float x_value(size_t i) {
    return std::ldexp(static_cast<float>(kx_of(i)), -16);
}

inline float y0_value(size_t i) {
    return std::ldexp(static_cast<float>(ky_of(i)), -8);
}

// ---------------------------------------------------------------------
// Contract-mandated CPU self-test (section 8): the integer coefficient
// bound and all 899 (=29*31) periodic (kx,ky) states must be exact FP32
// representable multiples of 2^-16 for every batches in [0, MAX_BATCHES].
// This only checks the algebraic bound, not a live run; it is cheap and
// runs once at process start.
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
            const long long coeff_at_max =
                256LL * ky + 3LL * MAX_BATCHES * kx;
            if (coeff_at_max > kMaxAbsCoefficient ||
                coeff_at_max < -kMaxAbsCoefficient) {
                throw std::runtime_error(
                    "AXPY contract self-test failed: coefficient out of "
                    "the exact-FP32-representable bound at MAX_BATCHES");
            }
            const float as_float = static_cast<float>(coeff_at_max);
            if (static_cast<long long>(as_float) != coeff_at_max) {
                throw std::runtime_error(
                    "AXPY contract self-test failed: coefficient not "
                    "exactly representable as FP32 integer");
            }
        }
    }
    // 899 = 29*31 periodic (kx,ky) states are exactly the ranges iterated
    // above (kx in [-14,14], ky in [-15,15]); no further enumeration over
    // index i is required since kx_of/ky_of are periodic in i with those
    // exact periods.
}

// ---------------------------------------------------------------------
// Mandatory OpenMP kernel structure (contract 6.2) -- verbatim shape.
// ---------------------------------------------------------------------

void axpy_batches(
    const float* __restrict__ x,
    float* __restrict__ y,
    size_t n,
    long long batches
) {
#pragma omp parallel
    {
        for (long long batch = 0; batch < batches; ++batch) {
#pragma omp for simd schedule(static)
            for (size_t i = 0; i < n; ++i) {
                y[i] = ALPHA * x[i] + y[i];
            }
        }
    }
}

void initialize_x(float* __restrict__ x, size_t n) {
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        x[i] = x_value(i);
    }
}

// Reset y to y0 outside the timing/energy window (contract 4.3). Runs
// under the currently configured thread count so first-touch placement
// matches the measured configuration (contract 6.4).
void reset_y(float* __restrict__ y, size_t n) {
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        y[i] = y0_value(i);
    }
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

void require_thread_team(int requested, int observed, const char* phase) {
    if (observed != requested) {
        throw std::runtime_error(
            std::string("OpenMP team mismatch during ") + phase +
            ": requested=" + std::to_string(requested) +
            ", active=" + std::to_string(observed));
    }
}

// ---------------------------------------------------------------------
// Exact FP32 checksum (contract 5) -- no algebraic batch collapse.
// ---------------------------------------------------------------------

struct ChecksumResult {
    bool ok{false};
    double max_abs_error{0.0};
    double max_rel_error{0.0};
};

std::vector<size_t> checksum_sample_indices(size_t n) {
    return {0, 1, n / 7, n / 3, n / 2, (2 * n) / 3, n - 2, n - 1};
}

ChecksumResult check_axpy(
    const float* x,
    const float* y,
    size_t n,
    long long batches
) {
    ChecksumResult result;
    result.ok = true;

    for (const size_t i : checksum_sample_indices(n)) {
        const int kx = kx_of(i);
        const int ky = ky_of(i);
        const long long expected_coefficient =
            256LL * ky + 3LL * batches * kx;
        const float expected_y =
            std::ldexp(static_cast<float>(expected_coefficient), -16);
        const float expected_x = x_value(i);
        const float actual_x = x[i];
        const float actual_y = y[i];

        const bool finite_ok =
            std::isfinite(actual_x) && std::isfinite(actual_y) &&
            std::isfinite(expected_x) && std::isfinite(expected_y);

        const double abs_err_x = std::abs(
            static_cast<double>(actual_x) - static_cast<double>(expected_x));
        const double abs_err_y = std::abs(
            static_cast<double>(actual_y) - static_cast<double>(expected_y));
        result.max_abs_error =
            std::max({result.max_abs_error, abs_err_x, abs_err_y});

        const double rel_err_x = abs_err_x /
            std::max(1.0, std::abs(static_cast<double>(expected_x)));
        const double rel_err_y = abs_err_y /
            std::max(1.0, std::abs(static_cast<double>(expected_y)));
        result.max_rel_error =
            std::max({result.max_rel_error, rel_err_x, rel_err_y});

        const bool exact_ok =
            finite_ok && (actual_x == expected_x) && (actual_y == expected_y);
        if (!exact_ok) {
            result.ok = false;
        }
    }
    return result;
}

// ---------------------------------------------------------------------
// Adaptive calibration (contract 8.2) -- exact contract formula, no
// additional per-step growth cap beyond MAX_BATCHES.
// ---------------------------------------------------------------------

long long axpy_scale_batches(double calibration_time_s, long long current) {
    const double safe_seconds = std::max(calibration_time_s, 1.0e-12);
    const long long estimate = static_cast<long long>(std::ceil(
        bench::TARGET_RUNTIME_S * static_cast<double>(current) / safe_seconds));
    const long long next = std::max<long long>(current + 1, estimate);
    return std::min<long long>(MAX_BATCHES, next);
}

long long calibrate(float* x, float* y, size_t n) {
    // Warm-up (contract 8.1): x already initialized, y already = y0.
    axpy_batches(x, y, n, 1);
    reset_y(y, n);

    long long batches = 1;
    for (int step = 0; step < MAX_CALIBRATION_STEPS_AXPY; ++step) {
        reset_y(y, n);  // reset outside the timing window, every step
        const auto start = std::chrono::steady_clock::now();
        axpy_batches(x, y, n, batches);
        const auto end = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration<double>(end - start).count();

        if (!std::isfinite(seconds) || seconds <= 0.0) {
            throw std::runtime_error(
                "Non-finite or non-positive calibration time for N=" +
                std::to_string(n));
        }
        if (seconds >= bench::TARGET_RUNTIME_S || batches == MAX_BATCHES) {
            return batches;
        }
        batches = axpy_scale_batches(seconds, batches);
    }
    return batches;
}

// ---------------------------------------------------------------------
// Anti-collapse scaling probe (contract 12.4). Opt-in via
// AXPY_ANTI_COLLAPSE_PROBE=1; runs once at N=1,000,000 using the
// platform-maximum thread count. Prints a structured PASS/FAIL line and
// never writes CSV rows (it augments, but is not part of, the regular
// quickcheck row count).
// ---------------------------------------------------------------------

void run_anti_collapse_probe() {
    const size_t n = 1000000;
    const int threads = platform_max_threads();

    omp_set_num_threads(threads);
    require_thread_team(
        threads, active_openmp_threads(), "anti-collapse thread guard");

    std::unique_ptr<float, void(*)(void*)> x(
        bench::allocate_aligned(n), &std::free);
    std::unique_ptr<float, void(*)(void*)> y(
        bench::allocate_aligned(n), &std::free);
    if (!x || !y) {
        throw std::runtime_error("Allocation failed for anti-collapse probe");
    }

    initialize_x(x.get(), n);
    reset_y(y.get(), n);
    const long long b_cal = calibrate(x.get(), y.get(), n);

    long long b_probe = std::min<long long>(
        std::max<long long>(100, b_cal / 4), MAX_BATCHES / 2);

    auto measure = [&](long long batches) {
        reset_y(y.get(), n);
        const auto start = std::chrono::steady_clock::now();
        axpy_batches(x.get(), y.get(), n, batches);
        const auto end = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration<double>(end - start).count();
        const ChecksumResult checksum = check_axpy(x.get(), y.get(), n, batches);
        return std::make_pair(seconds, checksum);
    };

    // PATCH F4: deterministically grow B_probe if the 20 ms floor is not
    // met, while 2*B_probe <= MAX_BATCHES (contract 12.4). The previous
    // version could recompute the same B_probe forever once it reached
    // MAX_BATCHES/2 (min(B_probe*2, MAX_BATCHES/2) == B_probe again),
    // looping without bound. This version hard-fails as soon as the cap
    // is reached without success, and additionally asserts strict forward
    // progress on every growth step as a defensive invariant.
    const long long max_probe = MAX_BATCHES / 2;
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
                "Anti-collapse probe made no forward progress "
                "(B_probe=" + std::to_string(b_probe) + ")");
        }
        b_probe = next;
    }

    const long long two_b_probe = 2 * b_probe;
    const auto [t2, c2] = measure(two_b_probe);

    const bool duration_ok = (t1 >= 0.020) && (t2 >= 0.020);
    const bool cap_ok = (two_b_probe <= MAX_BATCHES);
    const bool checksum_ok = c1.ok && c2.ok;
    const double ratio = t1 > 0.0
        ? t2 / t1
        : std::numeric_limits<double>::infinity();
    const bool ratio_ok = std::isfinite(ratio) && ratio >= 1.7 && ratio <= 2.3;
    const bool gate_pass = duration_ok && cap_ok && checksum_ok && ratio_ok;

    std::cout << std::setprecision(9)
              << "[ANTI_COLLAPSE] N=" << n
              << " threads=" << threads
              << " B_cal=" << b_cal
              << " B_probe=" << b_probe
              << " two_B_probe=" << two_b_probe
              << " t1=" << t1
              << " t2=" << t2
              << " ratio=" << ratio
              << " time_basis=wall_time_s"
              << " checksum1=" << (c1.ok ? "OK" : "FAIL")
              << " checksum2=" << (c2.ok ? "OK" : "FAIL")
              << " gate=" << (gate_pass ? "PASS" : "FAIL")
              << '\n';

    if (!gate_pass) {
        throw std::runtime_error("Anti-collapse gate FAILED (contract 12.4)");
    }
}

// ---------------------------------------------------------------------
// CSV row (contract 9.1 -- exact 45-column schema and 9.3/9.5 formulas
// and lossless serialization rules). Deliberately NOT reusing
// bench::ResultRow/write_row/make_cpu_row: those helpers aggregate
// total_energy_j (package+DRAM) into the power/energy formula columns
// and serialize flops_total/logical_bytes_per_op/energy fields in
// rounded scientific notation. The AXPY contract instead mandates
// device_energy_j-only formulas and exact/max_digits10 serialization,
// so this file defines its own row type and writer to match the
// contract exactly (contract precedence over reference conventions,
// per CODING-AUFTRAG "Autorisierte Grundlage").
// ---------------------------------------------------------------------

struct AxpyRow {
    std::string session_id;
    int sequence_index{};
    int run_id_global{};
    int repetition{};
    std::string device_name;
    int num_threads{};
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
    int clock_before_mhz{-1};
    int clock_after_mhz{-1};
    int temp_c{-1};
    int temp_before_c{-1};
    int temp_after_c{-1};
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

// Writes a double with lossless (max_digits10) precision, using default
// (non-scientific-rounded) formatting -- contract 9.5: "keine
// Scientific-6-Rundung fuer Formelanker".
void write_lossless_double(std::ofstream& output, double value) {
    output << std::defaultfloat
           << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
}

void write_axpy_row(std::ofstream& output, const AxpyRow& row) {
    output << bench::SCHEMA_VERSION << ',' << bench::timestamp() << ','
           << bench::csv_escape(row.session_id) << ','
           << row.sequence_index << ',' << row.run_id_global << ','
           << row.repetition << ','
           << bench::csv_escape("AXPY") << ','
           << bench::csv_escape("openmp_axpy_inplace_fp32") << ','
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

    // Exact decimal integers (contract 9.5) -- never floating formatted.
    output << row.flops_total << ',';
    write_lossless_double(output, row.gflops_per_s); output << ',';
    output << row.logical_bytes_per_op << ',';
    write_lossless_double(output, row.avg_power_w); output << ',';

    // PATCH F2: pcie_gen, pcie_width, sm_clock_mhz, mem_clock_mhz are
    // explicit -1 sentinels on CPU rows (contract 9.4), never blank cells.
    output << row.runtime_status << ','
           << -1 << ',' << -1 << ','           // pcie_gen, pcie_width
           << -1 << ','                        // sm_clock_mhz (not meaningful on CPU)
           << row.clock_before_mhz << ','
           << row.clock_after_mhz << ','
           << -1 << ','                        // mem_clock_mhz (GPU-only)
           << row.temp_c << ',' << row.temp_before_c << ',' << row.temp_after_c << ','
           << ','                               // throttle_reasons (CPU: none observed; blank permitted)
           << -1 << ',' << -1 << ',' << std::fixed << std::setprecision(6) << -1.0
           << std::defaultfloat << ',' << -1 << ','
           << (row.checksum_ok ? 't' : 'f') << '\n';
}

void print_axpy_result(const AxpyRow& row) {
    std::cout << "[AXPY] N=" << row.problem_size
              << " threads=" << row.num_threads
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
// Env-var config filters (BENCH_SIZE_FILTER / BENCH_THREAD_FILTER),
// following the established STREAM/REDUCTION reference convention.
// ---------------------------------------------------------------------

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

std::vector<int> parse_thread_filter(const char* name) {
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

}  // namespace

int main(int argc, char** argv) {
    try {
        run_contract_selftest();

        const bench::Options options =
            bench::parse_options(argc, argv, "axpy_amd.csv");

        const auto parent = std::filesystem::path(options.output_file).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Cannot open output file: " + options.output_file);
        }
        write_axpy_header(output);

        bench::Rapl rapl;
        bench::require_rapl(rapl);
        const std::string model = bench::cpu_model();

        omp_set_dynamic(0);
        if (omp_get_dynamic() != 0) {
            throw std::runtime_error("OpenMP dynamic teams could not be disabled");
        }

        const std::vector<size_t> size_filter = parse_size_filter("BENCH_SIZE_FILTER");
        const std::vector<int> thread_filter = parse_thread_filter("BENCH_THREAD_FILTER");

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
            throw std::runtime_error("No configurations remain after BENCH_*_FILTER");
        }

        bench::shuffle_configs(configs, options.seed);

        std::cout << "AXPY | " << model
                  << " | platform=AMD"
                  << " | implementation=openmp_axpy_inplace_fp32"
                  << " | session=" << options.session_id
                  << " | reps=" << options.repetitions
                  << " | configs=" << configs.size()
                  << " | DRAM-RAPL=" << (rapl.dram_available() ? "yes" : "no")
                  << '\n';

        int sequence = 0;
        for (const Config config : configs) {
            omp_set_num_threads(config.threads);
            const int active_threads = active_openmp_threads();
            require_thread_team(config.threads, active_threads, "thread guard");

            std::unique_ptr<float, void(*)(void*)> x(
                bench::allocate_aligned(config.n), &std::free);
            std::unique_ptr<float, void(*)(void*)> y(
                bench::allocate_aligned(config.n), &std::free);
            if (!x || !y) {
                throw std::runtime_error(
                    "Allocation failed for AXPY N=" + std::to_string(config.n));
            }

            initialize_x(x.get(), config.n);   // x initialized once, never modified again
            reset_y(y.get(), config.n);        // first-touch y = y0

            const long long batches = calibrate(x.get(), y.get(), config.n);
            std::cout << "[CALIBRATION] N=" << config.n
                      << " threads=" << config.threads
                      << " batches=" << batches << '\n';

            if (config.n == 1000000 && batches == MAX_BATCHES) {
                // Batchcap-at-1M hard-fail check is evaluated against the
                // calibration time basis (wall_time_s on CPU) per contract
                // 4.4 / CODING-AUFTRAG section 7. The actual < 0.75 s check
                // happens against the freshly measured value below, since
                // calibrate() does not return timing; recompute once more.
                reset_y(y.get(), config.n);
                const auto start = std::chrono::steady_clock::now();
                axpy_batches(x.get(), y.get(), config.n, batches);
                const auto end = std::chrono::steady_clock::now();
                const double seconds = std::chrono::duration<double>(end - start).count();
                if (seconds < bench::MIN_RUNTIME_S) {
                    throw std::runtime_error(
                        "Hard quickcheck failure: N=1M at MAX_BATCHES stays below "
                        "0.75s on the calibration time basis (contract 4.4)");
                }
            }

            for (int repetition = 1; repetition <= options.repetitions; ++repetition) {
                reset_y(y.get(), config.n);  // reset outside window before every rep

                const int clock_before = bench::average_online_cpu_frequency_mhz();
                const int temp_before = bench::cpu_temperature_c();
                const auto energy_before = rapl.read();
                const auto start = std::chrono::steady_clock::now();

                axpy_batches(x.get(), y.get(), config.n, batches);

                const auto end = std::chrono::steady_clock::now();
                const auto energy_after = rapl.read();
                const int clock_after = bench::average_online_cpu_frequency_mhz();
                const int temp_after = bench::cpu_temperature_c();

                const double seconds = std::chrono::duration<double>(end - start).count();
                const auto energy = rapl.delta(energy_before, energy_after);

                // PATCH F5: strict finite&positive package-energy validation.
                // The previous "< 0.0" check let 0.0 J and NaN through.
                if (!std::isfinite(energy.package_j) || energy.package_j <= 0.0) {
                    throw std::runtime_error(
                        "Invalid RAPL package energy (must be finite and > 0.0) for N=" +
                        std::to_string(config.n) +
                        ", threads=" + std::to_string(config.threads) +
                        ", repetition=" + std::to_string(repetition) +
                        ": device_energy_j=" + std::to_string(energy.package_j));
                }
                if (rapl.dram_available()) {
                    if (!std::isfinite(energy.dram_j) || energy.dram_j < 0.0) {
                        throw std::runtime_error(
                            "Invalid RAPL DRAM energy (must be finite and >= 0.0) for N=" +
                            std::to_string(config.n) +
                            ", threads=" + std::to_string(config.threads) +
                            ", repetition=" + std::to_string(repetition) +
                            ": dram_energy_j=" + std::to_string(energy.dram_j));
                    }
                }
                if (!std::isfinite(seconds) || seconds <= 0.0) {
                    throw std::runtime_error(
                        "Non-finite or non-positive measurement for N=" +
                        std::to_string(config.n));
                }

                const ChecksumResult checksum =
                    check_axpy(x.get(), y.get(), config.n, batches);

                const long long flops_total =
                    2LL * static_cast<long long>(config.n) * batches;
                const long long logical_bytes_per_op =
                    12LL * static_cast<long long>(config.n);
                const double total_energy_j =
                    energy.package_j + (energy.dram_j >= 0.0 ? energy.dram_j : 0.0);

                AxpyRow row;
                row.session_id = options.session_id;
                row.sequence_index = ++sequence;
                row.run_id_global = row.sequence_index;
                row.repetition = repetition;
                row.device_name = model;
                row.num_threads = active_threads;
                row.problem_size = static_cast<long long>(config.n);
                row.problem_spec =
                    "elements=" + std::to_string(config.n) +
                    ";alpha=3.0;x=period29*2^-16;y0=period31*2^-8;"
                    "reset=outside_window;max_batches=250000";
                row.batches = batches;
                row.e2e_time_s = seconds;
                row.kernel_time_s = seconds;
                row.wall_time_s = seconds;
                row.device_energy_j = energy.package_j;          // contract 6.6 / 9.3
                row.total_energy_j = total_energy_j;              // informational only
                row.dram_energy_j = energy.dram_j;                // -1 if unavailable (AMD)
                row.energy_per_op_j = row.device_energy_j / static_cast<double>(batches);
                row.energy_per_second_j = row.device_energy_j / row.wall_time_s;
                row.energy_per_flop_j = row.device_energy_j / static_cast<double>(flops_total);
                row.time_per_op_ms_kernel = 1000.0 * row.kernel_time_s / static_cast<double>(batches);
                row.time_per_op_ms_e2e = 1000.0 * row.e2e_time_s / static_cast<double>(batches);
                row.flops_total = flops_total;
                row.gflops_per_s = static_cast<double>(flops_total) / row.kernel_time_s / 1.0e9;
                row.logical_bytes_per_op = logical_bytes_per_op;
                row.avg_power_w = row.device_energy_j / row.wall_time_s;
                row.runtime_status = bench::runtime_status(row.e2e_time_s);
                row.clock_before_mhz = clock_before;
                row.clock_after_mhz = clock_after;
                row.temp_c = std::max(temp_before, temp_after);
                row.temp_before_c = temp_before;
                row.temp_after_c = temp_after;
                row.checksum_ok = checksum.ok;

                // PATCH F3: 'below' is a hard campaign-gate failure
                // (contract 8.3). A below-row must never be written out as
                // a valid campaign line.
                if (row.runtime_status == "below") {
                    throw std::runtime_error(
                        "Hard failure: runtime_status=below for N=" +
                        std::to_string(config.n) +
                        ", threads=" + std::to_string(config.threads) +
                        ", repetition=" + std::to_string(repetition) +
                        ", batches=" + std::to_string(batches) +
                        ", e2e_time_s=" + std::to_string(row.e2e_time_s));
                }

                write_axpy_row(output, row);
                output.flush();
                print_axpy_result(row);
                std::cout << "  max_abs_error=" << std::scientific << std::setprecision(3)
                          << checksum.max_abs_error
                          << " max_rel_error=" << checksum.max_rel_error
                          << std::defaultfloat << '\n';

                if (!checksum.ok) {
                    throw std::runtime_error(
                        "AXPY checksum failed for N=" + std::to_string(config.n) +
                        ", threads=" + std::to_string(config.threads));
                }
            }
        }

        if (env_flag_set("AXPY_ANTI_COLLAPSE_PROBE")) {
            run_anti_collapse_probe();
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        return 2;
    }
}
