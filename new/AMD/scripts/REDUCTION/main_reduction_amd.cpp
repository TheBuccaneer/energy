#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
static constexpr std::size_t BLOCK_SIZE = 4096;

struct RaplDomain {
    fs::path dir;
    fs::path energy_uj;
    fs::path max_energy_range_uj;
    std::string name;
};

static bool read_u64(const fs::path& p, uint64_t& out) {
    std::ifstream f(p);
    if (!f) return false;
    f >> out;
    return !f.fail();
}

static std::string read_line(const fs::path& p) {
    std::ifstream f(p);
    std::string s;
    if (f) std::getline(f, s);
    return s;
}

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return char(std::tolower(c)); });
    return s;
}

static std::vector<RaplDomain> discover_rapl() {
    std::vector<RaplDomain> out;
    fs::path root("/sys/class/powercap");
    if (!fs::exists(root)) return out;
    for (const auto& e : fs::recursive_directory_iterator(root)) {
        if (!e.is_directory()) continue;
        fs::path energy = e.path() / "energy_uj";
        if (!fs::exists(energy)) continue;
        out.push_back({e.path(), energy, e.path() / "max_energy_range_uj", read_line(e.path() / "name")});
    }
    return out;
}

static const RaplDomain* find_domain(const std::vector<RaplDomain>& domains, const std::string& needle) {
    for (const auto& d : domains) {
        if (lower(d.name).find(needle) != std::string::npos) return &d;
    }
    return nullptr;
}

static double delta_j(uint64_t before, uint64_t after, uint64_t max_range_uj) {
    uint64_t delta = 0;
    if (after >= before) {
        delta = after - before;
    } else if (max_range_uj > 0) {
        delta = (max_range_uj - before) + after;
    } else {
        delta = (std::numeric_limits<uint64_t>::max() - before) + after + 1ULL;
    }
    return double(delta) / 1e6;
}

static std::string now_iso() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &tm);
    return std::string(buf);
}

static float openmp_blocked_sum_fp32(const std::vector<float>& x, int threads) {
    const std::size_t N = x.size();
    const std::size_t blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<float> partial(blocks, 0.0f);

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (std::size_t b = 0; b < blocks; ++b) {
            const std::size_t begin = b * BLOCK_SIZE;
            const std::size_t end = std::min(begin + BLOCK_SIZE, N);
            float local = 0.0f;
#pragma omp simd reduction(+:local)
            for (std::size_t i = begin; i < end; ++i) {
                local += x[i];
            }
            partial[b] = local;
        }

#pragma omp barrier
#pragma omp single
        {
            float total = 0.0f;
            for (std::size_t b = 0; b < blocks; ++b) total += partial[b];
            partial[0] = total;
        }
#pragma omp barrier
    }
    return partial[0];
}

static double reference_sum(const std::vector<float>& x) {
    double s = 0.0;
    for (float v : x) s += double(v);
    return s;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " N threads reps output_csv\n";
        return 2;
    }

    const std::size_t N = std::stoull(argv[1]);
    const int threads = std::stoi(argv[2]);
    const int reps = std::stoi(argv[3]);
    const std::string output_csv = argv[4];
    if (N < 1 || threads < 1 || reps < 1) {
        std::cerr << "invalid N/threads/reps\n";
        return 2;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    std::vector<float> x(N);
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        const int centered = int(i % 17) - 8;
        x[i] = 1.0f + float(centered) * 1e-5f;
    }

    const double ref = reference_sum(x);
    volatile float warm = openmp_blocked_sum_fp32(x, threads);
    (void)warm;

    const auto domains = discover_rapl();
    const RaplDomain* pkg = find_domain(domains, "package");
    if (!pkg && !domains.empty()) pkg = &domains.front();
    const RaplDomain* dram = find_domain(domains, "dram");

    if (!pkg) {
        std::cerr << "ERROR: no RAPL package domain found under /sys/class/powercap. Run enable script / check permissions.\n";
        return 1;
    }

    uint64_t pkg_max = 0, dram_max = 0;
    read_u64(pkg->max_energy_range_uj, pkg_max);
    if (dram) read_u64(dram->max_energy_range_uj, dram_max);

    std::ofstream out(output_csv);
    if (!out) {
        std::cerr << "cannot open output csv: " << output_csv << "\n";
        return 2;
    }

    out << "timestamp,device,kernel,N,threads,reps,rep,block_size,blocks,"
        << "time_s,package_j,dram_j,total_j,power_w,logical_bytes,flops,"
        << "bandwidth_gbs,gflops,checksum,reference,rel_err,checksum_ok,"
        << "rapl_package_name,rapl_dram_name\n";

    for (int r = 1; r <= reps; ++r) {
        uint64_t pkg0 = 0, pkg1 = 0, dram0 = 0, dram1 = 0;
        if (!read_u64(pkg->energy_uj, pkg0)) {
            std::cerr << "ERROR: cannot read package energy: " << pkg->energy_uj << "\n";
            return 1;
        }
        bool has_dram0 = dram && read_u64(dram->energy_uj, dram0);

        auto t0 = std::chrono::steady_clock::now();
        float result = openmp_blocked_sum_fp32(x, threads);
        auto t1 = std::chrono::steady_clock::now();

        if (!read_u64(pkg->energy_uj, pkg1)) {
            std::cerr << "ERROR: cannot read package energy after run\n";
            return 1;
        }
        bool has_dram1 = dram && read_u64(dram->energy_uj, dram1);

        const double sec = std::chrono::duration<double>(t1 - t0).count();
        const double pkg_j = delta_j(pkg0, pkg1, pkg_max);
        const double dram_j = (has_dram0 && has_dram1) ? delta_j(dram0, dram1, dram_max) : 0.0;
        const double total_j = pkg_j + dram_j;
        const double power_w = total_j / sec;
        const double logical_bytes = double(4ULL * N + 4ULL);
        const double flops = double(N - 1ULL);
        const double bandwidth_gbs = logical_bytes / sec / 1e9;
        const double gflops = flops / sec / 1e9;
        const double rel_err = std::abs(double(result) - ref) / std::max(1.0, std::abs(ref));
        const bool checksum_ok = std::isfinite(result) && rel_err <= 1e-4;
        const std::size_t blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        out << now_iso() << ",AMD_CPU,openmp_blocked_sum_fp32,"
            << N << ',' << threads << ',' << reps << ',' << r << ','
            << BLOCK_SIZE << ',' << blocks << ','
            << std::setprecision(12) << sec << ',' << pkg_j << ',' << dram_j << ',' << total_j << ',' << power_w << ','
            << logical_bytes << ',' << flops << ',' << bandwidth_gbs << ',' << gflops << ','
            << result << ',' << ref << ',' << rel_err << ',' << (checksum_ok ? 1 : 0) << ','
            << '"' << pkg->name << '"' << ',' << '"' << (dram ? dram->name : "NONE") << '"' << '\n';

        if (!checksum_ok) {
            std::cerr << "ERROR: checksum failed, rel_err=" << rel_err << "\n";
            return 1;
        }
    }

    return 0;
}
