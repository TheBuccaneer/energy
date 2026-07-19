#pragma once

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cerrno>
#include <cstring>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

constexpr const char* SCHEMA_VERSION = "cpu-gpu-v2";
constexpr double TARGET_RUNTIME_S = 1.0;
constexpr double MIN_RUNTIME_S = 0.75;
constexpr double MAX_RUNTIME_S = 1.25;
constexpr int DEFAULT_REPETITIONS = 10;
constexpr int MAX_CALIBRATION_STEPS = 14;
inline const std::vector<int> THREAD_COUNTS{1, 2, 4, 8, 10, 16, 20, 32, 64};

struct Zone {
    std::string path;
    uint64_t max_uj{};
};

struct EnergySample {
    std::vector<uint64_t> package;
    std::vector<uint64_t> dram;
    uint64_t perf_package_raw{std::numeric_limits<uint64_t>::max()};
};

struct EnergyDelta {
    double package_j{-1.0};
    double dram_j{-1.0};
};

inline std::string read_text(const std::string& path) {
    std::ifstream file(path);
    std::string value;
    if (file) std::getline(file, value);
    return value;
}

inline bool read_u64(const std::string& path, uint64_t& value) {
    std::ifstream file(path);
    return static_cast<bool>(file >> value);
}

inline long perf_event_open(struct perf_event_attr* hw_event,
                            pid_t pid,
                            int cpu,
                            int group_fd,
                            unsigned long flags) {
    return syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

class Rapl {
public:
    Rapl() {
        init_powercap();
        if (packages_.empty()) init_perf_package();
    }

    ~Rapl() {
        if (perf_fd_ >= 0) {
            ioctl(perf_fd_, PERF_EVENT_IOC_DISABLE, 0);
            close(perf_fd_);
        }
    }

    Rapl(const Rapl&) = delete;
    Rapl& operator=(const Rapl&) = delete;

    [[nodiscard]] bool available() const {
        return !packages_.empty() || perf_fd_ >= 0;
    }

    [[nodiscard]] bool dram_available() const { return !drams_.empty(); }

    [[nodiscard]] std::string backend_name() const {
        if (!packages_.empty()) return "powercap";
        if (perf_fd_ >= 0) return "perf-energy-pkg";
        return "none";
    }

    [[nodiscard]] EnergySample read() const {
        EnergySample sample;
        sample.package = read_group(packages_);
        sample.dram = read_group(drams_);
        if (perf_fd_ >= 0) sample.perf_package_raw = read_perf_raw();
        return sample;
    }

    [[nodiscard]] EnergyDelta delta(const EnergySample& before,
                                    const EnergySample& after) const {
        if (perf_fd_ >= 0) {
            if (before.perf_package_raw == std::numeric_limits<uint64_t>::max() ||
                after.perf_package_raw == std::numeric_limits<uint64_t>::max() ||
                after.perf_package_raw < before.perf_package_raw) {
                return {-1.0, -1.0};
            }
            const long double raw_delta = static_cast<long double>(
                after.perf_package_raw - before.perf_package_raw);
            return {static_cast<double>(raw_delta * perf_scale_j_), -1.0};
        }

        return {
            delta_group(before.package, after.package, packages_),
            delta_group(before.dram, after.dram, drams_)
        };
    }

private:
    void init_powercap() {
        const std::filesystem::path base{"/sys/class/powercap"};
        if (!std::filesystem::exists(base)) return;

        // Powercap class entries are frequently symlinks. Scan only the class
        // entries and one child level, canonicalizing candidates to avoid
        // duplicate aliases and recursive sysfs loops.
        std::vector<std::filesystem::path> candidates;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(base, ec)) {
            if (ec) break;
            std::error_code type_ec;
            if (!entry.is_directory(type_ec)) continue;
            candidates.push_back(entry.path());

            std::error_code child_ec;
            for (const auto& child : std::filesystem::directory_iterator(entry.path(), child_ec)) {
                if (child_ec) break;
                std::error_code child_type_ec;
                if (child.is_directory(child_type_ec)) candidates.push_back(child.path());
            }
        }

        std::vector<std::string> seen;
        for (const auto& candidate : candidates) {
            std::error_code canonical_ec;
            const auto canonical = std::filesystem::weakly_canonical(candidate, canonical_ec);
            const std::string path = (canonical_ec ? candidate : canonical).string();
            if (std::find(seen.begin(), seen.end(), path) != seen.end()) continue;
            seen.push_back(path);

            const std::string name = read_text(path + "/name");
            uint64_t max_uj = 0;
            uint64_t test_energy = 0;
            if (!read_u64(path + "/max_energy_range_uj", max_uj) || max_uj == 0) continue;
            if (!read_u64(path + "/energy_uj", test_energy)) continue;

            if (name.find("package") != std::string::npos) {
                packages_.push_back({path, max_uj});
            } else if (name == "dram") {
                drams_.push_back({path, max_uj});
            }
        }
    }

    void init_perf_package() {
        const std::string base = "/sys/bus/event_source/devices/power";
        const std::string type_text = read_text(base + "/type");
        const std::string event_text = read_text(base + "/events/energy-pkg");
        const std::string scale_text = read_text(base + "/events/energy-pkg.scale");
        if (type_text.empty() || event_text.empty() || scale_text.empty()) return;

        try {
            const uint32_t type = static_cast<uint32_t>(std::stoul(type_text, nullptr, 0));
            const auto event_pos = event_text.find("event=");
            if (event_pos == std::string::npos) return;
            const auto value_start = event_pos + 6;
            const auto value_end = event_text.find(',', value_start);
            const std::string config_text = event_text.substr(
                value_start,
                value_end == std::string::npos ? std::string::npos : value_end - value_start);
            const uint64_t config = std::stoull(config_text, nullptr, 0);
            perf_scale_j_ = std::stold(scale_text);
            if (!(perf_scale_j_ > 0.0L)) return;

            struct perf_event_attr attr {};
            attr.type = type;
            attr.size = sizeof(attr);
            attr.config = config;
            attr.disabled = 1;
            attr.exclude_user = 0;
            attr.exclude_kernel = 0;
            attr.exclude_hv = 0;

            // Use the first representative CPU advertised by the power PMU.
            // On the current one-socket Threadripper this is normally CPU 0.
            int perf_cpu = 0;
            const std::string cpumask = read_text(base + "/cpumask");
            if (!cpumask.empty()) {
                try {
                    perf_cpu = std::stoi(cpumask);
                } catch (...) {
                    perf_cpu = 0;
                }
            }
            perf_fd_ = static_cast<int>(perf_event_open(
                &attr, -1, perf_cpu, -1, PERF_FLAG_FD_CLOEXEC));
            if (perf_fd_ < 0) {
                perf_fd_ = -1;
                return;
            }

            if (ioctl(perf_fd_, PERF_EVENT_IOC_RESET, 0) != 0 ||
                ioctl(perf_fd_, PERF_EVENT_IOC_ENABLE, 0) != 0) {
                close(perf_fd_);
                perf_fd_ = -1;
            }
        } catch (...) {
            perf_fd_ = -1;
            perf_scale_j_ = 0.0L;
        }
    }

    [[nodiscard]] uint64_t read_perf_raw() const {
        uint64_t value = std::numeric_limits<uint64_t>::max();
        if (perf_fd_ < 0) return value;
        const ssize_t bytes = ::read(perf_fd_, &value, sizeof(value));
        return bytes == static_cast<ssize_t>(sizeof(value))
                   ? value
                   : std::numeric_limits<uint64_t>::max();
    }

    static std::vector<uint64_t> read_group(const std::vector<Zone>& zones) {
        std::vector<uint64_t> values;
        values.reserve(zones.size());
        for (const auto& zone : zones) {
            uint64_t value = 0;
            values.push_back(read_u64(zone.path + "/energy_uj", value)
                                 ? value
                                 : std::numeric_limits<uint64_t>::max());
        }
        return values;
    }

    static double delta_group(const std::vector<uint64_t>& before,
                              const std::vector<uint64_t>& after,
                              const std::vector<Zone>& zones) {
        if (zones.empty()) return -1.0;
        if (before.size() != zones.size() || after.size() != zones.size()) return -1.0;

        long double total_uj = 0.0L;
        for (size_t i = 0; i < zones.size(); ++i) {
            if (before[i] == std::numeric_limits<uint64_t>::max() ||
                after[i] == std::numeric_limits<uint64_t>::max()) {
                return -1.0;
            }
            if (after[i] >= before[i]) {
                total_uj += static_cast<long double>(after[i] - before[i]);
            } else {
                total_uj += static_cast<long double>(after[i]) + zones[i].max_uj - before[i];
            }
        }
        return static_cast<double>(total_uj / 1.0e6L);
    }

    std::vector<Zone> packages_;
    std::vector<Zone> drams_;
    int perf_fd_{-1};
    long double perf_scale_j_{0.0L};
};

inline void require_rapl(const Rapl& rapl) {
    if (rapl.available()) return;
    throw std::runtime_error(
        "No readable CPU package-energy source. Run the matching 01_enable_CPU script; "
        "on AMD, power/energy-pkg/ must be accessible through perf_event_open.");
}

inline std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t raw = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_r(&raw, &local);
    std::ostringstream out;
    out << std::put_time(&local, "%Y-%m-%dT%H:%M:%S");
    return out.str();
}

inline std::string cpu_model() {
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto pos = line.find(':');
            return pos == std::string::npos ? "Unknown CPU" : line.substr(pos + 2);
        }
    }
    return "Unknown CPU";
}

inline int average_online_cpu_frequency_mhz() {
    long long sum_khz = 0;
    int count = 0;
    const std::filesystem::path cpu_base{"/sys/devices/system/cpu"};
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(cpu_base, ec)) {
        if (ec || !entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("cpu", 0) != 0 || name.size() <= 3 ||
            !std::all_of(name.begin() + 3, name.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
            continue;
        }
        std::ifstream file(entry.path() / "cpufreq/scaling_cur_freq");
        int khz = 0;
        if (file >> khz) {
            sum_khz += khz;
            ++count;
        }
    }
    return count > 0 ? static_cast<int>((sum_khz / count) / 1000) : -1;
}

inline int cpu_temperature_c() {
    const std::filesystem::path base{"/sys/class/hwmon"};
    if (!std::filesystem::exists(base)) return -1;
    int maximum = -1;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(base, ec)) {
        if (ec || !entry.is_directory()) continue;
        const std::string sensor = read_text(entry.path().string() + "/name");
        if (sensor != "coretemp" && sensor != "k10temp") continue;
        for (const auto& item : std::filesystem::directory_iterator(entry.path(), ec)) {
            if (ec) break;
            const std::string filename = item.path().filename().string();
            if (filename.rfind("temp", 0) != 0 ||
                filename.find("_input") == std::string::npos) {
                continue;
            }
            std::ifstream file(item.path());
            int milli = 0;
            if (file >> milli) maximum = std::max(maximum, milli / 1000);
        }
    }
    return maximum;
}

inline float* allocate_aligned(size_t count) {
    void* pointer = nullptr;
    if (count > std::numeric_limits<size_t>::max() / sizeof(float)) return nullptr;
    return posix_memalign(&pointer, 64, count * sizeof(float)) == 0
               ? static_cast<float*>(pointer)
               : nullptr;
}

inline std::string csv_escape(const std::string& value) {
    std::string escaped = value;
    size_t pos = 0;
    while ((pos = escaped.find('"', pos)) != std::string::npos) {
        escaped.insert(pos, 1, '"');
        pos += 2;
    }
    return '"' + escaped + '"';
}

inline std::string runtime_status(double seconds) {
    if (seconds < MIN_RUNTIME_S) return "below";
    if (seconds > MAX_RUNTIME_S) return "above";
    return "in_range";
}

struct Options {
    std::string output_file;
    int repetitions{DEFAULT_REPETITIONS};
    std::string session_id{"manual"};
    uint32_t seed{1};
};

inline Options parse_options(int argc, char** argv, const std::string& default_output) {
    Options options;
    options.output_file = argc > 1 ? argv[1] : default_output;
    if (argc > 2) options.repetitions = std::max(1, std::stoi(argv[2]));
    if (argc > 3) options.session_id = argv[3];
    if (argc > 4) options.seed = static_cast<uint32_t>(std::stoul(argv[4]));
    return options;
}

struct ResultRow {
    std::string session_id;
    int sequence_index{};
    int run_id_global{};
    int repetition{};
    std::string workload;
    std::string implementation;
    std::string execution_mode{"cpu_native"};
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
    double flops_total{};
    double gflops_per_s{};
    double logical_bytes_per_op{};
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
    long long cpu_cycles{-1};
    long long cpu_instructions{-1};
    double cpu_ipc{-1.0};
    long long cpu_cache_misses{-1};
    bool checksum_ok{};
};

inline void write_header(std::ofstream& output) {
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

inline void write_row(std::ofstream& output, const ResultRow& row) {
    output << SCHEMA_VERSION << ',' << timestamp() << ',' << csv_escape(row.session_id) << ','
           << row.sequence_index << ',' << row.run_id_global << ',' << row.repetition << ','
           << csv_escape(row.workload) << ',' << csv_escape(row.implementation) << ','
           << row.execution_mode << ',' << csv_escape(row.device_name) << ',' << row.num_threads << ','
           << row.problem_size << ',' << csv_escape(row.problem_spec) << ',' << row.batches << ','
           << std::fixed << std::setprecision(6)
           << row.e2e_time_s << ',' << row.kernel_time_s << ',' << row.wall_time_s << ','
           << row.device_energy_j << ',' << row.total_energy_j << ',' << row.dram_energy_j << ','
           << std::scientific << std::setprecision(6)
           << row.energy_per_op_j << ','
           << std::fixed << std::setprecision(6) << row.energy_per_second_j << ','
           << std::scientific << std::setprecision(6) << row.energy_per_flop_j << ','
           << std::fixed << std::setprecision(6)
           << row.time_per_op_ms_kernel << ',' << row.time_per_op_ms_e2e << ','
           << std::scientific << std::setprecision(6)
           << row.flops_total << ','
           << std::fixed << std::setprecision(2)
           << row.gflops_per_s << ','
           << std::scientific << std::setprecision(6)
           << row.logical_bytes_per_op << ','
           << std::fixed << std::setprecision(2)
           << row.avg_power_w << ',' << row.runtime_status << ',';

    if (row.pcie_gen >= 0) output << row.pcie_gen;
    output << ',';
    if (row.pcie_width >= 0) output << row.pcie_width;
    output << ',';
    if (row.sm_clock_mhz >= 0) output << row.sm_clock_mhz;
    output << ',';
    if (row.clock_before_mhz >= 0) output << row.clock_before_mhz;
    output << ',';
    if (row.clock_after_mhz >= 0) output << row.clock_after_mhz;
    output << ',';
    if (row.mem_clock_mhz >= 0) output << row.mem_clock_mhz;
    output << ',';
    if (row.temp_c >= 0) output << row.temp_c;
    output << ',';
    if (row.temp_before_c >= 0) output << row.temp_before_c;
    output << ',';
    if (row.temp_after_c >= 0) output << row.temp_after_c;
    output << ',' << row.throttle_reasons << ','
           << row.cpu_cycles << ',' << row.cpu_instructions << ','
           << std::fixed << std::setprecision(6) << row.cpu_ipc << ','
           << row.cpu_cache_misses << ',' << (row.checksum_ok ? 't' : 'f') << '\n';
}

inline ResultRow make_cpu_row(const Options& options,
                              int sequence_index,
                              int repetition,
                              const std::string& workload,
                              const std::string& implementation,
                              const std::string& device_name,
                              int num_threads,
                              long long problem_size,
                              const std::string& problem_spec,
                              long long batches,
                              double seconds,
                              const EnergyDelta& energy,
                              double flops_per_op,
                              double logical_bytes_per_op,
                              bool checksum_ok,
                              int clock_before_mhz,
                              int clock_after_mhz,
                              int temp_before_c,
                              int temp_after_c) {
    if (energy.package_j < 0.0) {
        throw std::runtime_error("RAPL package read failed during measurement");
    }
    if (!checksum_ok) {
        throw std::runtime_error("Checksum failed");
    }

    const double operations = static_cast<double>(batches);
    const double flops_total = flops_per_op * operations;
    const double total_energy_j = energy.package_j + (energy.dram_j >= 0.0 ? energy.dram_j : 0.0);
    ResultRow row;
    row.session_id = options.session_id;
    row.sequence_index = sequence_index;
    row.run_id_global = sequence_index;
    row.repetition = repetition;
    row.workload = workload;
    row.implementation = implementation;
    row.device_name = device_name;
    row.num_threads = num_threads;
    row.problem_size = problem_size;
    row.problem_spec = problem_spec;
    row.batches = batches;
    row.e2e_time_s = seconds;
    row.kernel_time_s = seconds;
    row.wall_time_s = seconds;
    row.device_energy_j = energy.package_j;
    row.total_energy_j = total_energy_j;
    row.dram_energy_j = energy.dram_j;
    row.energy_per_op_j = total_energy_j / operations;
    row.energy_per_second_j = total_energy_j / seconds;
    row.energy_per_flop_j = flops_total > 0.0 ? total_energy_j / flops_total : -1.0;
    row.time_per_op_ms_kernel = 1.0e3 * seconds / operations;
    row.time_per_op_ms_e2e = row.time_per_op_ms_kernel;
    row.flops_total = flops_total;
    row.gflops_per_s = flops_total / seconds / 1.0e9;
    row.logical_bytes_per_op = logical_bytes_per_op;
    row.avg_power_w = row.energy_per_second_j;
    row.runtime_status = runtime_status(seconds);
    row.clock_before_mhz = clock_before_mhz;
    row.clock_after_mhz = clock_after_mhz;
    row.temp_c = std::max(temp_before_c, temp_after_c);
    row.temp_before_c = temp_before_c;
    row.temp_after_c = temp_after_c;
    row.checksum_ok = checksum_ok;
    return row;
}

inline void print_result(const ResultRow& row) {
    std::cout << '[' << row.workload << "] N=" << row.problem_size
              << " threads=" << row.num_threads
              << " rep=" << row.repetition
              << " batches=" << row.batches
              << " time=" << std::fixed << std::setprecision(3) << row.wall_time_s << " s"
              << " | package=" << std::setprecision(3) << row.device_energy_j << " J"
              << " | dram=";
    if (row.dram_energy_j >= 0.0) std::cout << std::setprecision(6) << row.dram_energy_j;
    else std::cout << "NA";
    std::cout << " | total=" << std::setprecision(3) << row.total_energy_j << " J"
              << " | power=" << std::setprecision(1) << row.avg_power_w << " W"
              << " | runtime=" << row.runtime_status
              << " | checksum=OK\n";
}

template <typename Config>
inline void shuffle_configs(std::vector<Config>& configs, uint32_t seed) {
    std::mt19937 generator(seed);
    std::shuffle(configs.begin(), configs.end(), generator);
}

inline int scale_batches(double measured_seconds, int current, int maximum) {
    if (measured_seconds >= TARGET_RUNTIME_S) return current;
    const double safe_seconds = std::max(measured_seconds, 1.0e-9);
    const long long estimate = static_cast<long long>(
        std::ceil(TARGET_RUNTIME_S * static_cast<double>(current) / safe_seconds));
    const long long next = std::max<long long>(current + 1,
        std::min<long long>(estimate, static_cast<long long>(current) * 10));
    return static_cast<int>(std::min<long long>(maximum, next));
}

}  // namespace bench
