#include "common/benchmark_common.hpp"

#include <chrono>
#include <iostream>
#include <thread>

int main() {
    try {
        bench::Rapl energy;
        bench::require_rapl(energy);
        const std::string backend = energy.backend_name();
        if (backend != "powercap" && backend != "perf-energy-pkg") {
            std::cerr << "ERROR: Unsupported AMD package-energy backend: "
                      << backend << '\n';
            return 2;
        }
        const auto before = energy.read();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const auto after = energy.read();
        const auto delta = energy.delta(before, after);
        std::cout << "backend=" << backend
                  << " package_j=" << delta.package_j
                  << " dram_j=" << delta.dram_j << '\n';
        if (!(delta.package_j > 0.0)) {
            std::cerr << "ERROR: Package-energy delta was not positive.\n";
            return 3;
        }
        if (delta.dram_j >= 0.0) {
            std::cerr << "ERROR: Separate DRAM energy is not expected on this AMD setup.\n";
            return 4;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
