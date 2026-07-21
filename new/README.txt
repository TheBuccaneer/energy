REDUCTION deployment bundle v1.0

Four 02 runners with integrated quickchecks and all main files.
CPU quickcheck: 1M/64M/256M, 1/max threads, 2 reps.
GPU quickcheck: 1M/64M/256M, 2 reps.
Official sessions start only after the integrated quickcheck passes.

v1.1 correction:
- Canonical 45-column order follows the existing benchmark_common.hpp writer:
  device_energy_j before total_energy_j.
- CPU runner validators corrected.
- GPU writers and validators aligned to the same order.
- Measurement formulas and algorithms are unchanged.
