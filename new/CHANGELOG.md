# Changelog

## v1.4 — CPU provenance normalization fix

- Fixed a false all-platform preflight failure for the CPU sources.
- The AMD and Intel REDUCTION sources are still required to match after
  normalizing only two frozen platform-specific literals:
  - default output filename;
  - startup banner platform name.
- All kernel, calibration, checksum, formula, OpenMP, timing, and
  measurement code remains byte-sensitive after normalization.
- Added raw and normalized SHA-256 values to the preflight report.
- Added `cpu_source_normalized_diff.txt` on a genuine mismatch.
- No measurement or per-platform analysis data is changed.

## v1.3 — RTX 5060 Ti header-layout and runner-provenance fix

- Fixed the actual runner path capitalization:
  `scripts/02_run_GPU_5060Ti_REDUCTION_only.sh`.
- Replaced the overly rigid single-header-order gate with:
  - hard exact supported-layout validation for every session file;
  - hard cross-session header consistency;
  - a warning when the canonical order is not used.
- Accepted exactly one documented legacy RTX 5060 Ti layout in which
  `total_energy_j` and `device_energy_j` are swapped in position.
- Missing, duplicated, renamed, added, or otherwise reordered columns remain
  hard failures.
- Analysis continues to address columns by name; raw CSV files are not edited.
- No measurement rerun is required by this formatting-only deviation.

## v1.2 — CPU/GPU formula representation and GPU provenance

- Fixed false RTX 3090/RTX 5060 Ti formula failures introduced by v1.1.
- Frozen formulas remain `(N-1)*batches` and `4*N+4`.
- Strictly accepts only the two observed writer outputs:
  exact formula value or scientific-6 serialization.
- Rejects mixed distinguishable storage modes within a campaign.
- Accepts GPU source layouts that compute `flops_total` directly.
- Small CUDA-event/host-clock crossings remain warnings under the frozen
  material threshold.
- No raw measurement is edited, removed, or regenerated.

## v1.1 — REDUCTION scientific-notation validation fix

- Fixed false hard failures for `flops_total` and
  `logical_bytes_per_op`.
- The authoritative formulas remain exactly:
  - `flops_total = (N-1) * batches`
  - `logical_bytes_per_op = 4*N+4`
- The validator now reproduces the benchmark writer's
  `std::scientific << std::setprecision(6)` serialization before comparing
  these two CSV fields.
- Added independent source-level provenance gates for `N-1` and `4*N+4`.
- Formula-failure rows now report exact and serialized expectations.
- No measurement rows are modified, corrected or discarded.
- Existing runtime `above` rows remain warnings, not hard failures.

# REDUCTION analysis pipeline changelog

## v1 — 2026-07-22

- Created four individual platform pipelines and one combined all-platform audit.
- Adapted the final STREAM analysis architecture to frozen REDUCTION semantics.
- Enforced `REDUCTION`, `N-1`, `4*N+4`, `openmp_blocked_sum_fp32`,
  `cub_device_reduce_sum_fp32` and `gpu_resident`.
- Preserved the GPU cross-clock timing fix:
  hard failure only above `max(0.5 ms, 0.5% of E2E)`.
- Added validation for `energy_per_second_j` and `energy_per_flop_j`.
- Added descriptive exact runtime-winner versus energy-winner regret analysis.
- Added explicit warnings for internal partial/workspace traffic, post-selection,
  energy-domain asymmetry and resident-GPU scope.
- Added full synthetic end-to-end and mutation testing.
