# REDUCTION Analysis v1.4 — CPU Provenance Preflight Audit

## Observed failure

All four platforms passed individual validation and analysis. The combined
preflight had one hard failure:

```text
cpu_sources_semantically_identical
```

Both CPU sources passed every required-token check. The preflight's
normalizer handled the default filenames, but it searched for banner text
of the form `platform=AMD` / `platform=INTEL`.

The actual frozen source banners are:

```text
REDUCTION(sum) | AMD |
REDUCTION(sum) | Intel |
```

Therefore, otherwise matching source files retained a platform-specific
string and produced different normalized hashes.

## v1.4 correction

The normalizer now replaces exactly:

```text
reduction_amd.csv        -> reduction_PLATFORM.csv
reduction_intel.csv      -> reduction_PLATFORM.csv
REDUCTION(sum) | AMD |   -> REDUCTION(sum) | PLATFORM |
REDUCTION(sum) | Intel | -> REDUCTION(sum) | PLATFORM |
```

No broad `AMD`/`Intel` replacement is used. A difference in implementation,
calibration, checksum, formula, OpenMP, timing, or measurement logic still
causes a hard failure.

On a real mismatch the preflight writes:

```text
ALL AUDIT/REDUCTION/results/cpu_source_normalized_diff.txt
```

## Consequence

This is a preflight-validator correction. The completed AMD, Intel,
RTX 3090, and RTX 5060 Ti measurements do not require reruns.
