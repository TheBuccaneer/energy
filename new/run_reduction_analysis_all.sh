#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MPLBACKEND=Agg

for platform in AMD INTEL 3090 5060ti; do
    echo "=== REDUCTION individual analysis: ${platform} ==="
    "$ROOT/$platform/analyse/REDUCTION/run_all.sh"
done

echo "=== REDUCTION combined all-platform audit ==="
"$ROOT/ALL AUDIT/REDUCTION/analyse/run_all.sh"

echo "REDUCTION analysis pipeline complete."
echo "Combined results: $ROOT/ALL AUDIT/REDUCTION/results"
