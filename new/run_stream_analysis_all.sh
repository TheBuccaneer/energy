#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MPLBACKEND=Agg

for platform in AMD INTEL 3090 5060ti; do
    echo "=== STREAM individual analysis: ${platform} ==="
    "$ROOT/$platform/analyse/STREAM/run_all.sh"
done

echo "=== STREAM combined all-platform audit ==="
"$ROOT/ALL AUDIT/STREAM/analyse/run_all.sh"

echo "STREAM analysis pipeline complete."
echo "Combined results: $ROOT/ALL AUDIT/STREAM/results"
