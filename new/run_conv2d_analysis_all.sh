#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MPLBACKEND=Agg
for platform in AMD INTEL 3090 5060ti; do
    echo "=== CONV2D individual analysis: ${platform} ==="
    args=()
    case "$platform" in
      AMD) [[ -n "${AMD_CAMPAIGN:-}" ]] && args=(--campaign "$AMD_CAMPAIGN") ;;
      INTEL) [[ -n "${INTEL_CAMPAIGN:-}" ]] && args=(--campaign "$INTEL_CAMPAIGN") ;;
      3090) [[ -n "${GPU3090_CAMPAIGN:-}" ]] && args=(--campaign "$GPU3090_CAMPAIGN") ;;
      5060ti) [[ -n "${GPU5060TI_CAMPAIGN:-}" ]] && args=(--campaign "$GPU5060TI_CAMPAIGN") ;;
    esac
    "$ROOT/$platform/analyse/CONV2D/run_all.sh" "${args[@]}"
done

echo "=== CONV2D combined all-platform audit ==="
"$ROOT/ALL AUDIT/CONV2D/analyse/run_all.sh"
echo "CONV2D analysis pipeline complete."
echo "Combined results: $ROOT/ALL AUDIT/CONV2D/results"
