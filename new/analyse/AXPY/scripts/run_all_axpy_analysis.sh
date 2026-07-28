#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO="${1:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || pwd)}"
OUTPUT_REL="${2:-new/analyse/AXPY/all_platforms}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-5000}"
TIE_PERCENT="${TIE_PERCENT:-2.0}"
REQUIRED_SESSIONS="${REQUIRED_SESSIONS:-5}"
REQUIRED_REPETITIONS="${REQUIRED_REPETITIONS:-10}"
FINAL="$REPO/$OUTPUT_REL"
PARENT=$(dirname "$FINAL")
BASE=$(basename "$FINAL")
STAGE="$PARENT/.${BASE}.tmp.$$"
OLD="$PARENT/.${BASE}.old.$$"
FAILED="$PARENT/${BASE}_FAILED_LAST"
CAMPAIGN_ARGS=()
LOCK_ARGS_FILE=""

command -v python3 >/dev/null || {
  echo "ERROR: python3 fehlt." >&2
  exit 2
}

if [[ -n "${AXPY_CAMPAIGN_LOCK:-}" ]]; then
  [[ -f "$AXPY_CAMPAIGN_LOCK" ]] || {
    echo "ERROR: AXPY_CAMPAIGN_LOCK fehlt: $AXPY_CAMPAIGN_LOCK" >&2
    exit 2
  }
  LOCK_ARGS_FILE=$(mktemp)
  if ! python3 "$SCRIPT_DIR/05_verify_axpy_campaign_lock.py" \
      --lock "$AXPY_CAMPAIGN_LOCK" \
      --repo "$REPO" \
      --sessions "$REQUIRED_SESSIONS" \
      --repetitions "$REQUIRED_REPETITIONS" \
      --args-file "$LOCK_ARGS_FILE"; then
    rm -f "$LOCK_ARGS_FILE"
    exit 2
  fi
  mapfile -t CAMPAIGN_ARGS < "$LOCK_ARGS_FILE"
  rm -f "$LOCK_ARGS_FILE"
  LOCK_ARGS_FILE=""
  [[ ${#CAMPAIGN_ARGS[@]} -eq 16 ]] || {
    echo "ERROR: Campaign-Lock lieferte ${#CAMPAIGN_ARGS[@]} statt 16 Argumentzeilen." >&2
    exit 2
  }
  echo "[lock] Bytegenau verifizierte Kampagnen aus: $AXPY_CAMPAIGN_LOCK"
fi

echo "[selftest] Prüfe v5-Regressionsanker..."
python3 "$SCRIPT_DIR/selftest_axpy_analysis_v5.py"

mkdir -p "$PARENT"
rm -rf "$STAGE" "$OLD"
mkdir -p "$STAGE"

cleanup() {
  rc=$?
  [[ -n "$LOCK_ARGS_FILE" ]] && rm -f "$LOCK_ARGS_FILE"
  if [[ $rc -ne 0 && -d "$STAGE" ]]; then
    rm -rf "$FAILED"
    mv "$STAGE" "$FAILED"
    echo >&2
    echo "Analyse abgebrochen. Diagnose wurde erhalten unter:" >&2
    echo "  $FAILED" >&2
  else
    rm -rf "$STAGE"
  fi
}
trap cleanup EXIT INT TERM

echo "[1/5] Prüfe Provenienz, Logs, Manifeste, Rekalibrierung und Quickchecks..."
python3 "$SCRIPT_DIR/00_validate_axpy_provenance.py" \
  --repo "$REPO" \
  --output "$STAGE" \
  --sessions "$REQUIRED_SESSIONS" \
  --reps "$REQUIRED_REPETITIONS" \
  "${CAMPAIGN_ARGS[@]}"

echo "[2/5] Validiere Semantik, Sentinels, Telemetrie und aggregiere session-bewusst..."
python3 "$SCRIPT_DIR/01_validate_and_aggregate_axpy.py" \
  --repo "$REPO" \
  --output "$STAGE" \
  --sessions "$REQUIRED_SESSIONS" \
  --reps "$REQUIRED_REPETITIONS" \
  --bootstrap-resamples "$BOOTSTRAP_RESAMPLES" \
  "${CAMPAIGN_ARGS[@]}"

test -f "$STAGE/axpy_validation_complete.json" || {
  echo "ERROR: Validierungsmarker fehlt." >&2
  exit 2
}

echo "[3/5] Erzeuge Punktvergleich, Pareto-Tabellen und Plots..."
python3 "$SCRIPT_DIR/02_compare_axpy_all.py" \
  --input "$STAGE/axpy_config_summary.csv" \
  --output "$STAGE" \
  --tie-percent "$TIE_PERCENT"

echo "[4/5] Ergänze fixed-config-/oracle-/CI-/5x10-In-Range-Robustheit..."
python3 "$SCRIPT_DIR/03_assess_axpy_robustness.py" \
  --config "$STAGE/axpy_config_summary.csv" \
  --session "$STAGE/axpy_session_summary.csv" \
  --normalized "$STAGE/axpy_normalized_rows.csv" \
  --cross "$STAGE/axpy_cross_platform_by_size.csv" \
  --pareto "$STAGE/axpy_global_pareto.csv" \
  --comparison-report "$STAGE/axpy_comparison_report.md" \
  --output "$STAGE" \
  --tie-percent "$TIE_PERCENT" \
  --required-sessions "$REQUIRED_SESSIONS" \
  --required-repetitions "$REQUIRED_REPETITIONS"

python3 - "$STAGE" "$SCRIPT_DIR" "$BOOTSTRAP_RESAMPLES" "$TIE_PERCENT" "$REQUIRED_SESSIONS" "$REQUIRED_REPETITIONS" <<'PYMARK'
import csv
import hashlib
import json
import pathlib
import sys
import time

out = pathlib.Path(sys.argv[1])
scripts = pathlib.Path(sys.argv[2])
bootstrap_resamples = int(sys.argv[3])
tie_percent = float(sys.argv[4])
required_sessions = int(sys.argv[5])
required_repetitions = int(sys.argv[6])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(name):
    return json.loads((out/name).read_text(encoding='utf-8'))

def count_issues(name):
    path=out/name
    if not path.exists(): return {'FAIL':0,'WARN':0}
    counts={'FAIL':0,'WARN':0}
    with path.open(newline='',encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            severity=row.get('severity','')
            if severity in counts: counts[severity]+=1
    return counts

provenance=load('axpy_provenance_complete.json')
validation=load('axpy_validation_complete.json')
robustness=load('axpy_robustness_complete.json')
prov_issues=count_issues('axpy_provenance_issues.csv')
val_issues=count_issues('axpy_validation_issues.csv')
warning_count=(prov_issues['WARN']+val_issues['WARN']+int(robustness.get('warnings',0)))
fail_count=prov_issues['FAIL']+val_issues['FAIL']
status='FAIL' if fail_count else ('PASS_WITH_WARNINGS' if warning_count else 'PASS')
used = [
    '00_validate_axpy_provenance.py',
    '01_validate_and_aggregate_axpy.py',
    '02_compare_axpy_all.py',
    '03_assess_axpy_robustness.py',
    '04_build_axpy_handover.py',
    '05_verify_axpy_campaign_lock.py',
    'axpy_analysis_common.py',
    'run_all_axpy_analysis.sh',
    'selftest_axpy_analysis_v5.py',
]
marker = {
    'status': status,
    'created_unix': time.time(),
    'failures': fail_count,
    'warnings': warning_count,
    'analysis_parameters': {
        'bootstrap_resamples': bootstrap_resamples,
        'tie_percent': tie_percent,
        'required_sessions': required_sessions,
        'required_repetitions': required_repetitions,
        'aggregation': 'median_of_session_medians',
    },
    'component_status': {
        'provenance': provenance.get('status'),
        'validation': validation.get('status'),
        'robustness': robustness.get('status'),
    },
    'campaign_lock_sha256': sha(out/'axpy_campaign_lock.json'),
    'script_sha256': {name: sha(scripts/name) for name in used},
}
(out/'ANALYSIS_COMPLETE.json').write_text(json.dumps(marker,indent=2,sort_keys=True)+'\n')
PYMARK

echo "[5/5] Erzeuge eingefrorenes Handover-Archiv..."
python3 "$SCRIPT_DIR/04_build_axpy_handover.py" \
  --output-dir "$STAGE" \
  --scripts-dir "$SCRIPT_DIR"

if [[ -e "$FINAL" ]]; then
  mv "$FINAL" "$OLD"
fi
if ! mv "$STAGE" "$FINAL"; then
  [[ -e "$OLD" ]] && mv "$OLD" "$FINAL"
  exit 2
fi
rm -rf "$OLD" "$FAILED"
trap - EXIT INT TERM

FINAL_STATUS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$FINAL/ANALYSIS_COMPLETE.json")
echo
echo "$FINAL_STATUS: atomare AXPY-Auswertung abgeschlossen."
echo "  $FINAL/axpy_validation_report.md"
echo "  $FINAL/axpy_telemetry_report.md"
echo "  $FINAL/axpy_comparison_report.md"
echo "  $FINAL/axpy_robustness_report.md"
echo "  $FINAL/AXPY_ANALYSIS_HANDOVER.zip"
