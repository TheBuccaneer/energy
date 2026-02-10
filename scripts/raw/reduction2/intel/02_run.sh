#!/bin/bash
set -euo pipefail

BIN_NAME="reduction_cpu"
SRC_FILE="main.cpp"
# Default-Output-Datei im aktuellen Verzeichnis, kann optional als 1. Argument überschrieben werden
OUTPUT_FILE="${1:-$(pwd)/reduction_cpu_amd.csv}"

echo "=== 02_run_reduction_cpu ==="
echo "Binary:      $BIN_NAME"
echo "Source:      $SRC_FILE"
echo "Output CSV:  $OUTPUT_FILE"
echo

# Alte Binary löschen (falls vorhanden)
if [ -f "$BIN_NAME" ]; then
    echo "Removing old binary: $BIN_NAME"
    rm -f "$BIN_NAME"
fi

# Kompilieren
echo "Compiling $SRC_FILE ..."
g++ -O3 -march=native -std=c++17 -o "$BIN_NAME" "$SRC_FILE" -lopenblas -lpthread -lm
echo "Compile done."
echo

# Output-Verzeichnis sicherstellen
OUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "$OUT_DIR"

# Benchmark starten
echo "Running ./$BIN_NAME ..."
./"$BIN_NAME" --output "$OUTPUT_FILE"

echo
echo "Run finished. CSV written to:"
echo "  $OUTPUT_FILE"
