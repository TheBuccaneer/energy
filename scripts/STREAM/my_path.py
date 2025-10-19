#!/usr/bin/env python3
"""
Recursive CSV/TSV enricher: adds per-pass energy/time metrics + optional FLOP efficiency.
Robust handling of numeric types, divide-by-zero, non-square matrices, and missing power data.
Uses passes_e2e as the divisor (total number of E2E passes).
Includes STREAM-specific kernel metrics (GB/s_kernel, J/GB_kernel).
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(
        description="Enrich CSVs/TSVs with per-pass metrics (E, t, FLOPs)"
    )
    ap.add_argument("--root", required=True, help="Root directory to scan recursively")
    ap.add_argument("--inplace", action="store_true", default=True,
                    help="Overwrite original files (default: True, use --no-inplace for separate files)")
    ap.add_argument("--no-inplace", dest="inplace", action="store_false",
                    help="Create *_enriched.csv instead of overwriting")
    ap.add_argument("--sep", default="auto",
                    help="Delimiter: 'auto' (detect), ',' or '\\t' (default: auto)")
    ap.add_argument("--gemm-mode-values", default="gemm,GEMM,matmul,dgemm,sgemm,e2e",
                    help="Comma-separated mode values that trigger FLOP calculations")
    return ap.parse_args()


def detect_separator(file_path):
    """Auto-detect CSV (,) or TSV (\\t) by reading first line."""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline()
            if '\t' in first_line:
                return '\t'
            return ','
    except:
        return ','


def extract_matrix_dims(row):
    """
    Extract M, K, N from matrix_size column.
    Formats: '4096', '4096x4096', 'MxKxN'
    Returns: (M, K, N) or (N, N, N) for square, or (nan, nan, nan) on error.
    """
    try:
        ms = str(row.get("matrix_size", ""))
        if not ms or ms == "nan":
            return np.nan, np.nan, np.nan
        
        # Split by 'x' or 'X'
        parts = ms.lower().replace("x", " ").split()
        nums = [int(p) for p in parts if p.isdigit()]
        
        if len(nums) == 1:
            # Square: N×N×N
            n = nums[0]
            return n, n, n
        elif len(nums) == 2:
            # Rectangular: M×N, assume K=N
            m, n = nums
            return m, n, n
        elif len(nums) >= 3:
            # Full: M×K×N
            return nums[0], nums[1], nums[2]
        else:
            return np.nan, np.nan, np.nan
    except:
        return np.nan, np.nan, np.nan


def compute_batch_metrics(df, gemm_modes):
    """Add per-pass energy, time, and optional FLOP metrics."""
    
    # Force numeric types for all critical columns
    numeric_cols = ["energy_j", "seconds_wall", "passes_e2e", "avg_power_w", 
                    "seconds_gpu", "seconds_target", "seconds_kernel", "energy_kernel_j",
                    "bytes_total"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Replace zeros with NaN to avoid divide-by-zero
    df["passes_e2e"] = df["passes_e2e"].replace(0, np.nan)
    df["seconds_wall"] = df["seconds_wall"].replace(0, np.nan)
    
    # 1) Per-pass energy & time (using passes_e2e)
    df["energy_per_pass_j"] = df["energy_j"] / df["passes_e2e"]
    df["kWh_per_pass"] = df["energy_per_pass_j"] / 3.6e6
    df["seconds_per_pass"] = df["seconds_wall"] / df["passes_e2e"]
    
    # 2) Compute avg_power_w if missing
    if "avg_power_w" not in df.columns or df["avg_power_w"].isna().all():
        df["avg_power_w"] = df["energy_j"] / df["seconds_wall"]
    
    # 3) STREAM kernel metrics (sustained memory bandwidth)
    # Only compute if seconds_kernel, energy_kernel_j, and bytes_total exist
    if "seconds_kernel" in df.columns and "energy_kernel_j" in df.columns and "bytes_total" in df.columns:
        # Replace zeros to avoid division by zero
        df["seconds_kernel"] = df["seconds_kernel"].replace(0, np.nan)
        df["bytes_total"] = df["bytes_total"].replace(0, np.nan)
        
        # GB/s_kernel = bytes_total / seconds_kernel / 1e9
        df["gb_per_s_kernel"] = np.where(
            df["seconds_kernel"].notna() & (df["seconds_kernel"] > 0) & df["bytes_total"].notna(),
            df["bytes_total"] / df["seconds_kernel"] / 1e9,
            np.nan
        )
        
        # J/GB_kernel = energy_kernel_j / (bytes_total / 1e9)
        df["j_per_gb_kernel"] = np.where(
            df["energy_kernel_j"].notna() & (df["energy_kernel_j"] > 0) & 
            df["bytes_total"].notna() & (df["bytes_total"] > 0),
            df["energy_kernel_j"] / (df["bytes_total"] / 1e9),
            np.nan
        )
    
    # 4) Extract matrix dimensions (M, K, N)
    dims = df.apply(extract_matrix_dims, axis=1, result_type="expand")
    df["M"], df["K"], df["N"] = dims[0], dims[1], dims[2]
    
    # 5) FLOP metrics (only for GEMM modes)
    gemm_set = set(m.strip().lower() for m in gemm_modes.split(","))
    
    # Check if mode column exists, otherwise assume all rows are GEMM if matrix_size is present
    if "mode" in df.columns:
        is_gemm = df["mode"].astype(str).str.lower().isin(gemm_set)
    else:
        # If no mode column, treat all rows with valid matrix_size as GEMM
        is_gemm = df["matrix_size"].notna() & (df["matrix_size"].astype(str) != "nan")
    
    # FLOPs per pass = 2·M·N·K (general matrix multiply)
    has_dims = df["M"].notna() & df["K"].notna() & df["N"].notna()
    df["flops_per_pass"] = np.where(
        is_gemm & has_dims,
        2.0 * df["M"] * df["N"] * df["K"],
        np.nan
    )
    
    # Joule/GFLOP
    df["joule_per_gflop"] = np.where(
        df["flops_per_pass"].notna() & (df["flops_per_pass"] > 0),
        df["energy_per_pass_j"] / (df["flops_per_pass"] / 1e9),
        np.nan
    )
    
    # GFLOP/s (only if seconds_per_pass > 0)
    df["gflops"] = np.where(
        df["flops_per_pass"].notna() & (df["seconds_per_pass"] > 0),
        (df["flops_per_pass"] / 1e9) / df["seconds_per_pass"],
        np.nan
    )
    
    # FLOPS/W (Green500 style: GFLOP/s per Watt)
    df["flops_per_watt"] = np.where(
        df["gflops"].notna() & (df["avg_power_w"] > 0),
        df["gflops"] / df["avg_power_w"],
        np.nan
    )
    
    return df


def process_file(file_path, inplace, sep, gemm_modes):
    """Read, enrich, and save a single CSV/TSV."""
    try:
        # Auto-detect separator if needed
        if sep == "auto":
            detected_sep = detect_separator(file_path)
        else:
            detected_sep = sep
        
        df = pd.read_csv(file_path, sep=detected_sep)
        
        # Check required columns
        required = ["energy_j", "seconds_wall", "passes_e2e"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[SKIP] {file_path.name}: missing {missing}", file=sys.stderr)
            return False
        
        # Compute metrics
        df = compute_batch_metrics(df, gemm_modes)
        
        # Save (preserve separator for TSV)
        if inplace:
            out_path = file_path
        else:
            suffix = "_enriched.tsv" if detected_sep == '\t' else "_enriched.csv"
            out_path = file_path.parent / f"{file_path.stem}{suffix}"
        
        df.to_csv(out_path, index=False, sep=detected_sep)
        print(f"✓ {file_path.name} → {out_path.name} (sep='{detected_sep}')")
        return True
        
    except Exception as e:
        print(f"[ERROR] {file_path.name}: {e}", file=sys.stderr)
        return False


def main():
    args = parse_args()
    root = Path(args.root)
    
    if not root.exists():
        sys.exit(f"Error: {root} does not exist")
    
    # Find both CSV and TSV files
    csv_files = list(root.rglob("*.csv")) + list(root.rglob("*.tsv"))
    if not csv_files:
        sys.exit(f"No CSV/TSV files found under {root}")
    
    print(f"Found {len(csv_files)} files under {root}")
    
    success = 0
    for file_path in csv_files:
        if process_file(file_path, args.inplace, args.sep, args.gemm_mode_values):
            success += 1
    
    print(f"\nProcessed {success}/{len(csv_files)} files successfully")


if __name__ == "__main__":
    main()