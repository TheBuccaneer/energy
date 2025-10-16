#!/usr/bin/env python3
"""
Recursive CSV/TSV enricher: adds per-pass energy/time metrics + optional FLOP efficiency.
Robust handling of numeric types, divide-by-zero, non-square matrices, and missing power data.
Uses passes_e2e (or passes_kernel as fallback) and seconds_wall (or seconds_kernel as fallback).
Compatible with both CPU and GPU CSV formats.
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
        # Return NaN if matrix_size doesn't exist in row
        if "matrix_size" not in row.index:
            return np.nan, np.nan, np.nan
            
        ms = str(row.get("matrix_size", ""))
        if not ms or ms == "nan" or ms == "":
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
    except Exception as e:
        return np.nan, np.nan, np.nan


def compute_batch_metrics(df, gemm_modes):
    """Add per-pass energy, time, and optional FLOP metrics."""
    
    # Force numeric types for all critical columns
    numeric_cols = ["energy_j", "seconds_wall", "seconds_kernel", "passes_e2e", 
                    "passes_kernel", "avg_power_w", "seconds_gpu", "seconds_target", 
                    "energy_kernel_j", "bytes_total"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # CPU/GPU compatibility: Use passes_kernel if passes_e2e doesn't exist
    if "passes_e2e" not in df.columns and "passes_kernel" in df.columns:
        df["passes_e2e"] = df["passes_kernel"].copy()
        print("  [INFO] Using passes_kernel as passes_e2e (CPU mode)")
    
    # CPU/GPU compatibility: Use seconds_kernel if seconds_wall doesn't exist
    if "seconds_wall" not in df.columns and "seconds_kernel" in df.columns:
        df["seconds_wall"] = df["seconds_kernel"].copy()
        print("  [INFO] Using seconds_kernel as seconds_wall (CPU mode)")
    
    # Replace zeros with NaN to avoid divide-by-zero
    if "passes_e2e" in df.columns:
        df["passes_e2e"] = df["passes_e2e"].replace(0, np.nan)
    if "seconds_wall" in df.columns:
        df["seconds_wall"] = df["seconds_wall"].replace(0, np.nan)
    
    # 1) Total E2E energy in kWh
    df["kWh_e2e"] = df["energy_j"] / 3.6e6
    
    # 2) Per-pass energy & time (using passes_e2e)
    df["energy_per_pass_j"] = df["energy_j"] / df["passes_e2e"]
    df["kWh_per_pass"] = df["energy_per_pass_j"] / 3.6e6
    df["seconds_per_pass"] = df["seconds_wall"] / df["passes_e2e"]
    
    # 3) Compute avg_power_w if missing
    if "avg_power_w" not in df.columns or df["avg_power_w"].isna().all():
        df["avg_power_w"] = df["energy_j"] / df["seconds_wall"]
    
    # 4) STREAM kernel metrics (sustained memory bandwidth)
    # Only compute if seconds_kernel and bytes_total exist
    if "seconds_kernel" in df.columns and "bytes_total" in df.columns:
        # Replace zeros to avoid division by zero
        df["seconds_kernel"] = df["seconds_kernel"].replace(0, np.nan)
        df["bytes_total"] = df["bytes_total"].replace(0, np.nan)
        
        # GB/s_kernel = bytes_total / seconds_kernel / 1e9
        df["gb_per_s_kernel"] = np.where(
            df["seconds_kernel"].notna() & (df["seconds_kernel"] > 0) & df["bytes_total"].notna(),
            df["bytes_total"] / df["seconds_kernel"] / 1e9,
            np.nan
        )
        
        # J/GB_kernel: Use energy_kernel_j if available, else fall back to energy_j (CPU mode)
        if "energy_kernel_j" in df.columns:
            energy_for_kernel = df["energy_kernel_j"]
        else:
            energy_for_kernel = df["energy_j"]
            print("  [INFO] Using energy_j for j_per_gb_kernel (CPU mode)")
        
        # J/GB_kernel = energy / (bytes_total / 1e9)
        df["j_per_gb_kernel"] = np.where(
            energy_for_kernel.notna() & (energy_for_kernel > 0) & 
            df["bytes_total"].notna() & (df["bytes_total"] > 0),
            energy_for_kernel / (df["bytes_total"] / 1e9),
            np.nan
        )
    
    # 5) Extract matrix dimensions (M, K, N) - only if matrix_size column exists
    if "matrix_size" in df.columns:
        dims = df.apply(extract_matrix_dims, axis=1, result_type="expand")
        df["M"], df["K"], df["N"] = dims[0], dims[1], dims[2]
    else:
        # No matrix_size column - set all to NaN
        df["M"] = np.nan
        df["K"] = np.nan
        df["N"] = np.nan
    
    # 6) FLOP metrics (only for GEMM modes)
    gemm_set = set(m.strip().lower() for m in gemm_modes.split(","))
    
    # Check if mode column exists, otherwise assume all rows are GEMM if matrix_size is present
    if "mode" in df.columns:
        is_gemm = df["mode"].astype(str).str.lower().isin(gemm_set)
    else:
        # If no mode column, treat all rows with valid matrix_size as GEMM
        is_gemm = df["matrix_size"].notna() & (df["matrix_size"].astype(str) != "nan") if "matrix_size" in df.columns else False
    
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
        
        # Flexible column requirements - need energy_j and either passes_e2e OR passes_kernel
        has_energy = "energy_j" in df.columns
        has_passes = "passes_e2e" in df.columns or "passes_kernel" in df.columns
        has_time = "seconds_wall" in df.columns or "seconds_kernel" in df.columns
        
        if not (has_energy and has_passes and has_time):
            missing = []
            if not has_energy:
                missing.append("energy_j")
            if not has_passes:
                missing.append("passes_e2e or passes_kernel")
            if not has_time:
                missing.append("seconds_wall or seconds_kernel")
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
        import traceback
        traceback.print_exc()
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