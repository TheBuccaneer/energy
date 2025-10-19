#!/usr/bin/env python3
"""
CO₂ Intensity Enricher for measurement CSVs.
Maps hourly CO₂ intensities (DE/FR/PL) to measurement CSVs and calculates kgCO₂/Job.
Uses location-based (average) grid emissions per GHG Protocol Scope 2.
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo


def parse_args():
    ap = argparse.ArgumentParser(
        description="Enrich measurement CSVs with CO₂ intensities (Scope 2 location-based)"
    )
    ap.add_argument("--root", required=True, help="Root directory to scan recursively")
    ap.add_argument("--ci-file", required=True, help="Path to ci_DE_FR_PL_hourly.csv")
    ap.add_argument("--ci-mode", choices=['lifecycle', 'direct'], default='lifecycle',
                    help="CI column mode: lifecycle (default) or direct")
    return ap.parse_args()


def convert_2025_to_2024(timestamp_str):
    """
    Convert ANY 2025 timestamp to 2024 (keep month/day/time identical).
    Also converts October 2024 to September 2024 (same day/time).
    Always returns timezone-aware timestamp in Europe/Berlin.
    """
    try:
        dt = pd.to_datetime(timestamp_str)
        tz = ZoneInfo('Europe/Berlin')
        
        # Make timezone-aware if needed
        if dt.tz is None:
            dt = dt.tz_localize(tz)
        else:
            dt = dt.tz_convert(tz)
        
        # Convert 2025 -> 2024
        if dt.year == 2025:
            dt = dt.replace(year=2024)
        
        # Convert October -> September
        if dt.month == 10:
            if dt.day == 31:
                return pd.NaT
            dt = dt.replace(month=9)
        
        return dt
    except:
        return pd.NaT


def load_ci_data(ci_file, ci_mode):
    """
    Load and validate CO₂ intensity data.
    Converts UTC timestamps to Europe/Berlin for consistent joining.
    
    Args:
        ci_file: Path to ci_DE_FR_PL_hourly.csv
        ci_mode: 'lifecycle' or 'direct'
    
    Returns:
        DataFrame indexed by timestamp (Europe/Berlin), with columns:
        de_ci_g_per_kwh, fr_ci_g_per_kwh, pl_ci_g_per_kwh
    """
    try:
        # Read CI file
        df = pd.read_csv(ci_file)
        
        # Ensure required columns
        if 'ts' not in df.columns:
            sys.exit("Error: CI file missing 'ts' column")
        
        # Select columns based on mode
        if ci_mode == 'lifecycle':
            required = ['de_ci_lifecycle_g_per_kwh', 'fr_ci_lifecycle_g_per_kwh', 'pl_ci_lifecycle_g_per_kwh']
        else:  # direct
            required = ['de_ci_direct_g_per_kwh', 'fr_ci_direct_g_per_kwh', 'pl_ci_direct_g_per_kwh']
        
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"ERROR: Available columns: {df.columns.tolist()}", file=sys.stderr)
            sys.exit(f"Error: CI file missing columns: {missing}")
        
        print(f"Loading CI data (mode: {ci_mode})...")
        
        # Parse timestamps (UTC in file)
        df['ts'] = pd.to_datetime(df['ts'])
        
        # Convert UTC -> Europe/Berlin for joining with measurement timestamps
        tz_utc = ZoneInfo('UTC')
        tz_berlin = ZoneInfo('Europe/Berlin')
        
        if df['ts'].dt.tz is None:
            df['ts'] = df['ts'].dt.tz_localize(tz_utc)
        df['ts'] = df['ts'].dt.tz_convert(tz_berlin)
        
        # Floor timestamps to hour BEFORE renaming
        # Convert back to UTC temporarily to avoid DST ambiguity during floor operation
        ts_utc = df['ts'].dt.tz_convert('UTC')
        ts_utc_floored = ts_utc.dt.floor('h')
        df['ts_hour'] = ts_utc_floored.dt.tz_convert(tz_berlin)
        
        # Rename columns to standardized names
        df = df.rename(columns={
            required[0]: 'de_ci_g_per_kwh',
            required[1]: 'fr_ci_g_per_kwh',
            required[2]: 'pl_ci_g_per_kwh'
        })
        
        # Keep only timestamp and CI columns
        df = df[['ts_hour', 'de_ci_g_per_kwh', 'fr_ci_g_per_kwh', 'pl_ci_g_per_kwh']]
        
        # Convert to numeric
        for col in ['de_ci_g_per_kwh', 'fr_ci_g_per_kwh', 'pl_ci_g_per_kwh']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Plausibility check: typical range 0-1500 gCO₂/kWh
        for area in ['de', 'fr', 'pl']:
            col = f'{area}_ci_g_per_kwh'
            mask_bad = ~df[col].between(0, 1500)
            if mask_bad.any():
                bad_count = mask_bad.sum()
                print(f"⚠ Warning: {bad_count} {area.upper()} CI values outside range (0-1500)", file=sys.stderr)
                df.loc[mask_bad, col] = np.nan
        
        # Set index for fast lookup
        df = df.set_index('ts_hour').sort_index()
        
        # Remove duplicates (keep first)
        df = df[~df.index.duplicated(keep='first')]
        
        # Stats per area
        print(f"✓ Loaded {len(df)} hourly CO₂ intensities")
        for area in ['de', 'fr', 'pl']:
            col = f'{area}_ci_g_per_kwh'
            ci_min = df[col].min()
            ci_max = df[col].max()
            ci_mean = df[col].mean()
            print(f"  {area.upper()}: range {ci_min:.1f}-{ci_max:.1f} gCO₂/kWh (mean: {ci_mean:.1f})")
        
        # Show time range
        print(f"  Time range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(f"Error loading CI file: {e}")


def calculate_weighted_ci(run_start, run_duration_s, area, ci_df):
    """
    Calculate time-weighted CO₂ intensity for runs spanning multiple hours.
    Uses the same overlap-weighting logic as price calculation.
    
    Args:
        run_start: Timestamp (timezone-aware, Europe/Berlin)
        run_duration_s: Duration in seconds
        area: 'de', 'fr', or 'pl' (lowercase)
        ci_df: DataFrame indexed by hour timestamp, with columns *_ci_g_per_kwh
    
    Returns:
        Weighted average CO₂ intensity in gCO₂/kWh, or NaN if no data
    """
    if pd.isna(run_start) or run_duration_s <= 0 or not area:
        return np.nan
    
    ci_col = f'{area}_ci_g_per_kwh'
    
    if ci_col not in ci_df.columns:
        return np.nan
    
    run_end = run_start + timedelta(seconds=run_duration_s)
    
    # Floor to hour boundaries
    hour_start = run_start.replace(minute=0, second=0, microsecond=0)
    hour_end = run_end.replace(minute=0, second=0, microsecond=0)
    
    # Single hour case
    if hour_start == hour_end:
        try:
            return ci_df.loc[hour_start, ci_col]
        except KeyError:
            return np.nan
    
    # Multi-hour case: weighted by overlap
    total_weight = 0.0
    weighted_ci = 0.0
    
    current_hour = hour_start
    while current_hour <= hour_end:
        overlap_start = max(run_start, current_hour)
        overlap_end = min(run_end, current_hour + timedelta(hours=1))
        overlap_seconds = (overlap_end - overlap_start).total_seconds()
        
        if overlap_seconds > 0:
            try:
                ci = ci_df.loc[current_hour, ci_col]
                if pd.notna(ci):
                    weight = overlap_seconds / run_duration_s
                    weighted_ci += ci * weight
                    total_weight += weight
            except KeyError:
                pass
        
        current_hour += timedelta(hours=1)
    
    return weighted_ci if total_weight > 0 else np.nan


def detect_csv_format(file_path):
    """
    Robustly detect CSV format with fallback cascade.
    Returns: (sep, decimal) tuple
    """
    numeric_cols = ['seconds_wall', 'energy_j']
    
    test_configs = [
        (';', ','),
        (',', ','),
        (',', '.'),
        ('\t', '.')
    ]
    
    for sep, decimal in test_configs:
        try:
            if sep == ',' and decimal == ',':
                df_test = pd.read_csv(file_path, sep=sep, decimal=decimal, 
                                     quotechar='"', nrows=5)
            else:
                df_test = pd.read_csv(file_path, sep=sep, decimal=decimal, nrows=5)
            
            has_numeric = False
            for col in numeric_cols:
                if col in df_test.columns:
                    test_series = pd.to_numeric(df_test[col], errors='coerce')
                    if test_series.notna().sum() / len(test_series) >= 0.8:
                        has_numeric = True
                        break
            
            if has_numeric:
                return sep, decimal
                
        except Exception:
            continue
    
    return ',', '.'


def enrich_measurement_file(file_path, ci_df, issues_log, qa_log):
    """
    Process a single measurement CSV:
    1. Fix timestamps (2025->2024, Oct->Sep)
    2. Calculate weighted CO₂ intensities for all areas
    3. Calculate kgCO₂/Job = kWh × (gCO₂/kWh) / 1000
    4. Add new columns, preserving original format
    """
    try:
        # Detect original CSV format
        orig_sep, orig_decimal = detect_csv_format(file_path)
        
        read_kwargs = {'sep': orig_sep, 'decimal': orig_decimal}
        if orig_sep == ',' and orig_decimal == ',':
            read_kwargs['quotechar'] = '"'
        
        df = pd.read_csv(file_path, **read_kwargs)
        
        # Check required columns
        required = ['timestamp', 'seconds_wall', 'energy_j']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[SKIP] {file_path.name}: missing {missing}", file=sys.stderr)
            return False
        
        orig_rows = len(df)
        
        # === STEP 1: FIX TIMESTAMPS ===
        print(f"  Fixing timestamps...", end='')
        df['timestamp'] = df['timestamp'].apply(convert_2025_to_2024)
        valid_count = df['timestamp'].notna().sum()
        
        if valid_count == 0:
            print(f" FAILED: All timestamps invalid")
            return False
        
        print(f" {valid_count}/{orig_rows} valid")
        
        # === STEP 2: Force numeric types ===
        df['seconds_wall'] = pd.to_numeric(df['seconds_wall'], errors='coerce')
        df['energy_j'] = pd.to_numeric(df['energy_j'], errors='coerce')
        
        # Calculate kWh if missing
        if 'kWh_e2e' not in df.columns:
            df['kWh_e2e'] = df['energy_j'] / 3.6e6
        else:
            if df['kWh_e2e'].dtype == 'object':
                df['kWh_e2e'] = df['kWh_e2e'].astype(str).str.replace(',', '.', regex=False)
            df['kWh_e2e'] = pd.to_numeric(df['kWh_e2e'], errors='coerce')
        
        # === STEP 3: Calculate CO₂ intensities for ALL areas (DE, FR, PL) ===
        print(f"  Calculating CO₂ intensities for all areas...", end='')
        
        for area in ['de', 'fr', 'pl']:
            # Weighted CI per area
            df[f'{area}_ci_g_per_kwh'] = df.apply(
                lambda row: calculate_weighted_ci(
                    row['timestamp'],
                    row['seconds_wall'],
                    area,
                    ci_df
                ),
                axis=1
            )
            
            # Calculate kgCO₂/Job per area
            # Formula: kWh × (gCO₂/kWh) / 1000 = kgCO₂
            df[f'{area}_co2_job_kg'] = (
                df['kWh_e2e'] * df[f'{area}_ci_g_per_kwh'] / 1000.0
            )
        
        # === STEP 4: Batch metrics (if available) ===
        if 'batches' in df.columns and 'kWh_batch' in df.columns:
            df['batches'] = pd.to_numeric(df['batches'], errors='coerce')
            
            if df['kWh_batch'].dtype == 'object':
                df['kWh_batch'] = df['kWh_batch'].astype(str).str.replace(',', '.', regex=False)
            df['kWh_batch'] = pd.to_numeric(df['kWh_batch'], errors='coerce')
            
            # Calculate batch CO₂ for all areas
            for area in ['de', 'fr', 'pl']:
                df[f'{area}_co2_batch_kg'] = (
                    df['kWh_batch'] * df[f'{area}_ci_g_per_kwh'] / 1000.0
                )
                df[f'{area}_co2_job_from_batches_kg'] = (
                    df[f'{area}_co2_batch_kg'] * df['batches']
                )
            
            # QA: check ±1% tolerance
            for area in ['de', 'fr', 'pl']:
                mismatch = df[
                    (df[f'{area}_co2_job_kg'].notna()) & 
                    (df[f'{area}_co2_job_from_batches_kg'].notna()) &
                    (np.abs((df[f'{area}_co2_job_kg'] - df[f'{area}_co2_job_from_batches_kg']) 
                            / df[f'{area}_co2_job_kg']) > 0.01)
                ]
                if not mismatch.empty:
                    qa_log.write(f"{file_path.name} ({area.upper()}): {len(mismatch)} rows with >1% job/batch CO₂ mismatch\n")
        
        # === STEP 5: QA checks ===
        na_counts = {}
        for area in ['de', 'fr', 'pl']:
            na_count = df[f'{area}_ci_g_per_kwh'].isna().sum()
            na_counts[area] = na_count
            if na_count > 0:
                issues_log.write(f"{file_path.name} ({area.upper()}): {na_count}/{orig_rows} rows with missing CI\n")
        
        print(f" done (NAs: DE={na_counts['de']}, FR={na_counts['fr']}, PL={na_counts['pl']})")
        
        # === STEP 6: Numerical QA ===
        # Check formula accuracy: co2_job_kg ≈ kWh × ci / 1000
        for area in ['de', 'fr', 'pl']:
            recalc = df['kWh_e2e'] * df[f'{area}_ci_g_per_kwh'] / 1000.0
            diff = (df[f'{area}_co2_job_kg'] - recalc).abs()
            max_diff = diff.max()
            
            if max_diff > 1e-9:
                qa_log.write(f"{file_path.name} ({area.upper()}): max formula deviation {max_diff:.2e} kg\n")
        
        # === STEP 7: Save ===
        # New file with 'kab_' prefix (original with 'bak_' stays)
        out_path = file_path.parent / f"kab_{file_path.name.replace('bak_', '')}"
        
        save_kwargs = {'index': False, 'sep': orig_sep, 'decimal': orig_decimal}
        if orig_sep == ',' and orig_decimal == ',':
            save_kwargs['quotechar'] = '"'
            save_kwargs['quoting'] = 1
        
        df.to_csv(out_path, **save_kwargs)
        
        # === STEP 8: Summary statistics ===
        print(f"✓ {file_path.name} -> {out_path.name}")
        print(f"  Rows: {len(df)}")
        
        for area in ['de', 'fr', 'pl']:
            valid = df[f'{area}_co2_job_kg'].notna().sum()
            if valid > 0:
                co2_mean = df[f'{area}_co2_job_kg'].mean()
                co2_sum = df[f'{area}_co2_job_kg'].sum()
                print(f"  {area.upper()}: valid={valid}, mean={co2_mean*1000:.3f}g/job, total={co2_sum:.6f}kg")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {file_path.name}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    root = Path(args.root)
    ci_file = Path(args.ci_file)
    
    if not root.exists():
        sys.exit(f"Error: {root} does not exist")
    if not ci_file.exists():
        sys.exit(f"Error: CI file {ci_file} does not exist")
    
    # Load CI data
    ci_df = load_ci_data(ci_file, args.ci_mode)
    
    # Find measurement CSVs starting with 'bak_'
    csv_files = [f for f in root.rglob("bak_*.csv")]
    
    if not csv_files:
        sys.exit(f"No bak_*.csv measurement files found under {root}")
    
    print(f"\nFound {len(csv_files)} measurement files")
    print("=" * 60)
    
    # Open log files
    issues_log = open(root / "co2_join_issues.log", "w")
    qa_log = open(root / "co2_qa_warnings.log", "w")
    
    success = 0
    for file_path in csv_files:
        print(f"\nProcessing {file_path.name}...")
        if enrich_measurement_file(file_path, ci_df, issues_log, qa_log):
            success += 1
    
    issues_log.close()
    qa_log.close()
    
    print("\n" + "=" * 60)
    print(f"✓ Processed {success}/{len(csv_files)} files successfully")
    print(f"Logs: co2_join_issues.log, co2_qa_warnings.log")


if __name__ == "__main__":
    main()