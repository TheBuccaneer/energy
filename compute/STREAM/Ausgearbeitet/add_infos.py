#!/usr/bin/env python3
"""
Day-Ahead Price Enricher with 2025->2024 timestamp conversion.
Maps hourly electricity prices (DE/FR/PL) to measurement CSVs and calculates costs.
NEW: Converts all 2025 timestamps to 2024 before any processing.
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
        description="Enrich measurement CSVs with Day-Ahead prices (October->September fix)"
    )
    ap.add_argument("--root", required=True, help="Root directory to scan recursively")
    ap.add_argument("--price-file", required=True, help="Path to day_ahead_all_FIXED.csv")
    ap.add_argument("--inplace", action="store_true", default=True,
                    help="Overwrite original files with backup (default: True)")
    ap.add_argument("--no-inplace", dest="inplace", action="store_false",
                    help="Create *_enriched.csv instead of overwriting")
    return ap.parse_args()


def convert_2025_to_2024(timestamp_str):
    """
    Convert ANY 2025 timestamp to 2024 (keep month/day/time identical).
    Also converts October 2024 to September 2024 (same day/time).
    Always returns timezone-aware timestamp in Europe/Berlin.
    
    Examples:
    - 2025-09-15 14:30:00 -> 2024-09-15 14:30:00+02:00
    - 2024-10-15 14:30:00 -> 2024-09-15 14:30:00+02:00
    - 2024-10-31 XX:XX:XX -> NaT (September has only 30 days)
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
        
        # Convert October -> September (fixes incorrect mapping)
        if dt.month == 10:
            # Reject October 31 (September only has 30 days)
            if dt.day == 31:
                return pd.NaT
            dt = dt.replace(month=9)
        
        return dt
    except:
        return pd.NaT


def load_price_data(price_file):
    """
    Load and validate Day-Ahead price data.
    CONVERTS ALL 2025 TIMESTAMPS TO 2024 FIRST!
    Expected format: sep=',' (comma), but with quoted decimal values ("0,08408")
    Columns: local_start_CETCEST, price_eur_mwh, area, price_eur_kwh (DE/FR/PL)
    Returns: (DataFrame indexed by [timestamp, area], dict of sync prices per area)
    """
    try:
        # Read with comma separator and quoted strings
        df = pd.read_csv(price_file, sep=',', quotechar='"')
        
        # Rename columns for consistency
        df = df.rename(columns={
            'local_start_CETCEST': 'timestamp'
        })
        
        # Ensure required columns exist
        required = ['timestamp', 'area']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"ERROR: Available columns: {df.columns.tolist()}", file=sys.stderr)
            sys.exit(f"Error: Price file missing columns: {missing}")
        
        # === NEW: CONVERT 2025 -> 2024 FIRST (if needed) ===
        print("Checking/converting timestamps to 2024...", end='')
        df['timestamp'] = df['timestamp'].apply(convert_2025_to_2024)
        converted = df['timestamp'].notna().sum()
        print(f" {converted} valid timestamps")
        
        # === Handle both old and new price file formats ===
        if 'price_eur_kwh' in df.columns and 'price_eur_mwh' not in df.columns:
            # NEW FORMAT: Only price_eur_kwh (comma-decimal in quotes)
            print("  Detected new format: price_eur_kwh only")
            if df['price_eur_kwh'].dtype == 'object':
                df['price_eur_kwh'] = df['price_eur_kwh'].astype(str).str.replace(',', '.', regex=False)
            df['price_eur_kwh'] = pd.to_numeric(df['price_eur_kwh'], errors='coerce')
        
        elif 'price_eur_mwh' in df.columns:
            # OLD FORMAT: price_eur_mwh, derive price_eur_kwh
            print("  Detected old format: price_eur_mwh")
            if df['price_eur_mwh'].dtype == 'object':
                df['price_eur_mwh'] = df['price_eur_mwh'].astype(str).str.replace(',', '.', regex=False)
            df['price_eur_mwh'] = pd.to_numeric(df['price_eur_mwh'], errors='coerce')
            df['price_eur_kwh'] = df['price_eur_mwh'] / 1000.0
        
        else:
            sys.exit("Error: Price file must have either 'price_eur_kwh' or 'price_eur_mwh' column")
        
        # Plausibility check: typical range -0.20 to 1.00 EUR/kWh
        mask_bad = ~df['price_eur_kwh'].between(-0.20, 1.00)
        if mask_bad.any():
            bad_count = mask_bad.sum()
            bad_examples = df.loc[mask_bad, ['timestamp', 'area', 'price_eur_kwh']].head(5)
            print(f"⚠ Warning: {bad_count} prices outside plausible range (-0.20 to 1.00 EUR/kWh):", file=sys.stderr)
            print(bad_examples.to_string(index=False), file=sys.stderr)
            # Set out-of-range values to NaN
            df.loc[mask_bad, 'price_eur_kwh'] = np.nan
        
        # Uppercase area and strip whitespace
        df['area'] = df['area'].str.strip().str.upper()
        
        # Validate: should have 720 hours × 3 areas = 2160 rows
        expected_rows = 720 * 3
        if len(df) != expected_rows:
            print(f"⚠ Warning: Price file has {len(df)} rows, expected {expected_rows}", file=sys.stderr)
        
        # Check for duplicates per area
        duplicates = df[df.duplicated(subset=['timestamp', 'area'], keep=False)]
        if not duplicates.empty:
            print(f"⚠ Warning: Found {len(duplicates)} duplicate timestamp-area pairs", file=sys.stderr)
            df = df.drop_duplicates(subset=['timestamp', 'area'], keep='first')
        
        # Set multi-index for fast lookup
        df = df.set_index(['timestamp', 'area']).sort_index()
        
        # DEBUG: Show index structure
        print("\n  DEBUG: Price DataFrame Index Info:")
        print(f"    Index levels: {df.index.names}")
        print(f"    First 3 index entries:")
        for idx in list(df.index)[:3]:
            print(f"      {idx} -> {df.loc[idx, 'price_eur_kwh']:.5f} EUR/kWh")
        print(f"    Last index entry: {df.index[-1]}")
        
        # Check timezone info
        first_timestamp = df.index.get_level_values('timestamp')[0]
        print(f"    Timestamp timezone: {first_timestamp.tzinfo}")
        print(f"    Timestamp type: {type(first_timestamp)}")
        
        # Calculate sync price per area (mean over 720 hours each)
        price_sync = {}
        for area in ['DE', 'FR', 'PL']:
            try:
                area_prices = df.xs(area, level='area')['price_eur_kwh']
                price_sync[area] = area_prices.mean()
                print(f"✓ {area}: {len(area_prices)} hours, "
                      f"range {area_prices.min():.4f}-{area_prices.max():.4f} EUR/kWh, "
                      f"sync={price_sync[area]:.4f}")
            except KeyError:
                print(f"⚠ Warning: No data found for area {area}", file=sys.stderr)
                price_sync[area] = 0.0
        
        # Show year range for verification
        all_timestamps = df.reset_index()['timestamp']
        print(f"✓ Loaded {len(df)} hourly prices")
        print(f"  Date range: {all_timestamps.min()} to {all_timestamps.max()}")
        
        return df, price_sync
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(f"Error loading price file: {e}")


def infer_market_area(filename, row_market_area=None):
    """
    Determine market area from filename or existing column.
    Priority: 1) existing market_area column, 2) filename heuristic
    Uses word boundaries to avoid false positives (e.g., "framework" -> FR)
    """
    if pd.notna(row_market_area) and row_market_area in ['DE', 'FR', 'PL']:
        return row_market_area
    
    fname_lower = filename.lower()
    
    # Match on word boundaries: _fr_, -fr-, .fr., _fr., -fr., etc.
    import re
    if re.search(r'[_\-.]fr[_\-.]', fname_lower):
        return 'FR'
    elif re.search(r'[_\-.]pl[_\-.]', fname_lower):
        return 'PL'
    else:
        return 'DE'


def calculate_weighted_price(run_start, run_duration_s, area, price_df):
    """
    Calculate time-weighted price for runs spanning multiple hours.
    ALWAYS uses weighted calculation over affected hours, even for runs <1h.
    Single-hour optimization only when start AND end are in same clock hour.
    """
    if pd.isna(run_start) or run_duration_s <= 0 or not area:
        return np.nan
    
    run_end = run_start + timedelta(seconds=run_duration_s)
    
    # Floor to hour boundaries
    hour_start = run_start.replace(minute=0, second=0, microsecond=0)
    hour_end = run_end.replace(minute=0, second=0, microsecond=0)
    
    # Single hour case: ONLY if both start and end in same clock hour
    if hour_start == hour_end:
        try:
            return price_df.loc[(hour_start, area), 'price_eur_kwh']
        except KeyError:
            return np.nan
    
    # Multi-hour case: weighted by overlap (even if <3600s total duration)
    total_weight = 0.0
    weighted_price = 0.0
    
    current_hour = hour_start
    while current_hour <= hour_end:
        # Calculate overlap in seconds
        overlap_start = max(run_start, current_hour)
        overlap_end = min(run_end, current_hour + timedelta(hours=1))
        overlap_seconds = (overlap_end - overlap_start).total_seconds()
        
        if overlap_seconds > 0:
            try:
                price = price_df.loc[(current_hour, area), 'price_eur_kwh']
                if pd.notna(price):
                    weight = overlap_seconds / run_duration_s
                    weighted_price += price * weight
                    total_weight += weight
            except KeyError:
                pass
        
        current_hour += timedelta(hours=1)
    
    return weighted_price if total_weight > 0 else np.nan


def detect_csv_format(file_path):
    """
    Robustly detect CSV format with fallback cascade.
    Tests multiple combinations and validates numeric columns.
    Returns: (sep, decimal) tuple
    """
    # Expected numeric columns to validate
    numeric_cols = ['seconds_wall', 'energy_j']
    
    # Fallback cascade (in order of preference)
    test_configs = [
        # German: semicolon separator with comma decimal
        (';', ','),
        # German with quoting: comma separator, comma decimal, quoted values
        (',', ','),  # Will need quotechar='"' if this works
        # English: comma separator with dot decimal
        (',', '.'),
        # Tab-separated with dot decimal
        ('\t', '.')
    ]
    
    for sep, decimal in test_configs:
        try:
            # Try reading with this configuration
            if sep == ',' and decimal == ',':
                # This case needs quotechar for quoted comma-decimal values
                df_test = pd.read_csv(file_path, sep=sep, decimal=decimal, 
                                     quotechar='"', nrows=5)
            else:
                df_test = pd.read_csv(file_path, sep=sep, decimal=decimal, nrows=5)
            
            # Check if expected numeric columns are actually numeric
            has_numeric = False
            for col in numeric_cols:
                if col in df_test.columns:
                    # Try to convert to numeric
                    test_series = pd.to_numeric(df_test[col], errors='coerce')
                    # If at least 80% of values are numeric, it's valid
                    if test_series.notna().sum() / len(test_series) >= 0.8:
                        has_numeric = True
                        break
            
            if has_numeric:
                return sep, decimal
                
        except Exception:
            continue
    
    # Fallback to safe default
    return ',', '.'


def enrich_measurement_file(file_path, price_df, price_sync_dict, inplace, issues_log, qa_log):
    """
    Process a single measurement CSV:
    1. Convert ALL 2025 timestamps to 2024 FIRST
    2. Map area (DE/FR/PL)
    3. Calculate realized and sync costs per area
    4. Add new columns in-place, preserving original format
    """
    try:
        # Detect original CSV format
        orig_sep, orig_decimal = detect_csv_format(file_path)
        
        # Read with detected format (add quotechar for comma-decimal + comma-sep case)
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
        
        # === STEP 1: FIX TIMESTAMPS (2025->2024 AND October->September) ===
        print(f"  Fixing timestamps (Oct->Sep)...", end='')
        df['timestamp'] = df['timestamp'].apply(convert_2025_to_2024)
        valid_count = df['timestamp'].notna().sum()
        rejected_oct31 = orig_rows - valid_count
        
        # DEBUG: Check first timestamp
        if valid_count > 0:
            first_ts = df['timestamp'].iloc[0]
            if rejected_oct31 > 0:
                print(f" {valid_count}/{orig_rows} valid ({rejected_oct31} Oct 31 rejected)")
            else:
                print(f" {valid_count}/{orig_rows} valid")
            print(f"    First timestamp: {first_ts}")
        else:
            print(f" {valid_count}/{orig_rows} valid [ERROR: All timestamps failed!]")
            return False
        
        # === STEP 2: Force numeric types ===
        df['seconds_wall'] = pd.to_numeric(df['seconds_wall'], errors='coerce')
        df['energy_j'] = pd.to_numeric(df['energy_j'], errors='coerce')
        
        # Calculate kWh if missing - MUST be numeric for price calculations
        if 'kWh_e2e' not in df.columns:
            df['kWh_e2e'] = df['energy_j'] / 3.6e6
        else:
            # Force numeric conversion (handles comma-decimal strings)
            df['kWh_e2e'] = df['kWh_e2e'].astype(str).str.replace(',', '.', regex=False)
            df['kWh_e2e'] = pd.to_numeric(df['kWh_e2e'], errors='coerce')
        
        # === STEP 3: Infer market area ===
        if 'market_area' not in df.columns:
            df['market_area'] = infer_market_area(file_path.name)
        else:
            df['market_area'] = df.apply(
                lambda row: infer_market_area(file_path.name, row.get('market_area')),
                axis=1
            )
        
        # === STEP 4: Calculate prices for ALL areas (DE, FR, PL) ===
        print(f"  Calculating prices for all areas...")
        
        # DEBUG: Check first row lookup
        if len(df) > 0:
            test_row = df.iloc[0]
            test_ts = test_row['timestamp']
            test_dur = test_row['seconds_wall']
            test_area = test_row['market_area']
            
            print(f"  DEBUG: First row test:")
            print(f"    Timestamp: {test_ts} (tz: {test_ts.tzinfo})")
            print(f"    Duration: {test_dur}s")
            print(f"    Area: {test_area}")
            
            # Try direct lookup
            hour_ts = test_ts.replace(minute=0, second=0, microsecond=0)
            print(f"    Hour timestamp: {hour_ts}")
            
            try:
                lookup_price = price_df.loc[(hour_ts, test_area), 'price_eur_kwh']
                print(f"    ✓ Direct lookup successful: {lookup_price:.5f} EUR/kWh")
            except KeyError as e:
                print(f"    ✗ Direct lookup FAILED: {e}")
                # Show available keys near this timestamp
                print(f"    Available timestamps around this time:")
                price_timestamps = price_df.index.get_level_values('timestamp').unique()
                nearby = [t for t in price_timestamps if abs((t - hour_ts).total_seconds()) < 7200]
                for t in nearby[:5]:
                    print(f"      {t}")
            
            # Try weighted calculation
            try:
                weighted_price = calculate_weighted_price(test_ts, test_dur, test_area, price_df)
                print(f"    Weighted price result: {weighted_price}")
            except Exception as e:
                print(f"    Weighted calculation error: {e}")
        
        # Calculate for each area with country prefix
        for area in ['DE', 'FR', 'PL']:
            area_lower = area.lower()
            
            # Realized price per area
            df[f'{area_lower}_price_eur_kwh_run'] = df.apply(
                lambda row: calculate_weighted_price(
                    row['timestamp'],
                    row['seconds_wall'],
                    area,
                    price_df
                ),
                axis=1
            )
            
            # Realized cost per job per area
            df[f'{area_lower}_eur_job'] = df['kWh_e2e'] * df[f'{area_lower}_price_eur_kwh_run']
            
            # Sync cost per area (time-neutral reference)
            df[f'{area_lower}_price_sync'] = price_sync_dict[area]
            df[f'{area_lower}_eur_job_sync'] = df['kWh_e2e'] * price_sync_dict[area]
        
        # Keep the original market_area-based columns for backward compatibility
        df['price_eur_kwh_run'] = df.apply(
            lambda row: row[f'{row["market_area"].lower()}_price_eur_kwh_run'],
            axis=1
        )
        df['eur_job'] = df.apply(
            lambda row: row[f'{row["market_area"].lower()}_eur_job'],
            axis=1
        )
        df['price_sync'] = df['market_area'].map(price_sync_dict)
        df['eur_job_sync'] = df.apply(
            lambda row: row[f'{row["market_area"].lower()}_eur_job_sync'],
            axis=1
        )
        
        # === STEP 5: Batch metrics (if available) ===
        if 'batches' in df.columns and 'kWh_batch' in df.columns:
            # Force numeric conversion for batch columns
            df['batches'] = pd.to_numeric(df['batches'], errors='coerce')
            
            # Handle comma-decimal in kWh_batch if it's a string
            if df['kWh_batch'].dtype == 'object':
                df['kWh_batch'] = df['kWh_batch'].astype(str).str.replace(',', '.', regex=False)
            df['kWh_batch'] = pd.to_numeric(df['kWh_batch'], errors='coerce')
            
            # Calculate batch costs for all areas with country prefix
            for area in ['DE', 'FR', 'PL']:
                area_lower = area.lower()
                df[f'{area_lower}_price_eur_kwh_batch'] = df[f'{area_lower}_price_eur_kwh_run']
                df[f'{area_lower}_eur_batch'] = df['kWh_batch'] * df[f'{area_lower}_price_eur_kwh_batch']
                df[f'{area_lower}_eur_batch_sync'] = df['kWh_batch'] * price_sync_dict[area]
                df[f'{area_lower}_eur_job_from_batches'] = df[f'{area_lower}_eur_batch'] * df['batches']
            
            # Keep backward-compatible columns based on market_area
            df['price_eur_kwh_batch'] = df.apply(
                lambda row: row[f'{row["market_area"].lower()}_price_eur_kwh_batch'],
                axis=1
            )
            df['eur_batch'] = df.apply(
                lambda row: row[f'{row["market_area"].lower()}_eur_batch'],
                axis=1
            )
            df['eur_batch_sync'] = df.apply(
                lambda row: row[f'{row["market_area"].lower()}_eur_batch_sync'],
                axis=1
            )
            df['eur_job_from_batches'] = df.apply(
                lambda row: row[f'{row["market_area"].lower()}_eur_job_from_batches'],
                axis=1
            )
            
            # QA: check ±1% tolerance for market_area
            mismatch = df[
                (df['eur_job'].notna()) & 
                (df['eur_job_from_batches'].notna()) &
                (np.abs((df['eur_job'] - df['eur_job_from_batches']) / df['eur_job']) > 0.01)
            ]
            if not mismatch.empty:
                qa_log.write(f"{file_path.name}: {len(mismatch)} rows with >1% job/batch mismatch\n")
        
        # === STEP 6: QA checks ===
        na_count = df['price_eur_kwh_run'].isna().sum()
        if na_count > 0:
            issues_log.write(f"{file_path.name}: {na_count}/{orig_rows} rows with missing price\n")
        
        print(f" done ({na_count} NAs)")
        
        # === STEP 7: Save with reversed backup logic ===
        # Original file stays untouched, new file gets 'bak_' prefix
        if inplace:
            out_path = file_path.parent / f"bak_{file_path.name}"
        else:
            out_path = file_path.parent / f"{file_path.stem}_enriched.csv"
        
        # Save with original format (add quotechar for comma-decimal case)
        save_kwargs = {'index': False, 'sep': orig_sep, 'decimal': orig_decimal}
        if orig_sep == ',' and orig_decimal == ',':
            save_kwargs['quotechar'] = '"'
            save_kwargs['quoting'] = 1  # QUOTE_MINIMAL
        
        df.to_csv(out_path, **save_kwargs)
        
        # === STEP 8: Summary statistics per area ===
        print(f"✓ {file_path.name}")
        print(f"  Rows: {len(df)}, Valid prices: {len(df) - na_count}")
        
        for area in df['market_area'].unique():
            area_data = df[df['market_area'] == area]
            if len(area_data) > 0:
                eur_job_mean = area_data['eur_job'].mean()
                eur_sync_mean = area_data['eur_job_sync'].mean()
                print(f"  {area}: n={len(area_data)}, "
                      f"EUR/job realized={eur_job_mean:.6f}, "
                      f"sync={eur_sync_mean:.6f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {file_path.name}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    root = Path(args.root)
    price_file = Path(args.price_file)
    
    if not root.exists():
        sys.exit(f"Error: {root} does not exist")
    if not price_file.exists():
        sys.exit(f"Error: Price file {price_file} does not exist")
    
    # Load price data (with 2025->2024 conversion)
    price_df, price_sync_dict = load_price_data(price_file)
    
    # Find measurement CSVs (exclude price file itself)
    csv_files = [f for f in root.rglob("*.csv") if f != price_file and 'day_ahead' not in f.name.lower()]
    
    if not csv_files:
        sys.exit(f"No measurement CSV files found under {root}")
    
    print(f"\nFound {len(csv_files)} measurement files")
    print("=" * 60)
    
    # Open log files
    issues_log = open(root / "price_join_issues.log", "w")
    qa_log = open(root / "qa_warnings.log", "w")
    
    success = 0
    for file_path in csv_files:
        print(f"\nProcessing {file_path.name}...")
        if enrich_measurement_file(file_path, price_df, price_sync_dict, args.inplace, issues_log, qa_log):
            success += 1
    
    issues_log.close()
    qa_log.close()
    
    print("\n" + "=" * 60)
    print(f"✓ Processed {success}/{len(csv_files)} files successfully")
    print(f"Logs: price_join_issues.log, qa_warnings.log")


if __name__ == "__main__":
    main()