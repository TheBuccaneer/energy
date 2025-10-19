#!/usr/bin/env python3
"""
Day-Ahead Price Enricher for SpMV measurements (Sept 2024 format).
Maps hourly electricity prices (DE/FR/PL) to measurement CSVs and calculates costs.
Compatible with day_ahead_sept2024_final.csv format.
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
        description="Enrich SpMV measurement CSVs with Day-Ahead prices (Sept 2024)"
    )
    ap.add_argument("--root", required=True, help="Root directory to scan recursively")
    ap.add_argument("--price-file", required=True, help="Path to day_ahead_sept2024_final.csv")
    ap.add_argument("--inplace", action="store_true", default=True,
                    help="Create backup with 'bak_' prefix (default: True)")
    ap.add_argument("--no-inplace", dest="inplace", action="store_false",
                    help="Create *_enriched.csv instead")
    return ap.parse_args()


def load_price_data(price_file):
    """
    Load Day-Ahead price data in new Sept 2024 format.
    Expected columns:
    - timestamp, date
    - price_eur_kwh_de, price_eur_kwh_fr, price_eur_kwh_pl
    - price_sync_de, price_sync_fr, price_sync_pl
    
    Returns: (DataFrame indexed by timestamp, dict of sync prices per area)
    """
    try:
        df = pd.read_csv(price_file)
        
        # Check required columns
        required = ['timestamp', 'price_eur_kwh_de', 'price_eur_kwh_fr', 'price_eur_kwh_pl',
                   'price_sync_de', 'price_sync_fr', 'price_sync_pl']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"ERROR: Available columns: {df.columns.tolist()}", file=sys.stderr)
            sys.exit(f"Error: Price file missing columns: {missing}")
        
        # Parse timestamp and make timezone-aware (Europe/Berlin)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        tz = ZoneInfo('Europe/Berlin')
        
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(tz)
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert(tz)
        
        # Set timestamp as index for fast lookup
        df = df.set_index('timestamp').sort_index()
        
        # Extract sync prices (same in all rows)
        price_sync = {
            'DE': df['price_sync_de'].iloc[0],
            'FR': df['price_sync_fr'].iloc[0],
            'PL': df['price_sync_pl'].iloc[0]
        }
        
        print(f"✅ Loaded {len(df)} hourly prices")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Sync prices (Sept 2024 mean):")
        for area, price in price_sync.items():
            print(f"    {area}: {price:.6f} EUR/kWh")
        
        return df, price_sync
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(f"Error loading price file: {e}")


def convert_2025_to_2024(timestamp_str):
    """
    Convert ANY 2025 timestamp to 2024 (keep month/day/time identical).
    Also converts October 2024 to September 2024 (same day/time).
    Always returns timezone-aware timestamp in Europe/Berlin.
    """
    try:
        dt = pd.to_datetime(timestamp_str)
        tz = ZoneInfo('Europe/Berlin')
        
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


def infer_market_area(filename, row_market_area=None):
    """
    Determine market area from filename or existing column.
    Priority: 1) existing market_area column, 2) filename heuristic
    """
    if pd.notna(row_market_area) and row_market_area in ['DE', 'FR', 'PL']:
        return row_market_area
    
    fname_lower = filename.lower()
    
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
    Uses new format: price_eur_kwh_{area} columns indexed by timestamp.
    """
    if pd.isna(run_start) or run_duration_s <= 0 or not area:
        return np.nan
    
    area_lower = area.lower()
    price_col = f'price_eur_kwh_{area_lower}'
    
    if price_col not in price_df.columns:
        return np.nan
    
    run_end = run_start + timedelta(seconds=run_duration_s)
    
    # Floor to hour boundaries
    hour_start = run_start.replace(minute=0, second=0, microsecond=0)
    hour_end = run_end.replace(minute=0, second=0, microsecond=0)
    
    # Single hour case
    if hour_start == hour_end:
        try:
            return price_df.loc[hour_start, price_col]
        except KeyError:
            return np.nan
    
    # Multi-hour case: weighted by overlap
    total_weight = 0.0
    weighted_price = 0.0
    
    current_hour = hour_start
    while current_hour <= hour_end:
        overlap_start = max(run_start, current_hour)
        overlap_end = min(run_end, current_hour + timedelta(hours=1))
        overlap_seconds = (overlap_end - overlap_start).total_seconds()
        
        if overlap_seconds > 0:
            try:
                price = price_df.loc[current_hour, price_col]
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
    Returns: (sep, decimal) tuple
    """
    numeric_cols = ['seconds_wall', 'energy_j']
    
    test_configs = [
        (';', ','),  # German: semicolon + comma decimal
        (',', ','),  # German with quotes
        (',', '.'),  # English
        ('\t', '.')  # Tab-separated
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


def enrich_measurement_file(file_path, price_df, price_sync_dict, inplace, issues_log, qa_log):
    """
    Process a single measurement CSV:
    1. Convert timestamps (2025->2024, Oct->Sep)
    2. Map area (DE/FR/PL)
    3. Calculate realized and sync costs for ALL areas
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
        print(f"  Fixing timestamps (2025->2024, Oct->Sep)...", end='')
        df['timestamp'] = df['timestamp'].apply(convert_2025_to_2024)
        valid_count = df['timestamp'].notna().sum()
        rejected = orig_rows - valid_count
        
        if rejected > 0:
            print(f" {valid_count}/{orig_rows} valid ({rejected} rejected)")
        else:
            print(f" {valid_count}/{orig_rows} valid")
        
        if valid_count == 0:
            print(f"[ERROR] {file_path.name}: No valid timestamps", file=sys.stderr)
            return False
        
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
        
        # === STEP 3: Infer market area ===
        if 'market_area' not in df.columns:
            df['market_area'] = infer_market_area(file_path.name)
        else:
            df['market_area'] = df.apply(
                lambda row: infer_market_area(file_path.name, row.get('market_area')),
                axis=1
            )
        
        # === STEP 4: Calculate prices for ALL areas ===
        print(f"  Calculating prices for all areas...", end='')
        
        for area in ['DE', 'FR', 'PL']:
            area_lower = area.lower()
            
            # Realized price per area (time-weighted)
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
        
        # Backward-compatible columns (based on market_area)
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
            df['batches'] = pd.to_numeric(df['batches'], errors='coerce')
            
            if df['kWh_batch'].dtype == 'object':
                df['kWh_batch'] = df['kWh_batch'].astype(str).str.replace(',', '.', regex=False)
            df['kWh_batch'] = pd.to_numeric(df['kWh_batch'], errors='coerce')
            
            # Calculate batch costs for all areas
            for area in ['DE', 'FR', 'PL']:
                area_lower = area.lower()
                df[f'{area_lower}_price_eur_kwh_batch'] = df[f'{area_lower}_price_eur_kwh_run']
                df[f'{area_lower}_eur_batch'] = df['kWh_batch'] * df[f'{area_lower}_price_eur_kwh_batch']
                df[f'{area_lower}_eur_batch_sync'] = df['kWh_batch'] * price_sync_dict[area]
                df[f'{area_lower}_eur_job_from_batches'] = df[f'{area_lower}_eur_batch'] * df['batches']
            
            # Backward-compatible batch columns
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
            
            # QA: check ±1% tolerance
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
        
        # === STEP 7: Save ===
        if inplace:
            # Original stays untouched, backup gets 'bak_' prefix
            out_path = file_path.parent / f"bak_{file_path.name}"
        else:
            out_path = file_path.parent / f"{file_path.stem}_enriched.csv"
        
        save_kwargs = {'index': False, 'sep': orig_sep, 'decimal': orig_decimal}
        if orig_sep == ',' and orig_decimal == ',':
            save_kwargs['quotechar'] = '"'
            save_kwargs['quoting'] = 1
        
        df.to_csv(out_path, **save_kwargs)
        
        # === STEP 8: Summary ===
        print(f"✅ {file_path.name}")
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
    
    # Load price data
    price_df, price_sync_dict = load_price_data(price_file)
    
    # Find measurement CSVs (exclude price file itself and backups)
    csv_files = [f for f in root.rglob("*.csv") 
                 if f != price_file 
                 and 'day_ahead' not in f.name.lower()
                 and not f.name.startswith('bak_')
                 and not f.name.startswith('kab_')]
    
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
    print(f"✅ Processed {success}/{len(csv_files)} files successfully")
    print(f"Logs: price_join_issues.log, qa_warnings.log")


if __name__ == "__main__":
    main()