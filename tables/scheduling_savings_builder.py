#!/usr/bin/env python3
"""
Scheduling Savings Builder - ORIGINAL WORKING VERSION
Generates scheduling_savings_by_run.csv for Pareto-Front analysis
Values represent BATCH jobs (aggregate over n_measurements)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

ZONES = ['de', 'fr', 'pl']
WORKLOAD_FILES = [
    'GEMM_by_run.csv',
    'SPMV_by_run.csv',
    'STREAM_by_run.csv',
    'REDUCTION_by_run.csv'
]

MAX_WAIT_HOURS = 24

# ============================================================================
# LOAD DATA
# ============================================================================

def load_windows():
    """Load time-window data for all zones"""
    windows = {}
    for zone in ZONES:
        df = pd.read_csv(f'windows_{zone}.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert numeric columns from German format (comma) to float
        numeric_cols = ['price_eur_kwh', 'ci_g_per_kwh']
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        windows[zone] = df
    return windows

def load_jobs():
    """Load all job characteristics"""
    jobs = []
    for filename in WORKLOAD_FILES:
        try:
            df = pd.read_csv(filename)
            
            # Convert kWh from German format if needed
            if 'kWh_e2e_sum' in df.columns and df['kWh_e2e_sum'].dtype == 'object':
                df['kWh_e2e_sum'] = df['kWh_e2e_sum'].astype(str).str.replace(',', '.').astype(float)
            
            # Also convert n_measurements for reporting
            if 'n_measurements' in df.columns and df['n_measurements'].dtype == 'object':
                df['n_measurements'] = df['n_measurements'].astype(str).str.replace(',', '.').astype(float)
            
            jobs.append(df)
            print(f"  ✓ Loaded {filename}: {len(df)} rows")
        except FileNotFoundError:
            print(f"  ⚠️  Warning: {filename} not found, skipping...")
    
    if not jobs:
        raise FileNotFoundError("No job files found!")
    
    return pd.concat(jobs, ignore_index=True)

# ============================================================================
# CORE LOGIC
# ============================================================================

def precompute_optimal_windows(windows_df, max_wait):
    """
    Pre-compute all optimal windows once for performance
    """
    n_hours = len(windows_df)
    
    cheapest_result = []
    greenest_result = []
    overlap_result = []
    
    for hour_idx in range(n_hours):
        end_idx = min(hour_idx + max_wait + 1, n_hours)
        search_window = windows_df.iloc[hour_idx:end_idx]
        
        # Cheapest
        cheapest_idx = search_window['price_eur_kwh'].idxmin()
        cheapest_result.append({
            'start_hour': hour_idx,
            'target_hour': cheapest_idx,
            'wait_h': cheapest_idx - hour_idx
        })
        
        # Greenest
        greenest_idx = search_window['ci_g_per_kwh'].idxmin()
        greenest_result.append({
            'start_hour': hour_idx,
            'target_hour': greenest_idx,
            'wait_h': greenest_idx - hour_idx
        })
        
        # Overlap
        overlap_mask = search_window['overlap_flag'] == True
        if overlap_mask.any():
            overlap_idx = search_window[overlap_mask].index[0]
        else:
            overlap_idx = cheapest_idx
        
        overlap_result.append({
            'start_hour': hour_idx,
            'target_hour': overlap_idx,
            'wait_h': overlap_idx - hour_idx
        })
    
    return {
        'cheapest': pd.DataFrame(cheapest_result),
        'greenest': pd.DataFrame(greenest_result),
        'overlap': pd.DataFrame(overlap_result)
    }

def calculate_scenario(job_kwh, windows_df, hour_idx, scenario_type):
    """Calculate cost and CO2 for a specific scenario"""
    row = windows_df.iloc[hour_idx]
    
    return {
        'hour': hour_idx,
        'eur': job_kwh * row['price_eur_kwh'],
        'co2_kg': job_kwh * row['ci_g_per_kwh'] / 1000,
        'wait_h': 0
    }

def process_job_zone(job, zone, windows_df, optimal_windows):
    """Process one job for one zone across all hours"""
    job_id = job['setup_id']
    workload = job['workload']
    job_kwh = job['kWh_e2e_sum']  # NOTE: This is aggregate over n_measurements!
    
    scenarios = []
    
    for hour_idx in range(len(windows_df)):
        # AS-RUN
        asrun = calculate_scenario(job_kwh, windows_df, hour_idx, 'asrun')
        
        # CHEAPEST
        cheapest_row = optimal_windows['cheapest'].iloc[hour_idx]
        cheapest = calculate_scenario(job_kwh, windows_df, cheapest_row['target_hour'], 'cheapest')
        cheapest['wait_h'] = cheapest_row['wait_h']
        
        # GREENEST
        greenest_row = optimal_windows['greenest'].iloc[hour_idx]
        greenest = calculate_scenario(job_kwh, windows_df, greenest_row['target_hour'], 'greenest')
        greenest['wait_h'] = greenest_row['wait_h']
        
        # OVERLAP
        overlap_row = optimal_windows['overlap'].iloc[hour_idx]
        overlap = calculate_scenario(job_kwh, windows_df, overlap_row['target_hour'], 'overlap')
        overlap['wait_h'] = overlap_row['wait_h']
        
        scenarios.append({
            'job_id': job_id,
            'workload': workload,
            'zone': zone,
            'kwh_job': job_kwh,
            'hour_asrun': asrun['hour'],
            'eur_asrun': asrun['eur'],
            'co2_asrun_kg': asrun['co2_kg'],
            'hour_cheapest': cheapest['hour'],
            'eur_cheapest': cheapest['eur'],
            'co2_cheapest_kg': cheapest['co2_kg'],
            'wait_h_cheapest': cheapest['wait_h'],
            'hour_greenest': greenest['hour'],
            'eur_greenest': greenest['eur'],
            'co2_greenest_kg': greenest['co2_kg'],
            'wait_h_greenest': greenest['wait_h'],
            'hour_overlap': overlap['hour'],
            'eur_overlap': overlap['eur'],
            'co2_overlap_kg': overlap['co2_kg'],
            'wait_h_overlap': overlap['wait_h']
        })
    
    return scenarios

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("SCHEDULING SAVINGS BUILDER")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading time windows...")
    windows = load_windows()
    print(f"✓ Loaded {len(ZONES)} zones, {len(windows['de'])} hours each")
    
    print("\n[2/5] Pre-computing optimal windows...")
    optimal_windows = {}
    for zone in ZONES:
        optimal_windows[zone] = precompute_optimal_windows(windows[zone], MAX_WAIT_HOURS)
    print(f"✓ Pre-computed for {len(ZONES)} zones")
    
    print("\n[3/5] Loading job characteristics...")
    jobs_df = load_jobs()
    print(f"✓ Loaded {len(jobs_df)} jobs across {jobs_df['workload'].nunique()} workloads")
    print(f"  Workloads: {', '.join(jobs_df['workload'].unique())}")
    
    # Report batch info
    if 'n_measurements' in jobs_df.columns:
        print(f"\n  Batch info:")
        print(f"    Median n_measurements: {jobs_df['n_measurements'].median():.0f}")
        print(f"    Range: {jobs_df['n_measurements'].min():.0f} - {jobs_df['n_measurements'].max():.0f}")
        print(f"  NOTE: kWh_e2e_sum represents AGGREGATE over all measurements")
    
    print(f"\n  Energy ranges (AGGREGATE per batch):")
    print(f"    Median: {jobs_df['kWh_e2e_sum'].median():.6f} kWh")
    print(f"    Range: {jobs_df['kWh_e2e_sum'].min():.6f} - {jobs_df['kWh_e2e_sum'].max():.6f} kWh")
    
    # Process
    print("\n[4/5] Processing job×zone×hour scenarios...")
    all_scenarios = []
    
    total_jobs = len(jobs_df) * len(ZONES)
    processed = 0
    
    for _, job in jobs_df.iterrows():
        for zone in ZONES:
            scenarios = process_job_zone(job, zone, windows[zone], optimal_windows[zone])
            all_scenarios.extend(scenarios)
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Progress: {processed}/{total_jobs} job-zone pairs...")
    
    print(f"✓ Generated {len(all_scenarios)} scenario rows")
    
    # Convert to DataFrame
    print("\n[5/5] Saving results...")
    df_results = pd.DataFrame(all_scenarios)
    
    # Add savings columns (safe for negative prices)
    def safe_pct_savings(baseline, optimized):
        mask = baseline > 0.001
        result = pd.Series(np.nan, index=baseline.index)
        result[mask] = (baseline[mask] - optimized[mask]) / baseline[mask] * 100
        result = result.clip(lower=-100, upper=1000)
        return result
    
    df_results['savings_eur_cheapest_pct'] = safe_pct_savings(
        df_results['eur_asrun'], 
        df_results['eur_cheapest']
    )
    df_results['savings_co2_cheapest_pct'] = safe_pct_savings(
        df_results['co2_asrun_kg'],
        df_results['co2_cheapest_kg']
    )
    df_results['savings_eur_greenest_pct'] = safe_pct_savings(
        df_results['eur_asrun'],
        df_results['eur_greenest']
    )
    df_results['savings_co2_greenest_pct'] = safe_pct_savings(
        df_results['co2_asrun_kg'],
        df_results['co2_greenest_kg']
    )
    
    # Save
    output_file = 'scheduling_savings_by_run.csv'
    df_results.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (BATCH-LEVEL)")
    print("="*60)
    
    for zone in ZONES:
        zone_data = df_results[df_results['zone'] == zone]
        valid_cost = zone_data['savings_eur_cheapest_pct'].notna()
        
        print(f"\n{zone.upper()}:")
        print(f"  Avg wait (cheapest): {zone_data['wait_h_cheapest'].mean():.1f}h")
        print(f"  Avg wait (greenest): {zone_data['wait_h_greenest'].mean():.1f}h")
        print(f"  Median cost savings: {zone_data.loc[valid_cost, 'savings_eur_cheapest_pct'].median():.1f}%")
        print(f"  Median CO2 savings: {zone_data['savings_co2_greenest_pct'].median():.1f}%")
    
    print("\n✅ COMPLETE!")
    print(f"Generated {len(df_results)} scenario rows")
    print("\n📊 NOTE: Values represent BATCH jobs (aggregate over n_measurements)")
    print("   For per-single-run values, divide by median n_measurements (~25)")

if __name__ == '__main__':
    main()