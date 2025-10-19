#!/usr/bin/env python3
"""
Step 3: Scheduling Simulation - Calculate savings from shifting jobs to better time windows.

Methodology:
- Prices: EUR/kWh (originally EUR/MWh from ENTSO-E, converted via ÷1000)
- September 2024: Hourly resolution (PT60M); 15-min MTU starts 30.09.2025 per EPEX SPOT
- Scope 2 location-based: Grid-average carbon intensity (GHG Protocol)

Sources:
- ENTSO-E Transparency: https://transparency.entsoe.eu
- EPEX SPOT: https://www.epexspot.com/en/news/15-minute-products-live-epex-spot-day-ahead-markets
- GHG Protocol: https://ghgprotocol.org/sites/default/files/2023-03/Scope%202%20Guidance.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple


def load_price_and_ci_data() -> pd.DataFrame:
    """Load and merge price and CI data."""
    print("[1/5] Loading price and CI data...")
    
    # Load prices
    df_price = pd.read_csv("./day_ahead_sept2024_final.csv")
    df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
    
    # Load CI
    df_ci = pd.read_csv("./ci_DE_FR_PL_hourly.csv")
    df_ci['ts'] = pd.to_datetime(df_ci['ts']).dt.tz_localize(None)
    
    # Merge
    df = pd.merge(df_price, df_ci, left_on='timestamp', right_on='ts', how='inner')
    
    if len(df) == 0:
        raise ValueError("Merge resulted in 0 rows!")
    
    print(f"  Loaded {len(df)} hourly records")
    return df


def load_windows() -> Dict[str, pd.DataFrame]:
    """Load time windows (cheapest, greenest, overlap) per area."""
    print("[2/5] Loading time windows...")
    
    windows = {}
    for area in ['de', 'fr', 'pl']:
        # Try tables/ first, then current directory
        file = Path(f"tables/windows_{area}.csv")
        if not file.exists():
            file = Path(f"windows_{area}.csv")
        
        if not file.exists():
            raise FileNotFoundError(f"Missing windows file: windows_{area}.csv")
        
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        windows[area.upper()] = df
        
        print(f"  {area.upper()}: {len(df)} hours from {file}")
    
    return windows


def load_jobs() -> pd.DataFrame:
    """Load all job data from *_by_size.csv and *_by_matrix.csv."""
    print("[3/5] Loading job data...")
    
    # Define files: by_size for most, by_matrix for SPMV
    job_files = [
        ("GEMM_by_size.csv", "matrix_size"),
        ("STREAM_by_size.csv", "N"),
        ("REDUCTION_by_size.csv", "N"),
        ("SPMV_by_matrix.csv", "pattern")  # Special case: uses pattern instead of size
    ]
    
    all_jobs = []
    
    for file, size_col in job_files:
        path = Path(file)
        if not path.exists():
            print(f"  ⚠️ Skipping {file} (not found)")
            continue
        
        df = pd.read_csv(path)
        
        # Check required columns (median values for by_size)
        required = ['setup_id', 'kWh_e2e_median']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  ⚠️ Skipping {file} (missing columns: {missing})")
            continue
        
        # Extract workload type from filename
        workload = path.stem.split('_')[0]  # GEMM, STREAM, SPMV, REDUCTION
        df['workload'] = workload
        df['source_file'] = path.name
        
        # Add size/pattern identifier
        if size_col in df.columns:
            df['size_id'] = df[size_col].astype(str)
        else:
            df['size_id'] = 'unknown'
        
        all_jobs.append(df)
        print(f"  Loaded {len(df)} configurations from {path.name}")
    
    if not all_jobs:
        raise ValueError("No job data loaded!")
    
    jobs = pd.concat(all_jobs, ignore_index=True)
    
    # Create unique job_id (workload + setup + size/pattern)
    jobs['job_id'] = jobs.apply(
        lambda row: f"{row['workload']}_{row['setup_id']}_{row['size_id']}", 
        axis=1
    )
    
    # Rename kWh column (median per job)
    jobs = jobs.rename(columns={'kWh_e2e_median': 'kwh_job'})
    
    print(f"  Total: {len(jobs)} job configurations")
    return jobs


def assign_random_timestamps(jobs: pd.DataFrame, hourly_data: pd.DataFrame) -> pd.DataFrame:
    """
    Assign random September 2024 timestamps to jobs (as-run simulation).
    
    In real scenario, jobs would have actual execution timestamps.
    For simulation, we randomly distribute them across the month.
    """
    print("[4/5] Assigning random execution timestamps...")
    
    # Get all available timestamps
    all_timestamps = hourly_data['timestamp'].unique()
    
    # Randomly assign timestamps to jobs
    np.random.seed(42)  # Reproducibility
    jobs['timestamp_asrun'] = np.random.choice(all_timestamps, size=len(jobs))
    
    # Assign areas randomly (weighted by typical distribution)
    area_weights = {'DE': 0.5, 'FR': 0.3, 'PL': 0.2}
    jobs['area'] = np.random.choice(
        list(area_weights.keys()),
        size=len(jobs),
        p=list(area_weights.values())
    )
    
    print(f"  Assigned timestamps to {len(jobs)} jobs")
    print(f"  Area distribution: {jobs['area'].value_counts().to_dict()}")
    
    return jobs


def calculate_asrun_values(jobs: pd.DataFrame, hourly_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate as-run costs and emissions for each job."""
    print("[5/5] Calculating as-run and shift scenarios...")
    
    results = []
    
    for area in ['DE', 'FR', 'PL']:
        area_jobs = jobs[jobs['area'] == area].copy()
        
        if len(area_jobs) == 0:
            continue
        
        print(f"\n  {area}: {len(area_jobs)} jobs")
        
        # Get area-specific columns
        price_col = f'price_eur_kwh_{area.lower()}'
        ci_col = f'{area.lower()}_ci_lifecycle_g_per_kwh'
        
        # Merge with hourly data to get as-run values
        area_jobs = area_jobs.merge(
            hourly_data[['timestamp', price_col, ci_col]],
            left_on='timestamp_asrun',
            right_on='timestamp',
            how='left',
            suffixes=('', '_asrun')
        )
        
        # Calculate as-run values
        area_jobs['eur_asrun'] = area_jobs['kwh_job'] * area_jobs[price_col]
        area_jobs['co2_asrun_kg'] = area_jobs['kwh_job'] * area_jobs[ci_col] / 1000
        
        # Find best hours for each scenario
        area_hourly = hourly_data[['timestamp', price_col, ci_col]].copy()
        
        # Cheapest hour
        cheapest_hour = area_hourly.loc[area_hourly[price_col].idxmin()]
        area_jobs['eur_cheapest'] = area_jobs['kwh_job'] * cheapest_hour[price_col]
        area_jobs['co2_cheapest_kg'] = area_jobs['kwh_job'] * cheapest_hour[ci_col] / 1000
        
        # Greenest hour
        greenest_hour = area_hourly.loc[area_hourly[ci_col].idxmin()]
        area_jobs['eur_greenest'] = area_jobs['kwh_job'] * greenest_hour[price_col]
        area_jobs['co2_greenest_kg'] = area_jobs['kwh_job'] * greenest_hour[ci_col] / 1000
        
        # Overlap: find hour with best combined score (normalized)
        area_hourly['price_norm'] = (area_hourly[price_col] - area_hourly[price_col].min()) / (area_hourly[price_col].max() - area_hourly[price_col].min())
        area_hourly['ci_norm'] = (area_hourly[ci_col] - area_hourly[ci_col].min()) / (area_hourly[ci_col].max() - area_hourly[ci_col].min())
        area_hourly['combined_score'] = area_hourly['price_norm'] + area_hourly['ci_norm']
        
        overlap_hour = area_hourly.loc[area_hourly['combined_score'].idxmin()]
        area_jobs['eur_overlap'] = area_jobs['kwh_job'] * overlap_hour[price_col]
        area_jobs['co2_overlap_kg'] = area_jobs['kwh_job'] * overlap_hour[ci_col] / 1000
        
        # Calculate savings (absolute and relative)
        for scenario in ['cheapest', 'greenest', 'overlap']:
            # Absolute savings (always calculated)
            area_jobs[f'delta_eur_{scenario}'] = area_jobs['eur_asrun'] - area_jobs[f'eur_{scenario}']
            area_jobs[f'delta_co2_kg_{scenario}'] = area_jobs['co2_asrun_kg'] - area_jobs[f'co2_{scenario}_kg']
            
            # Relative savings (only for non-negative prices)
            # When prices are negative, relative savings don't make sense mathematically
            eps = 1e-10
            
            # Mask: only calculate relative for positive as-run costs
            positive_mask = area_jobs['eur_asrun'] > eps
            
            # Initialize with NaN
            area_jobs[f'rel_eur_{scenario}'] = np.nan
            area_jobs[f'rel_co2_{scenario}'] = np.nan
            
            # Calculate only for positive prices
            area_jobs.loc[positive_mask, f'rel_eur_{scenario}'] = (
                area_jobs.loc[positive_mask, f'delta_eur_{scenario}'] / 
                (area_jobs.loc[positive_mask, 'eur_asrun'] + eps)
            )
            
            # CO2 should always be positive (or zero), so calculate for all
            area_jobs[f'rel_co2_{scenario}'] = (
                area_jobs[f'delta_co2_kg_{scenario}'] / 
                (area_jobs['co2_asrun_kg'] + eps)
            )
            
            # Clip extreme relative values for sensible analysis
            # Relative savings beyond ±500% are not meaningful and likely due to tiny baseline values
            area_jobs[f'rel_eur_{scenario}'] = area_jobs[f'rel_eur_{scenario}'].clip(-1.0, 5.0)  # -100% to +500%
            area_jobs[f'rel_co2_{scenario}'] = area_jobs[f'rel_co2_{scenario}'].clip(-1.0, 5.0)
            
            # Worth waiting? (>5% EUR or >20% CO2 savings, only for valid rel values)
            area_jobs[f'worth_waiting_{scenario}'] = (
                (area_jobs[f'rel_eur_{scenario}'].fillna(0) >= 0.05) | 
                (area_jobs[f'rel_co2_{scenario}'] >= 0.20)
            )
        
        # Print summary for this area
        print(f"    As-run: EUR={area_jobs['eur_asrun'].sum():.2f}, CO2={area_jobs['co2_asrun_kg'].sum():.2f} kg")
        n_negative = (area_jobs['eur_asrun'] < 0).sum()
        if n_negative > 0:
            print(f"    ⚠️  {n_negative} jobs with negative prices (rel_eur set to NaN)")
        
        for scenario in ['cheapest', 'greenest', 'overlap']:
            eur_saved = area_jobs[f'delta_eur_{scenario}'].sum()
            co2_saved = area_jobs[f'delta_co2_kg_{scenario}'].sum()
            worth = area_jobs[f'worth_waiting_{scenario}'].sum()
            print(f"    {scenario.capitalize()}: Δ€={eur_saved:.2f}, ΔCO2={co2_saved:.2f} kg, worth_waiting={worth}/{len(area_jobs)}")
        
        results.append(area_jobs)
    
    return pd.concat(results, ignore_index=True)


def save_results(results: pd.DataFrame):
    """Save results to CSV."""
    # Save in current directory (no tables/ subfolder)
    outfile = 'scheduling_savings_by_run.csv'
    
    # Select output columns
    output_cols = [
        'job_id', 'workload', 'area', 'timestamp_asrun', 'kwh_job',
        'eur_asrun', 'co2_asrun_kg',
        'eur_cheapest', 'co2_cheapest_kg', 'delta_eur_cheapest', 'delta_co2_kg_cheapest',
        'rel_eur_cheapest', 'rel_co2_cheapest', 'worth_waiting_cheapest',
        'eur_greenest', 'co2_greenest_kg', 'delta_eur_greenest', 'delta_co2_kg_greenest',
        'rel_eur_greenest', 'rel_co2_greenest', 'worth_waiting_greenest',
        'eur_overlap', 'co2_overlap_kg', 'delta_eur_overlap', 'delta_co2_kg_overlap',
        'rel_eur_overlap', 'rel_co2_overlap', 'worth_waiting_overlap'
    ]
    
    results[output_cols].to_csv(outfile, index=False)
    print(f"\n✅ Saved: {outfile} ({len(results)} jobs)")


def create_plots(results: pd.DataFrame):
    """Create visualization plots."""
    # Save in current directory (no figs/ subfolder for now)
    
    print("\nCreating plots...")
    
    # 1. CDF of relative EUR savings (skip NaN values from negative prices)
    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario in ['cheapest', 'greenest', 'overlap']:
        data = results[f'rel_eur_{scenario}'].dropna().sort_values()  # Drop NaN!
        if len(data) > 0:
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data * 100, cdf, label=scenario.capitalize(), linewidth=2)
    
    ax.set_xlabel('Relative Cost Savings (%)', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('CDF: Cost Savings from Job Shifting (Positive Prices Only)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('savings_cdf_eur.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: savings_cdf_eur.png")
    
    # 2. CDF of relative CO2 savings
    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario in ['cheapest', 'greenest', 'overlap']:
        data = results[f'rel_co2_{scenario}'].sort_values()
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data * 100, cdf, label=scenario.capitalize(), linewidth=2)
    
    ax.set_xlabel('Relative CO₂ Savings (%)', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('CDF: CO₂ Savings from Job Shifting', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('savings_cdf_co2.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: savings_cdf_co2.png")
    
    # 3. Summary bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    scenarios = ['cheapest', 'greenest', 'overlap']
    eur_savings = [results[f'delta_eur_{s}'].sum() for s in scenarios]
    co2_savings = [results[f'delta_co2_kg_{s}'].sum() for s in scenarios]
    
    ax1.bar(scenarios, eur_savings, color=['steelblue', 'seagreen', 'coral'])
    ax1.set_ylabel('Total Cost Savings (EUR)', fontsize=11)
    ax1.set_title('Total Cost Savings by Scenario', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    ax2.bar(scenarios, co2_savings, color=['steelblue', 'seagreen', 'coral'])
    ax2.set_ylabel('Total CO₂ Savings (kg)', fontsize=11)
    ax2.set_title('Total CO₂ Savings by Scenario', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('savings_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: savings_summary.png")


def main():
    print("=" * 70)
    print("STEP 3: Scheduling Simulation")
    print("=" * 70)
    
    # Load data
    hourly_data = load_price_and_ci_data()
    windows = load_windows()
    jobs = load_jobs()
    
    # Assign random timestamps (simulate as-run execution)
    jobs = assign_random_timestamps(jobs, hourly_data)
    
    # Calculate savings
    results = calculate_asrun_values(jobs, hourly_data)
    
    # Save and visualize
    save_results(results)
    create_plots(results)
    
    print("\n" + "=" * 70)
    print("✅ DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()