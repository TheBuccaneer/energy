#!/usr/bin/env python3
"""
Step 4: CPU/GPU Decision Matrix + Time-of-Day Optimization
Combines hardware selection with time-shifting for optimal cost/carbon.

Output:
- Decision matrix: Best hardware × time window per workload/size
- Heatmaps: Cost/CO2 by hour-of-day and hardware
- Summary tables: Break-even analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 70)
print("STEP 4: CPU/GPU DECISION MATRIX")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/5] Loading data...")

# Load workload data (by_size files with CPU and GPU)
workload_files = {
    'GEMM': 'GEMM_by_size.csv',
    'STREAM': 'STREAM_by_size.csv',
    'REDUCTION': 'REDUCTION_by_size.csv',
    'SPMV': 'SPMV_by_matrix.csv'
}

workloads = {}
for wl, file in workload_files.items():
    if Path(file).exists():
        df = pd.read_csv(file)
        # Key columns: setup_id, kWh_e2e_median, matrix_size/N/pattern
        workloads[wl] = df
        print(f"  {wl}: {len(df)} configurations")

# Load windows (cheapest/greenest hours per area)
windows = {}
for area in ['de', 'fr', 'pl']:
    file = Path(f"windows_{area}.csv")
    if file.exists():
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        windows[area.upper()] = df

print(f"  Loaded windows for {len(windows)} areas")

# ============================================================================
# CPU vs GPU CLASSIFICATION
# ============================================================================

print("\n[2/5] Classifying CPU vs GPU setups...")

def classify_hardware(setup_id):
    """Classify setup as CPU or GPU."""
    if 'CPU-only' in setup_id:
        return 'CPU'
    else:
        return 'GPU'

# Add hardware type to all workloads
for wl in workloads:
    workloads[wl]['hw_type'] = workloads[wl]['setup_id'].apply(classify_hardware)
    n_cpu = (workloads[wl]['hw_type'] == 'CPU').sum()
    n_gpu = (workloads[wl]['hw_type'] == 'GPU').sum()
    print(f"  {wl}: {n_cpu} CPU, {n_gpu} GPU configs")

# ============================================================================
# BEST HARDWARE PER SIZE/AREA/METRIC
# ============================================================================

print("\n[3/5] Finding optimal hardware per scenario...")

def find_best_hardware(wl_df, area, metric='energy'):
    """
    Find best hardware (lowest kWh) per size/pattern.
    
    Args:
        wl_df: Workload dataframe
        area: DE/FR/PL
        metric: 'energy', 'cost', or 'co2'
    
    Returns:
        DataFrame with best setup per size
    """
    # Determine size column
    size_col = None
    for col in ['matrix_size', 'N', 'pattern']:
        if col in wl_df.columns:
            size_col = col
            break
    
    if size_col is None:
        return None
    
    # Group by size and find minimum energy
    best = []
    for size in wl_df[size_col].unique():
        size_df = wl_df[wl_df[size_col] == size].copy()
        
        if len(size_df) == 0:
            continue
        
        # Find best (minimum kWh)
        best_idx = size_df['kWh_e2e_median'].idxmin()
        best_row = size_df.loc[best_idx]
        
        # Get price/CI for this area from windows
        if area in windows:
            win = windows[area]
            # Use cheapest hour for cost, greenest for CO2
            if metric == 'cost':
                best_hour = win.loc[win['cheapest_flag'] == True]
            elif metric == 'co2':
                best_hour = win.loc[win['greenest_flag'] == True]
            else:  # energy
                best_hour = win  # doesn't matter, energy is fixed
            
            if len(best_hour) > 0:
                price_col = f'price_eur_kwh'
                ci_col = f'ci_g_per_kwh'
                
                if price_col in best_hour.columns:
                    price = best_hour[price_col].median()
                    best_row['cost_eur'] = best_row['kWh_e2e_median'] * price
                
                if ci_col in best_hour.columns:
                    ci = best_hour[ci_col].median()
                    best_row['co2_kg'] = best_row['kWh_e2e_median'] * ci / 1000
        
        best_row['size'] = size
        best_row['area'] = area
        best.append(best_row)
    
    return pd.DataFrame(best) if best else None

# Create decision matrices for each workload
decision_matrices = {}

for wl_name, wl_df in workloads.items():
    print(f"\n  {wl_name}:")
    wl_decisions = []
    
    for area in ['DE', 'FR', 'PL']:
        # Find best for energy (hardware selection)
        best_energy = find_best_hardware(wl_df, area, 'energy')
        if best_energy is not None:
            wl_decisions.append(best_energy)
            
            cpu_count = (best_energy['hw_type'] == 'CPU').sum()
            gpu_count = (best_energy['hw_type'] == 'GPU').sum()
            print(f"    {area}: {len(best_energy)} sizes - CPU wins: {cpu_count}, GPU wins: {gpu_count}")
    
    if wl_decisions:
        decision_matrices[wl_name] = pd.concat(wl_decisions, ignore_index=True)

# ============================================================================
# TIME-OF-DAY OPTIMIZATION
# ============================================================================

print("\n[4/5] Creating time-of-day heatmaps...")

Path('decision_plots').mkdir(exist_ok=True)

# For each area, show cost/CO2 by hour
for area in ['DE', 'FR', 'PL']:
    if area not in windows:
        continue
    
    win = windows[area]
    
    # Create hour-based summary
    hourly = win.groupby('hour').agg({
        'price_eur_kwh': 'mean',
        'ci_g_per_kwh': 'mean'
    }).reset_index()
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    color1 = 'steelblue'
    ax1.set_xlabel('Hour of Day', fontsize=11)
    ax1.set_ylabel('Price (EUR/kWh)', color=color1, fontsize=11)
    ax1.plot(hourly['hour'], hourly['price_eur_kwh'], 
             color=color1, linewidth=2, marker='o', label='Price')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2 = ax1.twinx()
    color2 = 'seagreen'
    ax2.set_ylabel('CO₂ Intensity (g/kWh)', color=color2, fontsize=11)
    ax2.plot(hourly['hour'], hourly['ci_g_per_kwh'], 
             color=color2, linewidth=2, marker='s', label='CO₂')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Highlight best windows
    cheapest_hours = win[win['cheapest_flag'] == True]['hour'].values
    greenest_hours = win[win['greenest_flag'] == True]['hour'].values
    
    ax1.axvspan(cheapest_hours.min(), cheapest_hours.max(), 
                alpha=0.2, color='blue', label='Cheapest Window')
    ax1.axvspan(greenest_hours.min(), greenest_hours.max(), 
                alpha=0.2, color='green', label='Greenest Window')
    
    plt.title(f'{area} - Price & CO₂ by Hour of Day', fontsize=12, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'decision_plots/hourly_profile_{area.lower()}.png', 
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: decision_plots/hourly_profile_{area.lower()}.png")

# ============================================================================
# DECISION SUMMARY TABLES
# ============================================================================

print("\n[5/5] Creating decision summary tables...")

# Summary: CPU vs GPU wins by workload and area
summary_rows = []

for wl_name, dm in decision_matrices.items():
    for area in ['DE', 'FR', 'PL']:
        area_dm = dm[dm['area'] == area]
        
        if len(area_dm) == 0:
            continue
        
        cpu_wins = (area_dm['hw_type'] == 'CPU').sum()
        gpu_wins = (area_dm['hw_type'] == 'GPU').sum()
        total = len(area_dm)
        
        # Average metrics
        avg_kwh = area_dm['kWh_e2e_median'].mean()
        
        summary_rows.append({
            'Workload': wl_name,
            'Area': area,
            'Total_Configs': total,
            'CPU_Wins': cpu_wins,
            'GPU_Wins': gpu_wins,
            'CPU_Win_Rate_%': cpu_wins / total * 100 if total > 0 else 0,
            'Avg_kWh': avg_kwh
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('decision_plots/cpu_gpu_summary.csv', index=False)
print(f"\n✅ Saved: decision_plots/cpu_gpu_summary.csv")

# Print summary
print("\n" + "=" * 70)
print("CPU vs GPU SUMMARY")
print("=" * 70)
print(summary_df.to_string(index=False))

# Overall statistics
print("\n" + "=" * 70)
print("OVERALL STATISTICS")
print("=" * 70)
total_configs = summary_df['Total_Configs'].sum()
total_cpu_wins = summary_df['CPU_Wins'].sum()
total_gpu_wins = summary_df['GPU_Wins'].sum()

print(f"Total configurations analyzed: {total_configs}")
print(f"CPU wins: {total_cpu_wins} ({total_cpu_wins/total_configs*100:.1f}%)")
print(f"GPU wins: {total_gpu_wins} ({total_gpu_wins/total_configs*100:.1f}%)")

print("\nBy Workload:")
for wl in summary_df['Workload'].unique():
    wl_df = summary_df[summary_df['Workload'] == wl]
    cpu = wl_df['CPU_Wins'].sum()
    gpu = wl_df['GPU_Wins'].sum()
    total = cpu + gpu
    print(f"  {wl:10} - CPU: {cpu:3} ({cpu/total*100:5.1f}%), GPU: {gpu:3} ({gpu/total*100:5.1f}%)")

print("\nBy Area:")
for area in ['DE', 'FR', 'PL']:
    area_df = summary_df[summary_df['Area'] == area]
    cpu = area_df['CPU_Wins'].sum()
    gpu = area_df['GPU_Wins'].sum()
    total = cpu + gpu
    if total > 0:
        print(f"  {area} - CPU: {cpu:3} ({cpu/total*100:5.1f}%), GPU: {gpu:3} ({gpu/total*100:5.1f}%)")

print("\n" + "=" * 70)
print("✅ DONE! Check decision_plots/ for outputs")
print("=" * 70)