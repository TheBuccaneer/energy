#!/usr/bin/env python3
"""
GPU Breakdown Analysis: Compare 3090 vs 5050 vs CPU
Extended: Runs for ALL workloads (GEMM, STREAM, REDUCTION, SPMV)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

WORKLOAD_FILES = {
    'GEMM': 'GEMM_by_size.csv',
    'STREAM': 'STREAM_by_size.csv',
    'REDUCTION': 'REDUCTION_by_size.csv',
    'SPMV': 'SPMV_by_matrix.csv'
}

OUTPUT_BASE = Path('gpu_analysis_multi')
OUTPUT_BASE.mkdir(exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def classify_hardware_detailed(setup_id):
    """Classify into CPU, 3090, 5050."""
    if 'CPU-only' in setup_id or 'CPU' in setup_id:
        return 'CPU'
    elif '3090' in setup_id:
        return '3090'
    elif '5050' in setup_id:
        return '5050'
    else:
        return 'Other'

def get_size_column(df):
    """Auto-detect size column."""
    # Special case for SpMV: prefer effective_size over pattern
    if 'effective_size' in df.columns and 'pattern' in df.columns:
        if df['effective_size'].notna().any() and df['effective_size'].nunique() > 1:
            return 'effective_size'
    
    # Priority: matrix_size > effective_size > N (if > 0) > nnz_first > pattern
    if 'matrix_size' in df.columns:
        return 'matrix_size'
    
    # For SpMV: prefer effective_size (N²) if available
    if 'effective_size' in df.columns and df['effective_size'].notna().any():
        return 'effective_size'
    
    if 'N' in df.columns:
        # Check if N has meaningful values
        if df['N'].nunique() > 1 and df['N'].max() > 0:
            return 'N'
    
    # For SPMV: use nnz_first as numeric size
    if 'nnz_first' in df.columns:
        return 'nnz_first'
    
    # Last resort: pattern (categorical)
    if 'pattern' in df.columns:
        return 'pattern'
    
    return None

# ============================================================================
# ANALYSIS FUNCTION (per workload)
# ============================================================================

def analyze_workload(workload_name, data_file):
    """Run full breakdown analysis for one workload."""
    
    print("\n" + "=" * 70)
    print(f"ANALYZING: {workload_name}")
    print("=" * 70)
    
    if not Path(data_file).exists():
        print(f"⚠️  File not found: {data_file}")
        return None
    
    # Load data
    df = pd.read_csv(data_file)
    df['hw_class'] = df['setup_id'].apply(classify_hardware_detailed)
    df = df[df['hw_class'].isin(['CPU', '3090', '5050'])]  # Filter out "Other"
    
    print(f"\nLoaded {len(df)} configurations")
    print(f"Hardware: {df['hw_class'].value_counts().to_dict()}")
    
    # Create output directory
    out_dir = OUTPUT_BASE / workload_name
    out_dir.mkdir(exist_ok=True)
    
    # Detect size column
    size_col = get_size_column(df)
    if size_col is None:
        print("❌ No size column found!")
        return None
    
    df['size'] = df[size_col]
    
    # Check if we have meaningful size variation
    if df['size'].nunique() <= 1:
        print(f"⚠️  Only one unique size value ({df['size'].unique()}) - skipping plots")
        return None
    
    # Determine if size is numeric or categorical
    is_numeric = pd.api.types.is_numeric_dtype(df['size'])
    
    print(f"  Size column: {size_col} ({'numeric' if is_numeric else 'categorical'})")
    print(f"  Unique sizes: {df['size'].nunique()}")
    print(f"  Sample sizes: {sorted(df['size'].unique())[:5]}")
    if is_numeric:
        print(f"  Size range: {df['size'].min()} - {df['size'].max()}")
    else:
        print(f"  Categories: {list(df['size'].unique()[:5])}")
    
    # --- ANALYSIS 1: Size Comparison ---
    print(f"\n[1/4] Creating size comparison...")
    
    size_comparison = []
    for size in sorted(df['size'].unique()):
        size_df = df[df['size'] == size]
        for hw in ['CPU', '3090', '5050']:
            hw_df = size_df[size_df['hw_class'] == hw]
            if len(hw_df) == 0:
                continue
            best = hw_df.loc[hw_df['kWh_e2e_median'].idxmin()]
            size_comparison.append({
                'size': size,
                'hw_class': hw,
                'kwh': best['kWh_e2e_median'],
                'setup_id': best['setup_id']
            })
    
    comp_df = pd.DataFrame(size_comparison)
    
    # Find winners
    winners = []
    for size in comp_df['size'].unique():
        size_df = comp_df[comp_df['size'] == size]
        winner = size_df.loc[size_df['kwh'].idxmin()]
        winners.append(winner)
    
    winners_df = pd.DataFrame(winners)
    
    print("\nWinner per Size:")
    for hw in ['CPU', '3090', '5050']:
        wins = (winners_df['hw_class'] == hw).sum()
        total = len(winners_df)
        print(f"  {hw:6}: {wins:2}/{total} ({wins/total*100:5.1f}%)")
    
    # Save comparison
    comp_df.to_csv(out_dir / 'size_comparison.csv', index=False)
    
    # --- PLOT 1: Energy vs Size ---
    print(f"[2/4] Creating energy vs size plot...")
    
    colors = {'CPU': 'steelblue', '3090': 'coral', '5050': 'seagreen'}
    markers = {'CPU': 'o', '3090': 's', '5050': '^'}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if is_numeric:
        # Numeric: standard line plot with log scale
        for hw in ['CPU', '3090', '5050']:
            hw_df = comp_df[comp_df['hw_class'] == hw].sort_values('size')
            if len(hw_df) > 0:
                ax.plot(hw_df['size'], hw_df['kwh'] * 1000, 
                        marker=markers[hw], markersize=8, linewidth=2,
                        label=hw, color=colors[hw])
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        # Categorical: bar plot
        width = 0.25
        sizes = sorted(comp_df['size'].unique())
        x = np.arange(len(sizes))
        
        for i, hw in enumerate(['CPU', '3090', '5050']):
            hw_data = []
            for size in sizes:
                val = comp_df[(comp_df['size'] == size) & (comp_df['hw_class'] == hw)]['kwh']
                hw_data.append(val.values[0] * 1000 if len(val) > 0 else 0)
            
            ax.bar(x + i*width, hw_data, width, label=hw, color=colors[hw])
        
        ax.set_xticks(x + width)
        ax.set_xticklabels(sizes, rotation=45, ha='right')
        ax.set_yscale('log')
    
    ax.set_xlabel(f'{size_col.replace("_", " ").title()}', fontsize=11)
    ax.set_ylabel('Energy per Job (Wh)', fontsize=11)
    ax.set_title(f'{workload_name} - Energy vs Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', which='both' if is_numeric else 'major')
    plt.tight_layout()
    plt.savefig(out_dir / 'energy_vs_size.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # --- PLOT 2: GPU/CPU Ratio ---
    print(f"[3/4] Creating GPU/CPU ratio plot...")
    
    if is_numeric:
        # Numeric: line plot with log scale
        fig, ax = plt.subplots(figsize=(12, 6))
        for hw in ['3090', '5050']:
            ratios = []
            sizes = []
            for size in sorted(comp_df['size'].unique()):
                cpu_kwh = comp_df[(comp_df['size'] == size) & (comp_df['hw_class'] == 'CPU')]['kwh']
                gpu_kwh = comp_df[(comp_df['size'] == size) & (comp_df['hw_class'] == hw)]['kwh']
                if len(cpu_kwh) > 0 and len(gpu_kwh) > 0:
                    ratios.append(gpu_kwh.values[0] / cpu_kwh.values[0])
                    sizes.append(size)
            
            if len(sizes) > 0:
                ax.plot(sizes, ratios, marker=markers[hw], markersize=8, linewidth=2,
                        label=f'{hw}/CPU', color=colors[hw])
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Break-even')
        ax.set_xscale('log')
    else:
        # Categorical: bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.35
        sizes = sorted(comp_df['size'].unique())
        x = np.arange(len(sizes))
        
        for i, hw in enumerate(['3090', '5050']):
            ratios = []
            for size in sizes:
                cpu_kwh = comp_df[(comp_df['size'] == size) & (comp_df['hw_class'] == 'CPU')]['kwh']
                gpu_kwh = comp_df[(comp_df['size'] == size) & (comp_df['hw_class'] == hw)]['kwh']
                if len(cpu_kwh) > 0 and len(gpu_kwh) > 0:
                    ratios.append(gpu_kwh.values[0] / cpu_kwh.values[0])
                else:
                    ratios.append(0)
            
            ax.bar(x + i*width, ratios, width, label=f'{hw}/CPU', color=colors[hw])
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Break-even')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(sizes, rotation=45, ha='right')
    
    ax.set_xlabel(f'{size_col.replace("_", " ").title()}', fontsize=11)
    ax.set_ylabel('Energy Ratio (GPU/CPU)', fontsize=11)
    ax.set_title(f'{workload_name} - GPU vs CPU Efficiency', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_dir / 'gpu_vs_cpu_ratio.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # --- PLOT 3: Power Distribution ---
    print(f"[4/4] Creating power distribution plot...")
    
    if 'seconds_wall_median' in df.columns:
        df['avg_power_w'] = (df['kWh_e2e_median'] * 1000) / (df['seconds_wall_median'] / 3600)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for hw in ['CPU', '3090', '5050']:
            hw_power = df[df['hw_class'] == hw]['avg_power_w'].dropna()
            if len(hw_power) > 0:
                ax.hist(hw_power, bins=20, alpha=0.5, label=hw, color=colors[hw])
        
        ax.set_xlabel('Average Power (W)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{workload_name} - Power Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.tight_layout()
        plt.savefig(out_dir / 'power_distribution.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    # --- Break-even Analysis ---
    print(f"\n{workload_name} Break-even:")
    for hw in ['3090', '5050']:
        found = False
        for size in sorted(comp_df['size'].unique()):
            cpu_kwh = comp_df[(comp_df['size'] == size) & (comp_df['hw_class'] == 'CPU')]['kwh']
            gpu_kwh = comp_df[(comp_df['size'] == size) & (comp_df['hw_class'] == hw)]['kwh']
            if len(cpu_kwh) > 0 and len(gpu_kwh) > 0:
                if gpu_kwh.values[0] < cpu_kwh.values[0]:
                    ratio = gpu_kwh.values[0] / cpu_kwh.values[0]
                    print(f"  {hw}: First win at {size} (ratio: {ratio:.3f})")
                    found = True
                    break
        if not found:
            print(f"  {hw}: ❌ Never wins")
    
    print(f"\n✅ Saved outputs to: {out_dir}/")
    
    return {
        'workload': workload_name,
        'winners': winners_df,
        'comparison': comp_df
    }

# ============================================================================
# MAIN: Run for all workloads
# ============================================================================

print("=" * 70)
print("GPU BREAKDOWN ANALYSIS - MULTI-WORKLOAD")
print("=" * 70)

all_results = {}

for wl_name, wl_file in WORKLOAD_FILES.items():
    result = analyze_workload(wl_name, wl_file)
    if result is not None:
        all_results[wl_name] = result

# ============================================================================
# COMBINED SUMMARY
# ============================================================================

if len(all_results) > 1:
    print("\n" + "=" * 70)
    print("COMBINED SUMMARY (All Workloads)")
    print("=" * 70)
    
    # Aggregate win rates
    summary_rows = []
    for wl_name, result in all_results.items():
        winners = result['winners']
        for hw in ['CPU', '3090', '5050']:
            wins = (winners['hw_class'] == hw).sum()
            total = len(winners)
            summary_rows.append({
                'Workload': wl_name,
                'Hardware': hw,
                'Wins': wins,
                'Total': total,
                'Win_Rate_%': wins / total * 100 if total > 0 else 0
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_BASE / 'combined_summary.csv', index=False)
    print("\n✅ Saved: gpu_analysis_multi/combined_summary.csv")
    
    # Print summary table
    print("\nWin Rates by Workload:")
    pivot = summary_df.pivot(index='Workload', columns='Hardware', values='Win_Rate_%')
    print(pivot.to_string())
    
    # Overall statistics
    print("\n" + "-" * 70)
    print("Overall (All Workloads Combined):")
    overall = summary_df.groupby('Hardware')[['Wins', 'Total']].sum()
    overall['Win_Rate_%'] = overall['Wins'] / overall['Total'] * 100
    print(overall.to_string())

print("\n" + "=" * 70)
print("✅ DONE! Check gpu_analysis_multi/ for all outputs")
print("=" * 70)