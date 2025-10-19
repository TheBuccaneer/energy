#!/usr/bin/env python3
"""
Thread Efficiency Analysis (Clean Version)
Focuses on key insights with cleaner visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 70)
print("THREAD EFFICIENCY ANALYSIS (CLEAN)")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================

WORKLOAD_FILES = {
    'GEMM': 'GEMM_by_size.csv',
    'SPMV': 'SPMV_by_matrix.csv'
}

PRICING_AREA = 'de'
OUTPUT_DIR = Path('thread_efficiency_clean')
OUTPUT_DIR.mkdir(exist_ok=True)

# Filter settings for cleaner plots
MIN_SIZE = 512  # Only plot sizes >= 512 (removes small overhead-dominated sizes)

# CPU-specific thread filters (instead of power-of-2)
INTEL_THREADS = [1, 2, 4, 8, 10, 16, 20]  # Intel Xeon common config
AMD_THREADS = [1, 2, 4, 8, 16, 32, 64]     # AMD Ryzen/EPYC common config

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_thread_count(setup_id):
    """Extract thread count from setup_id."""
    import re
    match = re.search(r'(\d+)t', setup_id)
    return int(match.group(1)) if match else None

def is_power_of_two(n):
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0

def filter_threads_by_cpu(df, cpu_type):
    """Filter threads based on CPU type."""
    if cpu_type == 'Intel':
        return df[df['threads'].isin(INTEL_THREADS)]
    elif cpu_type == 'AMD':
        return df[df['threads'].isin(AMD_THREADS)]
    else:
        return df  # No filter for unknown CPU

def get_size_column(df):
    """Auto-detect size column."""
    for col in ['matrix_size', 'effective_size', 'N']:
        if col in df.columns and df[col].notna().any():
            return col
    return None

# ============================================================================
# LOAD AND FILTER DATA
# ============================================================================

print("\n[1/3] Loading and filtering data...")

all_workloads = {}

for wl_name, file_name in WORKLOAD_FILES.items():
    if not Path(file_name).exists():
        print(f"  ⚠️  Skipping {wl_name}: not found")
        continue
    
    df = pd.read_csv(file_name)
    
    # Identify CPU type (Intel or AMD)
    if 'cpu_type' in df.columns:
        cpu_types = df['cpu_type'].unique()
    else:
        # Infer from setup_id
        df['cpu_type'] = df['setup_id'].apply(
            lambda x: 'Intel' if 'Intel' in x else ('AMD' if 'AMD' in x else 'Unknown')
        )
        cpu_types = df['cpu_type'].unique()
    
    print(f"\n  {wl_name}:")
    print(f"    CPU types found: {list(cpu_types)}")
    
    # Process each CPU type separately
    for cpu_type in cpu_types:
        if cpu_type == 'Unknown':
            continue
        
        cpu_df = df[df['cpu_type'] == cpu_type].copy()
        
        # Filter CPU-only
        cpu_only = cpu_df[cpu_df['setup_id'].str.contains('CPU-only', na=False)].copy()
        if len(cpu_only) == 0:
            continue
        
        # Extract threads
        cpu_only['threads'] = cpu_only['setup_id'].apply(extract_thread_count)
        cpu_only = cpu_only[cpu_only['threads'].notna()]
        
        # Get size column
        size_col = get_size_column(cpu_only)
        if size_col is None:
            continue
        
        cpu_only['size'] = cpu_only[size_col]
        
        # Filter: Only sizes >= MIN_SIZE
        cpu_only = cpu_only[cpu_only['size'] >= MIN_SIZE]
        
        # Filter: CPU-specific thread counts
        cpu_only = filter_threads_by_cpu(cpu_only, cpu_type)
        
        # Filter: Only sizes with multiple thread counts
        size_thread_counts = cpu_only.groupby('size')['threads'].nunique()
        valid_sizes = size_thread_counts[size_thread_counts >= 3].index  # At least 3 thread counts
        cpu_only = cpu_only[cpu_only['size'].isin(valid_sizes)]
        
        if len(cpu_only) == 0:
            print(f"    ⚠️  {cpu_type}: No data after filtering")
            continue
        
        # Store with CPU type in key
        key = f"{wl_name}_{cpu_type}"
        all_workloads[key] = cpu_only
        
        print(f"    ✅ {cpu_type}: {len(cpu_only)} configs, "
              f"Threads: {sorted(cpu_only['threads'].unique())}, "
              f"Sizes: {sorted(cpu_only['size'].unique())[:3]}...")

if not all_workloads:
    print("\n❌ No data available after filtering!")
    exit(1)

# ============================================================================
# CLEANER VISUALIZATIONS
# ============================================================================

print("\n[2/3] Creating clean visualizations...")

cost_col = f'{PRICING_AREA}_eur_job_median'

for full_key, df in all_workloads.items():
    # Parse key: "GEMM_Intel" or "GEMM_AMD"
    parts = full_key.rsplit('_', 1)
    wl_name = parts[0]
    cpu_type = parts[1] if len(parts) > 1 else 'Unknown'
    
    print(f"\n{wl_name} ({cpu_type}):")
    
    wl_dir = OUTPUT_DIR / f"{wl_name}_{cpu_type}"
    wl_dir.mkdir(exist_ok=True)
    
    # Select top 4 largest sizes for clarity
    top_sizes = sorted(df['size'].unique())[-4:]
    df_plot = df[df['size'].isin(top_sizes)].copy()
    
    # Compute efficiency metric
    df_plot['jobs_per_wh'] = 1000.0 / (df_plot['kWh_e2e_median'] * 1000)
    
    # ========================================================================
    # PLOT 1: Energy Efficiency (Jobs/Wh) - BAR CHART
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    thread_counts = sorted(df_plot['threads'].unique())
    x = np.arange(len(thread_counts))
    width = 0.2
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, size in enumerate(top_sizes):
        size_df = df_plot[df_plot['size'] == size].sort_values('threads')
        
        if len(size_df) > 0:
            # Align data with all thread counts (fill missing with NaN)
            data = []
            for tc in thread_counts:
                val = size_df[size_df['threads'] == tc]['jobs_per_wh']
                data.append(val.values[0] if len(val) > 0 else np.nan)
            
            ax.bar(x + idx * width, data, width, 
                   label=f'N={int(size)}', color=colors[idx])
    
    ax.set_xlabel('Thread Count', fontsize=11, fontweight='bold')
    ax.set_ylabel('Jobs per Wh (higher = better)', fontsize=11, fontweight='bold')
    ax.set_title(f'{wl_name} ({cpu_type}) - Energy Efficiency by Thread Count', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(thread_counts)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(wl_dir / 'efficiency_bar_chart.png', dpi=250, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 2: Normalized Energy (to 1 thread) - LINE CHART
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, size in enumerate(top_sizes):
        size_df = df_plot[df_plot['size'] == size].sort_values('threads')
        
        if len(size_df) > 0:
            # Normalize to 1-thread (or minimum thread)
            min_threads = size_df['threads'].min()
            baseline = size_df[size_df['threads'] == min_threads]['kWh_e2e_median'].values
            
            if len(baseline) > 0:
                normalized = size_df['kWh_e2e_median'] / baseline[0]
                ax.plot(size_df['threads'], normalized, 
                        marker='o', linewidth=2.5, markersize=9,
                        label=f'N={int(size)}', color=colors[idx])
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, 
               label='Baseline (1t)', alpha=0.7)
    ax.set_xlabel('Thread Count', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Energy (to 1 thread)', fontsize=11, fontweight='bold')
    ax.set_title(f'{wl_name} ({cpu_type}) - Energy Scaling (1.0 = single thread baseline)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, frameon=True, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(wl_dir / 'energy_scaling.png', dpi=250, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 3: Heatmap - Threads × Size
    # ========================================================================
    
    # Create pivot table
    pivot = df_plot.pivot_table(
        index='threads', 
        columns='size', 
        values='jobs_per_wh',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f'{int(s)}' for s in pivot.columns])
    ax.set_yticklabels([f'{int(t)}t' for t in pivot.index])
    
    ax.set_xlabel('Problem Size (N)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Thread Count', fontsize=11, fontweight='bold')
    ax.set_title(f'{wl_name} ({cpu_type}) - Energy Efficiency Heatmap (Jobs/Wh)', 
                 fontsize=13, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Jobs per Wh', fontsize=10)
    
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.0f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(wl_dir / 'efficiency_heatmap.png', dpi=250, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 4: Sweet-Spot Analysis (Single size, all metrics)
    # ========================================================================
    
    # Pick largest size
    largest_size = top_sizes[-1]
    size_df = df_plot[df_plot['size'] == largest_size].sort_values('threads')
    
    if len(size_df) >= 3:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        threads = size_df['threads']
        
        # Subplot 1: Energy
        ax1.plot(threads, size_df['kWh_e2e_median'] * 1000, 
                 marker='o', linewidth=2.5, markersize=10, color='#E63946')
        ax1.set_ylabel('Energy per Job (Wh)', fontsize=10, fontweight='bold')
        ax1.set_title(f'{wl_name} ({cpu_type}) - Metrics vs Thread Count (N={int(largest_size)})', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Subplot 2: Time
        if 'seconds_wall_median' in size_df.columns:
            ax2.plot(threads, size_df['seconds_wall_median'], 
                     marker='s', linewidth=2.5, markersize=10, color='#06A77D')
            ax2.set_ylabel('Time per Job (s)', fontsize=10, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Subplot 3: Cost
        if cost_col in size_df.columns:
            ax3.plot(threads, size_df[cost_col] * 1000,  # cents
                     marker='^', linewidth=2.5, markersize=10, color='#F18F01')
            ax3.set_ylabel('Cost per Job (EUR cents)', fontsize=10, fontweight='bold')
            ax3.set_xlabel('Thread Count', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(wl_dir / 'sweet_spot_analysis.png', dpi=250, bbox_inches='tight')
        plt.close()
    
    print(f"  ✅ Created 4 clean plots in {wl_dir}/")

# ============================================================================
# SUMMARY INSIGHTS
# ============================================================================

print("\n[3/3] Summary insights...")
print("=" * 70)

for full_key, df in all_workloads.items():
    # Parse key
    parts = full_key.rsplit('_', 1)
    wl_name = parts[0]
    cpu_type = parts[1] if len(parts) > 1 else 'Unknown'
    
    print(f"\n{wl_name} ({cpu_type}):")
    
    # Find optimal thread count (best efficiency)
    df['jobs_per_wh'] = 1000.0 / (df['kWh_e2e_median'] * 1000)
    
    # Group by size, find best thread count
    for size in sorted(df['size'].unique())[-2:]:  # Last 2 sizes
        size_df = df[df['size'] == size]
        
        best_idx = size_df['jobs_per_wh'].idxmax()
        best = size_df.loc[best_idx]
        
        worst_idx = size_df['jobs_per_wh'].idxmin()
        worst = size_df.loc[worst_idx]
        
        improvement = (best['jobs_per_wh'] - worst['jobs_per_wh']) / worst['jobs_per_wh'] * 100
        
        print(f"  Size N={int(size)}:")
        print(f"    Best: {int(best['threads'])}t → {best['jobs_per_wh']:.0f} jobs/Wh")
        print(f"    Worst: {int(worst['threads'])}t → {worst['jobs_per_wh']:.0f} jobs/Wh")
        print(f"    Improvement: {improvement:.1f}% (choosing optimal thread count)")

print("\n" + "=" * 70)
print(f"✅ DONE! Clean plots saved to {OUTPUT_DIR}/")
print("=" * 70)