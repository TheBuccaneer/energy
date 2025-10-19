#!/usr/bin/env python3
"""
Detailed analysis of scheduling savings: breakdowns by area and workload.
Reads scheduling_savings_by_run.csv and creates detailed plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
print("Loading scheduling results...")
df = pd.read_csv('scheduling_savings_by_run.csv')
print(f"Loaded {len(df)} jobs\n")

# Create output directory
Path('analysis_plots').mkdir(exist_ok=True)

# ============================================================================
# PART A: BY AREA (DE, FR, PL)
# ============================================================================

print("=" * 70)
print("PART A: ANALYSIS BY AREA")
print("=" * 70)

areas = ['DE', 'FR', 'PL']
scenarios = ['cheapest', 'greenest', 'overlap']

# A1: Cost Savings CDF per Area
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, area in enumerate(areas):
    ax = axes[i]
    area_df = df[df['area'] == area]
    
    for scenario in scenarios:
        data = area_df[f'rel_eur_{scenario}'].dropna().sort_values()
        if len(data) > 0:
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data * 100, cdf, label=scenario.capitalize(), linewidth=2)
    
    ax.set_xlabel('Relative Cost Savings (%)', fontsize=10)
    ax.set_ylabel('Cumulative Probability', fontsize=10)
    ax.set_title(f'{area} - Cost Savings', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('analysis_plots/cost_cdf_by_area.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: analysis_plots/cost_cdf_by_area.png")

# A2: CO2 Savings CDF per Area
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, area in enumerate(areas):
    ax = axes[i]
    area_df = df[df['area'] == area]
    
    for scenario in scenarios:
        data = area_df[f'rel_co2_{scenario}'].dropna().sort_values()
        if len(data) > 0:
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data * 100, cdf, label=scenario.capitalize(), linewidth=2)
    
    ax.set_xlabel('Relative CO₂ Savings (%)', fontsize=10)
    ax.set_ylabel('Cumulative Probability', fontsize=10)
    ax.set_title(f'{area} - CO₂ Savings', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('analysis_plots/co2_cdf_by_area.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: analysis_plots/co2_cdf_by_area.png")

# A3: Summary statistics per Area
print("\nSummary Statistics by Area:")
for area in areas:
    area_df = df[df['area'] == area]
    print(f"\n{area} ({len(area_df)} jobs):")
    
    for scenario in scenarios:
        eur_savings = area_df[f'delta_eur_{scenario}'].sum()
        co2_savings = area_df[f'delta_co2_kg_{scenario}'].sum()
        rel_eur_median = area_df[f'rel_eur_{scenario}'].dropna().median() * 100
        rel_co2_median = area_df[f'rel_co2_{scenario}'].dropna().median() * 100
        
        print(f"  {scenario.capitalize():9} - Cost: {eur_savings:8.4f} EUR ({rel_eur_median:6.1f}% median), "
              f"CO₂: {co2_savings:7.4f} kg ({rel_co2_median:6.1f}% median)")

# ============================================================================
# PART B: BY WORKLOAD (GEMM, STREAM, REDUCTION, SPMV)
# ============================================================================

print("\n" + "=" * 70)
print("PART B: ANALYSIS BY WORKLOAD")
print("=" * 70)

workloads = df['workload'].unique()
print(f"\nWorkloads found: {sorted(workloads)}")

# B1: Cost Savings CDF per Workload
n_workloads = len(workloads)
cols = min(n_workloads, 4)
rows = (n_workloads + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
if n_workloads == 1:
    axes = [axes]
else:
    axes = axes.flatten() if n_workloads > 1 else [axes]

for i, workload in enumerate(sorted(workloads)):
    ax = axes[i]
    wl_df = df[df['workload'] == workload]
    
    for scenario in scenarios:
        data = wl_df[f'rel_eur_{scenario}'].dropna().sort_values()
        if len(data) > 0:
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data * 100, cdf, label=scenario.capitalize(), linewidth=2)
    
    ax.set_xlabel('Relative Cost Savings (%)', fontsize=10)
    ax.set_ylabel('Cumulative Probability', fontsize=10)
    ax.set_title(f'{workload} - Cost Savings ({len(wl_df)} jobs)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

# Hide empty subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('analysis_plots/cost_cdf_by_workload.png', dpi=200, bbox_inches='tight')
plt.close()
print("\n✅ Saved: analysis_plots/cost_cdf_by_workload.png")

# B2: CO2 Savings CDF per Workload
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
if n_workloads == 1:
    axes = [axes]
else:
    axes = axes.flatten() if n_workloads > 1 else [axes]

for i, workload in enumerate(sorted(workloads)):
    ax = axes[i]
    wl_df = df[df['workload'] == workload]
    
    for scenario in scenarios:
        data = wl_df[f'rel_co2_{scenario}'].dropna().sort_values()
        if len(data) > 0:
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data * 100, cdf, label=scenario.capitalize(), linewidth=2)
    
    ax.set_xlabel('Relative CO₂ Savings (%)', fontsize=10)
    ax.set_ylabel('Cumulative Probability', fontsize=10)
    ax.set_title(f'{workload} - CO₂ Savings ({len(wl_df)} jobs)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

# Hide empty subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('analysis_plots/co2_cdf_by_workload.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: analysis_plots/co2_cdf_by_workload.png")

# B3: Summary statistics per Workload
print("\nSummary Statistics by Workload:")
for workload in sorted(workloads):
    wl_df = df[df['workload'] == workload]
    print(f"\n{workload} ({len(wl_df)} jobs):")
    
    for scenario in scenarios:
        eur_savings = wl_df[f'delta_eur_{scenario}'].sum()
        co2_savings = wl_df[f'delta_co2_kg_{scenario}'].sum()
        rel_eur_median = wl_df[f'rel_eur_{scenario}'].dropna().median() * 100
        rel_co2_median = wl_df[f'rel_co2_{scenario}'].dropna().median() * 100
        
        print(f"  {scenario.capitalize():9} - Cost: {eur_savings:8.4f} EUR ({rel_eur_median:6.1f}% median), "
              f"CO₂: {co2_savings:7.4f} kg ({rel_co2_median:6.1f}% median)")

# ============================================================================
# PART C: COMBINED HEATMAP (Area × Workload)
# ============================================================================

print("\n" + "=" * 70)
print("PART C: AREA × WORKLOAD HEATMAPS")
print("=" * 70)

# Create pivot tables for heatmaps
for metric, label in [('delta_eur_cheapest', 'Cost Savings (EUR)'), 
                      ('delta_co2_kg_greenest', 'CO₂ Savings (kg)')]:
    
    pivot = df.pivot_table(
        values=metric,
        index='workload',
        columns='area',
        aggfunc='sum'
    )
    
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlGn')
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Labels
    ax.set_xlabel('Area', fontsize=11)
    ax.set_ylabel('Workload', fontsize=11)
    ax.set_title(f'Total {label} by Area × Workload', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, fontsize=10)
    
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.4f}',
                             ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    filename = f"analysis_plots/heatmap_{metric.split('_')[1]}_{metric.split('_')[2]}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {filename}")

print("\n" + "=" * 70)
print("✅ DONE! All detailed analyses saved in analysis_plots/")
print("=" * 70)