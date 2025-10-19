#!/usr/bin/env python3
"""
GPU Breakdown Analysis: Compare 3090 vs 5050 vs CPU
Identify size thresholds where GPUs become competitive.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 70)
print("GPU BREAKDOWN: 3090 vs 5050 vs CPU")
print("=" * 70)

# Load GEMM data (most comprehensive)
gemm = pd.read_csv('GEMM_by_size.csv')
print(f"\nLoaded {len(gemm)} GEMM configurations")

# Classify hardware types
def classify_hardware_detailed(setup_id):
    """Classify into CPU, 3090, 5050."""
    if 'CPU-only' in setup_id:
        return 'CPU'
    elif '3090' in setup_id:
        return '3090'
    elif '5050' in setup_id:
        return '5050'
    else:
        return 'Other GPU'

gemm['hw_class'] = gemm['setup_id'].apply(classify_hardware_detailed)

# Distribution
print("\nHardware Distribution:")
print(gemm['hw_class'].value_counts())

# ============================================================================
# ANALYSIS 1: Energy per Size
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 1: ENERGY BY PROBLEM SIZE")
print("=" * 70)

# Group by size and hardware
size_comparison = []

for size in sorted(gemm['matrix_size'].unique()):
    size_df = gemm[gemm['matrix_size'] == size]
    
    for hw in ['CPU', '3090', '5050']:
        hw_df = size_df[size_df['hw_class'] == hw]
        if len(hw_df) == 0:
            continue
        
        # Get best (minimum energy) for this hw
        best = hw_df.loc[hw_df['kWh_e2e_median'].idxmin()]
        
        size_comparison.append({
            'size': size,
            'hw_class': hw,
            'kwh': best['kWh_e2e_median'],
            'setup_id': best['setup_id']
        })

comp_df = pd.DataFrame(size_comparison)

# Find winner per size
winners = []
for size in comp_df['size'].unique():
    size_df = comp_df[comp_df['size'] == size]
    winner_idx = size_df['kwh'].idxmin()
    winner = size_df.loc[winner_idx].copy()
    winners.append(winner)

winners_df = pd.DataFrame(winners)

print("\nWinner per Matrix Size:")
print(winners_df[['size', 'hw_class', 'kwh']].to_string(index=False))

# Count wins
print("\nWin Summary:")
for hw in ['CPU', '3090', '5050']:
    wins = (winners_df['hw_class'] == hw).sum()
    total = len(winners_df)
    print(f"  {hw:6} wins: {wins:2}/{total} ({wins/total*100:5.1f}%)")

# ============================================================================
# ANALYSIS 2: Energy vs Size Plot
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOTS...")
print("=" * 70)

Path('gpu_analysis').mkdir(exist_ok=True)

# Plot 1: Energy vs Size (all hardware)
fig, ax = plt.subplots(figsize=(12, 6))

colors = {'CPU': 'steelblue', '3090': 'coral', '5050': 'seagreen'}
markers = {'CPU': 'o', '3090': 's', '5050': '^'}

for hw in ['CPU', '3090', '5050']:
    hw_df = comp_df[comp_df['hw_class'] == hw].sort_values('size')
    if len(hw_df) > 0:
        ax.plot(hw_df['size'], hw_df['kwh'] * 1000, 
                marker=markers[hw], markersize=8, linewidth=2,
                label=hw, color=colors[hw])

ax.set_xlabel('Matrix Size (N×N)', fontsize=11)
ax.set_ylabel('Energy per Job (Wh)', fontsize=11)
ax.set_title('Energy Consumption vs Problem Size', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', which='both')
plt.tight_layout()
plt.savefig('gpu_analysis/energy_vs_size.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: gpu_analysis/energy_vs_size.png")

# Plot 2: Relative Energy (normalized to CPU)
fig, ax = plt.subplots(figsize=(12, 6))

for hw in ['3090', '5050']:
    relative_energy = []
    sizes = []
    
    for size in sorted(comp_df['size'].unique()):
        cpu_kwh = comp_df[(comp_df['size'] == size) & (comp_df['hw_class'] == 'CPU')]['kwh']
        gpu_kwh = comp_df[(comp_df['size'] == size) & (comp_df['hw_class'] == hw)]['kwh']
        
        if len(cpu_kwh) > 0 and len(gpu_kwh) > 0:
            ratio = gpu_kwh.values[0] / cpu_kwh.values[0]
            relative_energy.append(ratio)
            sizes.append(size)
    
    if len(sizes) > 0:
        ax.plot(sizes, relative_energy, 
                marker=markers[hw], markersize=8, linewidth=2,
                label=f'{hw} / CPU', color=colors[hw])

ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Break-even')
ax.set_xlabel('Matrix Size (N×N)', fontsize=11)
ax.set_ylabel('Energy Ratio (GPU/CPU)', fontsize=11)
ax.set_title('GPU Energy Efficiency vs CPU (Ratio < 1 = GPU Wins)', 
             fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('gpu_analysis/gpu_vs_cpu_ratio.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: gpu_analysis/gpu_vs_cpu_ratio.png")

# ============================================================================
# ANALYSIS 3: Break-even Size
# ============================================================================

print("\n" + "=" * 70)
print("BREAK-EVEN ANALYSIS")
print("=" * 70)

for hw in ['3090', '5050']:
    # Find first size where GPU < CPU
    breakeven_size = None
    
    for size in sorted(comp_df['size'].unique()):
        cpu_kwh = comp_df[(comp_df['size'] == size) & (comp_df['hw_class'] == 'CPU')]['kwh']
        gpu_kwh = comp_df[(comp_df['size'] == size) & (comp_df['hw_class'] == hw)]['kwh']
        
        if len(cpu_kwh) > 0 and len(gpu_kwh) > 0:
            if gpu_kwh.values[0] < cpu_kwh.values[0]:
                breakeven_size = size
                ratio = gpu_kwh.values[0] / cpu_kwh.values[0]
                savings = (1 - ratio) * 100
                print(f"\n{hw}:")
                print(f"  First win at size: {size}×{size}")
                print(f"  Energy ratio: {ratio:.3f} (saves {savings:.1f}%)")
                break
    
    if breakeven_size is None:
        print(f"\n{hw}:")
        print(f"  ❌ Never competitive (CPU wins at all sizes)")

# ============================================================================
# ANALYSIS 4: Power Draw Comparison
# ============================================================================

print("\n" + "=" * 70)
print("POWER DRAW ANALYSIS")
print("=" * 70)

# Estimate average power from energy and time
if 'seconds_wall_median' in gemm.columns:
    gemm['avg_power_w'] = (gemm['kWh_e2e_median'] * 1000) / (gemm['seconds_wall_median'] / 3600)
    
    power_summary = gemm.groupby('hw_class')['avg_power_w'].agg(['mean', 'median', 'min', 'max'])
    print("\nAverage Power Draw by Hardware:")
    print(power_summary.to_string())
    
    # Plot power distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for hw in ['CPU', '3090', '5050']:
        hw_power = gemm[gemm['hw_class'] == hw]['avg_power_w'].dropna()
        if len(hw_power) > 0:
            ax.hist(hw_power, bins=20, alpha=0.5, label=hw, color=colors[hw])
    
    ax.set_xlabel('Average Power Draw (W)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Power Draw Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig('gpu_analysis/power_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("\n✅ Saved: gpu_analysis/power_distribution.png")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

summary = gemm.groupby('hw_class').agg({
    'kWh_e2e_median': ['mean', 'min', 'max'],
    'setup_id': 'count'
}).round(6)

summary.columns = ['Avg_kWh', 'Min_kWh', 'Max_kWh', 'N_Configs']
print(summary)

# Save detailed comparison
comp_df.to_csv('gpu_analysis/size_comparison.csv', index=False)
print("\n✅ Saved: gpu_analysis/size_comparison.csv")

print("\n" + "=" * 70)
print("✅ DONE! Check gpu_analysis/ for all outputs")
print("=" * 70)