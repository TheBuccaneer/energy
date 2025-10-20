#!/usr/bin/env python3
"""
Pareto-Front Visualizer - COMPLETE WORKING VERSION
Handles correct units (mg) and German CSV format
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

ZONE_CONFIG = {
    'de': {'name': 'Germany', 'color': '#FF8C00'},
    'fr': {'name': 'France', 'color': '#1E90FF'},
    'pl': {'name': 'Poland', 'color': '#DC143C'}
}

# ============================================================================
# PARETO LOGIC
# ============================================================================

def compute_pareto_front_max(df, x_col, y_col):
    """Compute Pareto-front for MAXIMIZATION"""
    df_sorted = df.sort_values(x_col, ascending=False).reset_index(drop=True)
    
    pareto_indices = []
    current_max_y = -float('inf')
    
    for idx, row in df_sorted.iterrows():
        if row[y_col] > current_max_y:
            pareto_indices.append(idx)
            current_max_y = row[y_col]
    
    return df_sorted.iloc[pareto_indices]

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_pareto_front_savings(df, output_file='pareto_front_savings.png'):
    """Pareto-Front with CORRECT mg units"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (zone, config) in enumerate(ZONE_CONFIG.items()):
        ax = axes[idx]
        zone_data = df[df['zone'] == zone].copy()
        
        scenarios = ['cheapest', 'greenest', 'overlap']
        scenario_colors = {
            'cheapest': '#4285F4',
            'greenest': '#0F9D58',
            'overlap': '#F4B400'
        }
        scenario_markers = {
            'cheapest': 's',
            'greenest': '^',
            'overlap': 'D'
        }
        
        all_savings = []
        
        for scenario in scenarios:
            # Calculate savings
            co2_savings_kg = zone_data['co2_asrun_kg'] - zone_data[f'co2_{scenario}_kg']
            cost_savings_eur = zone_data['eur_asrun'] - zone_data[f'eur_{scenario}']
            
            # Clip to non-negative
            co2_savings_kg = co2_savings_kg.clip(lower=0)
            cost_savings_eur = cost_savings_eur.clip(lower=0)
            
            # Convert to display units (g and €ct) - NOT mg!
            co2_savings_g = co2_savings_kg * 1000  # kg → g
            cost_savings_eurcent = cost_savings_eur * 100
            
            savings_data = pd.DataFrame({
                'co2_savings_g': co2_savings_g,
                'cost_savings_eurcent': cost_savings_eurcent
            })
            
            # SANITY CHECK
            baseline_g = zone_data['co2_asrun_kg'] * 1000  # kg → g
            max_possible = baseline_g.max()
            max_actual = savings_data['co2_savings_g'].max()
            
            if max_actual > max_possible * 1.01:
                print(f"⚠️  {zone.upper()}-{scenario}: Savings ({max_actual:.3f} g) > Baseline ({max_possible:.3f} g)!")
            
            # Plot
            ax.scatter(
                savings_data['co2_savings_g'],
                savings_data['cost_savings_eurcent'],
                alpha=0.3,
                s=20,
                color=scenario_colors[scenario],
                marker=scenario_markers[scenario],
                label=scenario.capitalize()
            )
            
            all_savings.append(savings_data)
        
        # Compute Pareto-front
        combined = pd.concat(all_savings, ignore_index=True)
        pareto = compute_pareto_front_max(combined, 'co2_savings_g', 'cost_savings_eurcent')
        
        # Plot Pareto-front
        pareto_sorted = pareto.sort_values('co2_savings_g')
        ax.plot(
            pareto_sorted['co2_savings_g'],
            pareto_sorted['cost_savings_eurcent'],
            color=config['color'],
            linewidth=2.5,
            marker='o',
            markersize=6,
            label='Pareto-Front',
            zorder=10
        )
        
        # Fill dominated region
        if len(pareto_sorted) > 1:
            x_fill = [0] + list(pareto_sorted['co2_savings_g']) + [pareto_sorted['co2_savings_g'].iloc[-1], 0]
            y_fill = [0] + list(pareto_sorted['cost_savings_eurcent']) + [0, 0]
            ax.fill(x_fill, y_fill, alpha=0.1, color='red', label='Dominated', zorder=1)
        
        # Styling
        ax.set_xlabel('CO₂ Savings per Batch (g)', fontweight='bold')
        ax.set_ylabel('Cost Savings per Batch (€ct)', fontweight='bold')
        ax.set_title(f'{config["name"]} ({zone.upper()})', fontweight='bold', fontsize=12)
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        # Debug output
        print(f"  {zone.upper()} CO₂ range: {combined['co2_savings_g'].min():.3f} - {combined['co2_savings_g'].max():.3f} g")
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\n✓ Saved {output_file}")
    plt.close()

def plot_savings_distribution(df, output_file='savings_distribution.png'):
    """CDF of savings percentages"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    df_valid = df[
        (df['savings_co2_greenest_pct'].notna()) &
        (df['savings_eur_cheapest_pct'].notna()) &
        (df['savings_eur_cheapest_pct'] < 1000)
    ].copy()
    
    # CO2 Savings CDF
    ax1 = axes[0]
    for zone, config in ZONE_CONFIG.items():
        zone_data = df_valid[df_valid['zone'] == zone]
        savings = zone_data['savings_co2_greenest_pct'].sort_values()
        cdf = np.arange(1, len(savings)+1) / len(savings)
        ax1.plot(savings, cdf, label=config['name'], color=config['color'], linewidth=2)
    
    ax1.set_xlabel('CO₂ Savings (%)', fontweight='bold')
    ax1.set_ylabel('Cumulative Probability', fontweight='bold')
    ax1.set_title('CDF: CO₂ Savings (Greenest Strategy)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cost Savings CDF
    ax2 = axes[1]
    for zone, config in ZONE_CONFIG.items():
        zone_data = df_valid[df_valid['zone'] == zone]
        savings = zone_data['savings_eur_cheapest_pct'].sort_values()
        cdf = np.arange(1, len(savings)+1) / len(savings)
        ax2.plot(savings, cdf, label=config['name'], color=config['color'], linewidth=2)
    
    ax2.set_xlabel('Cost Savings (%)', fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontweight='bold')
    ax2.set_title('CDF: Cost Savings (Cheapest Strategy)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()

def plot_wait_time_analysis(df, output_file='wait_time_analysis.png'):
    """Wait time vs savings scatter plots"""
    df_valid = df[
        (df['savings_co2_greenest_pct'].notna()) &
        (df['savings_eur_cheapest_pct'].notna()) &
        (df['savings_eur_cheapest_pct'] < 1000)
    ].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for col_idx, (zone, config) in enumerate(ZONE_CONFIG.items()):
        zone_data = df_valid[df_valid['zone'] == zone]
        
        # CO2
        ax1 = axes[0, col_idx]
        ax1.scatter(zone_data['wait_h_greenest'], zone_data['savings_co2_greenest_pct'],
                   alpha=0.5, s=10, color=config['color'])
        ax1.set_xlabel('Wait Time (h)')
        ax1.set_ylabel('CO₂ Savings (%)')
        ax1.set_title(f'{config["name"]} - CO₂')
        ax1.grid(True, alpha=0.3)
        
        # Cost
        ax2 = axes[1, col_idx]
        ax2.scatter(zone_data['wait_h_cheapest'], zone_data['savings_eur_cheapest_pct'],
                   alpha=0.5, s=10, color=config['color'])
        ax2.set_xlabel('Wait Time (h)')
        ax2.set_ylabel('Cost Savings (%)')
        ax2.set_title(f'{config["name"]} - Cost')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-50, 500)
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()

def generate_summary_table(df):
    """Summary statistics table"""
    df_valid = df[
        (df['savings_co2_greenest_pct'].notna()) &
        (df['savings_eur_cheapest_pct'].notna()) &
        (df['savings_eur_cheapest_pct'] < 1000)
    ].copy()
    
    summary = []
    for zone in ZONE_CONFIG.keys():
        zone_data = df_valid[df_valid['zone'] == zone]
        summary.append({
            'Zone': zone.upper(),
            'Avg Wait (Cheapest) [h]': f"{zone_data['wait_h_cheapest'].mean():.1f}",
            'Avg Wait (Greenest) [h]': f"{zone_data['wait_h_greenest'].mean():.1f}",
            'Median Cost Savings [%]': f"{zone_data['savings_eur_cheapest_pct'].median():.1f}",
            'Median CO2 Savings [%]': f"{zone_data['savings_co2_greenest_pct'].median():.1f}",
            'Max Cost Savings [%]': f"{zone_data['savings_eur_cheapest_pct'].quantile(0.95):.1f}",
            'Max CO2 Savings [%]': f"{zone_data['savings_co2_greenest_pct'].max():.1f}"
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('scheduling_summary_stats.csv', index=False)
    print("\n✓ Summary statistics:")
    print(summary_df.to_string(index=False))
    print(f"✓ Saved to scheduling_summary_stats.csv")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("PARETO-FRONT VISUALIZER (FIXED)")
    print("="*60)
    
    print("\n[1/4] Loading data...")
    df = pd.read_csv('scheduling_savings_by_run.csv')
    print(f"✓ Loaded {len(df)} scenario rows")
    
    print("\n[2/4] Generating Pareto-Front plots...")
    plot_pareto_front_savings(df)
    
    print("\n[3/4] Generating Wait-Time analysis...")
    plot_wait_time_analysis(df)
    
    print("\n[4/4] Generating Savings distributions...")
    plot_savings_distribution(df)
    
    generate_summary_table(df)
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. pareto_front_savings.png")
    print("  2. wait_time_analysis.png")
    print("  3. savings_distribution.png")
    print("  4. scheduling_summary_stats.csv")

if __name__ == '__main__':
    main()