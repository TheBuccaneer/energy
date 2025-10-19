#!/usr/bin/env python3
"""
Table 4: Annual Cost Scenarios
Calculate yearly costs assuming 1000 jobs/day for CPU, 3090, 5050.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("TABLE 4: ANNUAL COST SCENARIOS")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Jobs per year
JOBS_PER_DAY = 1000
DAYS_PER_YEAR = 365
ANNUAL_JOBS = JOBS_PER_DAY * DAYS_PER_YEAR

# Input file (use GEMM as most representative workload)
DATA_FILE = 'GEMM_by_size.csv'

# Output directory
OUTPUT_DIR = Path('tables')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\nAssumptions:")
print(f"  Jobs per day: {JOBS_PER_DAY:,}")
print(f"  Days per year: {DAYS_PER_YEAR}")
print(f"  Total annual jobs: {ANNUAL_JOBS:,}")

# ============================================================================
# HARDWARE CLASSIFICATION
# ============================================================================

def classify_hardware(setup_id):
    """Classify into CPU, 3090, 5050."""
    if 'CPU-only' in setup_id or 'CPU' in setup_id:
        return 'CPU'
    elif '3090' in setup_id:
        return '3090'
    elif '5050' in setup_id:
        return '5050'
    else:
        return 'Other'

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n[1/4] Loading data from {DATA_FILE}...")

if not Path(DATA_FILE).exists():
    print(f"❌ ERROR: {DATA_FILE} not found!")
    exit(1)

df = pd.read_csv(DATA_FILE)
df['hw_class'] = df['setup_id'].apply(classify_hardware)
df = df[df['hw_class'].isin(['CPU', '3090', '5050'])]

print(f"  Loaded {len(df)} configurations")
print(f"  Hardware: CPU={len(df[df['hw_class']=='CPU'])}, "
      f"3090={len(df[df['hw_class']=='3090'])}, "
      f"5050={len(df[df['hw_class']=='5050'])}")

# ============================================================================
# CALCULATE MEDIAN COSTS PER HARDWARE
# ============================================================================

print("\n[2/4] Calculating median cost per job...")

results = []

for area in ['de', 'fr', 'pl']:
    cost_col = f'{area}_eur_job_median'
    
    if cost_col not in df.columns:
        print(f"  ⚠️  Skipping {area.upper()}: {cost_col} not found")
        continue
    
    print(f"\n  {area.upper()}:")
    
    for hw in ['CPU', '3090', '5050']:
        hw_df = df[df['hw_class'] == hw]
        
        if len(hw_df) == 0:
            print(f"    {hw:6}: No data")
            continue
        
        # Calculate median cost per job across all sizes
        median_cost = hw_df[cost_col].median()
        
        # Annual cost
        annual_cost = median_cost * ANNUAL_JOBS
        
        results.append({
            'Area': area.upper(),
            'Hardware': hw,
            'Cost_per_Job_EUR': median_cost,
            'Annual_Cost_EUR': annual_cost,
            'N_Configs': len(hw_df)
        })
        
        print(f"    {hw:6}: €{median_cost:.6f}/job → €{annual_cost:,.2f}/year "
              f"({len(hw_df)} configs)")

results_df = pd.DataFrame(results)

# ============================================================================
# CALCULATE DIFFERENCES TO CPU
# ============================================================================

print("\n[3/4] Calculating GPU extra costs vs CPU...")

enhanced_results = []

for area in ['DE', 'FR', 'PL']:
    area_df = results_df[results_df['Area'] == area]
    
    # Get CPU baseline
    cpu_row = area_df[area_df['Hardware'] == 'CPU']
    
    if len(cpu_row) == 0:
        print(f"  ⚠️  No CPU baseline for {area}")
        continue
    
    cpu_annual = cpu_row['Annual_Cost_EUR'].values[0]
    cpu_per_job = cpu_row['Cost_per_Job_EUR'].values[0]
    
    print(f"\n  {area} Baseline (CPU): €{cpu_annual:,.2f}/year")
    
    for _, row in area_df.iterrows():
        hw = row['Hardware']
        annual = row['Annual_Cost_EUR']
        per_job = row['Cost_per_Job_EUR']
        
        # Calculate differences
        extra_annual = annual - cpu_annual
        extra_per_job = per_job - cpu_per_job
        ratio = annual / cpu_annual if cpu_annual > 0 else 0
        
        enhanced_results.append({
            'Area': area,
            'Hardware': hw,
            'Cost_per_Job_EUR': per_job,
            'Annual_Cost_EUR': annual,
            'Extra_vs_CPU_per_Job_EUR': extra_per_job,
            'Extra_vs_CPU_Annual_EUR': extra_annual,
            'Cost_Ratio_vs_CPU': ratio,
            'N_Configs': row['N_Configs']
        })
        
        if hw != 'CPU':
            print(f"    {hw:6}: €{extra_annual:+,.2f}/year ({ratio:.2f}× CPU cost)")

enhanced_df = pd.DataFrame(enhanced_results)

# ============================================================================
# CREATE FORMATTED TABLE
# ============================================================================

print("\n[4/4] Creating formatted table...")

# Pivot for better readability
table_data = []

for area in ['DE', 'FR', 'PL']:
    area_df = enhanced_df[enhanced_df['Area'] == area]
    
    row = {'Area': area}
    
    for hw in ['CPU', '3090', '5050']:
        hw_row = area_df[area_df['Hardware'] == hw]
        
        if len(hw_row) > 0:
            annual = hw_row['Annual_Cost_EUR'].values[0]
            extra = hw_row['Extra_vs_CPU_Annual_EUR'].values[0]
            
            row[f'{hw}_Annual_EUR'] = annual
            row[f'{hw}_Extra_EUR'] = extra
        else:
            row[f'{hw}_Annual_EUR'] = None
            row[f'{hw}_Extra_EUR'] = None
    
    table_data.append(row)

table_df = pd.DataFrame(table_data)

# Save detailed data
enhanced_df.to_csv(OUTPUT_DIR / 'table4_annual_costs_detailed.csv', index=False)
print(f"\n✅ Saved: {OUTPUT_DIR / 'table4_annual_costs_detailed.csv'}")

# Save formatted table
table_df.to_csv(OUTPUT_DIR / 'table4_annual_costs_formatted.csv', index=False)
print(f"✅ Saved: {OUTPUT_DIR / 'table4_annual_costs_formatted.csv'}")

# ============================================================================
# PRINT PAPER-READY TABLE
# ============================================================================

print("\n" + "=" * 70)
print("TABLE 4: ANNUAL COST SCENARIOS (1000 jobs/day)")
print("=" * 70)

print("\nDetailed Breakdown:")
print("-" * 70)
print(enhanced_df.to_string(index=False))

print("\n" + "=" * 70)
print("PAPER-READY TABLE (LaTeX-friendly)")
print("=" * 70)

# Create LaTeX-style table
print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Annual Cost Scenarios (1000 jobs/day, 365 days/year)}")
print("\\label{tab:annual_costs}")
print("\\begin{tabular}{lccc}")
print("\\toprule")
print("Area & CPU (EUR/year) & RTX 3090 (EUR/year) & RTX 5050 (EUR/year) \\\\")
print("\\midrule")

for _, row in table_df.iterrows():
    area = row['Area']
    cpu = row['CPU_Annual_EUR']
    gpu3090 = row['3090_Annual_EUR']
    gpu5050 = row['5050_Annual_EUR']
    
    extra3090 = row['3090_Extra_EUR']
    extra5050 = row['5050_Extra_EUR']
    
    print(f"{area} & {cpu:,.2f} & {gpu3090:,.2f} (+{extra3090:,.0f}) & {gpu5050:,.2f} (+{extra5050:,.0f}) \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print("\nAverage Extra Costs (across all areas):")
for hw in ['3090', '5050']:
    hw_df = enhanced_df[enhanced_df['Hardware'] == hw]
    avg_extra = hw_df['Extra_vs_CPU_Annual_EUR'].mean()
    avg_ratio = hw_df['Cost_Ratio_vs_CPU'].mean()
    
    print(f"\n{hw}:")
    print(f"  Average extra cost: €{avg_extra:,.2f}/year")
    print(f"  Average cost ratio: {avg_ratio:.2f}× CPU cost")
    print(f"  Per-job extra: €{avg_extra/ANNUAL_JOBS:.6f}")

print("\nWorst-case scenario (highest extra cost):")
worst = enhanced_df[enhanced_df['Hardware'] != 'CPU'].nlargest(1, 'Extra_vs_CPU_Annual_EUR')
print(worst[['Area', 'Hardware', 'Extra_vs_CPU_Annual_EUR', 'Cost_Ratio_vs_CPU']].to_string(index=False))

print("\nBest-case GPU scenario (lowest extra cost):")
best = enhanced_df[enhanced_df['Hardware'] != 'CPU'].nsmallest(1, 'Extra_vs_CPU_Annual_EUR')
print(best[['Area', 'Hardware', 'Extra_vs_CPU_Annual_EUR', 'Cost_Ratio_vs_CPU']].to_string(index=False))

print("\n" + "=" * 70)
print("✅ DONE!")
print("=" * 70)