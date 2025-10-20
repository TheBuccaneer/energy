#!/usr/bin/env python3
"""
Sanity-Check für Scheduling-Daten
Prüft ob CO2-Savings physikalisch möglich sind
"""

import pandas as pd
import numpy as np

def validate_scheduling_data():
    """
    Prüft ob CO2-Savings < CO2-Baseline (physikalisch unmöglich sonst!)
    """
    print("="*60)
    print("SANITY-CHECK: Scheduling Savings Data")
    print("="*60)
    
    # Load data - output from builder is now in standard format
    df = pd.read_csv('scheduling_savings_by_run.csv')
    
    print(f"\n[1] Loaded {len(df)} rows")
    print(f"    Zones: {df['zone'].unique()}")
    print(f"    Workloads: {df['workload'].unique()}")
    
    # Check baseline values
    print("\n[2] Baseline CO2 emissions (in GRAMS, not mg!):")
    for zone in ['de', 'fr', 'pl']:
        zone_data = df[df['zone'] == zone]
        baseline_kg = zone_data['co2_asrun_kg']
        baseline_g = baseline_kg * 1000  # kg → g (NOT mg!)
        
        print(f"\n  {zone.upper()}:")
        print(f"    Median baseline: {baseline_g.median():.3f} g")
        print(f"    Range: {baseline_g.min():.3f} - {baseline_g.max():.3f} g")
        print(f"    Mean kWh/job: {zone_data['kwh_job'].mean():.6f} kWh")
        print(f"    MAX kWh/job: {zone_data['kwh_job'].max():.6f} kWh")
    
    # Check savings vs baseline
    print("\n[3] Checking if savings exceed baseline (INVALID!):")
    
    scenarios = ['cheapest', 'greenest', 'overlap']
    errors_found = False
    
    for zone in ['de', 'fr', 'pl']:
        zone_data = df[df['zone'] == zone].copy()
        baseline_kg = zone_data['co2_asrun_kg']
        
        print(f"\n  {zone.upper()}:")
        
        for scenario in scenarios:
            scenario_kg = zone_data[f'co2_{scenario}_kg']
            savings_kg = baseline_kg - scenario_kg
            savings_mg = savings_kg * 1_000_000
            baseline_mg = baseline_kg * 1_000_000
            
            # Check for violations
            invalid_mask = savings_kg > baseline_kg
            n_invalid = invalid_mask.sum()
            
            if n_invalid > 0:
                errors_found = True
                print(f"    ❌ {scenario}: {n_invalid} rows with savings > baseline!")
                print(f"       Max invalid savings: {savings_mg[invalid_mask].max():.3f} mg")
                print(f"       Corresponding baseline: {baseline_mg[invalid_mask].max():.3f} mg")
            else:
                print(f"    ✅ {scenario}: All savings <= baseline")
                print(f"       Median savings: {savings_mg.median():.3f} mg")
                print(f"       Max savings: {savings_mg.max():.3f} mg")
    
    # Expected ranges based on audit
    print("\n[4] Expected ranges for LARGEST jobs (~125 Wh):")
    print("    FR:  ~5 g     (ΔCI ~40 g/kWh × 0.125 kWh)")
    print("    DE:  ~51 g    (ΔCI ~410 g/kWh × 0.125 kWh)")
    print("    PL:  ~66 g    (ΔCI ~526 g/kWh × 0.125 kWh)")
    print("\n    For MEDIAN jobs (~36 Wh):")
    print("    FR:  ~1.4 g")
    print("    DE:  ~15 g")
    print("    PL:  ~19 g")
    
    if errors_found:
        print("\n" + "="*60)
        print("❌ VALIDATION FAILED!")
        print("="*60)
        print("\nPossible causes:")
        print("  1. kWh values in job CSVs are wrong (too high?)")
        print("  2. CI values in windows CSVs are wrong (factor 1000?)")
        print("  3. Calculation logic in scheduling_savings_builder.py")
        return False
    else:
        print("\n" + "="*60)
        print("✅ VALIDATION PASSED!")
        print("="*60)
        return True

def check_input_files():
    """
    Prüft die Input-Dateien auf Plausibilität
    """
    print("\n" + "="*60)
    print("CHECKING INPUT FILES")
    print("="*60)
    
    # Check job files
    print("\n[A] Job characteristics:")
    job_files = ['GEMM_by_run.csv', 'SPMV_by_run.csv', 'STREAM_by_run.csv', 'REDUCTION_by_run.csv']
    
    for filename in job_files:
        try:
            # Read and convert German format
            df = pd.read_csv(filename)
            
            # Convert kWh column if it exists
            if 'kWh_e2e_sum' in df.columns:
                # Handle both German (comma) and standard (dot) format
                if df['kWh_e2e_sum'].dtype == 'object':
                    df['kWh_e2e_sum'] = df['kWh_e2e_sum'].astype(str).str.replace(',', '.').astype(float)
                kwh = df['kWh_e2e_sum']
                print(f"\n  {filename}:")
                print(f"    Median: {kwh.median():.6f} kWh")
                print(f"    Range: {kwh.min():.6f} - {kwh.max():.6f} kWh")
                
                # Convert to expected CO2 for DE (medium CI ~300 g/kWh)
                expected_co2_mg = kwh.median() * 300 * 1000  # kWh * g/kWh * mg/g
                print(f"    Expected CO2 @ 300g/kWh: {expected_co2_mg:.3f} mg")
        except FileNotFoundError:
            print(f"  ⚠️  {filename} not found")
    
    # Check window files
    print("\n[B] Time windows (CI values):")
    for zone in ['de', 'fr', 'pl']:
        try:
            # Read and convert German format
            df = pd.read_csv(f'windows_{zone}.csv')
            
            # Convert CI column if it exists
            if 'ci_g_per_kwh' in df.columns:
                if df['ci_g_per_kwh'].dtype == 'object':
                    df['ci_g_per_kwh'] = df['ci_g_per_kwh'].astype(str).str.replace(',', '.').astype(float)
                ci = df['ci_g_per_kwh']
                print(f"\n  {zone.upper()}:")
                print(f"    Median CI: {ci.median():.1f} g/kWh")
                print(f"    Range: {ci.min():.1f} - {ci.max():.1f} g/kWh")
                print(f"    Delta (max-min): {ci.max() - ci.min():.1f} g/kWh")
        except FileNotFoundError:
            print(f"  ⚠️  windows_{zone}.csv not found")

if __name__ == '__main__':
    # Check input files first
    check_input_files()
    
    # Then validate scheduling data
    print("\n\n")
    validate_scheduling_data()