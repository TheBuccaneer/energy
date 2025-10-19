#!/usr/bin/env python3
"""
Create time windows (cheapest 20%, greenest 20%, overlap) for each area.
Quick helper to generate windows_*.csv files for scheduling simulation.
"""

import pandas as pd
import numpy as np

# Load data
print("Loading data...")
df_price = pd.read_csv("day_ahead_sept2024_final.csv")
df_ci = pd.read_csv("ci_DE_FR_PL_hourly.csv")

# Harmonize timestamps
df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
df_ci['ts'] = pd.to_datetime(df_ci['ts']).dt.tz_localize(None)

# Merge
df = pd.merge(df_price, df_ci, left_on='timestamp', right_on='ts', how='inner')
print(f"Merged: {len(df)} hours\n")

# Process each area
for area in ['de', 'fr', 'pl']:
    print(f"Processing {area.upper()}...")
    
    price_col = f'price_eur_kwh_{area}'
    ci_col = f'{area}_ci_lifecycle_g_per_kwh'
    
    # Extract relevant columns
    area_df = df[['timestamp', price_col, ci_col]].copy()
    area_df.columns = ['timestamp', 'price_eur_kwh', 'ci_g_per_kwh']
    
    # Calculate percentiles (20% = top quintile)
    price_threshold = area_df['price_eur_kwh'].quantile(0.20)  # Cheapest 20%
    ci_threshold = area_df['ci_g_per_kwh'].quantile(0.20)      # Greenest 20%
    
    # Flag windows
    area_df['cheapest_flag'] = area_df['price_eur_kwh'] <= price_threshold
    area_df['greenest_flag'] = area_df['ci_g_per_kwh'] <= ci_threshold
    area_df['overlap_flag'] = area_df['cheapest_flag'] & area_df['greenest_flag']
    
    # Add hour column
    area_df['hour'] = area_df['timestamp'].dt.hour
    
    # Stats
    n_cheap = area_df['cheapest_flag'].sum()
    n_green = area_df['greenest_flag'].sum()
    n_overlap = area_df['overlap_flag'].sum()
    
    print(f"  Cheapest hours: {n_cheap} ({n_cheap/len(area_df)*100:.1f}%)")
    print(f"  Greenest hours: {n_green} ({n_green/len(area_df)*100:.1f}%)")
    print(f"  Overlap hours: {n_overlap} ({n_overlap/len(area_df)*100:.1f}%)")
    print(f"  Price threshold: {price_threshold:.4f} EUR/kWh")
    print(f"  CI threshold: {ci_threshold:.1f} g/kWh")
    
    # Save
    outfile = f"windows_{area}.csv"
    area_df.to_csv(outfile, index=False)
    print(f"  ✅ Saved: {outfile}\n")

print("✅ DONE! You can now run simulate_scheduling.py")