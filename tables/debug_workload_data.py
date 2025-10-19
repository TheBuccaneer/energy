#!/usr/bin/env python3
"""
Debug: Inspect workload CSV files to find why plots are empty.
"""

import pandas as pd
from pathlib import Path

WORKLOAD_FILES = {
    'GEMM': 'GEMM_by_size.csv',
    'STREAM': 'STREAM_by_size.csv',
    'REDUCTION': 'REDUCTION_by_size.csv',
    'SPMV': 'SPMV_by_matrix.csv'
}

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

print("=" * 70)
print("WORKLOAD DATA INSPECTION")
print("=" * 70)

for wl_name, wl_file in WORKLOAD_FILES.items():
    print(f"\n{'='*70}")
    print(f"{wl_name}: {wl_file}")
    print('='*70)
    
    if not Path(wl_file).exists():
        print("❌ File not found!")
        continue
    
    df = pd.read_csv(wl_file)
    
    print(f"\n[1] Basic Info:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    
    print(f"\n[2] Columns:")
    print(f"  {list(df.columns)}")
    
    print(f"\n[3] Sample setup_id values (first 5 unique):")
    for sid in df['setup_id'].unique()[:5]:
        hw_class = classify_hardware_detailed(sid)
        print(f"  '{sid}' → {hw_class}")
    
    print(f"\n[4] Hardware classification counts:")
    df['hw_class'] = df['setup_id'].apply(classify_hardware_detailed)
    print(df['hw_class'].value_counts().to_string())
    
    print(f"\n[5] Size column detection:")
    for col in ['matrix_size', 'N', 'pattern']:
        if col in df.columns:
            print(f"  ✅ Found: {col}")
            print(f"     Unique values: {sorted(df[col].unique())}")
            break
    else:
        print("  ❌ No size column found!")
    
    print(f"\n[6] Energy column check:")
    if 'kWh_e2e_median' in df.columns:
        print(f"  ✅ kWh_e2e_median exists")
        print(f"     Non-null: {df['kWh_e2e_median'].notna().sum()}/{len(df)}")
        print(f"     Range: {df['kWh_e2e_median'].min():.6f} - {df['kWh_e2e_median'].max():.6f}")
    else:
        print(f"  ❌ kWh_e2e_median NOT found")
    
    print(f"\n[7] Sample rows (first 3):")
    cols_to_show = ['setup_id', 'hw_class']
    for col in ['matrix_size', 'N', 'pattern']:
        if col in df.columns:
            cols_to_show.append(col)
            break
    if 'kWh_e2e_median' in df.columns:
        cols_to_show.append('kWh_e2e_median')
    
    print(df[cols_to_show].head(3).to_string(index=False))
    
    print(f"\n[8] Data after filtering (CPU/3090/5050 only):")
    filtered = df[df['hw_class'].isin(['CPU', '3090', '5050'])]
    print(f"  Rows remaining: {len(filtered)}/{len(df)}")
    if len(filtered) > 0:
        print(f"  Hardware counts:")
        print(filtered['hw_class'].value_counts().to_string())

print("\n" + "=" * 70)
print("✅ DONE")
print("=" * 70)