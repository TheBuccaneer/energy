import pandas as pd
df = pd.read_csv('SPMV_by_matrix.csv')

print("=== CPU Data ===")
cpu = df[df['setup_id'].str.contains('CPU-only', na=False)]
print(f"CPU rows: {len(cpu)}")
print(f"CPU with effective_size: {cpu['effective_size'].notna().sum()}")
print(f"CPU unique effective_size: {cpu['effective_size'].dropna().unique()[:5]}")

print("\n=== GPU Data ===")
gpu = df[~df['setup_id'].str.contains('CPU-only', na=False)]
print(f"GPU rows: {len(gpu)}")
print(f"GPU with effective_size: {gpu['effective_size'].notna().sum()}")
print(f"GPU unique effective_size: {gpu['effective_size'].dropna().unique()[:5]}")