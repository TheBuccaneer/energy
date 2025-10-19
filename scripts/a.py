import pandas as pd

df = pd.read_csv('scheduling_savings_by_run.csv')

print("=== DATA OVERVIEW ===")
print(f"Total jobs: {len(df)}")
print(f"\nWorkload distribution:")
print(df['workload'].value_counts())
print(f"\nArea distribution:")
print(df['area'].value_counts())

print("\n=== EXTREME VALUES ===")
print("\nrel_eur_cheapest:")
print(f"  Min: {df['rel_eur_cheapest'].min():.2e}")
print(f"  Max: {df['rel_eur_cheapest'].max():.2e}")
print(f"  Median: {df['rel_eur_cheapest'].median():.4f}")

print("\nTop 5 extreme rel_eur values:")
extreme = df.nlargest(5, 'rel_eur_cheapest')[['job_id', 'area', 'kwh_job', 'eur_asrun', 'eur_cheapest', 'rel_eur_cheapest']]
print(extreme.to_string())

print("\n=== JOBS WITH TINY COSTS ===")
tiny = df[df['eur_asrun'] < 1e-8].sort_values('eur_asrun')
print(f"Jobs with eur_asrun < 1e-8: {len(tiny)}")
if len(tiny) > 0:
    print(tiny[['job_id', 'area', 'kwh_job', 'eur_asrun', 'rel_eur_cheapest']].head(10).to_string())