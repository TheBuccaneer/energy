import pandas as pd
import matplotlib.pyplot as plt
import os

# Load both files
price_file = "./day_ahead_sept2024_final.csv"
ci_file = "./ci_DE_FR_PL_hourly.csv"

df_price = pd.read_csv(price_file)
df_ci = pd.read_csv(ci_file)

print(f"Loaded {len(df_price)} price rows, {len(df_ci)} CI rows")

# Convert timestamps to datetime
df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
df_ci['ts'] = pd.to_datetime(df_ci['ts'])

# Remove timezone from CI timestamps to match price timestamps
df_ci['ts'] = df_ci['ts'].dt.tz_localize(None)

print("\n=== After conversion ===")
print(f"Price range: {df_price['timestamp'].min()} to {df_price['timestamp'].max()}")
print(f"CI range: {df_ci['ts'].min()} to {df_ci['ts'].max()}")

# Merge on timestamp
df = pd.merge(df_price, df_ci, left_on='timestamp', right_on='ts', how='inner')
print(f"\n✅ Merged: {len(df)} rows")

if len(df) == 0:
    print("❌ ERROR: No matching timestamps after merge!")
    print("\nFirst 5 price timestamps:")
    print(df_price['timestamp'].head())
    print("\nFirst 5 CI timestamps (in Sept 2024):")
    sept_ci = df_ci[(df_ci['ts'] >= '2024-09-01') & (df_ci['ts'] < '2024-10-01')]
    print(sept_ci['ts'].head())
    exit(1)

# Column pairs per zone
pairs = {
    "DE": ("price_eur_kwh_de", "de_ci_lifecycle_g_per_kwh"),
    "FR": ("price_eur_kwh_fr", "fr_ci_lifecycle_g_per_kwh"),
    "PL": ("price_eur_kwh_pl", "pl_ci_lifecycle_g_per_kwh"),
}

os.makedirs("figs", exist_ok=True)

for zone, (px, ci) in pairs.items():
    if px in df.columns and ci in df.columns:
        sub = df[[px, ci]].dropna()
        
        if len(sub) == 0:
            print(f"⚠️ {zone}: No data after dropna")
            continue
        
        print(f"\n✅ {zone}: {len(sub)} points")
        print(f"   Price range: {sub[px].min():.4f} - {sub[px].max():.4f} EUR/kWh")
        print(f"   CI range: {sub[ci].min():.1f} - {sub[ci].max():.1f} g/kWh")
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(sub[px], sub[ci], s=12, alpha=0.5, edgecolors='none')
        plt.xlabel("Preis (EUR/kWh)", fontsize=11)
        plt.ylabel("CO₂-Intensität (g/kWh, Lifecycle)", fontsize=11)
        plt.title(f"Preis vs. CO₂-Intensität — {zone} (Sep 2024)", fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save
        outfile = f"figs/scatter_price_vs_ci_{zone.lower()}.png"
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {outfile}")
    else:
        missing = []
        if px not in df.columns:
            missing.append(px)
        if ci not in df.columns:
            missing.append(ci)
        print(f"❌ {zone}: Missing columns: {missing}")

print("\n✅ DONE! Check figs/ folder for scatter plots.")