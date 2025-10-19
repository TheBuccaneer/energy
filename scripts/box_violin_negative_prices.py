import pandas as pd
import matplotlib.pyplot as plt
import os

# Load both files
price_file = "./day_ahead_sept2024_final.csv"
ci_file = "./ci_DE_FR_PL_hourly.csv"

df_price = pd.read_csv(price_file)
df_ci = pd.read_csv(ci_file)

# Convert timestamps and remove timezone
df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
df_ci['ts'] = pd.to_datetime(df_ci['ts']).dt.tz_localize(None)

# Merge
df = pd.merge(df_price, df_ci, left_on='timestamp', right_on='ts', how='inner')
print(f"Merged: {len(df)} rows")

# Zone definitions
zones = {
    "DE": ("price_eur_kwh_de", "de_ci_lifecycle_g_per_kwh"),
    "FR": ("price_eur_kwh_fr", "fr_ci_lifecycle_g_per_kwh"),
    "PL": ("price_eur_kwh_pl", "pl_ci_lifecycle_g_per_kwh"),
}

os.makedirs("figs", exist_ok=True)

for z, (px, ci) in zones.items():
    if px in df.columns and ci in df.columns:
        sub = df[[px, ci]].dropna()
        
        if len(sub) == 0:
            print(f"⚠️ {z}: No data")
            continue
        
        # Create negative price indicator
        sub["neg"] = sub[px] < 0
        
        n_neg = sub["neg"].sum()
        n_nonneg = (~sub["neg"]).sum()
        
        print(f"\n{z}:")
        print(f"  Negative prices: {n_neg} hours ({n_neg/len(sub)*100:.1f}%)")
        print(f"  Non-negative: {n_nonneg} hours ({n_nonneg/len(sub)*100:.1f}%)")
        
        if n_neg == 0:
            print(f"  ⚠️ No negative prices in {z}, skipping plots")
            continue
        
        # Prepare data for plots
        data_neg = sub.loc[sub["neg"], ci]
        data_nonneg = sub.loc[~sub["neg"], ci]
        data = [data_neg, data_nonneg]
        
        print(f"  CO₂ (neg): median={data_neg.median():.1f}, mean={data_neg.mean():.1f} g/kWh")
        print(f"  CO₂ (≥0): median={data_nonneg.median():.1f}, mean={data_nonneg.mean():.1f} g/kWh")
        
        # Boxplot
        plt.figure(figsize=(7, 5))
        bp = plt.boxplot(data, labels=["Price < 0", "Price ≥ 0"], 
                         showfliers=False, patch_artist=True)
        # Color boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        plt.ylabel("CO₂ Intensity (g/kWh, Lifecycle)", fontsize=11)
        plt.title(f"CO₂ vs. Price Sign — {z} (Sep 2024)", fontsize=12, fontweight='bold')
        plt.grid(True, axis="y", alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"figs/box_ci_neg_vs_nonneg_{z.lower()}.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figs/box_ci_neg_vs_nonneg_{z.lower()}.png")
        
        # Violin plot
        plt.figure(figsize=(7, 5))
        parts = plt.violinplot(data, showmeans=True, showmedians=True)
        # Color violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(['lightblue', 'lightcoral'][i])
            pc.set_alpha(0.7)
        plt.xticks([1, 2], ["Price < 0", "Price ≥ 0"])
        plt.ylabel("CO₂ Intensity (g/kWh, Lifecycle)", fontsize=11)
        plt.title(f"CO₂ vs. Price Sign (Violin) — {z} (Sep 2024)", fontsize=12, fontweight='bold')
        plt.grid(True, axis="y", alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"figs/violin_ci_neg_vs_nonneg_{z.lower()}.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figs/violin_ci_neg_vs_nonneg_{z.lower()}.png")

print("\n✅ DONE! Check figs/ folder for box and violin plots.")