import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu  # nonparametric test

# Load data
dfp = pd.read_csv("./day_ahead_sept2024_final.csv", parse_dates=["timestamp"])
dfc = pd.read_csv("./ci_DE_FR_PL_hourly.csv", parse_dates=["ts"])
dfc["ts"] = dfc["ts"].dt.tz_localize(None)

# Merge
df = dfp.merge(dfc, left_on="timestamp", right_on="ts", how="inner")
print(f"Merged: {len(df)} rows\n")

# Zone definitions
zones = {
    "DE": ("price_eur_kwh_de", "de_ci_lifecycle_g_per_kwh"),
    "FR": ("price_eur_kwh_fr", "fr_ci_lifecycle_g_per_kwh"),
    "PL": ("price_eur_kwh_pl", "pl_ci_lifecycle_g_per_kwh"),
}

out = []
for z, (px, ci) in zones.items():
    sub = df[[px, ci]].dropna().copy()
    sub["neg"] = sub[px] < 0
    
    x = sub.loc[sub["neg"], ci].to_numpy()
    y = sub.loc[~sub["neg"], ci].to_numpy()
    
    print(f"{z}:")
    print(f"  Negative prices: n={len(x)}")
    print(f"  Non-negative: n={len(y)}")
    
    if len(x) == 0 or len(y) == 0:
        print(f"  ⚠️ Insufficient data for test\n")
        out.append((z, len(x), len(y), np.nan, np.nan, np.nan, np.nan, np.nan))
        continue
    
    # Descriptive stats
    med_neg = np.median(x)
    med_nonneg = np.median(y)
    mean_neg = np.mean(x)
    mean_nonneg = np.mean(y)
    
    print(f"  CO₂ (negative): median={med_neg:.1f}, mean={mean_neg:.1f} g/kWh")
    print(f"  CO₂ (≥0): median={med_nonneg:.1f}, mean={mean_nonneg:.1f} g/kWh")
    
    # Mann-Whitney U test (two-sided)
    u, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    
    # Cliff's Delta (effect size)
    # delta = P(X>Y) - P(X<Y); approximation via rank statistic
    nx, ny = len(x), len(y)
    delta = (u / (nx * ny)) * 2 - 1
    
    print(f"  Mann-Whitney U: U={u:.1f}, p={p:.4f}")
    print(f"  Cliff's Delta: {delta:.3f}")
    
    # Effect size interpretation
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        effect = "negligible"
    elif abs_delta < 0.33:
        effect = "small"
    elif abs_delta < 0.474:
        effect = "medium"
    else:
        effect = "large"
    print(f"  Effect size: {effect}\n")
    
    out.append((z, len(x), len(y), med_neg, med_nonneg, u, p, delta))

# Create results dataframe
res = pd.DataFrame(out, columns=[
    "zone", "n_neg", "n_nonneg", 
    "median_neg", "median_nonneg",
    "U", "p_value", "cliffs_delta"
])

# Save results
res.to_csv("figs/neg_vs_nonneg_mannwhitney.csv", index=False)
print("=" * 70)
print(res.to_string(index=False))
print("\n✅ Saved: figs/neg_vs_nonneg_mannwhitney.csv")

# Summary interpretation
print("\n" + "=" * 70)
print("INTERPRETATION (α = 0.05):")
print("=" * 70)
for _, row in res.iterrows():
    z = row['zone']
    p = row['p_value']
    delta = row['cliffs_delta']
    
    if pd.isna(p):
        print(f"{z}: Insufficient data for test")
        continue
    
    if p < 0.05:
        direction = "lower" if delta < 0 else "higher"
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            effect = "negligible"
        elif abs_delta < 0.33:
            effect = "small"
        elif abs_delta < 0.474:
            effect = "medium"
        else:
            effect = "large"
        
        print(f"{z}: Significant difference (p={p:.4f}). CO₂ intensity at negative "
              f"prices is {direction} ({effect} effect, δ={delta:.3f}).")
    else:
        print(f"{z}: No significant difference (p={p:.4f}).")