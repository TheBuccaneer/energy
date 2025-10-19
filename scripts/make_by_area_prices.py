#!/usr/bin/env python3
"""
Generate price statistics table by area from day_ahead_sept2024_final.csv
Computes: N, mean, median, min, max, p10, p90, share_negative per zone
"""
import pandas as pd
from pathlib import Path

# === CONFIG ===
INFILE = Path("day_ahead_sept2024_final.csv")
OUT = Path("tables/by_area_prices.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"📂 LOADING: {INFILE}")
print("=" * 70)

# Daten laden
df = pd.read_csv(INFILE, parse_dates=["timestamp"])
print(f"✅ Loaded {len(df)} rows")

# Define price columns and flag columns per area
areas = ['DE', 'FR', 'PL']
price_cols = {area: f'price_eur_kwh_{area.lower()}' for area in areas}
flag_cols = {area: f'is_negative_{area.lower()}' for area in areas}

# Convert prices to numeric
print("\n💱 Converting prices to numeric...")
for area, col in price_cols.items():
    df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"  ✅ {area}: {col}")

# Check if flag columns exist, if not create them
print("\n🚩 Checking negative price flags...")
for area, col in flag_cols.items():
    if col not in df.columns:
        price_col = price_cols[area]
        df[col] = df[price_col] < 0
        print(f"  ⚠ Created {col} (not found in file)")
    else:
        print(f"  ✅ {area}: {col} exists")

# === COMPUTE STATISTICS PER AREA ===
print("\n📊 Computing statistics per area...")

stats_data = []

for area in areas:
    price_col = price_cols[area]
    flag_col = flag_cols[area]
    
    # Filter out NaN values for statistics
    valid_prices = df[price_col].dropna()
    
    if len(valid_prices) == 0:
        print(f"  ⚠ Warning: No valid prices for {area}")
        continue
    
    stats = {
        'area': area,
        'N': len(valid_prices),
        'mean_eur_kwh': valid_prices.mean(),
        'median_eur_kwh': valid_prices.median(),
        'min_eur_kwh': valid_prices.min(),
        'max_eur_kwh': valid_prices.max(),
        'p10_eur_kwh': valid_prices.quantile(0.10),
        'p90_eur_kwh': valid_prices.quantile(0.90),
        'share_negative': df[flag_col].mean()  # Use all rows (including NaN as False)
    }
    
    stats_data.append(stats)
    print(f"  ✅ {area}: {stats['N']} valid prices")

# Create DataFrame
by_area = pd.DataFrame(stats_data)

# Round for readability
by_area = by_area.round({
    'mean_eur_kwh': 4,
    'median_eur_kwh': 4,
    'min_eur_kwh': 4,
    'max_eur_kwh': 4,
    'p10_eur_kwh': 4,
    'p90_eur_kwh': 4,
    'share_negative': 4
})

# === SAVE ===
print(f"\n💾 Saving to: {OUT}")
by_area.to_csv(OUT, index=False)
print("✅ Done!")

# === DISPLAY RESULTS ===
print("\n" + "=" * 70)
print("=== PREIS-STATISTIKEN JE AREA ===")
print("=" * 70)
print()
print(by_area.to_string(index=False))
print()
print("Legende:")
print("  N              : Anzahl gültiger Stunden")
print("  mean_eur_kwh   : Arithmetisches Mittel")
print("  median_eur_kwh : Median (50. Perzentil)")
print("  min_eur_kwh    : Minimum")
print("  max_eur_kwh    : Maximum")
print("  p10_eur_kwh    : 10. Perzentil")
print("  p90_eur_kwh    : 90. Perzentil")
print("  share_negative : Anteil negativer Preise (0-1)")
print()
print("=" * 70)

# === ADDITIONAL INSIGHTS ===
print("\n📈 Zusätzliche Insights:")

# Price spread (p90 - p10)
by_area['price_spread_p90_p10'] = by_area['p90_eur_kwh'] - by_area['p10_eur_kwh']

# Coefficient of variation (if mean > 0)
for idx, row in by_area.iterrows():
    area = row['area']
    price_col = price_cols[area]
    valid_prices = df[price_col].dropna()
    
    if row['mean_eur_kwh'] > 0:
        cv = valid_prices.std() / row['mean_eur_kwh']
        by_area.at[idx, 'cv'] = cv
    else:
        by_area.at[idx, 'cv'] = float('nan')

print("\nPrice Spread (p90 - p10):")
for _, row in by_area.iterrows():
    print(f"  {row['area']}: {row['price_spread_p90_p10']:.4f} EUR/kWh")

print("\nCoefficient of Variation (std/mean):")
for _, row in by_area.iterrows():
    if pd.notna(row['cv']):
        print(f"  {row['area']}: {row['cv']:.4f}")
    else:
        print(f"  {row['area']}: N/A")

print("\n✅ COMPLETE!")