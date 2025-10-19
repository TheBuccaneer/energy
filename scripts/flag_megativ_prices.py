#!/usr/bin/env python3
"""
Flag negative Day-Ahead prices in day_ahead_sept2024_final.csv
Adds is_negative_{de,fr,pl} columns and generates summary statistics.
"""
import pandas as pd
from pathlib import Path

# === CONFIG ===
input_file = Path("day_ahead_sept2024_final.csv")
output_dir = Path("tables")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("📂 LOADING DATA")
print("=" * 70)

# CSV einlesen
df = pd.read_csv(input_file, parse_dates=["timestamp"])
print(f"✅ Loaded {len(df)} rows from {input_file}")
print(f"   Columns: {df.columns.tolist()}")

# === CONVERT PRICES TO NUMERIC ===
print("\n💱 Converting prices to numeric...")
price_cols = ['price_eur_kwh_de', 'price_eur_kwh_fr', 'price_eur_kwh_pl']

for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    n_missing = df[col].isna().sum()
    if n_missing > 0:
        print(f"  ⚠ Warning: {n_missing} NaN values in {col}")

# === FLAG NEGATIVE PRICES PER ZONE ===
print("\n🚩 Flagging negative prices per zone...")

df['is_negative_de'] = df['price_eur_kwh_de'] < 0
df['is_negative_fr'] = df['price_eur_kwh_fr'] < 0
df['is_negative_pl'] = df['price_eur_kwh_pl'] < 0

# Count negatives per zone
neg_de = df['is_negative_de'].sum()
neg_fr = df['is_negative_fr'].sum()
neg_pl = df['is_negative_pl'].sum()

print(f"  DE: {neg_de} negative hours ({neg_de/len(df)*100:.1f}%)")
print(f"  FR: {neg_fr} negative hours ({neg_fr/len(df)*100:.1f}%)")
print(f"  PL: {neg_pl} negative hours ({neg_pl/len(df)*100:.1f}%)")

# === SAVE UPDATED FILE ===
print(f"\n💾 Saving updated file: {input_file}")
df.to_csv(input_file, index=False)
print("✅ Done! Added columns: is_negative_de, is_negative_fr, is_negative_pl")

# === GENERATE SUMMARY ===
print("\n📊 Generating summary statistics...")

# Reshape data for area-based summary
summary_data = []
for area in ['DE', 'FR', 'PL']:
    area_lower = area.lower()
    price_col = f'price_eur_kwh_{area_lower}'
    flag_col = f'is_negative_{area_lower}'
    
    total_hours = df[price_col].notna().sum()
    negative_hours = df[flag_col].sum()
    share_negative = negative_hours / total_hours if total_hours > 0 else 0
    
    min_price = df[price_col].min()
    max_price = df[price_col].max()
    mean_price = df[price_col].mean()
    
    summary_data.append({
        'area': area,
        'total_hours': total_hours,
        'negative_hours': negative_hours,
        'share_negative_hours': share_negative,
        'min_price_eur_kwh': min_price,
        'max_price_eur_kwh': max_price,
        'mean_price_eur_kwh': mean_price
    })

summary = pd.DataFrame(summary_data)

# Save summary
summary_file = output_dir / "neg_share_by_area.csv"
summary.to_csv(summary_file, index=False)
print(f"✅ Summary saved: {summary_file}")

# === DETAILED OUTPUT ===
print("\n" + "=" * 70)
print("=== ANTEIL NEGATIVER STROMPREISE JE ZONE ===")
print("=" * 70)

for _, row in summary.iterrows():
    print(f"\n{row['area']}:")
    print(f"  Negative Stunden: {int(row['negative_hours'])} von {int(row['total_hours'])} ({row['share_negative_hours']:.2%})")
    print(f"  Preisspanne: {row['min_price_eur_kwh']:.4f} bis {row['max_price_eur_kwh']:.4f} EUR/kWh")
    print(f"  Durchschnitt: {row['mean_price_eur_kwh']:.4f} EUR/kWh")

# === ADDITIONAL STATISTICS ===
print("\n" + "=" * 70)
print("=== ZUSÄTZLICHE STATISTIKEN ===")
print("=" * 70)

print(f"Gesamtanzahl Datensätze: {len(df)}")
print(f"Zeitraum: {df['timestamp'].min()} bis {df['timestamp'].max()}")
print(f"Anzahl Zonen: 3 (DE, FR, PL)")

# Check for hours where ALL zones are negative
all_negative = df['is_negative_de'] & df['is_negative_fr'] & df['is_negative_pl']
n_all_neg = all_negative.sum()
print(f"\nStunden wo ALLE Zonen negativ: {n_all_neg} ({n_all_neg/len(df)*100:.1f}%)")

# Check for hours where ANY zone is negative
any_negative = df['is_negative_de'] | df['is_negative_fr'] | df['is_negative_pl']
n_any_neg = any_negative.sum()
print(f"Stunden wo MINDESTENS EINE Zone negativ: {n_any_neg} ({n_any_neg/len(df)*100:.1f}%)")

# Most negative hour per zone
print("\n=== NIEDRIGSTE PREISE ===")
for area in ['DE', 'FR', 'PL']:
    area_lower = area.lower()
    price_col = f'price_eur_kwh_{area_lower}'
    
    min_idx = df[price_col].idxmin()
    min_row = df.loc[min_idx]
    
    print(f"{area}: {min_row[price_col]:.4f} EUR/kWh am {min_row['timestamp']}")

print("\n✅ DONE!")