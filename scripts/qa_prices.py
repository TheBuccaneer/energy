#!/usr/bin/env python3
"""
QA-Check für day_ahead_sept2024_final.csv (Wide Format: DE/FR/PL als Spalten)
Prüft: NaNs, Duplikate, Wertebereich, Zeitabdeckung pro Zone
"""
import pandas as pd
from pathlib import Path

# === CONFIG ===
INFILE = Path("day_ahead_sept2024_final.csv")
OUT_SUMMARY = Path("tables/qa_prices_summary.csv")
OUT_ISSUES = Path("tables/qa_prices_issues.csv")
OUT_COVERAGE = Path("tables/qa_prices_coverage_by_area.csv")
OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"📂 LOADING: {INFILE}")
print("=" * 70)

# Laden
df = pd.read_csv(INFILE, parse_dates=['timestamp'])
print(f"✅ Loaded {len(df)} rows")
print(f"   Columns: {df.columns.tolist()}")

# Define price columns for each area
price_cols = {
    'DE': 'price_eur_kwh_de',
    'FR': 'price_eur_kwh_fr',
    'PL': 'price_eur_kwh_pl'
}

# === CONVERT TYPES ===
print("\n📊 Converting types...")
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

for area, col in price_cols.items():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"  ✅ {area}: {col}")
    else:
        print(f"  ⚠ Warning: {col} not found!")

# === CHECKS ===
issues = []

print("\n" + "=" * 70)
print("=== QA CHECKS ===")
print("=" * 70)

# 1) FEHLENDE WERTE
print("\n1️⃣ FEHLENDE WERTE (NaNs)")
n_ts_na = df['timestamp'].isna().sum()
print(f"   Timestamp NaNs: {n_ts_na}")

if n_ts_na > 0:
    issues.append({
        'check': 'missing_timestamp',
        'area': 'ALL',
        'count': int(n_ts_na),
        'severity': 'HIGH'
    })

for area, col in price_cols.items():
    if col in df.columns:
        n_price_na = df[col].isna().sum()
        pct = n_price_na / len(df) * 100
        print(f"   {area} ({col}): {n_price_na} NaNs ({pct:.1f}%)")
        
        if n_price_na > 0:
            issues.append({
                'check': 'missing_price',
                'area': area,
                'count': int(n_price_na),
                'severity': 'MEDIUM'
            })

# 2) DUPLIKATE (timestamp)
print("\n2️⃣ DUPLIKATE")
dup_mask = df.duplicated(subset=['timestamp'], keep=False)
n_dups = int(dup_mask.sum())
print(f"   Duplicate timestamps: {n_dups}")

if n_dups > 0:
    issues.append({
        'check': 'duplicates',
        'area': 'ALL',
        'count': n_dups,
        'severity': 'HIGH'
    })
    
    # Show examples
    dup_ts = df[dup_mask]['timestamp'].unique()[:5]
    print(f"   Examples: {dup_ts}")

# 3) WERTEBEREICH
print("\n3️⃣ WERTEBEREICH (EPEX Day-Ahead: -0.5 bis 2.0 EUR/kWh)")

range_summary = []
for area, col in price_cols.items():
    if col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        
        # Out of range: < -0.5 or > 2.0
        out_of_range = df[(df[col] < -0.5) | (df[col] > 2.0)]
        n_out = len(out_of_range)
        
        print(f"   {area}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f} EUR/kWh")
        print(f"         Out of range [-0.5, 2.0]: {n_out} rows")
        
        range_summary.append({
            'area': area,
            'min_eur_kwh': min_val,
            'max_eur_kwh': max_val,
            'mean_eur_kwh': mean_val,
            'out_of_range': n_out
        })
        
        if n_out > 0:
            issues.append({
                'check': 'range_violation',
                'area': area,
                'count': n_out,
                'severity': 'LOW'
            })

# 4) ZEITABDECKUNG
print("\n4️⃣ ZEITABDECKUNG")

# Floor timestamps to hour (Europe/Berlin timezone)
if df['timestamp'].dt.tz is None:
    df['ts_hour'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin').dt.floor('H')
else:
    df['ts_hour'] = df['timestamp'].dt.tz_convert('Europe/Berlin').dt.floor('H')

# Count unique hours
n_unique_hours = df['ts_hour'].nunique()

# Expected hours (September 2024 = 720 hours)
time_range = df['timestamp'].max() - df['timestamp'].min()
expected_hours = int(time_range.total_seconds() / 3600) + 1

print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Unique hours present: {n_unique_hours}")
print(f"   Expected hours: {expected_hours}")
print(f"   Coverage: {n_unique_hours/expected_hours*100:.1f}%")

missing_hours = expected_hours - n_unique_hours
if missing_hours > 0:
    issues.append({
        'check': 'incomplete_coverage',
        'area': 'ALL',
        'count': missing_hours,
        'severity': 'MEDIUM'
    })

# Coverage per area (check for NaN prices per hour)
coverage_data = []
for area, col in price_cols.items():
    if col in df.columns:
        valid_hours = df[df[col].notna()]['ts_hour'].nunique()
        missing = expected_hours - valid_hours
        pct = valid_hours / expected_hours * 100
        
        coverage_data.append({
            'area': area,
            'hours_present': valid_hours,
            'expected_hours': expected_hours,
            'missing_hours': missing,
            'coverage_pct': pct
        })
        
        print(f"   {area}: {valid_hours}/{expected_hours} hours ({pct:.1f}%)")

coverage_df = pd.DataFrame(coverage_data)

# 5) NEGATIVE PREISE (INFO ONLY)
print("\n5️⃣ NEGATIVE PREISE (Informational)")

for area, col in price_cols.items():
    if col in df.columns:
        flag_col = f'is_negative_{area.lower()}'
        if flag_col in df.columns:
            n_neg = df[flag_col].sum()
            pct = n_neg / len(df) * 100
            print(f"   {area}: {n_neg} negative hours ({pct:.1f}%)")
        else:
            n_neg = (df[col] < 0).sum()
            pct = n_neg / len(df) * 100
            print(f"   {area}: {n_neg} negative hours ({pct:.1f}%)")

# === SUMMARY ===
print("\n" + "=" * 70)
print("=== ZUSAMMENFASSUNG ===")
print("=" * 70)

summary = pd.DataFrame({
    'total_rows': [len(df)],
    'areas': [len(price_cols)],
    'unique_hours': [n_unique_hours],
    'expected_hours': [expected_hours],
    'missing_hours': [missing_hours],
    'n_ts_nans': [int(n_ts_na)],
    'n_duplicates': [n_dups],
    'time_range_start': [df['timestamp'].min()],
    'time_range_end': [df['timestamp'].max()],
    'total_issues': [len(issues)]
})

# === SAVE RESULTS ===
print(f"\n💾 Saving results...")

summary.to_csv(OUT_SUMMARY, index=False)
print(f"  ✅ {OUT_SUMMARY}")

coverage_df.to_csv(OUT_COVERAGE, index=False)
print(f"  ✅ {OUT_COVERAGE}")

if issues:
    issues_df = pd.DataFrame(issues)
    issues_df.to_csv(OUT_ISSUES, index=False)
    print(f"  ⚠ {OUT_ISSUES} ({len(issues)} issues)")
else:
    pd.DataFrame([{'check': 'ok', 'area': 'ALL', 'count': 0, 'severity': 'NONE'}]).to_csv(OUT_ISSUES, index=False)
    print(f"  ✅ {OUT_ISSUES} (no issues)")

# === FINAL STATUS ===
print("\n" + "=" * 70)
if issues:
    high = sum(1 for i in issues if i.get('severity') == 'HIGH')
    med = sum(1 for i in issues if i.get('severity') == 'MEDIUM')
    low = sum(1 for i in issues if i.get('severity') == 'LOW')
    
    print(f"⚠ QA COMPLETE: {len(issues)} issues found")
    print(f"  HIGH: {high}, MEDIUM: {med}, LOW: {low}")
else:
    print("✅ QA COMPLETE: No issues found!")
print("=" * 70)