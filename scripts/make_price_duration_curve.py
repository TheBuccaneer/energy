#!/usr/bin/env python3
"""
Generate Price-Duration-Curves (PDC) from day_ahead_sept2024_final.csv
PDC = Prices sorted descending vs. share of time (%)
Separate plots for each area (DE/FR/PL) + combined plot
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIG ===
INFILE = Path("day_ahead_sept2024_final.csv")
OUTDIR = Path("figs")
OUTDIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"📂 LOADING: {INFILE}")
print("=" * 70)

# Load data
df = pd.read_csv(INFILE, parse_dates=["timestamp"])
print(f"✅ Loaded {len(df)} rows")

# Define areas and price columns
areas = ['DE', 'FR', 'PL']
price_cols = {area: f'price_eur_kwh_{area.lower()}' for area in areas}

# Convert to numeric
print("\n💱 Converting prices to numeric...")
for area, col in price_cols.items():
    df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"  ✅ {area}: {col}")


def plot_pdc(s: pd.Series, title: str, outpath: Path, color='#2E86AB'):
    """
    Plot Price-Duration-Curve (PDC).
    PDC = Prices sorted descending vs. share of time.
    
    Args:
        s: Price series
        title: Plot title
        outpath: Output file path
        color: Line color
    """
    # Sort descending (highest prices on left)
    s_sorted = s.sort_values(ascending=False).reset_index(drop=True)
    
    # X-axis: Share of hours in %
    x = (s_sorted.index + 1) / len(s_sorted) * 100.0
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, s_sorted.values, linewidth=2.5, color=color, label='Price')
    
    # Zero line (for negative prices)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='0 EUR/kWh')
    
    # Labels and styling
    plt.xlabel("Share of Hours [%]", fontsize=12, fontweight='bold')
    plt.ylabel("Price [EUR/kWh]", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    plt.legend(loc='best', fontsize=11)
    
    # Add statistics as text box
    stats_text = (
        f"Min: {s.min():.4f} EUR/kWh\n"
        f"Max: {s.max():.4f} EUR/kWh\n"
        f"Mean: {s.mean():.4f} EUR/kWh\n"
        f"Negative: {(s < 0).sum()} h ({(s < 0).sum()/len(s)*100:.1f}%)"
    )
    plt.text(0.98, 0.97, stats_text, 
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ {outpath.name}")


# === 1) PDC PER AREA ===
print("\n📊 Creating Price-Duration-Curves per area...")

area_colors = {
    'DE': '#1f77b4',  # Blue
    'FR': '#ff7f0e',  # Orange
    'PL': '#2ca02c'   # Green
}

for area in areas:
    col = price_cols[area]
    prices = df[col].dropna()
    
    if len(prices) == 0:
        print(f"  ⚠ Warning: No valid prices for {area}")
        continue
    
    # Individual PDC
    plot_pdc(
        prices,
        f"Price-Duration-Curve (September 2024, {area})",
        OUTDIR / f"price_duration_{area}.png",
        color=area_colors[area]
    )
    
    # Statistics
    n_neg = (prices < 0).sum()
    pct_neg = n_neg / len(prices) * 100
    print(f"    {area}: {len(prices)} hours, {n_neg} negative ({pct_neg:.1f}%)")


# === 2) COMBINED PDC (ALL AREAS) ===
print("\n📊 Creating combined Price-Duration-Curve...")

plt.figure(figsize=(12, 7))

for area in areas:
    col = price_cols[area]
    prices = df[col].dropna()
    
    if len(prices) == 0:
        continue
    
    # Sort descending
    s_sorted = prices.sort_values(ascending=False).reset_index(drop=True)
    x = (s_sorted.index + 1) / len(s_sorted) * 100.0
    
    # Plot
    plt.plot(x, s_sorted.values, linewidth=2.5, 
             color=area_colors[area], label=area, alpha=0.8)

# Zero line
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='0 EUR/kWh')

# Labels and styling
plt.xlabel("Share of Hours [%]", fontsize=12, fontweight='bold')
plt.ylabel("Price [EUR/kWh]", fontsize=12, fontweight='bold')
plt.title("Price-Duration-Curves (September 2024, All Areas)", fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
plt.legend(loc='best', fontsize=11, framealpha=0.9)
plt.tight_layout()

# Save combined plot
combined_path = OUTDIR / "price_duration_all_areas.png"
plt.savefig(combined_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✅ {combined_path.name}")


# === SUMMARY ===
print("\n" + "=" * 70)
print("✅ ALL PLOTS SAVED")
print("=" * 70)
print(f"\nOutput directory: {OUTDIR}/")
print("\nGenerated files:")
for area in areas:
    print(f"  - price_duration_{area}.png")
print(f"  - price_duration_all_areas.png")

print("\n📖 Note: PDC shows prices sorted in descending order:")
print("  - Left (0%): Highest prices")
print("  - Right (100%): Lowest prices")
print("  - Negative prices are normal during oversupply (EPEX Day-Ahead)")
print("\n" + "=" * 70)