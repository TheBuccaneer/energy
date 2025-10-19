from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

# Paths
INFILE = Path("co_2_ci_DE_FR_PL_hourly.csv")
OUTDIR = Path("figs")
OUTDIR.mkdir(parents=True, exist_ok=True)
MONTH_TAG = "2024-01"  # Adjust based on your data

print(f"Loading {INFILE}...")
df = pd.read_csv(INFILE, parse_dates=["ts"])

# Transform from wide to long format
# Columns: de_ci_lifecycle_g_per_kwh, fr_ci_lifecycle_g_per_kwh, pl_ci_lifecycle_g_per_kwh
df_long = pd.melt(
    df,
    id_vars=["ts"],
    value_vars=["de_ci_lifecycle_g_per_kwh", "fr_ci_lifecycle_g_per_kwh", "pl_ci_lifecycle_g_per_kwh"],
    var_name="area_raw",
    value_name="ci_g_per_kwh"
)

# Extract area code (DE, FR, PL) from column name
df_long["area"] = df_long["area_raw"].str.extract(r"^([a-z]{2})_", flags=re.IGNORECASE)[0].str.upper()

# Clean numeric column
df_long["ci_g_per_kwh"] = pd.to_numeric(
    df_long["ci_g_per_kwh"].astype(str).str.replace(",", ".").str.strip(),
    errors="coerce"
)

def plot_duration(s: pd.Series, title: str, outpath: Path, ylabel: str):
    """
    Plot duration curve (descending sorted values).
    """
    s_sorted = s.dropna().sort_values(ascending=False).reset_index(drop=True)
    if len(s_sorted) == 0:
        print(f"  ⚠ No data for {title}, skipping")
        return
    
    x = (s_sorted.index + 1) / len(s_sorted) * 100.0
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, s_sorted.values, linewidth=2, color='#A23B72')
    plt.xlabel("Share of Hours [%]", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    plt.xticks([0, 20, 40, 60, 80, 100])
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {outpath.name}")

# Plot 1: All areas combined
print("\nCreating CO₂-Duration-Curves...")
plot_duration(
    df_long["ci_g_per_kwh"],
    "CO₂-Duration-Curve (All Areas)",
    OUTDIR / f"ci_duration_{MONTH_TAG}_ALL.png",
    "Carbon Intensity [gCO₂/kWh]"
)

# Plot 2: Per area
areas = df_long["area"].dropna().unique()
print(f"\nCreating CO₂-Duration-Curves for {len(areas)} areas:")
for area in sorted(areas):
    area_data = df_long[df_long["area"] == area]["ci_g_per_kwh"]
    
    if len(area_data.dropna()) > 0:
        # Statistics
        mean_ci = area_data.mean()
        median_ci = area_data.median()
        min_ci = area_data.min()
        max_ci = area_data.max()
        
        plot_duration(
            area_data,
            f"CO₂-Duration-Curve ({area})",
            OUTDIR / f"ci_duration_{MONTH_TAG}_{area}.png",
            "Carbon Intensity [gCO₂/kWh]"
        )
        
        print(f"    {area}: {len(area_data)} hours, "
              f"mean={mean_ci:.1f}, median={median_ci:.1f}, "
              f"range=[{min_ci:.1f}, {max_ci:.1f}] gCO₂/kWh")

print(f"\n✓ All plots saved in: {OUTDIR}/")
print("\nNote: CO₂-Duration-Curve shows carbon intensity sorted in descending order.")
print("  - Left (0%): Highest CI (dirtiest grid)")
print("  - Right (100%): Lowest CI (cleanest grid)")
print("  - Useful for identifying carbon-aware scheduling opportunities")