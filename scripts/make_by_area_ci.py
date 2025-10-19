from pathlib import Path
import pandas as pd
import re

# Paths
CANDIDATES = [
    Path("ci_DE_FR_PL_hourly.csv"),
    Path("co_2_ci_DE_FR_PL_hourly.csv"),
]

INFILE = next((p for p in CANDIDATES if p.exists()), None)
if INFILE is None:
    raise SystemExit("No CI file found. Looking for: " + ", ".join(str(c) for c in CANDIDATES))

OUT = Path("tables/by_area_ci.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"Loading {INFILE}...")
df = pd.read_csv(INFILE, parse_dates=["ts"])

def to_long(df):
    """
    Transform CI data to long format.
    Handles both wide format (de_ci_..., fr_ci_..., pl_ci_...) 
    and long format (area, ci_g_per_kwh columns).
    """
    # Case A: Already in long format with 'area' column
    ci_cols = [c for c in df.columns if re.search(r"(?:ci|intensity).*g.*per.*kwh", c, re.I)]
    if "area" in df.columns and len(ci_cols) == 1:
        area_col = "area"
        ci_col = ci_cols[0]
        out = df[[area_col, ci_col]].copy()
        out.rename(columns={area_col: "area", ci_col: "ci_g_per_kwh"}, inplace=True)
        return out

    # Case B: Wide format (de_ci_..., fr_ci_..., pl_ci_...)
    value_vars = [c for c in df.columns if re.match(r"(de|fr|pl)_.*ci.*g.*per.*kwh", c, re.I)]
    if value_vars:
        print(f"Found {len(value_vars)} CI columns in wide format")
        long = pd.melt(
            df, 
            id_vars=["ts"], 
            value_vars=value_vars,
            var_name="area_raw", 
            value_name="ci_g_per_kwh"
        )
        long["area"] = long["area_raw"].str.extract(r"^([a-z]{2})_", flags=re.I)[0].str.upper()
        return long[["area", "ci_g_per_kwh"]]

    raise SystemExit("CI columns not found – check file structure.")

# Transform to long format
long = to_long(df)

# Robust numeric conversion
long["ci_g_per_kwh"] = pd.to_numeric(
    long["ci_g_per_kwh"].astype(str).str.replace(",", ".").str.strip(),
    errors="coerce"
)

print(f"Calculating statistics for {long['area'].nunique()} areas...")

# Aggregate by area
by_area = (
    long.dropna(subset=["ci_g_per_kwh"])
        .groupby("area")
        .agg(
            N=("ci_g_per_kwh", "count"),
            mean_g_per_kwh=("ci_g_per_kwh", "mean"),
            median_g_per_kwh=("ci_g_per_kwh", "median"),
            min_g_per_kwh=("ci_g_per_kwh", "min"),
            p10_g_per_kwh=("ci_g_per_kwh", lambda s: s.quantile(0.10)),
            p90_g_per_kwh=("ci_g_per_kwh", lambda s: s.quantile(0.90)),
            max_g_per_kwh=("ci_g_per_kwh", "max"),
        )
        .reset_index()
        .round(1)
)

# Save
by_area.to_csv(OUT, index=False)
print(f"✓ {OUT} saved\n")

# Console output
print("=== Carbon Intensity Statistics by Area ===")
print(by_area.to_string(index=False))
print("\nLegend:")
print("  N: Number of hours")
print("  p10/p90: 10th and 90th percentile")
print("  All values in gCO₂/kWh")
print("\nTypical ranges (for validation):")
print("  FR (France): ~70-90 gCO₂/kWh (nuclear-dominated)")
print("  DE (Germany): ~300-450 gCO₂/kWh (mixed)")
print("  PL (Poland): ~600-800 gCO₂/kWh (coal-dominated)")