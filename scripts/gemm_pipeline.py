#!/usr/bin/env python3
"""
GEMM Pipeline: Aggregate GEMM measurements with prices & CO2 for all areas
Input: gemm_*.csv files with price/CO2 columns (from add_infos + add_co2 scripts)
Output: GEMM_by_run.csv (per setup) + GEMM_by_size.csv (per matrix size)
"""
from pathlib import Path
import pandas as pd
import re

# Paths
FREEZE = Path(".")  # Current directory (where you are running the script)
OUTDIR = Path("tables")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Output files
out_by_run = OUTDIR / "GEMM_by_run.csv"
out_by_size = OUTDIR / "GEMM_by_size.csv"

# Columns to load (saves memory)
wanted = [
    "timestamp", "host", "gpu_name", "matrix_size", "mode", "batches",
    "seconds_wall", "energy_j", "kWh_e2e", 
    "pcie_gen_current", "pcie_width_current",
    "temp_c", "throttle_reasons",
    # Price & Carbon for all areas
    "de_price_eur_kwh_run", "de_eur_job", "de_co2_job_kg", "de_price_sync", "de_eur_job_sync",
    "fr_price_eur_kwh_run", "fr_eur_job", "fr_co2_job_kg", "fr_price_sync", "fr_eur_job_sync",
    "pl_price_eur_kwh_run", "pl_eur_job", "pl_co2_job_kg", "pl_price_sync", "pl_eur_job_sync",
    # Performance metrics
    "gflops", "joule_per_gflop"
]

# Find all GEMM files
files = [p for p in FREEZE.glob("*.csv") if "gemm" in p.stem.lower()]
if not files:
    raise SystemExit(f"No GEMM files found in {FREEZE}")

print(f"Found {len(files)} GEMM files:")
for f in sorted(files):
    print(f"  - {f.name}")

# Storage for aggregated data
agg_by_run_all = []
agg_by_size_all = []

for p in sorted(files):
    print(f"\nProcessing {p.name}...")
    
    # Read header to check available columns
    header = pd.read_csv(p, nrows=0).columns.tolist()
    usecols = [c for c in wanted if c in header]
    
    # Load data - German CSV format (comma as decimal separator!)
    df = pd.read_csv(p, usecols=usecols, decimal=',', thousands=None)
    print(f"  Loaded {len(df)} rows")
    
    # Check if data was loaded correctly (decimal=',' should handle it)
    if 'energy_j' in df.columns:
        valid_energy = df['energy_j'].notna().sum()
        print(f"  ✅ Valid energy_j values: {valid_energy}/{len(df)}")
    
    # Convert numeric columns (should already be correct with decimal=',')
    numeric_cols = [
        "kWh_e2e", "seconds_wall", "energy_j",
        "de_eur_job", "fr_eur_job", "pl_eur_job",
        "de_eur_job_sync", "fr_eur_job_sync", "pl_eur_job_sync",
        "de_co2_job_kg", "fr_co2_job_kg", "pl_co2_job_kg",
        "de_price_eur_kwh_run", "fr_price_eur_kwh_run", "pl_price_eur_kwh_run",
        "de_price_sync", "fr_price_sync", "pl_price_sync",
        "matrix_size", "gflops", "joule_per_gflop"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Remove rows with NaN in critical columns (energy, cost, CO2)
    critical_cols = ["energy_j"]
    existing_critical = [c for c in critical_cols if c in df.columns]
    
    if existing_critical:
        before = len(df)
        df = df.dropna(subset=existing_critical)
        after = len(df)
        if before != after:
            print(f"  ℹ Removed {before - after} rows with missing energy_j ({after} rows remain)")
    
    if len(df) == 0:
        print(f"  ⚠ WARNING: No valid rows after filtering, skipping {p.name}")
        continue
    
    # Identify setup from filename
    cpu_type = "AMD" if "amd" in p.stem.lower() else ("Intel" if "intel" in p.stem.lower() else "Unknown")
    df["cpu_type"] = cpu_type
    
    # Extract thread count from filename for CPU-only runs
    # Pattern: gemm_<cpu>_cpu_<number>.csv
    thread_match = re.search(r'cpu_(\d+)\.csv$', p.name.lower())
    threads = int(thread_match.group(1)) if thread_match else None
    df["threads"] = threads
    
    # Create setup identifier
    def make_setup_id(row):
        parts = [row["cpu_type"]]
        gpu_name = row.get("gpu_name")
        
        # Check if it's actually a GPU (not a CPU in gpu_name column)
        is_gpu = False
        if pd.notna(gpu_name) and str(gpu_name).strip() != "":
            gpu_str = str(gpu_name).lower()
            # Real GPUs contain these keywords
            if any(keyword in gpu_str for keyword in ['rtx', 'gtx', '3090', '4090', '5050', 'quadro', 'tesla', 'geforce', 'radeon', 'nvidia']):
                is_gpu = True
        
        if is_gpu:
            gpu = str(gpu_name)
            if "3090" in gpu:
                parts.append("3090")
            elif "5050" in gpu:
                parts.append("5050")
            else:
                parts.append("GPU")
            
            pcie_gen = row.get("pcie_gen_current")
            if pd.notna(pcie_gen):
                parts.append(f"PCIe{int(pcie_gen)}")
        else:
            parts.append("CPU-only")
            # Add thread count for CPU-only setups
            threads = row.get("threads")
            if pd.notna(threads):
                parts.append(f"{int(threads)}t")
        
        return "_".join(parts)
    
    df["setup_id"] = df.apply(make_setup_id, axis=1, result_type='reduce')
    
    # --- Aggregation 1: By Setup (all matrix sizes combined) ---
    group_cols_run = ["setup_id", "cpu_type", "gpu_name", "pcie_gen_current", "mode", "threads"]
    
    agg_dict_run = {}
    # For each area, sum energy/cost/carbon (both run and sync)
    for area in ["de", "fr", "pl"]:
        if f"{area}_eur_job" in df.columns:
            agg_dict_run[f"{area}_eur_job_sum"] = (f"{area}_eur_job", "sum")
            agg_dict_run[f"{area}_co2_job_kg_sum"] = (f"{area}_co2_job_kg", "sum")
        if f"{area}_eur_job_sync" in df.columns:
            agg_dict_run[f"{area}_eur_job_sync_sum"] = (f"{area}_eur_job_sync", "sum")
        if f"{area}_price_sync" in df.columns:
            agg_dict_run[f"{area}_price_sync"] = (f"{area}_price_sync", "first")
    
    if "kWh_e2e" in df.columns:
        agg_dict_run["kWh_e2e_sum"] = ("kWh_e2e", "sum")
    if "seconds_wall" in df.columns:
        agg_dict_run["seconds_wall_median"] = ("seconds_wall", "median")
    if "gflops" in df.columns:
        agg_dict_run["gflops_mean"] = ("gflops", "mean")
    
    by_run = df.groupby(group_cols_run, dropna=False, as_index=False).agg(**agg_dict_run)
    by_run["n_measurements"] = df.groupby(group_cols_run, dropna=False).size().values
    by_run["workload"] = "GEMM"
    by_run["source_file"] = p.name
    agg_by_run_all.append(by_run)
    
    # --- Aggregation 2: By Matrix Size ---
    if "matrix_size" in df.columns:
        group_cols_size = ["setup_id", "cpu_type", "matrix_size", "mode", "gpu_name", "pcie_gen_current", "threads"]
        
        agg_dict_size = {}
        # For each area, take median (both run and sync)
        for area in ["de", "fr", "pl"]:
            if f"{area}_eur_job" in df.columns:
                agg_dict_size[f"{area}_eur_job_median"] = (f"{area}_eur_job", "median")
                agg_dict_size[f"{area}_co2_job_kg_median"] = (f"{area}_co2_job_kg", "median")
                agg_dict_size[f"{area}_price_eur_kwh"] = (f"{area}_price_eur_kwh_run", "median")
            if f"{area}_eur_job_sync" in df.columns:
                agg_dict_size[f"{area}_eur_job_sync_median"] = (f"{area}_eur_job_sync", "median")
            if f"{area}_price_sync" in df.columns:
                agg_dict_size[f"{area}_price_sync"] = (f"{area}_price_sync", "first")
        
        if "kWh_e2e" in df.columns:
            agg_dict_size["kWh_e2e_median"] = ("kWh_e2e", "median")
        if "seconds_wall" in df.columns:
            agg_dict_size["seconds_wall_median"] = ("seconds_wall", "median")
        if "gflops" in df.columns:
            agg_dict_size["gflops_median"] = ("gflops", "median")
        if "joule_per_gflop" in df.columns:
            agg_dict_size["joule_per_gflop_median"] = ("joule_per_gflop", "median")
        
        by_size = df.groupby(group_cols_size, dropna=False, as_index=False).agg(**agg_dict_size)
        by_size["n_measurements"] = df.groupby(group_cols_size, dropna=False).size().values
        by_size["workload"] = "GEMM"
        by_size["source_file"] = p.name
        agg_by_size_all.append(by_size)
        print(f"  ✅ Aggregated {len(by_size)} matrix size configurations")

# Combine and save
print("\n" + "=" * 70)
print("=== SAVING RESULTS ===")
print("=" * 70)

# By Run
final_by_run = pd.concat(agg_by_run_all, ignore_index=True)

# Ensure numeric columns are actually numeric (fix any string issues)
numeric_output_cols = [col for col in final_by_run.columns 
                      if any(x in col for x in ['eur_job', 'co2_job', 'kwh', 'seconds', 'gflops', 'price'])]
for col in numeric_output_cols:
    final_by_run[col] = pd.to_numeric(final_by_run[col], errors='coerce')

# Smart rounding: more precision for money/CO2 (very small values), less for others
money_co2_cols = [c for c in final_by_run.columns 
                  if any(x in c for x in ['eur_job', 'co2_job', 'price_eur_kwh', 'price_sync'])]
other_numeric = [c for c in numeric_output_cols if c not in money_co2_cols]

# Round money/CO2 to 10 decimals (preserve micro-euros), others to 4
round_dict = {c: 10 for c in money_co2_cols}
round_dict.update({c: 4 for c in other_numeric})
final_by_run = final_by_run.round(round_dict)

# Sort for better readability: CPU setups by threads, GPU setups by name
final_by_run = final_by_run.sort_values(
    by=["cpu_type", "gpu_name", "threads", "pcie_gen_current"],
    na_position="last"
)
final_by_run.to_csv(out_by_run, index=False)
print(f"✅ {out_by_run}")
print(f"  {len(final_by_run)} setup configurations")

# By Size
if agg_by_size_all:
    final_by_size = pd.concat(agg_by_size_all, ignore_index=True)
    
    # Ensure numeric columns are actually numeric (fix any string issues)
    numeric_output_cols = [col for col in final_by_size.columns 
                          if any(x in col for x in ['eur_job', 'co2_job', 'kwh', 'seconds', 'gflops', 'price', 'joule'])]
    for col in numeric_output_cols:
        final_by_size[col] = pd.to_numeric(final_by_size[col], errors='coerce')
    
    # Smart rounding: more precision for money/CO2 (very small values), less for others
    money_co2_cols = [c for c in final_by_size.columns 
                      if any(x in c for x in ['eur_job', 'co2_job', 'price_eur_kwh', 'price_sync'])]
    other_numeric = [c for c in numeric_output_cols if c not in money_co2_cols]
    
    # Round money/CO2 to 10 decimals (preserve micro-euros), others to 4
    round_dict = {c: 10 for c in money_co2_cols}
    round_dict.update({c: 4 for c in other_numeric})
    final_by_size = final_by_size.round(round_dict)
    
    # Sort for better readability
    final_by_size = final_by_size.sort_values(
        by=["cpu_type", "gpu_name", "threads", "matrix_size", "pcie_gen_current"],
        na_position="last"
    )
    final_by_size.to_csv(out_by_size, index=False)
    print(f"✅ {out_by_size}")
    print(f"  {len(final_by_size)} matrix size configurations")

print("\n" + "=" * 70)
print("=== SUMMARY ===")
print("=" * 70)
print(f"Setups found: {sorted(final_by_run['setup_id'].unique().tolist())}")
if agg_by_size_all:
    print(f"Matrix sizes: {sorted(final_by_size['matrix_size'].dropna().unique())}")
print("\n✅ DONE!")