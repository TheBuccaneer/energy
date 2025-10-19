import pandas as pd
from pathlib import Path

tables = Path(".")
for f in tables.glob("*_by_run.csv"):
    df = pd.read_csv(f)
    print(f"\n{f.name}:")
    for area in ["de", "fr", "pl"]:
        col = f"{area}_price_sync"
        if col in df.columns:
            n_unique = df[col].nunique()
            print(f"  {area}: {n_unique} unique value(s)")
        else:
            print(f"  {area}: missing")