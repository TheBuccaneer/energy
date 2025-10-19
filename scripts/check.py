import pandas as pd
df = pd.read_csv("figs/../day_ahead_sept2024_final.csv")  # Pfad ggf. anpassen
for z in ["de","fr","pl"]:
    c = f"price_eur_kwh_{z}"
    print(z.upper(), "min/max €/kWh:", df[c].min(), df[c].max(), "neg%:", (df[c]<0).mean())