import pandas as pd
from pathlib import Path

# ---- Config ----
in_csv  = "stars.csv"            # input dataset
out_csv = "stars_for_plot.csv"   # final output (top 2000 brightest only)

# ---- Columns to keep ----
cols_to_keep = [
    "ID", "Source",
    "RA_ICRS", "DE_ICRS",
    "BPmag", "RPmag",
    "Gmag", "Rad"
]

# ---- Read and filter columns ----
df = pd.read_csv(in_csv, usecols=lambda c: c in cols_to_keep)

# ---- Derive BP–RP color index ----
df["BP_RP"] = df["BPmag"] - df["RPmag"]

# ---- Drop rows with missing key data ----
df = df.dropna(subset=["RA_ICRS", "DE_ICRS", "Gmag"])

# ---- Keep only 2000 brightest (lowest Gmag values) ----
df = df.sort_values(by="Gmag", ascending=True).head(2000)

# ---- Reorder columns for clarity ----
df = df[["ID", "Source", "RA_ICRS", "DE_ICRS", "BP_RP", "Gmag", "Rad", "BPmag", "RPmag"]]

# ---- Save final dataset ----
df.to_csv(out_csv, index=False)
print(f"🌟 Saved top 2000 brightest stars to {Path(out_csv).resolve()}")
