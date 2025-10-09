import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import cm
import matplotlib as mpl

# =========================
# Config
# =========================
IN_CSV  = "stars.csv"            # input dataset (full)
OUT_CSV = "stars_plot_ready.csv"      # final plotting file (top 2000, ready to plot)

# If your CSV has a header like the one you pasted, these exist:
# ID,RA_ICRS,DE_ICRS,Source,...,BPmag,RPmag,Gmag,Rad,...
COLS_TO_KEEP = [
    "ID", "Source",
    "RA_ICRS", "DE_ICRS",
    "BP_RP",          # keep if present (some datasets include it directly)
    "BPmag", "RPmag", # used to derive BP_RP if BP_RP not present
    "Gmag"            # for brightness/size and for top-2000 selection
]

# Continuous colormap and ranges
COLORMAP     = "turbo"  # try 'turbo', 'plasma', 'viridis', 'magma', ...
BP_RP_MIN    = -0.5     # clamp/normalize lower bound for BP_RP
BP_RP_MAX    = 4.0      # clamp/normalize upper bound for BP_RP

# Dot size (pixels) scaled from Gmag via flux ~ 10^(-0.4 G)
MIN_SIZE_PX  = 2.0
MAX_SIZE_PX  = 10.0
BRIGHT_TOP_N = 2000     # keep N brightest by Gmag (smallest Gmag)

# =========================
# Helpers
# =========================
def normalize(arr, vmin, vmax):
    arr = np.asarray(arr, dtype=float)
    return np.clip((arr - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)

def bp_rp_to_rgb(bp_rp_series, vmin, vmax, cmap_name):
    """
    Map BP_RP to continuous RGB via a Matplotlib colormap.
    Returns (R, G, B) as integer arrays in [0,255].
    """
    cmap = mpl.colormaps[cmap_name]
    t = normalize(bp_rp_series.values, vmin, vmax)  # 0..1
    rgba = cmap(t)                                  # Nx4 in [0,1]
    rgb = (rgba[:, :3] * 255.0).round().astype(int)
    return rgb[:, 0], rgb[:, 1], rgb[:, 2]

def gmag_to_size(g_series, min_size=MIN_SIZE_PX, max_size=MAX_SIZE_PX):
    """
    Scale point size from Gmag using flux ~ 10^(-0.4 G).
    Uses robust clipping at 1st/99th percentiles to prevent outliers dominating.
    """
    g = g_series.astype(float)
    flux = 10.0 ** (-0.4 * g)
    fmin = np.percentile(flux, 1)
    fmax = np.percentile(flux, 99)
    fmin = max(fmin, 1e-12)
    fmax = max(fmax, fmin + 1e-12)
    flux = np.clip(flux, fmin, fmax)
    t = (flux - fmin) / (fmax - fmin)
    return min_size + t * (max_size - min_size)

# =========================
# Pipeline
# =========================
def main():
    # -- Load and keep only relevant columns if present
    df_all = pd.read_csv(IN_CSV)
    have = [c for c in COLS_TO_KEEP if c in df_all.columns]
    if not {"ID", "Source", "RA_ICRS", "DE_ICRS", "Gmag"}.issubset(have):
        missing = {"ID", "Source", "RA_ICRS", "DE_ICRS", "Gmag"} - set(have)
        raise ValueError(f"Missing required columns in {IN_CSV}: {sorted(missing)}")

    df = df_all[have].copy()

    # -- Ensure BP_RP exists: prefer provided BP_RP; else derive from BPmag - RPmag
    if "BP_RP" not in df.columns:
        if {"BPmag", "RPmag"}.issubset(df.columns):
            df["BP_RP"] = df["BPmag"] - df["RPmag"]
        else:
            raise ValueError("No BP_RP column and cannot derive it (BPmag/RPmag missing).")

    # -- Drop rows missing essentials
    df = df.dropna(subset=["RA_ICRS", "DE_ICRS", "Gmag", "BP_RP"]).copy()

    # -- Select the top N brightest by Gmag (ascending = brighter first)
    df = df.sort_values(by="Gmag", ascending=True).head(BRIGHT_TOP_N)

    # -- Compute continuous RGB from BP_RP and compute dot size from Gmag
    R, G, B = bp_rp_to_rgb(df["BP_RP"], BP_RP_MIN, BP_RP_MAX, COLORMAP)
    df["R"], df["G"], df["B"] = R, G, B
    df["size"] = gmag_to_size(df["Gmag"])

    # -- Rename coordinates for plotting
    df["x"] = df["RA_ICRS"]
    df["y"] = df["DE_ICRS"]

    # -- Keep only final plotting columns, in order
    final_cols = ["ID", "Source", "x", "y", "R", "G", "B", "size"]
    df_out = df[final_cols].copy()

    # -- Save final ready-to-plot table
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"Wrote {len(df_out)} rows to {Path(OUT_CSV).resolve()} "
          f"(top {BRIGHT_TOP_N} largest, colormap='{COLORMAP}')")

if __name__ == "__main__":
    main()
