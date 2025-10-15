# plot_real.py
# -----------------------------------------------------------
# Real-sky constellations at a given place/time using IAU defs
# Loads stars_plot_ready.csv (ID,Source,x,y,R,G,B,size)
# Outputs: real_sky_<timestamp>.png and visible_constellations_<timestamp>.csv
# -----------------------------------------------------------


"""

example usage:
at Indy:
python plot_real.py \
  --lat 39.7684 --lon -86.1581 \
  --time "2025-10-15 02:00:00" \
  --min_alt 10 \
  --label_constellations --max_labels 25
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_constellation


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot real constellations visible from a location at a given time."
    )
    p.add_argument("--csv", default="stars_plot_ready.csv",
                   help="Input CSV with ID,Source,x(=RA deg),y(=Dec deg),R,G,B,size")
    p.add_argument("--lat", type=float, required=True, help="Observer latitude in degrees (positive north)")
    p.add_argument("--lon", type=float, required=True, help="Observer longitude in degrees (positive east)")
    p.add_argument("--elev", type=float, default=0.0, help="Elevation in meters")
    p.add_argument(
        "--time",
        default=None,
        help="Observation time e.g. '2025-10-15 01:00:00' (assumed UTC unless you append offset like '+00:00')",
    )
    p.add_argument("--min_alt", type=float, default=0.0, help="Minimum altitude to include (degrees)")
    p.add_argument("--title", default=None, help="Optional plot title override")
    p.add_argument("--outdir", default="real_sky_out", help="Output folder")
    p.add_argument("--label_constellations", action="store_true",
                   help="Overlay constellation name labels at cluster centroids")
    p.add_argument("--max_labels", type=int, default=30,
                   help="Max number of constellation labels to draw (largest groups first)")
    return p.parse_args()


def load_stars(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = {"ID", "Source", "x", "y", "R", "G", "B", "size"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return df


def compute_altaz(df: pd.DataFrame, loc: EarthLocation, t: Time) -> pd.DataFrame:
    # x: RA deg, y: Dec deg
    coords = SkyCoord(ra=df["x"].to_numpy() * u.deg,
                      dec=df["y"].to_numpy() * u.deg,
                      frame="icrs")
    altaz_frame = AltAz(obstime=t, location=loc)
    altaz = coords.transform_to(altaz_frame)
    df = df.copy()
    df["Alt_deg"] = altaz.alt.degree
    df["Az_deg"] = altaz.az.degree
    return df


def assign_constellations(df: pd.DataFrame) -> pd.Series:
    coords = SkyCoord(ra=df["x"].to_numpy() * u.deg,
                      dec=df["y"].to_numpy() * u.deg,
                      frame="icrs")
    names = get_constellation(coords)  # IAU official names (e.g., 'Orion')
    return pd.Series(names, index=df.index, name="Constellation")


def sky_plot(df_vis: pd.DataFrame, out_png: Path, title: str,
             label_constellations: bool = False, max_labels: int = 30):
    # Prepare colors (0..1)
    rgb = df_vis[["R", "G", "B"]].to_numpy() / 255.0
    sizes = df_vis["size"].to_numpy()

    # Figure
    plt.figure(figsize=(12, 8), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    # Scatter: Az (x) vs Alt (y); invert Az to match planisphere feel if desired
    sc = ax.scatter(
        df_vis["Az_deg"].to_numpy(),
        df_vis["Alt_deg"].to_numpy(),
        s=sizes, c=rgb, alpha=0.9, edgecolors="none"
    )

    # Cosmetic
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 90)
    ax.set_xlabel("Azimuth (°)", color="white")
    ax.set_ylabel("Altitude (°)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(title, color="white", pad=12)
    ax.grid(alpha=0.2, color="white", linestyle=":")

    # Optional: label constellation names at centroids (by visible stars)
    if label_constellations and "Constellation" in df_vis.columns:
        groups = (
            df_vis.groupby("Constellation")
            .agg(count=("ID", "size"),
                 Alt=("Alt_deg", "mean"),
                 Az=("Az_deg", "mean"))
            .sort_values("count", ascending=False)
            .head(max_labels)
        )
        for name, row in groups.iterrows():
            ax.text(row["Az"], row["Alt"], name,
                    color="white", fontsize=9, ha="center", va="center", alpha=0.8)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Time handling
    if args.time is None:
        # default to current UTC time
        t = Time(datetime.utcnow(), scale="utc")
        ts_label = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    else:
        # If the string includes an offset like '+01:00', astropy parses it; else treat as UTC
        t = Time(args.time)
        # Make a safe label (remove colons/spaces)
        ts_label = str(t.isot).replace(":", "").replace("-", "").replace("T", "T").replace("Z", "Z")

    # Observer location
    loc = EarthLocation(lat=args.lat * u.deg, lon=args.lon * u.deg, height=args.elev * u.m)

    # Load and compute
    df = load_stars(csv_path)

    # Compute Alt/Az for the specified time/location
    df = compute_altaz(df, loc, t)

    # Filter above horizon (and optional min_alt)
    df_vis = df[df["Alt_deg"] >= float(args.min_alt)].copy()

    # Assign official IAU constellation names
    df_vis["Constellation"] = assign_constellations(df_vis)

    # Save mapping CSV
    map_csv = outdir / f"visible_constellations_{ts_label}.csv"
    df_vis[["ID", "Source", "Alt_deg", "Az_deg", "Constellation"]].to_csv(map_csv, index=False)

    # Build a title
    title = args.title or f"Visible Constellations @ lat {args.lat:.3f}, lon {args.lon:.3f}  —  {t.isot}"

    # Plot real sky with labels
    out_png = outdir / f"real_sky_{ts_label}.png"
    sky_plot(
        df_vis,
        out_png=out_png,
        title=title,
        label_constellations=args.label_constellations,
        max_labels=args.max_labels
    )

    print(f"Visible stars: {len(df_vis)}  (min_alt = {args.min_alt}°)")
    print(f"Saved constellation map: {out_png.resolve()}")
    print(f"Saved mapping CSV:       {map_csv.resolve()}")


if __name__ == "__main__":
    main()
