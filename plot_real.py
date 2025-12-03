"""plot_constellations_from_csv.py (week9: location + algorithmic bbox)
-----------------------------------------------------------
Purpose:
  Generate constellation "stick" plots from stars_plot_ready.csv.
  Always plots background stars, overlays the largest connected
  component per constellation, and labels names with offsets.

previous upgrades (effective knobs):
  (1) Location-based visibility filter:
      --apply_visibility with --observer_lat/--observer_lon/--observer_elev_m
      and --utc_time keeps only stars above the horizon (>0° altitude).

  (2) Adjustable number of plotted stars:
      --top_n limits brightest stars before per-constellation processing.

  (3) Algorithmic per-constellation bounding boxes:
      Two-pass inside each constellation:
        pass-1: graph on all -> largest component -> bbox (+--mask_pad_deg)
        pass-2: FILTER nodes to bbox -> rebuild graph -> largest component
      Final edges/labels are from pass-2 (bbox-constrained).
      Masks are saved to out/masks.csv and can be drawn with --draw_masks.

Example:
  python plot_real.py \
    --csv stars_plot_ready.csv --top_n 1000 \
    --apply_visibility \
    --observer_lat 39.77 --observer_lon -86.16 --observer_elev_m 220 \
    --utc_time 2025-11-06T02:00:00 \
    --mask_pad_deg 5 --draw_masks \
    --label_dx -15 --label_dy 1.5 --outdir out_week9

Requirements:
  astropy, numpy, pandas, matplotlib, scipy
-----------------------------------------------------------"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord, get_constellation
from astropy.coordinates import AltAz, EarthLocation
from astropy.time import Time

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib.patches as mpatches


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Constellation stick plot with optional visibility filter and algorithmic bbox."
    )
    p.add_argument("--csv", default="stars_plot_ready.csv",
                   help="Input CSV with columns: ID,Source,x(=RA°),y(=Dec°),R,G,B,size")
    p.add_argument("--outdir", default="out", help="Output directory")
    p.add_argument("--outfile", default="constellations_from_csv.png", help="Output PNG filename")

    # Data selection
    p.add_argument("--top_n", type=int, default=1200,
                   help="Limit to N brightest stars by 'size' BEFORE per-constellation processing (None = all)")

    # Graph knobs
    p.add_argument("--knn_k", type=int, default=6, help="k for kNN inside each constellation")
    p.add_argument("--edge_pct", type=float, default=80.0,
                   help="Keep edges <= this percentile of length (per-constellation)")
    p.add_argument("--degree_cap", type=int, default=3, help="Max degree per node")

    # Visuals: stars & labels
    p.add_argument("--bg_star_scale", type=float, default=1.2, help="Multiply sizes for background stars")
    p.add_argument("--fg_star_scale", type=float, default=1.2, help="Multiply sizes for kept-component stars")
    p.add_argument("--bg_star_alpha", type=float, default=0.55, help="Alpha for background stars")
    p.add_argument("--fg_star_alpha", type=float, default=0.9, help="Alpha for kept-component stars")
    p.add_argument("--line_lw", type=float, default=0.9, help="Constellation line width")
    p.add_argument("--label_max", type=int, default=40, help="Max constellation labels to draw")
    p.add_argument("--label_dx", type=float, default=-5.0, help="Label offset in RA degrees (negative = left)")
    p.add_argument("--label_dy", type=float, default=1.5, help="Label offset in Dec degrees (up)")

    # (1) Location/time visibility
    p.add_argument("--apply_visibility", action="store_true",
                   help="If set, filter to stars above horizon for given observer/time.")
    p.add_argument("--observer_lat", type=float, default=0.0, help="Observer latitude in degrees")
    p.add_argument("--observer_lon", type=float, default=0.0, help="Observer longitude in degrees (East positive)")
    p.add_argument("--observer_elev_m", type=float, default=0.0, help="Observer elevation in meters")
    p.add_argument("--utc_time", type=str, default="2025-01-01T00:00:00",
                   help="UTC ISO time for visibility, e.g., 2025-01-01T00:00:00")

    # (3) BBox controls (now algorithmic)
    p.add_argument("--mask_pad_deg", type=float, default=3.0,
                   help="Padding (degrees) added to bbox around pass-1 largest component before pass-2 rebuild.")
    p.add_argument("--draw_masks", action="store_true", help="Draw bbox rectangles on the plot.")
    return p.parse_args()


# ------------- helpers -------------
def load_stars(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"ID", "Source", "x", "y", "R", "G", "B", "size"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return df


def normalize_xy_ra_dec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """
    Make a quasi-Euclidean plane for local neighbor finding.
    Scale RA by cos(mean Dec) to reduce distortion.
    """
    y = dec_deg.astype(float)
    x = ra_deg.astype(float)
    scale = np.cos(np.deg2rad(np.mean(y))) if len(y) else 1.0
    return np.column_stack([x * scale, y])


def unwrap_ra_local(ra_deg: np.ndarray) -> np.ndarray:
    """
    Reduce 0/360 wrap for a single constellation cluster.
    If span > 180°, shift the low side by +360.
    """
    ra = np.array(ra_deg, dtype=float)
    if len(ra) < 2:
        return ra
    span = np.max(ra) - np.min(ra)
    if span > 180.0:
        med = np.median(ra)
        mask = ra < med - 180.0
        ra[mask] += 360.0
    return ra


def edge_lengths(pts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) == 0:
        return np.array([])
    a = pts[edges[:, 0]]
    b = pts[edges[:, 1]]
    return np.linalg.norm(a - b, axis=1)


def prune_by_length(edges: np.ndarray, lengths: np.ndarray, pct: float) -> np.ndarray:
    if len(edges) == 0:
        return edges
    thr = np.percentile(lengths, pct)
    return edges[lengths <= thr]


def cap_degree(edges: np.ndarray, lengths: np.ndarray, max_deg: int) -> np.ndarray:
    """Remove longest incident edges until all node degrees <= max_deg."""
    if max_deg is None or len(edges) == 0:
        return edges
    E, L = edges.copy(), lengths.copy()
    while True:
        deg = {}
        for i, j in E:
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
        offenders = [n for n, d in deg.items() if d > max_deg]
        if not offenders:
            return E
        # build adjacency
        inc = {}
        for idx, (i, j) in enumerate(E):
            inc.setdefault(i, []).append(idx)
            inc.setdefault(j, []).append(idx)
        drop = set()
        for n in offenders:
            idxs = inc.get(n, [])
            if idxs:
                # drop the longest incident edge
                best = max(idxs, key=lambda t: L[t])
                drop.add(best)
        if not drop:
            return E
        keep = [k for k in range(len(E)) if k not in drop]
        E, L = E[keep], L[keep]


def largest_component_mask(n_nodes: int, edges: np.ndarray):
    if len(edges) == 0 or n_nodes == 0:
        return np.zeros(n_nodes, dtype=bool)
    data = np.ones(len(edges) * 2, dtype=int)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    A = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(A, directed=False)
    if n_comp <= 1:
        return np.ones(n_nodes, dtype=bool)
    counts = np.bincount(labels, minlength=n_comp)
    keep_label = np.argmax(counts)
    return labels == keep_label


def knn_edges(pts: np.ndarray, k: int) -> np.ndarray:
    if len(pts) < 2:
        return np.empty((0, 2), dtype=int)
    k = min(k, max(1, len(pts) - 1))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)  # include self
    I = np.repeat(np.arange(len(pts)), k)
    J = idx[:, 1:].reshape(-1)
    pairs = np.sort(np.column_stack([I, J]), axis=1)
    return np.unique(pairs, axis=0)


def filter_visible(df: pd.DataFrame, lat: float, lon: float, elev_m: float, utc_iso: str) -> pd.DataFrame:
    """Keep rows whose AltAz altitude > 0 deg at given Earth location/time."""
    if len(df) == 0:
        return df
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev_m * u.m)
    t = Time(utc_iso)
    altaz = AltAz(obstime=t, location=loc)
    sc = SkyCoord(ra=df["x"].to_numpy() * u.deg, dec=df["y"].to_numpy() * u.deg, frame="icrs")
    altaz_sc = sc.transform_to(altaz)
    mask = altaz_sc.alt.deg > 0.0
    return df.loc[mask].reset_index(drop=True)


def _apply_bbox_filter(ra_unw: np.ndarray, dec: np.ndarray, kept_local: np.ndarray, pad_deg: float):
    """
    Build a bbox on *unwrapped* RA around the kept component, padded by pad_deg,
    and return a boolean mask selecting nodes inside that bbox, plus bbox values.
    """
    min_ra_u = float(np.min(ra_unw[kept_local]))
    max_ra_u = float(np.max(ra_unw[kept_local]))
    min_dec = float(np.min(dec[kept_local]))
    max_dec = float(np.max(dec[kept_local]))

    min_ra = min_ra_u - pad_deg
    max_ra = max_ra_u + pad_deg
    min_d = min_dec - pad_deg
    max_d = max_dec + pad_deg

    in_box = (ra_unw >= min_ra) & (ra_unw <= max_ra) & (dec >= min_d) & (dec <= max_d)
    return in_box, (min_ra, max_ra, min_d, max_d)


def _plot_bbox_wrapped(ax, min_ra, max_ra, min_dec, max_dec, lw=0.6, alpha=0.5):
    """
    Draw bbox while respecting RA wrap for axes in [0,360] with ax.invert_xaxis().
    Splits rectangle if it crosses 0/360.
    """
    def add_rect(x0, x1):
        w = x1 - x0
        h = max_dec - min_dec
        rect = mpatches.Rectangle((x0, min_dec), w, h, fill=False, lw=lw, ec=(1, 1, 1), alpha=alpha, zorder=2)
        ax.add_patch(rect)

    # Normalize to [0, 360)
    a0 = min_ra % 360.0
    a1 = max_ra % 360.0

    if min_ra <= max_ra and (max_ra - min_ra) < 360:
        # may cross wrap if a0 > a1 after modulo
        if a0 <= a1:
            add_rect(a0, a1)
        else:
            # split into [0,a1] and [a0,360)
            add_rect(0.0, a1)
            add_rect(a0, 360.0)
    else:
        # degenerate: cover full RA
        add_rect(0.0, 360.0)


# ------------- pipeline -------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / args.outfile
    out_csv = outdir / "constellation_components.csv"
    out_masks = outdir / "masks.csv"

    # ---- Load ----
    df = load_stars(Path(args.csv))
    n_all = len(df)

    # (1) Optional visibility filtering before any constellation assignment
    if args.apply_visibility:
        df = filter_visible(df, args.observer_lat, args.observer_lon, args.observer_elev_m, args.utc_time)
    n_vis = len(df)

    # Background selection by brightness (guarantee stars to draw)
    if args.top_n is not None:
        df_bg = df.sort_values("size", ascending=False).head(args.top_n).reset_index(drop=True)
    else:
        df_bg = df.copy()
    n_bg = len(df_bg)

    # Assign IAU constellation per star by RA/Dec
    sc = SkyCoord(ra=df_bg["x"].to_numpy() * u.deg, dec=df_bg["y"].to_numpy() * u.deg, frame="icrs")
    df_bg["Constellation"] = get_constellation(sc)

    kept_rows = []
    edge_segments = []
    label_rows = []
    mask_rows = []  # (Constellation, min_ra, max_ra, min_dec, max_dec, pad_deg, n_kept, center_ra, center_dec)

    # ---- Per-constellation: pass-1 (all) -> bbox -> pass-2 (inside bbox) ----
    for cname, g in df_bg.groupby("Constellation"):
        if len(g) < 3:
            continue

        # First pass on full constellation
        ra_unw_all = unwrap_ra_local(g["x"].to_numpy(float))
        dec_all = g["y"].to_numpy(float)
        pts_all = normalize_xy_ra_dec(ra_unw_all, dec_all)

        E1 = knn_edges(pts_all, k=args.knn_k)
        if len(E1) == 0:
            continue
        L1 = edge_lengths(pts_all, E1)
        E1 = prune_by_length(E1, L1, args.edge_pct)
        if len(E1) == 0:
            continue
        L1 = edge_lengths(pts_all, E1)
        E1 = cap_degree(E1, L1, args.degree_cap)
        if len(E1) == 0:
            continue

        # Largest component (pass 1)
        mask_keep1 = largest_component_mask(len(pts_all), E1)
        if not mask_keep1.any():
            continue
        kept_local1 = np.where(mask_keep1)[0]

        # Build bbox around pass-1 kept component and FILTER nodes (algorithmic effect)
        in_box, (min_ra, max_ra, min_d, max_d) = _apply_bbox_filter(
            ra_unw_all, dec_all, kept_local1, pad_deg=float(args.mask_pad_deg)
        )
        idx_box_local = np.where(in_box)[0]
        if len(idx_box_local) < 3:
            continue  # not enough nodes to rebuild

        # Pass-2 subset
        ra_unw_2 = ra_unw_all[idx_box_local]
        dec_2 = dec_all[idx_box_local]
        pts_2 = normalize_xy_ra_dec(ra_unw_2, dec_2)

        E2 = knn_edges(pts_2, k=args.knn_k)
        if len(E2) == 0:
            continue
        L2 = edge_lengths(pts_2, E2)
        E2 = prune_by_length(E2, L2, args.edge_pct)
        if len(E2) == 0:
            continue
        L2 = edge_lengths(pts_2, E2)
        E2 = cap_degree(E2, L2, args.degree_cap)
        if len(E2) == 0:
            continue

        # Largest component (pass 2, INSIDE bbox) → this defines what we DRAW
        mask_keep2 = largest_component_mask(len(pts_2), E2)
        if not mask_keep2.any():
            continue
        kept_local2 = np.where(mask_keep2)[0]  # indices within bbox-subset
        kept_in_g = idx_box_local[kept_local2]  # indices within group g
        kept_global = g.index.values[kept_in_g]
        kept_rows.extend(kept_global.tolist())

        # Edges to draw (only edges whose endpoints are in kept set)
        g_xy = g[["x", "y"]].to_numpy(float)
        keep_set = set(kept_in_g.tolist())
        for i2, j2 in E2:
            gi = int(idx_box_local[i2])
            gj = int(idx_box_local[j2])
            if gi in keep_set and gj in keep_set:
                x0, y0 = g_xy[gi]; x1, y1 = g_xy[gj]
                edge_segments.append((x0, y0, x1, y1))

        # Label position: center of pass-2 kept nodes (original RA/Dec)
        cx = g_xy[list(keep_set), 0].mean()
        cy = g_xy[list(keep_set), 1].mean()
        label_rows.append((cname, cx + args.label_dx, cy + args.label_dy, len(keep_set)))

        # Save applied bbox
        mask_rows.append((cname, float(min_ra), float(max_ra), float(min_d), float(max_d),
                          float(args.mask_pad_deg), int(len(keep_set)), float(cx), float(cy)))

    # Foreground kept stars
    df_keep = df_bg.loc[sorted(set(kept_rows))].copy()

    # ---- Plot in RA/Dec ----
    plt.figure(figsize=(12, 8), facecolor="black")
    ax = plt.gca(); ax.set_facecolor("black")

    # 1) Background: ALL selected stars (faint), so you ALWAYS see stars
    if len(df_bg):
        bg_rgb = df_bg[["R", "G", "B"]].to_numpy() / 255.0
        bg_sizes = df_bg["size"].to_numpy() * args.bg_star_scale
        ax.scatter(df_bg["x"], df_bg["y"], s=bg_sizes, c=bg_rgb,
                   edgecolors="none", alpha=args.bg_star_alpha, zorder=1)

    # 2) Edges for kept pass-2 components
    for (x0, y0, x1, y1) in edge_segments:
        ax.plot([x0, x1], [y0, y1], lw=args.line_lw, alpha=0.98, color=(1, 1, 1), zorder=3)

    # 3) Foreground: kept stars (brighter)
    if len(df_keep):
        fg_rgb = df_keep[["R", "G", "B"]].to_numpy() / 255.0
        fg_sizes = df_keep["size"].to_numpy() * args.fg_star_scale
        ax.scatter(df_keep["x"], df_keep["y"], s=fg_sizes, c=fg_rgb,
                   edgecolors="none", alpha=args.fg_star_alpha, zorder=4)

    # 4) Labels (top-N by kept size)
    if label_rows:
        label_rows.sort(key=lambda t: t[3], reverse=True)
        for cname, lx, ly, _n in label_rows[:args.label_max]:
            ax.text(lx, ly, cname, color="white", fontsize=9,
                    ha=("right" if args.label_dx < 0 else "left"),
                    va="center", alpha=0.9, zorder=5)

    # 5) Optional: draw applied masks (plotted with RA wrap handling)
    if args.draw_masks and mask_rows:
        for cname, min_ra, max_ra, min_d, max_d, pad, n_kept, cx, cy in mask_rows:
            _plot_bbox_wrapped(ax, min_ra, max_ra, min_d, max_d, lw=0.9, alpha=0.6)

    # Cosmetics
    ax.set_xlabel("Right Ascension (deg)", color="white")
    ax.set_ylabel("Declination (deg)", color="white")
    ax.tick_params(colors="white")
    title_bits = ["Largest Connected Component per Constellation — RA/Dec", "(bbox-constrained)"]
    if args.apply_visibility:
        title_bits.append("(visibility filtered)")
    ax.set_title(" ".join(title_bits), color="white", pad=12)
    ax.grid(alpha=0.15, color="white", linestyle=":")
    ax.invert_xaxis()  # mimic sky view


    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()

    # Save CSV of kept stars (foreground)
    if len(df_keep):
        df_keep[["ID", "Source", "x", "y", "Constellation"]].to_csv(out_csv, index=False)
    else:
        pd.DataFrame(columns=["ID", "Source", "x", "y", "Constellation"]).to_csv(out_csv, index=False)

    # Save masks
    cols = ["Constellation", "min_ra", "max_ra", "min_dec", "max_dec", "pad_deg", "n_kept", "center_ra", "center_dec"]
    pd.DataFrame(mask_rows, columns=cols).to_csv(out_masks, index=False)

    # Logs
    print(f"Total stars (input): {n_all}")
    if args.apply_visibility:
        print(f"Visible at lat={args.observer_lat}, lon={args.observer_lon}, elev(m)={args.observer_elev_m}, utc={args.utc_time}: {n_vis}")
    print(f"Background stars plotted (after top_n): {n_bg}")
    print(f"Kept stars (bbox-constrained largest components): {len(df_keep)}")
    print(f"Saved plot: {out_png.resolve()}")
    print(f"Saved stars CSV:  {out_csv.resolve()}")
    print(f"Saved masks CSV:  {out_masks.resolve()}")

def run_constellation_pipeline(
    csv_path,
    top_n=1200,
    apply_visibility=False,
    lat=0.0,
    lon=0.0,
    elev_m=0.0,
    utc_time="2025-01-01T00:00:00",
    mask_pad_deg=3.0,
    draw_masks=True,
    knn_k=6,
    edge_pct=80.0,
    degree_cap=3,
):
    """
    Runs the constellation-building pipeline and returns:
      fig     : Matplotlib Figure object
      df_keep : DataFrame of kept stars (foreground)
      masks_df: DataFrame of applied bounding boxes
      stats   : dict of key counts
    """

    import matplotlib.pyplot as plt
    from pathlib import Path

    if isinstance(csv_path, pd.DataFrame):
        df = csv_path.copy()
    else:
        df = load_stars(Path(csv_path))

    # ---- Load stars ----
    # df = load_stars(Path(csv_path))
    n_all = len(df)

    # ---- Visibility filter ----
    if apply_visibility:
        df = filter_visible(df, lat, lon, elev_m, utc_time)
    n_vis = len(df)

    # Higher elevation = clearer air = brighter stars
    if elev_m is not None and elev_m > 0:
        # Linear scale: +25% brightness per 1000m
        brightness_factor = 1 + 0.0005 * elev_m
        df["size"] = df["size"] * brightness_factor

    # ---- Limit to top_n by brightness ----
    if top_n is not None:
        df_bg = df.sort_values("size", ascending=False).head(top_n).reset_index(drop=True)
    else:
        df_bg = df.copy()
    n_bg = len(df_bg)

    # ---- Assign constellation ----
    sc = SkyCoord(ra=df_bg["x"].to_numpy()*u.deg,
                  dec=df_bg["y"].to_numpy()*u.deg,
                  frame="icrs")
    df_bg["Constellation"] = get_constellation(sc)

    kept_rows, edge_segments, label_rows, mask_rows = [], [], [], []

    # ---- Per-constellation algorithmic bbox ----
    for cname, g in df_bg.groupby("Constellation"):
        if len(g) < 3:
            continue

        ra_unw_all = unwrap_ra_local(g["x"].to_numpy(float))
        dec_all = g["y"].to_numpy(float)
        pts_all = normalize_xy_ra_dec(ra_unw_all, dec_all)

        E1 = knn_edges(pts_all, k=knn_k)
        if len(E1) == 0:
            continue
        L1 = edge_lengths(pts_all, E1)
        E1 = prune_by_length(E1, L1, edge_pct)
        if len(E1) == 0:
            continue
        L1 = edge_lengths(pts_all, E1)
        E1 = cap_degree(E1, L1, degree_cap)
        if len(E1) == 0:
            continue

        mask_keep1 = largest_component_mask(len(pts_all), E1)
        if not mask_keep1.any():
            continue
        kept_local1 = np.where(mask_keep1)[0]

        in_box, (min_ra, max_ra, min_d, max_d) = _apply_bbox_filter(
            ra_unw_all, dec_all, kept_local1, pad_deg=float(mask_pad_deg)
        )
        idx_box_local = np.where(in_box)[0]
        if len(idx_box_local) < 3:
            continue

        ra_unw_2 = ra_unw_all[idx_box_local]
        dec_2 = dec_all[idx_box_local]
        pts_2 = normalize_xy_ra_dec(ra_unw_2, dec_2)

        E2 = knn_edges(pts_2, k=knn_k)
        if len(E2) == 0:
            continue
        L2 = edge_lengths(pts_2, E2)
        E2 = prune_by_length(E2, L2, edge_pct)
        if len(E2) == 0:
            continue
        L2 = edge_lengths(pts_2, E2)
        E2 = cap_degree(E2, L2, degree_cap)
        if len(E2) == 0:
            continue

        mask_keep2 = largest_component_mask(len(pts_2), E2)
        if not mask_keep2.any():
            continue
        kept_local2 = np.where(mask_keep2)[0]
        kept_in_g = idx_box_local[kept_local2]
        kept_global = g.index.values[kept_in_g]
        kept_rows.extend(kept_global.tolist())

        g_xy = g[["x", "y"]].to_numpy(float)
        keep_set = set(kept_in_g.tolist())
        for i2, j2 in E2:
            gi = int(idx_box_local[i2])
            gj = int(idx_box_local[j2])
            if gi in keep_set and gj in keep_set:
                x0, y0 = g_xy[gi]; x1, y1 = g_xy[gj]
                edge_segments.append((x0, y0, x1, y1))

        cx = g_xy[list(keep_set), 0].mean()
        cy = g_xy[list(keep_set), 1].mean()
        label_rows.append((cname, cx, cy, len(keep_set)))
        mask_rows.append((cname, min_ra, max_ra, min_d, max_d,
                          mask_pad_deg, len(keep_set), cx, cy))

    # ---- Foreground ----
    df_keep = df_bg.loc[sorted(set(kept_rows))].copy()
    masks_df = pd.DataFrame(mask_rows, columns=[
        "Constellation", "min_ra", "max_ra", "min_dec", "max_dec",
        "pad_deg", "n_kept", "center_ra", "center_dec"
    ])

    # ---- Plot in RA/Dec ----
    fig, ax = plt.subplots(figsize=(8.0, 5.12), facecolor="black")
    ax.set_facecolor("black")

    # ---- Visual constants (copied from argparse defaults) ----
    bg_star_scale = 1.5       # Background star size multiplier
    fg_star_scale = 2       # Foreground star size multiplier
    bg_star_alpha = 0.8      # Transparency for background stars
    fg_star_alpha = 1       # Transparency for foreground stars
    line_lw = 0.9             # Line width for constellation edges
    label_max = 40            # Max number of constellation labels
    label_dx = -5.0           # Label offset (RA degrees)
    label_dy = 1.5            # Label offset (Dec degrees)

    # 1) Background: ALL selected stars (faint)
    if len(df_bg):
        bg_rgb = df_bg[["R", "G", "B"]].to_numpy() / 255.0
        bg_sizes = df_bg["size"].to_numpy() * bg_star_scale
        ax.scatter(df_bg["x"], df_bg["y"], s=bg_sizes, c=bg_rgb,
                   edgecolors="none", alpha=bg_star_alpha, zorder=1)

    # 2) Edges for kept pass-2 components
    for (x0, y0, x1, y1) in edge_segments:
        ax.plot([x0, x1], [y0, y1], lw=line_lw, alpha=0.98, color=(1, 1, 1), zorder=3)

    # 3) Foreground: kept stars (brighter)
    if len(df_keep):
        fg_rgb = df_keep[["R", "G", "B"]].to_numpy() / 255.0
        fg_sizes = df_keep["size"].to_numpy() * fg_star_scale
        ax.scatter(df_keep["x"], df_keep["y"], s=fg_sizes, c=fg_rgb,
                   edgecolors="none", alpha=fg_star_alpha, zorder=4)

    # 4) Labels (top-N by kept size)
    if label_rows:
        label_rows.sort(key=lambda t: t[3], reverse=True)
        for cname, lx, ly, _n in label_rows[:label_max]:
            ax.text(lx, ly, cname, color="white", fontsize=9,
                    ha=("right" if label_dx < 0 else "left"),
                    va="center", alpha=0.9, zorder=5)

    # 5) Optional: draw applied masks (bounding boxes)
    if draw_masks:
        for cname, min_ra, max_ra, min_d, max_d, pad, n_kept, cx, cy in mask_rows:
            _plot_bbox_wrapped(ax, min_ra, max_ra, min_d, max_d, lw=0.9, alpha=0.6)

    # Cosmetics
    ax.set_xlabel("Right Ascension (deg)", color="white")
    ax.set_ylabel("Declination (deg)", color="white")
    ax.tick_params(colors="white")
    title_bits = ["Largest Connected Component per Constellation — RA/Dec", "(bbox-constrained)"]
    if apply_visibility:
        title_bits.append("(visibility filtered)")
    ax.set_title(" ".join(title_bits), color="white", pad=12)
    ax.grid(alpha=0.15, color="white", linestyle=":")
    ax.invert_xaxis()  # mimic sky view
    
    ax.set_xlim(360, 0)   # because invert_xaxis flips RA
    ax.set_ylim(-90, 90)

    plt.tight_layout()

    # ---- Prepare outputs for Streamlit ----
    masks_df = pd.DataFrame(mask_rows, columns=[
        "Constellation", "min_ra", "max_ra", "min_dec", "max_dec",
        "pad_deg", "n_kept", "center_ra", "center_dec"
    ])

    stats = {
        "total_stars": n_all,
        "visible_stars": n_vis,
        "background": len(df_bg),
        "kept": len(df_keep),
        "constellations": len(set(df_keep["Constellation"])) if len(df_keep) else 0
    }

    return fig, df_keep, masks_df, stats

if __name__ == "__main__":
    main()
