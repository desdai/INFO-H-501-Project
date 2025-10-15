# plot_constellations_from_csv.py (fixed)
# -----------------------------------------------------------
# Build constellation "stick" drawings from stars_plot_ready.csv only,
# no geolocation. ALWAYS plots stars (background), then overlays the
# largest connected component per constellation (edges + brighter stars),
# and labels shifted to the left to avoid overlaps.
#
# Requires: astropy, numpy, pandas, matplotlib, scipy
# -----------------------------------------------------------



"""
python plot_real_from_csv.py \
  --csv stars_plot_ready.csv \
  --top_n 2000 \
  --knn_k 6 --edge_pct 80 --degree_cap 3 \
  --label_dx -15 --label_dy 1.5 \
  --bg_star_scale 1.0 --fg_star_scale 1.2

dataset, top_n stars, parameter for ploting graph, label location, star visibility
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord, get_constellation

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot constellation-like connections using only RA/Dec (no geolocation)."
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
    p.add_argument("--bg_star_scale", type=float, default=0.6, help="Multiply sizes for background stars")
    p.add_argument("--fg_star_scale", type=float, default=1.2, help="Multiply sizes for kept-component stars")
    p.add_argument("--bg_star_alpha", type=float, default=0.35, help="Alpha for background stars")
    p.add_argument("--fg_star_alpha", type=float, default=0.9, help="Alpha for kept-component stars")
    p.add_argument("--line_lw", type=float, default=0.9, help="Constellation line width")
    p.add_argument("--label_max", type=int, default=40, help="Max constellation labels to draw")
    p.add_argument("--label_dx", type=float, default=-5.0, help="Label offset in RA degrees (negative = left)")
    p.add_argument("--label_dy", type=float, default=1.5, help="Label offset in Dec degrees (up)")
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
    a = pts[edges[:,0]]
    b = pts[edges[:,1]]
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
    data = np.ones(len(edges)*2, dtype=int)
    rows = np.concatenate([edges[:,0], edges[:,1]])
    cols = np.concatenate([edges[:,1], edges[:,0]])
    A = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(A, directed=False)
    if n_comp <= 1:
        return np.ones(n_nodes, dtype=bool)
    counts = np.bincount(labels, minlength=n_comp)
    keep_label = np.argmax(counts)
    return labels == keep_label

def knn_edges(pts: np.ndarray, k: int) -> np.ndarray:
    if len(pts) < 2:
        return np.empty((0,2), dtype=int)
    k = min(k, max(1, len(pts)-1))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k+1)  # include self
    I = np.repeat(np.arange(len(pts)), k)
    J = idx[:,1:].reshape(-1)
    pairs = np.sort(np.column_stack([I, J]), axis=1)
    return np.unique(pairs, axis=0)

# ------------- pipeline -------------
def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / args.outfile
    out_csv = outdir / "constellation_components.csv"

    # ---- Load & preselect ----
    df = load_stars(Path(args.csv))

    # ALWAYS have some stars to draw: background = all (or top_n) stars
    if args.top_n is not None:
        df_bg = df.sort_values("size", ascending=False).head(args.top_n).reset_index(drop=True)
    else:
        df_bg = df.copy()

    # Assign IAU constellation per star by RA/Dec
    sc = SkyCoord(ra=df_bg["x"].to_numpy()*u.deg, dec=df_bg["y"].to_numpy()*u.deg, frame="icrs")
    df_bg["Constellation"] = get_constellation(sc)

    kept_rows = []
    edge_segments = []
    label_rows = []

    # ---- Per-constellation graph → largest component ----
    for cname, g in df_bg.groupby("Constellation"):
        if len(g) < 3:
            continue

        # unwrap RA locally & build quasi-Euclidean coords
        ra_unw = unwrap_ra_local(g["x"].to_numpy(float))
        dec    = g["y"].to_numpy(float)
        pts    = normalize_xy_ra_dec(ra_unw, dec)

        # edges
        E = knn_edges(pts, k=args.knn_k)
        if len(E) == 0:
            continue

        L = edge_lengths(pts, E)
        E = prune_by_length(E, L, args.edge_pct)
        if len(E) == 0:
            continue

        L = edge_lengths(pts, E)
        E = cap_degree(E, L, args.degree_cap)
        if len(E) == 0:
            continue

        # largest component
        mask_keep = largest_component_mask(len(pts), E)
        if not mask_keep.any():
            continue

        kept_local = np.where(mask_keep)[0]
        kept_global = g.index.values[kept_local]
        kept_rows.extend(kept_global.tolist())

        # edges to draw (use original RA/Dec for display)
        g_xy = g[["x","y"]].to_numpy(float)
        for i, j in E:
            if mask_keep[i] and mask_keep[j]:
                x0, y0 = g_xy[i]; x1, y1 = g_xy[j]
                edge_segments.append((x0, y0, x1, y1))

        # label position (shift left & up)
        cx = g_xy[kept_local, 0].mean()
        cy = g_xy[kept_local, 1].mean()
        label_rows.append((cname, cx + args.label_dx, cy + args.label_dy, len(kept_local)))

    # Foreground kept stars
    df_keep = df_bg.loc[sorted(set(kept_rows))].copy()

    # ---- Plot in RA/Dec ----
    plt.figure(figsize=(12, 8), facecolor="black")
    ax = plt.gca(); ax.set_facecolor("black")

    # 1) Background: ALL selected stars (faint), so you ALWAYS see stars
    bg_rgb = df_bg[["R","G","B"]].to_numpy()/255.0
    bg_sizes = df_bg["size"].to_numpy() * args.bg_star_scale
    ax.scatter(df_bg["x"], df_bg["y"], s=bg_sizes, c=bg_rgb,
               edgecolors="none", alpha=args.bg_star_alpha, zorder=1)

    # 2) Edges for largest components
    for (x0,y0,x1,y1) in edge_segments:
        ax.plot([x0,x1],[y0,y1], lw=args.line_lw, alpha=0.98, color=(1,1,1), zorder=3)

    # 3) Foreground: kept stars (brighter)
    if len(df_keep):
        fg_rgb = df_keep[["R","G","B"]].to_numpy()/255.0
        fg_sizes = df_keep["size"].to_numpy() * args.fg_star_scale
        ax.scatter(df_keep["x"], df_keep["y"], s=fg_sizes, c=fg_rgb,
                   edgecolors="none", alpha=args.fg_star_alpha, zorder=4)

    # 4) Labels (to the LEFT by default)
    if label_rows:
        label_rows.sort(key=lambda t: t[3], reverse=True)
        for cname, lx, ly, _n in label_rows[:args.label_max]:
            ax.text(lx, ly, cname, color="white", fontsize=9,
                    ha=("right" if args.label_dx < 0 else "left"),
                    va="center", alpha=0.9, zorder=5)

    # Cosmetics
    ax.set_xlabel("Right Ascension (deg)", color="white")
    ax.set_ylabel("Declination (deg)", color="white")
    ax.tick_params(colors="white")
    ax.set_title("Largest Connected Component per Constellation — RA/Dec", color="white", pad=12)
    ax.grid(alpha=0.15, color="white", linestyle=":")
    ax.invert_xaxis()  # optional: mimic sky view

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()

    # Save CSV of kept stars (foreground)
    if len(df_keep):
        df_keep[["ID","Source","x","y","Constellation"]].to_csv(out_csv, index=False)
    else:
        pd.DataFrame(columns=["ID","Source","x","y","Constellation"]).to_csv(out_csv, index=False)

    print(f"Background stars plotted: {len(df_bg)}")
    print(f"Kept stars (largest components across constellations): {len(df_keep)}")
    print(f"Saved plot: {out_png.resolve()}")
    print(f"Saved CSV:  {out_csv.resolve()}")

if __name__ == "__main__":
    main()
