#!/usr/bin/env python3
# all_sky_from_csv.py (spherical + brightness-aware)
# Build an all-sky (Mollweide) constellation map using ONLY stars_plot_ready.csv.
# Background = all stars; Foreground = largest (brightness-weighted) component per IAU constellation.
# Requires: astropy, numpy, pandas, matplotlib, scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord, get_constellation

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csgraph

# ---------------- helpers ----------------
def load_stars(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"ID", "Source", "x", "y", "R", "G", "B", "size"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return df

# spherical geometry
def _sph_xyz(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg); dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])

def knn_edges_sphere(ra_deg, dec_deg, k):
    xyz = _sph_xyz(ra_deg, dec_deg)
    if len(xyz) < 2:
        return np.empty((0, 2), dtype=int)
    k = min(k, max(1, len(xyz) - 1))
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)  # include self
    I = np.repeat(np.arange(len(xyz)), k)
    J = idx[:, 1:].reshape(-1)
    pairs = np.sort(np.column_stack([I, J]), axis=1)
    return np.unique(pairs, axis=0)

def edge_angles_deg(ra_deg, dec_deg, edges):
    if len(edges) == 0:
        return np.array([])
    v = _sph_xyz(ra_deg, dec_deg)
    a = v[edges[:, 0]]
    b = v[edges[:, 1]]
    cosang = np.einsum("ij,ij->i", a, b).clip(-1.0, 1.0)
    return np.rad2deg(np.arccos(cosang))

def prune_by_percentile(edges, ra_deg, dec_deg, pct):
    if len(edges) == 0:
        return edges
    L = edge_angles_deg(ra_deg, dec_deg, edges)
    thr = np.percentile(L, pct)
    return edges[L <= thr]

def prune_by_maxdeg(edges, ra_deg, dec_deg, max_deg):
    if len(edges) == 0:
        return edges
    L = edge_angles_deg(ra_deg, dec_deg, edges)
    return edges[L <= max_deg]

def cap_degree_brightness(edges, ra_deg, dec_deg, size, max_deg):
    if max_deg is None or len(edges) == 0:
        return edges
    E = edges.copy()
    while True:
        deg = {}
        for i, j in E:
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
        offenders = [n for n, d in deg.items() if d > max_deg]
        if not offenders:
            return E
        # edge incidence
        inc = {}
        for idx, (i, j) in enumerate(E):
            inc.setdefault(i, []).append(idx)
            inc.setdefault(j, []).append(idx)
        to_drop = set()
        for n in offenders:
            idxs = inc.get(n, [])
            if not idxs:
                continue
            worst_idx, worst_score = None, -1.0
            for e in idxs:
                i, j = E[e]
                other = j if i == n else i
                # angular length (deg) penalized by brightness of other node
                L = edge_angles_deg(ra_deg, dec_deg, E[[e]])[0]
                score = L * (1.0 / max(float(size[other]), 1e-6))
                if score > worst_score:
                    worst_idx, worst_score = e, score
            if worst_idx is not None:
                to_drop.add(worst_idx)
        if not to_drop:
            return E
        keep = [k for k in range(len(E)) if k not in to_drop]
        E = E[keep]

def brightest_component_mask(n_nodes, edges, size):
    if len(edges) == 0 or n_nodes == 0:
        return np.zeros(n_nodes, dtype=bool)
    data = np.ones(len(edges) * 2, dtype=int)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    A = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    n_comp, labels = csgraph.connected_components(A, directed=False)
    best_mask = np.zeros(n_nodes, dtype=bool)
    best_sum = -1.0
    for lab in range(n_comp):
        m = (labels == lab)
        s = float(size[m].sum())
        if s > best_sum:
            best_sum = s
            best_mask = m
    return best_mask

# Mollweide projection (RA increases to the left)
def ra_dec_to_mollweide(ra_deg: np.ndarray, dec_deg: np.ndarray):
    x = np.deg2rad(-(ra_deg - 180.0))
    y = np.deg2rad(dec_deg)
    x = (x + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    return x, y

# ---------------- main ----------------
def main():
    CSV = "stars_plot_ready.csv"
    OUTDIR = Path("out"); OUTDIR.mkdir(parents=True, exist_ok=True)
    OUTPNG = OUTDIR / "all_sky_mollweide.png"

    # Tunables (Gemini-friendly defaults)
    TOP_N       = 5000
    KNN_K       = 8
    EDGE_PCT    = 70.0   # keep shortest this percentile
    EDGE_MAX_DEG= 12.0   # absolute max edge on sky
    DEGREE_CAP  = 3
    LABEL_MAX   = 40

    df = load_stars(Path(CSV)).sort_values("size", ascending=False).head(TOP_N).reset_index(drop=True)

    # IAU constellation per star
    sc = SkyCoord(ra=df["x"].to_numpy()*u.deg, dec=df["y"].to_numpy()*u.deg, frame="icrs")
    df["Constellation"] = get_constellation(sc)

    # Build edges per constellation using spherical/brightness-aware pipeline
    edge_segments = []
    labels = []

    for cname, g in df.groupby("Constellation"):
        if len(g) < 3:
            continue

        ra = g["x"].to_numpy(float)
        dec = g["y"].to_numpy(float)
        size = g["size"].to_numpy(float)

        # 1) spherical kNN
        E = knn_edges_sphere(ra, dec, k=KNN_K)
        if len(E) == 0: continue

        # 2) percentile prune (short edges kept)
        E = prune_by_percentile(E, ra, dec, EDGE_PCT)
        if len(E) == 0: continue

        # 3) absolute max sky length
        E = prune_by_maxdeg(E, ra, dec, EDGE_MAX_DEG)
        if len(E) == 0: continue

        # 4) degree cap with brightness bias
        E = cap_degree_brightness(E, ra, dec, size, DEGREE_CAP)
        if len(E) == 0: continue

        # 5) choose brightness-weighted largest component
        mask_keep = brightest_component_mask(len(ra), E, size)
        if not mask_keep.any(): continue

        kept = np.where(mask_keep)[0]
        g_xy = g[["x", "y"]].to_numpy(float)

        # draw only edges fully inside kept component
        keep_set = set(kept.tolist())
        for i, j in E:
            if i in keep_set and j in keep_set:
                x0, y0 = g_xy[i]; x1, y1 = g_xy[j]
                edge_segments.append((x0, y0, x1, y1))

        # label center (brightness-weighted)
        w = size[kept] / (size[kept].sum() or 1.0)
        cx = (g_xy[kept, 0] * w).sum()
        cy = (g_xy[kept, 1] * w).sum()
        labels.append((cname, cx, cy, len(kept)))

    # -------- plot (Mollweide) --------
    fig = plt.figure(figsize=(12, 6), facecolor="black")
    ax = plt.subplot(111, projection="mollweide"); ax.set_facecolor("black")

    # background stars
    bx, by = ra_dec_to_mollweide(df["x"].to_numpy(), df["y"].to_numpy())
    bg_rgb = df[["R", "G", "B"]].to_numpy()/255.0
    bg_sizes = df["size"].to_numpy() * 0.55
    ax.scatter(bx, by, s=bg_sizes, c=bg_rgb, alpha=0.35, edgecolors="none", zorder=1)

    # edges
    for (ra0, dec0, ra1, dec1) in edge_segments:
        x0, y0 = ra_dec_to_mollweide(np.array([ra0]), np.array([dec0]))
        x1, y1 = ra_dec_to_mollweide(np.array([ra1]), np.array([dec1]))
        ax.plot([x0[0], x1[0]], [y0[0], y1[0]], lw=0.9, color="lime", alpha=0.9, zorder=3)

    # labels (top by kept size)
    labels.sort(key=lambda t: t[3], reverse=True)
    for cname, cx, cy, _n in labels[:LABEL_MAX]:
        lx, ly = ra_dec_to_mollweide(np.array([cx]), np.array([cy]))
        ax.text(lx[0], ly[0], cname, color="white", fontsize=7,
                ha="center", va="center", alpha=0.9, zorder=5)

    ax.grid(color="white", alpha=0.15, linestyle=":")
    ax.set_title("All-Sky Constellation Sticks — Mollweide (spherical kNN, brightness-aware)", color="white", pad=12)
    for spine in ax.spines.values(): spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(OUTPNG, dpi=300, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUTPNG.resolve())

if __name__ == "__main__":
    main()
