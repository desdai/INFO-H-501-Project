# constellation_drawings.py
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# =========================
# Config (tune these)
# =========================
CSV_PATH   = "stars_plot_ready.csv"   # expects: ID,Source,x,y,R,G,B,size
OUT_DIR    = "constellations_out"

# keep only the brightest N stars (by 'size' from your CSV)
TOP_N      = 2000

# k for kNN candidate edges (used in MST and MSF)
KNN_K      = 6

# prune long edges by length percentile (smaller = sparser)
EDGE_PCT   = 80   # keep edges <= 80th percentile length

# cap node degree to avoid hairy junctions (None to disable)
DEGREE_CAP = 3

# RNG/GG tolerances to avoid brittle decisions
EPS        = 1e-9

# brightness weighting in MSF: weight = dist / (brightness_i + brightness_j)^ALPHA
ALPHA      = 0.6


# =========================
# Utility helpers
# =========================
def normalize_xy(xy):
    """Scale x by cos(mean(y)) to reduce RA/Dec distortion on a plane."""
    x = xy[:, 0]
    y = xy[:, 1]
    y_mean = np.deg2rad(np.mean(y))
    x_scaled = x * np.cos(y_mean)
    return np.column_stack([x_scaled, y])

def edge_lengths(xy, edges):
    """Compute Euclidean length for each (i,j) edge."""
    pi = xy[edges[:, 0]]
    pj = xy[edges[:, 1]]
    return np.linalg.norm(pi - pj, axis=1)

def prune_by_length(edges, lengths, pct=80):
    """Keep only edges whose length <= percentile p."""
    if len(lengths) == 0:
        return edges
    thresh = np.percentile(lengths, pct)
    mask = lengths <= thresh
    return edges[mask]

def cap_degree(edges, max_deg):
    """Iteratively remove the longest incident edges until all degrees <= max_deg."""
    if max_deg is None:
        return edges
    if len(edges) == 0:
        return edges
    # Work on a copy with lengths cached
    e = edges.copy()
    return _cap_degree_impl(e, max_deg)

def _cap_degree_impl(edges, max_deg):
    # Count degree
    while True:
        if len(edges) == 0: 
            return edges
        deg = {}
        for i, j in edges:
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
        offenders = [n for n, d in deg.items() if d > max_deg]
        if not offenders:
            return edges
        # For each offender, drop its longest incident edge
        # Compute lengths once per loop
        # (We need coords; stash in closure or compute when called—here we pass via global)
        # Instead, compute using a temporary trick: mark lengths as 1 for now; we'll remove far edges by a proxy:
        # Better approach: remove oldest edge; but we want "longest". We'll require a lengths array. 
        # To keep function independent, we do a heuristic: remove the edge with the largest index difference.
        # However, for quality, we can ask caller to sort by length desc before cap_degree.
        # Here: assume edges already sorted by length ascending; we remove from the end first.
        # Remove one edge per offender per loop
        to_remove_idx = set()
        # Build adjacency map edge indices per node
        adj = {}
        for idx, (i, j) in enumerate(edges):
            adj.setdefault(i, []).append(idx)
            adj.setdefault(j, []).append(idx)
        # Remove longest (highest index) for each offender
        for node in offenders:
            idxs = adj.get(node, [])
            if idxs:
                to_remove_idx.add(max(idxs))
        if not to_remove_idx:
            return edges
        keep = [k for k in range(len(edges)) if k not in to_remove_idx]
        edges = edges[keep]

def unique_edges(pairs):
    """Sort node indices in each pair and remove duplicates."""
    if len(pairs) == 0:
        return pairs
    a = np.sort(pairs, axis=1)
    a = np.unique(a, axis=0)
    return a

def sort_edges_by_length(xy, edges):
    """Return edges sorted by ascending geometric length."""
    L = edge_lengths(xy, edges)
    order = np.argsort(L)
    return edges[order], L[order]


# =========================
# Candidate edge builders
# =========================
def knn_edges(xy, k=6):
    tree = cKDTree(xy)
    d, idx = tree.query(xy, k=k+1)  # first neighbor is itself
    I = np.repeat(np.arange(len(xy)), k)
    J = idx[:, 1:].reshape(-1)
    E = np.column_stack([I, J])
    return unique_edges(E)

def delaunay_edges(xy):
    tri = Delaunay(xy)
    # Each simplex has 3 edges
    simplices = tri.simplices
    e = np.vstack([
        simplices[:, [0, 1]],
        simplices[:, [1, 2]],
        simplices[:, [0, 2]],
    ])
    return unique_edges(e)

def gabriel_graph_edges(xy, candidate_edges=None, eps=1e-9):
    """Keep ij if the disk with ij as diameter contains no other point."""
    if candidate_edges is None:
        candidate_edges = delaunay_edges(xy)
    tree = cKDTree(xy)
    keep = []
    for i, j in candidate_edges:
        ci = xy[i]; cj = xy[j]
        c = (ci + cj) / 2.0
        r = np.linalg.norm(ci - cj) / 2.0 + eps
        # count points in the open disk
        idxs = tree.query_ball_point(c, r)
        # allow i and j only
        if len(idxs) <= 2:
            keep.append([i, j])
    return np.array(keep, dtype=int)

def rng_edges(xy, candidate_edges=None, eps=1e-9):
    """
    Relative Neighborhood Graph:
    keep ij if there is no k such that max(d(i,k), d(j,k)) < d(i,j)
    """
    if candidate_edges is None:
        candidate_edges = delaunay_edges(xy)
    tree = cKDTree(xy)
    keep = []
    for i, j in candidate_edges:
        pi = xy[i]; pj = xy[j]
        dij = np.linalg.norm(pi - pj)
        # candidates near i within dij
        idxs = tree.query_ball_point(pi, dij - eps)
        ok = True
        for k in idxs:
            if k == i or k == j:
                continue
            if np.linalg.norm(xy[j] - xy[k]) < dij - eps:
                ok = False
                break
        if ok:
            keep.append([i, j])
    return np.array(keep, dtype=int)


# =========================
# Graph constructors (5 variants)
# =========================
def graph_mst_knn(xy, k=KNN_K, pct=EDGE_PCT, degree_cap=DEGREE_CAP):
    # 1) kNN edges
    E = knn_edges(xy, k)
    # MST on this sparse graph
    L = edge_lengths(xy, E)
    n = len(xy)
    W = coo_matrix((L, (E[:, 0], E[:, 1])), shape=(n, n))
    W = W + W.T
    mst = minimum_spanning_tree(W).tocoo()
    E_mst = np.column_stack([mst.row, mst.col])
    E_mst, L_mst = sort_edges_by_length(xy, E_mst)
    # prune + degree cap
    E_mst = prune_by_length(E_mst, L_mst, pct)
    E_mst = cap_degree(E_mst, degree_cap)
    return E_mst

def graph_gabriel(xy, pct=EDGE_PCT, degree_cap=DEGREE_CAP):
    cand = delaunay_edges(xy)
    E = gabriel_graph_edges(xy, cand, eps=EPS)
    E, L = sort_edges_by_length(xy, E)
    E = prune_by_length(E, L, pct)
    E = cap_degree(E, degree_cap)
    return E

def graph_rng(xy, pct=EDGE_PCT, degree_cap=DEGREE_CAP):
    cand = delaunay_edges(xy)
    E = rng_edges(xy, cand, eps=EPS)
    E, L = sort_edges_by_length(xy, E)
    E = prune_by_length(E, L, pct)
    E = cap_degree(E, degree_cap)
    return E

def graph_delaunay_pruned(xy, pct=EDGE_PCT, degree_cap=DEGREE_CAP):
    E = delaunay_edges(xy)
    E, L = sort_edges_by_length(xy, E)
    E = prune_by_length(E, L, pct)
    E = cap_degree(E, degree_cap)
    return E

def graph_brightness_msf(xy, brightness, k=KNN_K, pct=EDGE_PCT, degree_cap=DEGREE_CAP, alpha=ALPHA):
    """
    Weighted MSF biased to connect bright stars:
    weight = dist / (brightness_i + brightness_j)^alpha
    """
    E = knn_edges(xy, k)
    pi = xy[E[:, 0]]
    pj = xy[E[:, 1]]
    dij = np.linalg.norm(pi - pj, axis=1)
    bsum = (brightness[E[:, 0]] + brightness[E[:, 1]]) ** alpha
    w = dij / np.maximum(bsum, 1e-9)

    n = len(xy)
    W = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(n, n))
    W = W + W.T
    mst = minimum_spanning_tree(W).tocoo()
    E_mst = np.column_stack([mst.row, mst.col])
    # prune by geometric length, not weight
    E_mst, L_mst = sort_edges_by_length(xy, E_mst)
    E_mst = prune_by_length(E_mst, L_mst, pct)
    E_mst = cap_degree(E_mst, degree_cap)
    return E_mst


# =========================
# Plotting
# =========================
def plot_constellation(xy_raw, rgb, sizes, edges, title, out_path):
    plt.figure(figsize=(10, 8), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")
    # stars
    ax.scatter(xy_raw[:, 0], xy_raw[:, 1],
               s=sizes, c=rgb, edgecolors="none", alpha=0.9)
    # lines
    for i, j in edges:
        x0, y0 = xy_raw[i]
        x1, y1 = xy_raw[j]
        ax.plot([x0, x1], [y0, y1], lw=0.6, alpha=0.9, color=(0.95, 0.95, 0.95))
    ax.set_title(title, color="white", pad=12)
    ax.invert_xaxis()  # sky convention
    ax.axis("off")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved: {out_path.resolve()}")


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(CSV_PATH)

    # Select top-N brightest by 'size' (already a brightness proxy)
    df = df.sort_values("size", ascending=False).head(TOP_N).reset_index(drop=True)

    # Prepare arrays
    xy = df[["x", "y"]].to_numpy(float)
    rgb = (df[["R", "G", "B"]].to_numpy(float) / 255.0)
    sizes = df["size"].astype(float).to_numpy()
    # brightness proxy for MSF
    brightness = sizes.copy()

    # Geometry normalization for edge building
    xy_norm = normalize_xy(xy)

    # Build edges for 5 methods (sparse + pruned by design)
    E1 = graph_mst_knn(xy_norm, k=KNN_K, pct=EDGE_PCT, degree_cap=DEGREE_CAP)
    E2 = graph_gabriel(xy_norm, pct=EDGE_PCT, degree_cap=DEGREE_CAP)
    E3 = graph_rng(xy_norm, pct=EDGE_PCT, degree_cap=DEGREE_CAP)
    E4 = graph_delaunay_pruned(xy_norm, pct=EDGE_PCT, degree_cap=DEGREE_CAP)
    E5 = graph_brightness_msf(xy_norm, brightness, k=KNN_K, pct=EDGE_PCT, degree_cap=DEGREE_CAP, alpha=ALPHA)

    # Plot over original (un-normalized) coordinates to keep the sky look
    out = Path(OUT_DIR)
    plot_constellation(xy, rgb, sizes, E1, "MST on kNN (pruned)", out / "constellation_mst_knn.png")
    plot_constellation(xy, rgb, sizes, E2, "Gabriel Graph (pruned)", out / "constellation_gabriel.png")
    plot_constellation(xy, rgb, sizes, E3, "Relative Neighborhood Graph (pruned)", out / "constellation_rng.png")
    plot_constellation(xy, rgb, sizes, E4, "Delaunay (pruned)", out / "constellation_delaunay.png")
    plot_constellation(xy, rgb, sizes, E5, "Brightness-weighted MSF", out / "constellation_bright_msf.png")

if __name__ == "__main__":
    main()
