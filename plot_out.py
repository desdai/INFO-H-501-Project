# constellation_3x5_named.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# =========================
# Config
# =========================
CSV_PATH   = "stars_plot_ready.csv"   # expects: ID,Source,x,y,R,G,B,size
OUT_DIR    = "out"

TOP_N      = 1200     # number of brightest stars (by 'size'); None = all
KNN_K      = 6        # k for kNN-based algorithms
EDGE_PCT   = 80       # keep edges <= this percentile of geometric length
DEGREE_CAP = 3        # limit max degree for each node
ALPHA      = 0.6      # brightness bias for MSF
EPS        = 1e-9

# =========================
# Utility functions
# =========================
def normalize_xy(xy):
    """scale RA by cos(mean(Dec)) for projection"""
    x, y = xy[:, 0], xy[:, 1]
    y_mean = np.deg2rad(np.mean(y))
    return np.column_stack([x * np.cos(y_mean), y])

def unique_edges(pairs):
    if len(pairs) == 0: return pairs
    a = np.sort(pairs, axis=1)
    return np.unique(a, axis=0)

def edge_lengths(xy, edges):
    if len(edges) == 0: return np.array([])
    pi, pj = xy[edges[:, 0]], xy[edges[:, 1]]
    return np.linalg.norm(pi - pj, axis=1)

def prune_edges(edges, lengths, pct):
    if len(edges) == 0: return edges
    thr = np.percentile(lengths, pct)
    return edges[lengths <= thr]

def cap_degree(edges, lengths, max_deg):
    if max_deg is None or len(edges) == 0: return edges
    E, L = edges.copy(), lengths.copy()
    while True:
        deg = {}
        for i, j in E:
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
        offenders = {n for n, d in deg.items() if d > max_deg}
        if not offenders: return E
        idxs_by_node = {}
        for idx, (i, j) in enumerate(E):
            idxs_by_node.setdefault(i, []).append(idx)
            idxs_by_node.setdefault(j, []).append(idx)
        drop = set()
        for n in offenders:
            idxs = idxs_by_node.get(n, [])
            if idxs: drop.add(max(idxs, key=lambda t: L[t]))
        if not drop: return E
        keep = [k for k in range(len(E)) if k not in drop]
        E, L = E[keep], L[keep]

# =========================
# Core graph builders
# =========================
def knn_edges(X, k=6):
    tree = cKDTree(X)
    _, idx = tree.query(X, k=k+1)
    I = np.repeat(np.arange(len(X)), k)
    J = idx[:, 1:].reshape(-1)
    return unique_edges(np.column_stack([I, J]))

def mutual_knn_edges(X, k=6):
    tree = cKDTree(X)
    _, idx = tree.query(X, k=k+1)
    nbrs = [set(row[1:]) for row in idx]
    pairs = [[i, j] for i in range(len(X)) for j in nbrs[i] if i in nbrs[j]]
    return unique_edges(np.array(pairs, int)) if pairs else np.empty((0,2), int)

def delaunay_edges(X2):
    if X2.shape[1] != 2 or len(X2) < 3: return np.empty((0,2), int)
    tri = Delaunay(X2)
    e = np.vstack([tri.simplices[:, [0,1]],
                   tri.simplices[:, [1,2]],
                   tri.simplices[:, [0,2]]])
    return unique_edges(e)

def mst_knn(xy_draw, X, k=KNN_K):
    E = knn_edges(X, k)
    L = edge_lengths(xy_draw, E)
    n = len(X)
    W = coo_matrix((L, (E[:,0], E[:,1])), shape=(n,n))
    W = W + W.T
    mst = minimum_spanning_tree(W).tocoo()
    return np.column_stack([mst.row, mst.col])

def bright_msf(xy_draw, X, brightness, k=KNN_K, alpha=ALPHA):
    E = knn_edges(X, k)
    n = len(X)
    pi, pj = xy_draw[E[:,0]], xy_draw[E[:,1]]
    dij = np.linalg.norm(pi - pj, axis=1)
    bsum = (brightness[E[:,0]] + brightness[E[:,1]]) ** alpha
    w = dij / np.maximum(bsum, 1e-9)
    W = coo_matrix((w, (E[:,0], E[:,1])), shape=(n,n)); W = W + W.T
    mst = minimum_spanning_tree(W).tocoo()
    return np.column_stack([mst.row, mst.col])

def gabriel_edges(X, k=KNN_K):
    E = knn_edges(X, k)
    tree = cKDTree(X)
    keep = []
    for i, j in E:
        ci, cj = X[i], X[j]
        c = (ci + cj) / 2.0
        r = np.linalg.norm(ci - cj) / 2.0 + EPS
        idxs = tree.query_ball_point(c, r)
        if len(idxs) <= 2: keep.append([i,j])
    return np.array(keep, int) if keep else np.empty((0,2), int)

def rng_edges(X, k=KNN_K):
    E = knn_edges(X, k)
    tree = cKDTree(X)
    keep = []
    for i, j in E:
        xi, xj = X[i], X[j]
        dij = np.linalg.norm(xi - xj)
        cand = set(tree.query_ball_point(xi, dij+EPS)) | set(tree.query_ball_point(xj, dij+EPS))
        ok = True
        for t in cand:
            if t == i or t == j: continue
            if max(np.linalg.norm(X[i]-X[t]), np.linalg.norm(X[j]-X[t])) < dij - EPS:
                ok = False; break
        if ok: keep.append([i,j])
    return np.array(keep, int) if keep else np.empty((0,2), int)

# =========================
# Plot
# =========================
def plot_constellation(xy, rgb, sizes, edges, title, filename):
    plt.figure(figsize=(10,8), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")
    ax.scatter(xy[:,0], xy[:,1], s=sizes, c=rgb, edgecolors="none", alpha=0.55)
    for i, j in edges:
        x0,y0 = xy[i]; x1,y1 = xy[j]
        ax.plot([x0,x1],[y0,y1],lw=0.8,alpha=0.95,color=(1,1,1))
    ax.set_title(title, color="white", pad=12)
    ax.invert_xaxis(); ax.axis("off")
    plt.tight_layout()
    Path(OUT_DIR).mkdir(exist_ok=True)
    plt.savefig(Path(OUT_DIR)/filename, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()
    print("Saved:", filename)

# =========================
# Main
# =========================
def main():
    df = pd.read_csv(CSV_PATH)
    if TOP_N: df = df.sort_values("size", ascending=False).head(TOP_N).reset_index(drop=True)
    xy = df[["x","y"]].to_numpy(float)
    xy_n = normalize_xy(xy)
    rgb = df[["R","G","B"]].to_numpy(float)/255.0
    sizes = df["size"].astype(float).to_numpy()
    brightness = sizes.copy()

    # --- feature metrics ---
    metrics = {
        "geo": xy_n,
        "color": rgb,
        "size": ((sizes - sizes.mean())/ (sizes.std()+1e-9)).reshape(-1,1)
    }

    # --- algorithm functions ---
    algos = {
        "mst": lambda X: mst_knn(xy, X),
        "gabriel": lambda X: gabriel_edges(X),
        "rng": lambda X: rng_edges(X),
        "delaunay": lambda X: delaunay_edges(X) if X.shape[1]==2 else mutual_knn_edges(X),
        "bright_msf": lambda X: bright_msf(xy, X, brightness)
    }

    # --- generate all 15 ---
    for algo_name, algo_func in algos.items():
        for metric_name, X in metrics.items():
            E = algo_func(X)
            L = edge_lengths(xy, E)
            E = prune_edges(E, L, EDGE_PCT)
            E = cap_degree(E, edge_lengths(xy, E), DEGREE_CAP)
            plot_constellation(
                xy, rgb, sizes, E,
                f"{algo_name.upper()} ({metric_name})",
                f"{algo_name}_{metric_name}.png"
            )

if __name__ == "__main__":
    main()
