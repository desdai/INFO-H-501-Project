import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Paths ----
csv_path = Path("stars_plot_ready.csv")
out_path = csv_path.parent / "brightest_stars_map.png"   # save in same repo folder

# ---- Load data ----
df = pd.read_csv(csv_path)
colors = df[["R", "G", "B"]].values / 255.0

# ---- Plot ----
plt.figure(figsize=(10, 8), facecolor="black")
plt.scatter(
    df["x"], df["y"],
    s=df["size"],
    c=colors,
    alpha=0.9, edgecolors="none"
)
plt.gca().set_facecolor("black")
plt.axis("off")
plt.title("Top 2000 Brightest Stars (Gaia DR3)", color="white", fontsize=14, pad=12)
plt.gca().invert_xaxis()
plt.tight_layout()

# ---- Save ----
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="black")
plt.close()

print(f"✅ Saved star map to {out_path.resolve()}")
