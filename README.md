# INFO-H-501-Project
Download data from ```https://www.kaggle.com/datasets/realkiller69/gaia-stars-dataset-from-dr3-data-release-3```, and name the csv ``stars.csv``.

# Columns to Keep for Visualization

To visualize the **Gaia DR3 stars dataset** on a 2D scatter plot that captures **spatial position, color, and size**, preserve the following columns:

---

### Position (x, y coordinates)
Use these to plot the stars’ positions on the plane:

- **`RA_ICRS`** — Right ascension (x-axis)  
- **`DE_ICRS`** — Declination (y-axis)

*(These define the celestial coordinates in degrees.)*

---

### Color (for RGB mapping)
Use the color indices to derive a realistic star color:

- **`BP-RP`** — Gaia’s color index (bluer = smaller value, redder = larger value)  
    Map this to RGB using an astronomical color scale or a Matplotlib colormap such as `plasma`, `coolwarm`, or a custom one.

---

### Size (dot diameter)
Base the star’s visual size on brightness or physical radius:

- **Apparent brightness** → `Gmag`  
  - Inverse relation: brighter stars (smaller Gmag) = larger dots.


To execute above, run
```bash
python prepare_for_plot.py
```
This create x,y-coordinates, size of the stars, color (RGB) of the stars. For clarity, we only show the largest 2000 stars.

Plot with:
```bash
python plot.py
```

# Algorithm to outline constellations

Todo (this week): We need to propose a few candidate algorithms for connecting these stars, so that they show constellations.

# Web Deployment
Todo (at least 2 weeks from now) Build interactable webpage for the project