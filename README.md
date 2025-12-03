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
(This week's progress)

```bash
plot_out.py
```

Plots the output by geo-location, size, and color with 5 candidate algorithms.

```bash
  python plot_real.py \
    --csv stars_plot_ready.csv --top_n 1000 \
    --apply_visibility \
    --observer_lat 39.77 --observer_lon -86.16 --observer_elev_m 220 \
    --utc_time 2025-11-06T02:00:00 \
    --mask_pad_deg 5 --draw_masks \
    --label_dx -15 --label_dy 1.5 --outdir out_week9
```
Plots the constellation from a real boundary (the lines are still connected by us). Using the astropy library, we first identify the real constellation boundary, then carry out geo-location connection algorithms (partly from plot_out.py)


# Steamlit web deployment
run with version 0.1:
```bash
streamlit run app.py
```
You need to upload the csv file ```stars_plot_ready.csv``` to plot.


# GIF and Video

Run 
```bash
python make_animation.py
```