# Constellation Graph Explorer
INFO-H 501 – Final Project

## Overview
This project builds an interactive Streamlit web application that visualizes the night sky using a simplified subset of the Gaia DR3 star catalog. Users can:

- Upload a star dataset or use a provided sample file  
- Visualize a sky projection (Right Ascension and Declination)  
- Reconstruct constellation graphs using a kNN-based algorithm  
- Apply visibility filtering based on latitude, longitude, elevation, and time  
- Explore animated simulations of star movement and observer-location changes  
- Download processed star tables for further analysis  

Stakeholders who benefit from this tool include astronomy students, instructors teaching celestial mechanics, and data science learners studying visualization and coordinate transformations. The application provides an intuitive interface for exploring how constellations appear differently depending on the observer’s location and the time of observation.

---

## Data Description
We use a processed subset of the Gaia DR3 star catalog.

Original dataset source:  
https://www.kaggle.com/datasets/realkiller69/gaia-stars-dataset-from-dr3-data-release-3

From the full dataset, we extracted the following key columns:

- `RA_ICRS`: Right ascension (degrees)  
- `DE_ICRS`: Declination (degrees)  
- `BP-RP`: Gaia color index  
- `Gmag`: Apparent magnitude  

We preprocessed these fields into:

- `x`, `y`: Projected RA/Dec for plotting  
- `R`, `G`, `B`: Realistic star colors  
- `size`: Dot size scaled by brightness  

A helper script (`prepare_for_plot.py`) performs this preprocessing and outputs `stars_plot_ready.csv`.

---

## Algorithm Description
The application reconstructs constellation shapes using a multi-stage geometric algorithm:

1. **Brightness filtering**  
   Select the top N brightest stars.

2. **Constellation assignment**  
   Use `astropy.coordinates.get_constellation` to assign each star to its IAU constellation.

3. **kNN-based edge construction**  
   - Build edges between stars using k-nearest neighbors  
   - Remove outlier edges using length-percentile pruning  
   - Cap node degree to reduce visual clutter  

4. **Bounding-box filtering**  
   - Compute bounding box around the largest connected component  
   - Re-run the algorithm inside the bounding box (to remove noise)  
   - Select only edges inside the refined region  

5. **Visibility computation**  
   Apply observer-based filtering using `astroplan`:
   - Convert RA/Dec to Alt/Az  
   - Keep only stars above the horizon (altitude > 0°)

6. **Final rendering**  
   Show background stars, constellation edges, and labels.

In addition, the project includes animation scripts to generate videos simulating:

- Changing observation time  
- Changing latitude  
- Changing longitude  
- Changing elevation  

---

## Tools Used

### Python Libraries
- **pandas** – Data cleaning and table manipulation  
- **numpy** – Numerical calculations  
- **matplotlib** – Plotting sky projections  
- **astropy** – Celestial coordinates and constellation lookup  
- **astroplan** – Visibility calculations (observer → Alt/Az)  
- **scipy** – kNN graph operations  
- **streamlit** – Interactive web application  
- **imageio** – GIF/MP4 animation generation  

### Additional Tools
- **Git + GitHub** – Version control  
- **.gitignore** – Prevents committing large data, environment files  
- **.env** (ignored) – Stores local paths or API keys if needed  

---

## Ethical Concerns

### 1. Data Source and Attribution  
Gaia DR3 star data was collected for astronomical research.  
Although the subset we use is publicly available, proper attribution is required.  
We provide dataset credit and avoid redistributing the original large dataset.

### 2. Computational Misinterpretation  
Constellation lines in the app are not official IAU shapes; they are generated algorithmically.  
Users might mistakenly interpret them as canonical.  
We mitigate this by documenting that the edges are approximate.

### 3. Accessibility  
We include:  
- Clear UI explanations  
- High-contrast visualizations (dark mode for night-sky)  
- Avoid technical jargon in the user interface  
Still, more could be done for screen-reader support.

### 4. Fair Use of Resources  
The app allows users to upload custom CSV files.  
We mitigate risks of large or malicious uploads by validating column names and catching errors.

---

## Running the Application

### Install dependencies
```
conda env create -f star.yml
```
and if you are using mac:
```
conda env create -f star_mac.yml
```

### Launch Streamlit app
```
streamlit run app.py
```

---

## Project Structure
```
project/
│
├── old_testing_files      # Code and results during designing of project
├── frames_xxx             # Folders containing frames and animations created
├── test                   # Folder including a unit test, as required (run pytest)
├── app.py                 # Streamlit front end
├── plot_real.py           # Constellation reconstruction module
├── prepare_for_plot.py    # Preprocessing script
├── make_animation.py      # GIF/MP4 generation
├── stars_plot_ready.csv   # Sample data (small subset)
└── README.md              # Documentation
```

---

## License
This project is for educational use under the INFO-H 501 course.  
Gaia DR3 data remains subject to ESA licensing.

