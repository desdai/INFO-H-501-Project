#!/usr/bin/env bash
set -euo pipefail

# === Config ===
PY="plot_real.py"                    # or plot_constellations_from_csv.py
CSV="stars_plot_ready.csv"
DATE="2025-11-06"                    # change if needed

# 4-point equal-interval grids (N/S, E/W, low->high elev)
LATS=( -60 60 )
LONS=( -120 120 )             # West neg, East pos
ELEV=( 1000 3000 )

# Before vs after midnight (UTC). Same calendar date for simplicity.
TIMES=( "${DATE}T22:00:00" "${DATE}T02:00:00" )

# Mask sizes (deg): small vs large
MASKS=( 2 8 )

# Star counts
TOPNS=( 1000 2000 )

# Extra plotting/graph defaults (tweak as you like)
KNN_K=6
EDGE_PCT=80
DEGREE_CAP=3
LABEL_DX=-15
LABEL_DY=1.5

OUTROOT="out_gridrun_week9"
mkdir -p "$OUTROOT"
LOG="$OUTROOT/run_index.csv"
echo "lat,lon,elev_m,utc_time,mask_pad_deg,top_n,outdir,outfile" > "$LOG"

# === Sweep ===
run_idx=0
for lat in "${LATS[@]}"; do
  for lon in "${LONS[@]}"; do
    for h in "${ELEV[@]}"; do
      for t in "${TIMES[@]}"; do
        for pad in "${MASKS[@]}"; do
          for nstars in "${TOPNS[@]}"; do
            run_idx=$((run_idx+1))
            tag="lat${lat}_lon${lon}_h${h}_t$(echo "$t" | sed 's/[:\-]//g')_pad${pad}_n${nstars}"
            outdir="$OUTROOT/$tag"
            outfile="constellations_${tag}.png"
            mkdir -p "$outdir"

            echo "[${run_idx}] $tag"

            python "$PY" \
                --csv "$CSV" \
                --top_n "$nstars" \
                --apply_visibility \
                --observer_lat "$lat" --observer_lon "$lon" --observer_elev_m "$h" \
                --utc_time "$t" \
                --mask_pad_deg "$pad" --draw_masks \
                --knn_k "$KNN_K" --edge_pct "$EDGE_PCT" --degree_cap "$DEGREE_CAP" \
                --label_dx "$LABEL_DX" --label_dy "$LABEL_DY" \
                --outdir "$outdir" --outfile "$outfile"

            echo "$lat,$lon,$h,$t,$pad,$nstars,$outdir,$outfile" >> "$LOG"
          done
        done
      done
    done
  done
done

echo "Done. Index saved to: $LOG"
