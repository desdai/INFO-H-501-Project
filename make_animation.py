#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

import imageio
import imageio.v3 as iio

from plot_real import run_constellation_pipeline


# ----------------------------------------
# GLOBAL CONFIG
# ----------------------------------------
CSV_FILE = "stars_plot_ready.csv"
N_FRAMES = 60
TOP_N = 1200
MASK_PAD = 3


# ----------------------------------------
# Helper for frame naming
# ----------------------------------------
def frame_name(outdir, i):
    return outdir / f"frame_{i:04d}.png"


# ----------------------------------------
# TIME SWEEP
# ----------------------------------------
def animate_time(df):
    outdir = Path("frames_time")
    outdir.mkdir(exist_ok=True)

    start = datetime(2025, 11, 6, 18, 0, 0)
    dt = timedelta(minutes=15)

    for i in range(N_FRAMES):
        utc_str = (start + i * dt).strftime("%Y-%m-%dT%H:%M:%S")
        print(f"[time] Frame {i}, utc={utc_str}")

        fig, df_keep, masks, stats = run_constellation_pipeline(
            df,
            top_n=TOP_N,
            apply_visibility=True,
            lat=39.7,
            lon=30,
            elev_m=220,
            utc_time=utc_str,
            mask_pad_deg=MASK_PAD,
            draw_masks=False,
        )
        fig.savefig(frame_name(outdir, i), dpi=200, bbox_inches="tight")
        fig.clf()

    build_gif_mp4(outdir, "time")


# ----------------------------------------
# LATITUDE SWEEP
# ----------------------------------------
def animate_lat(df):
    outdir = Path("frames_lat")
    outdir.mkdir(exist_ok=True)

    lats = np.linspace(-60, 60, N_FRAMES)

    for i, lat in enumerate(lats):
        print(f"[lat] Frame {i}, lat={lat:.2f}")

        fig, df_keep, masks, stats = run_constellation_pipeline(
            df,
            top_n=TOP_N,
            apply_visibility=True,
            lat=float(lat),
            lon=30,
            elev_m=220,
            utc_time="2025-11-06T02:00:00",
            mask_pad_deg=MASK_PAD,
            draw_masks=False,
        )
        fig.savefig(frame_name(outdir, i), dpi=200, bbox_inches="tight")
        fig.clf()

    build_gif_mp4(outdir, "lat")


# ----------------------------------------
# LONGITUDE SWEEP
# ----------------------------------------
def animate_lon(df):
    outdir = Path("frames_lon")
    outdir.mkdir(exist_ok=True)

    lons = np.linspace(0, 360, N_FRAMES)

    for i, lon in enumerate(lons):
        print(f"[lon] Frame {i}, lon={lon:.2f}")

        fig, df_keep, masks, stats = run_constellation_pipeline(
            df,
            top_n=TOP_N,
            apply_visibility=True,
            lat=39.7,
            lon=float(lon),
            elev_m=220,
            utc_time="2025-11-06T02:00:00",
            mask_pad_deg=MASK_PAD,
            draw_masks=False,
        )
        fig.savefig(frame_name(outdir, i), dpi=200, bbox_inches="tight")
        fig.clf()

    build_gif_mp4(outdir, "lon")


# ----------------------------------------
# ELEVATION SWEEP
# ----------------------------------------
def animate_elev(df):
    outdir = Path("frames_elev")
    outdir.mkdir(exist_ok=True)

    elevs = np.linspace(0, 4000, 20)

    for i, el in enumerate(elevs):
        print(f"[elev] Frame {i}, elev={el:.1f}")

        fig, df_keep, masks, stats = run_constellation_pipeline(
            df,
            top_n=TOP_N,
            apply_visibility=True,
            lat=39.7,
            lon=30,
            elev_m=float(el),
            utc_time="2025-11-06T02:00:00",
            mask_pad_deg=MASK_PAD,
            draw_masks=False,
        )
        fig.savefig(frame_name(outdir, i), dpi=200, bbox_inches="tight")
        fig.clf()

    build_gif_mp4(outdir, "elev")


# ----------------------------------------
# CREATE GIF + MP4 FOR ONE MODE
# ----------------------------------------
def build_gif_mp4(outdir, mode):
    print(f"Building GIF/MP4 for mode={mode} ...")

    frame_files = sorted([f for f in outdir.iterdir() if f.suffix == ".png"])

    # GIF
    gif_path = outdir / f"{mode}_animation.gif"
    with imageio.get_writer(gif_path, mode="I", duration=0.08) as writer:
        for f in frame_files:
            writer.append_data(iio.imread(f))
    print("GIF saved:", gif_path)

    # MP4
    mp4_path = outdir / f"{mode}_animation.mp4"
    with imageio.get_writer(mp4_path, fps=12, codec="libx264") as writer:
        for f in frame_files:
            writer.append_data(iio.imread(f))
    print("MP4 saved:", mp4_path)


# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} stars")

    # animate_time(df)
    # animate_lat(df)
    # animate_lon(df)
    animate_elev(df)

    print("\n==== All animations finished ====")
