import pandas as pd
from plot_real import run_constellation_pipeline

def test_basic_pipeline():
    df = pd.read_csv("stars_plot_ready.csv")
    fig, df_keep, masks, stats = run_constellation_pipeline(
        df, top_n=200, apply_visibility=False
    )
    assert "total_stars" in stats
    assert len(df_keep) > 0
