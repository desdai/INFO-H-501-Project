import pandas as pd
import random
star_data = pd.read_csv('stars.csv')
# subset = star_data.sample(n=1000, random_state=42)
# subset.to_csv("sample.csv", index=False)


df_top2000 = star_data.sort_values(by="Gmag", ascending=True).head(2000)
df_top2000.to_csv("bright_stars.csv", index=False)