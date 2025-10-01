import pandas as pd
import random
star_data = pd.read_csv('stars.csv')
subset = star_data.sample(n=1000, random_state=42)
subset.to_csv("sample.csv", index=False)