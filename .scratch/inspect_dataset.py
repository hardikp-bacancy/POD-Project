import pandas as pd, numpy as np
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)

df = pd.read_parquet("data/spotify_tracks.parquet")
print("=" * 70); print("SHAPE:", df.shape); print("=" * 70)
print("\n--- DTYPES ---")
print(df.dtypes.to_string())
print("\n--- MISSING VALUES (non-zero only) ---")
m = df.isna().sum(); m = m[m > 0]
print(m.to_string() if len(m) else "No missing values in any column")
print("\n--- SAMPLE (3 rows, transposed) ---")
print(df.head(3).T.to_string())
