import pandas as pd, numpy as np
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 50)
df = pd.read_parquet("data/spotify_tracks.parquet")

print("=== DUPLICATE ANALYSIS ===")
print("rows total:                       ", len(df))
print("unique track_id:                  ", df.track_id.nunique())
print("dup rows by track_id:             ", int(df.duplicated('track_id').sum()))
print("unique (track_name, artists):     ", len(df.drop_duplicates(['track_name','artists'])))
print("dup rows by (track_name,artists): ", int(df.duplicated(['track_name','artists']).sum()))
print("fully identical rows (excl index):", int(df.drop(columns=['Unnamed: 0']).duplicated().sum()))

print("\n=== WHY DUPLICATES EXIST? same track_id across genres ===")
g = df.groupby('track_id').track_genre.nunique()
print("track_ids appearing in >1 genre:", int((g > 1).sum()))
ex = g[g > 1].index[0]
print("\nexample track_id", ex)
print(df[df.track_id == ex][['track_name','artists','track_genre','popularity','energy']].to_string(index=False))

print("\n=== GENRES ===")
print("n unique genres:", df.track_genre.nunique())
print("tracks per genre -> min/max:", df.track_genre.value_counts().min(), df.track_genre.value_counts().max())
print("\nfirst 40 genres:", sorted(df.track_genre.unique())[:40])

print("\n=== NUMERIC FEATURE RANGES (the ones we'll model) ===")
feats = ['danceability','energy','loudness','speechiness','acousticness',
         'instrumentalness','liveness','valence','tempo','popularity','duration_ms']
print(df[feats].describe().T[['min','25%','50%','75%','max']].to_string())

print("\n=== EXPLICIT FLAG ===")
print(df.explicit.value_counts().to_string())

print("\n=== DOES 'Blinding Lights' EXIST? (test query from the brief) ===")
for q in ["Blinding Lights", "Shape of You"]:
    hit = df[df.track_name.fillna('').str.lower().str.contains(q.lower(), regex=False)]
    print(f"\n'{q}': {len(hit)} row(s)")
    if len(hit): print(hit[['track_name','artists','track_genre','energy','valence','tempo']].head(4).to_string(index=False))
