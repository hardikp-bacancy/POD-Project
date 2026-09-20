import pandas as pd
pd.set_option("display.width", 220)
df = pd.read_parquet("data/spotify_tracks.parquet")

print("=== IS THE ORIGINAL 'Blinding Lights' BY THE WEEKND PRESENT? ===")
w = df[df.artists.fillna('').str.contains("Weeknd", case=False, regex=False)]
print("The Weeknd rows:", len(w))
print(w[['track_name','album_name','track_genre','popularity']].drop_duplicates('track_name').head(12).to_string(index=False))

print("\n=== DATA-QUALITY PROBLEMS ===")
print("tempo == 0        :", int((df.tempo == 0).sum()))
print("duration_ms == 0  :", int((df.duration_ms == 0).sum()))
print("duration < 30s    :", int((df.duration_ms < 30000).sum()))
print("popularity == 0   :", int((df.popularity == 0).sum()))
print("time_signature==0 :", int((df.time_signature == 0).sum()))
print("rows with ANY of (tempo0|dur0):", int(((df.tempo==0)|(df.duration_ms==0)).sum()))

print("\n=== SANITY: do genres behave the way moods expect? (mean feature by genre) ===")
sub = df[df.track_genre.isin(['study','sleep','ambient','classical','edm','workout','death-metal','romance','sad','happy','chill','acoustic'])]
print(sub.groupby('track_genre')[['energy','valence','danceability','acousticness','instrumentalness','tempo']].mean().round(3).to_string())

print("\n=== POPULAR ARTISTS AVAILABLE (top by track count) ===")
print(df.artists.value_counts().head(15).to_string())
