# Dataset

**Source:** [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) on Hugging Face (BSD licence).

- 114,000 rows, 21 columns, covering 125 Spotify genres (114 distinct genre tags actually present), ~1,000 tracks scraped per genre.
- Columns: `track_id`, `artists`, `album_name`, `track_name`, `popularity`, `duration_ms`, `explicit`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`, `track_genre`.

## How it gets here

`src/data_loader.py` downloads the Parquet conversion of this dataset automatically the first time the app or notebook runs, and caches it as `spotify_tracks.parquet` in this folder. No API key is required. Subsequent runs read the local cache, so the app works offline after the first launch.

If you already have a copy of the CSV/Parquet file, drop it in this folder as `spotify_tracks.parquet` (or pass a path to `load_raw_data()`), and the loader will use it instead of downloading.

## Known data-quality notes (handled in `src/preprocessing.py`)

- ~16,000 songs are listed once per genre playlist they appeared on (the same `track_id` under multiple `track_genre` values). `clean_data()` collapses these into one row per song and merges the genres into a list.
- 157 rows have `tempo == 0` and 1 row has `duration_ms == 0` — extraction artifacts, dropped during cleaning.
- 3 rows are missing `track_name`/`artists`/`album_name` — dropped (unusable for display) or filled with `"Unknown"` depending on the column.
