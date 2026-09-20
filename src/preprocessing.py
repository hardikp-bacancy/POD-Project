"""
preprocessing.py
-----------------
Turns the raw Spotify DataFrame into a clean catalogue ready for similarity
search: deduplicated songs, scaled audio features, and a TF-IDF text index.

Preprocessing decisions (see notebooks/exploration.ipynb Step 1 for the
data that motivated each one):

1. Drop rows missing track_name/artists -- can't recommend or display a
   song with no name.
2. Deduplicate by (track_name, artists): the raw data lists the same
   recording once per genre playlist it appeared on (16k+ tracks are
   duplicated this way). We collapse those rows into ONE song and merge
   their genres into a list, so a query still finds the song under any of
   its genres, but a recommendation list never contains the same song twice.
3. Drop physically-invalid rows: tempo == 0 or duration_ms == 0 are
   extraction failures, not real songs with those values.
4. Scale the 9 numeric audio features TWICE, for two different jobs:
     - MinMaxScaler (0-1 range) for MOOD matching, where a mood profile is
       a "target point" in a bounded space (e.g. "energy around 0.3").
     - StandardScaler (z-score) for SONG-TO-SONG cosine similarity. This
       matters more than it sounds: cosine similarity measures the ANGLE
       between vectors, and these audio features are all non-negative, so
       on a raw/MinMax scale every song vector points into the same narrow
       cone -- cosine similarity between almost any two songs comes out
       above 0.85, which is technically correct but useless for ranking or
       for a user-facing "similarity %". Centering features at zero
       (StandardScaler) lets vectors point in genuinely different
       directions, so cosine similarity actually discriminates between
       songs. This was verified empirically during development (see
       notebooks/exploration.ipynb): song-to-song neighbours are the same
       set either way, but z-score cosine spreads scores across a much
       wider, more interpretable range.
5. Build a "text document" per song = artists + track_name + album_name +
   genre(s), and fit TF-IDF over it for the text-matching pathway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# The audio features used for content-based (song-to-song / mood) similarity.
# Deliberately excludes key/mode/time_signature (categorical, not a musical
# "distance") and popularity (a quality signal, not a taste signal).
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and de-duplicate songs that appear under multiple genres."""
    df = df.copy()

    # 1) Rows with no name/artist are useless -- can't show or search them.
    df = df.dropna(subset=["track_name", "artists"])

    # 2) Physically invalid audio features (0 tempo / 0 duration = bad extraction).
    if "tempo" in df.columns:
        df = df[df["tempo"] > 0]
    if "duration_ms" in df.columns:
        df = df[df["duration_ms"] > 0]

    # Fill remaining gaps in optional text columns rather than dropping rows.
    for col in ["album_name", "track_genre"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # 3) Merge duplicate (track_name, artists) rows: same recording listed
    #    once per genre it was scraped under. Keep first row's audio
    #    features (they're identical across duplicates) but union the genres.
    if "track_genre" in df.columns:
        genre_lists = (
            df.groupby(["track_name", "artists"])["track_genre"]
            .apply(lambda s: sorted(set(s)))
        )
        df = df.drop_duplicates(subset=["track_name", "artists"], keep="first").copy()
        df = df.set_index(["track_name", "artists"])
        df["genres"] = genre_lists
        df = df.reset_index()
        # Keep a single primary genre column too (first alphabetically) for
        # simple filtering in the UI, alongside the full "genres" list.
        df["track_genre"] = df["genres"].apply(lambda g: g[0] if g else "Unknown")
    else:
        df = df.drop_duplicates(subset=["track_name", "artists"], keep="first")
        df["genres"] = [[] for _ in range(len(df))]

    df = df.reset_index(drop=True)
    return df


def scale_audio_features(df: pd.DataFrame, features: list[str] | None = None):
    """
    Fit both scalers on the available audio features and return
    (df_with_scaled_columns, minmax_scaler, standard_scaler, feature_list_actually_used).

    Only scales columns that actually exist in this dataset -- if a future
    dataset is missing e.g. 'liveness', we adapt instead of crashing.
    """
    features = features or AUDIO_FEATURES
    available = [f for f in features if f in df.columns]
    if not available:
        raise ValueError(
            "None of the expected audio-feature columns are present in the "
            f"dataset. Expected one of: {features}"
        )

    df = df.copy()
    # Median-fill any stray NaNs so the scalers never choke on them.
    df[available] = df[available].fillna(df[available].median())

    minmax_scaler = MinMaxScaler()
    minmax_scaled = minmax_scaler.fit_transform(df[available])
    minmax_cols = [f"{f}_scaled" for f in available]
    df[minmax_cols] = minmax_scaled

    standard_scaler = StandardScaler()
    z_scaled = standard_scaler.fit_transform(df[available])
    z_cols = [f"{f}_z" for f in available]
    df[z_cols] = z_scaled

    return df, minmax_scaler, standard_scaler, available


def build_text_corpus(df: pd.DataFrame) -> pd.Series:
    """
    Build one text "document" per song combining artist, title, album and
    genre(s) -- the fields a user's free-text query is most likely to mention.
    """
    def genres_to_text(g):
        if isinstance(g, (list, tuple, np.ndarray)):
            return " ".join(str(x) for x in g)
        return str(g) if pd.notna(g) else ""

    parts = []
    for col in ["artists", "track_name", "album_name"]:
        parts.append(df[col].fillna("").astype(str) if col in df.columns else pd.Series([""] * len(df)))
    if "genres" in df.columns:
        parts.append(df["genres"].apply(genres_to_text))
    elif "track_genre" in df.columns:
        parts.append(df["track_genre"].fillna("").astype(str))

    corpus = parts[0]
    for p in parts[1:]:
        corpus = corpus + " " + p
    return corpus


def build_tfidf_matrix(corpus: pd.Series):
    """Fit a TF-IDF vectorizer over the text corpus. Returns (matrix, vectorizer)."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams, e.g. "chill acoustic"
        min_df=1,
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer


def preprocess_dataset(raw_df: pd.DataFrame) -> dict:
    """
    Run the full preprocessing pipeline once. Returns a dict bundling
    everything MusicRecommender needs, so app.py only calls this once
    and reuses the result.
    """
    df = clean_data(raw_df)
    df, minmax_scaler, standard_scaler, feature_cols = scale_audio_features(df)
    corpus = build_text_corpus(df)
    tfidf_matrix, vectorizer = build_tfidf_matrix(corpus)

    return {
        "df": df,
        "scaler": minmax_scaler,               # used for mood target-distance
        "standard_scaler": standard_scaler,     # used for song-to-song cosine similarity
        "audio_features": feature_cols,
        "scaled_columns": [f"{f}_scaled" for f in feature_cols],   # MinMax, for mood matching
        "z_columns": [f"{f}_z" for f in feature_cols],             # z-score, for cosine similarity
        "tfidf_matrix": tfidf_matrix,
        "tfidf_vectorizer": vectorizer,
    }
