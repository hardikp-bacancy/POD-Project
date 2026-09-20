"""
recommender.py
---------------
Core recommendation logic, deliberately kept separate from any UI or NLP
code so it can be tested and reasoned about on its own.

MusicRecommender exposes four strategies plus a hybrid combiner:

  recommend_by_song(song_name)   -- audio-feature cosine similarity
                                     ("songs similar to Blinding Lights")
  recommend_by_mood(mood)        -- distance to a target feature profile
                                     ("chill songs", "energetic workout songs")
  recommend_by_query(text)       -- TF-IDF cosine similarity on metadata text
                                     ("songs by Ariana Grande", "pop hits")
  recommend_by_genre(genre)      -- exact/fuzzy genre filter, ranked by popularity
  recommend_hybrid(...)          -- blends audio + text similarity with
                                     configurable weights

Why three different similarity metrics for three different signals
--------------------------------------------------------------------
* Song-to-song and hybrid text matching use COSINE SIMILARITY: it compares
  the *direction/shape* of two feature vectors regardless of magnitude,
  which is exactly what "sounds like this other song" means, and it is the
  standard metric for sparse TF-IDF vectors.
* Mood matching uses WEIGHTED EUCLIDEAN DISTANCE to a *target point*
  (e.g. "energy should be around 0.3"). A mood is a destination, not a
  direction -- cosine similarity cannot express "close to 0.3" (a song
  with energy 0.9 can still look "cosine-similar" to a 0.3-energy target
  if other features line up). Distance-to-target is the right tool for
  "how close is this song to what I asked for".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import MOOD_PROFILES


class MusicRecommender:
    def __init__(self, df: pd.DataFrame, scaler, audio_features: list[str],
                 scaled_columns: list[str], tfidf_matrix, tfidf_vectorizer,
                 z_columns: list[str] | None = None):
        self.df = df.reset_index(drop=True)
        self.scaler = scaler  # MinMaxScaler, used for mood target-distance
        self.audio_features = audio_features
        self.scaled_columns = scaled_columns  # MinMax columns, for mood matching
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_vectorizer = tfidf_vectorizer
        # MinMax feature matrix (mood distance) and z-score feature matrix
        # (song-to-song / hybrid cosine similarity -- see preprocessing.py
        # for why cosine needs centered features to be discriminating).
        self._feature_matrix = self.df[scaled_columns].to_numpy()
        z_columns = z_columns or scaled_columns
        self._z_matrix = self.df[z_columns].to_numpy() if all(c in self.df.columns for c in z_columns) else self._feature_matrix
        # Fast case-insensitive lookup: normalized name -> list of row indices.
        self._name_index: dict[str, list[int]] = {}
        for i, name in enumerate(self.df["track_name"].astype(str)):
            self._name_index.setdefault(name.strip().lower(), []).append(i)

    # ------------------------------------------------------------------ #
    # Lookup helpers
    # ------------------------------------------------------------------ #

    def find_song(self, song_name: str) -> pd.DataFrame:
        """Exact (case-insensitive) then substring match on track_name."""
        if not song_name or not song_name.strip():
            return self.df.iloc[0:0]
        key = song_name.strip().lower()
        if key in self._name_index:
            idx = self._name_index[key]
            return self.df.iloc[idx]
        mask = self.df["track_name"].astype(str).str.lower().str.contains(key, regex=False, na=False)
        return self.df[mask]

    def find_artist(self, artist_name: str) -> pd.DataFrame:
        if not artist_name or not artist_name.strip():
            return self.df.iloc[0:0]
        key = artist_name.strip().lower()
        mask = self.df["artists"].astype(str).str.lower().str.contains(key, regex=False, na=False)
        return self.df[mask]

    # ------------------------------------------------------------------ #
    # Strategy 1: song-to-song content similarity
    # ------------------------------------------------------------------ #

    def recommend_by_song(self, song_name: str, n: int = 5) -> dict:
        matches = self.find_song(song_name)
        if matches.empty:
            return {"ok": False, "reason": "song_not_found", "results": []}

        # If multiple recordings share the name, use the most popular one
        # as the reference -- most likely what the user meant.
        ref_idx = matches["popularity"].idxmax() if "popularity" in matches.columns else matches.index[0]
        ref_vec = self._z_matrix[ref_idx].reshape(1, -1)

        sims = cosine_similarity(ref_vec, self._z_matrix)[0]
        sims = self._friendly_similarity(sims)
        results = self._rank(sims, exclude_idx={ref_idx}, n=n)
        results = self._attach_explanations_song(results, ref_idx)
        return {
            "ok": True,
            "reference": self._row_to_dict(self.df.loc[ref_idx]),
            "results": results,
        }

    # ------------------------------------------------------------------ #
    # Strategy 2: mood-based recommendation
    # ------------------------------------------------------------------ #

    def recommend_by_mood(self, mood: str, n: int = 5, overrides: dict | None = None) -> dict:
        profile = MOOD_PROFILES.get(mood)
        if profile is None:
            return {"ok": False, "reason": "unknown_mood", "results": []}

        profile = dict(profile)
        if overrides:
            profile.update(overrides)  # e.g. feature hints like "acoustic" -> acousticness

        target_scaled, used_features = self._profile_to_scaled_vector(profile)

        sub_matrix = self.df[[f"{f}_scaled" for f in used_features]].to_numpy()
        dist = np.linalg.norm(sub_matrix - target_scaled, axis=1)
        # Convert distance to a 0-1 "closeness" score for a friendly display.
        max_possible = np.sqrt(len(used_features))  # worst case: every axis off by 1.0
        closeness = 1 - (dist / max_possible)

        results = self._rank(closeness, exclude_idx=set(), n=n)
        results = self._attach_explanations_mood(results, mood, used_features)
        return {"ok": True, "mood": mood, "results": results}

    def _profile_to_scaled_vector(self, profile: dict) -> tuple[np.ndarray, list[str]]:
        """Convert a raw-unit mood profile into scaled space using the fitted scaler."""
        used_features = [f for f in self.audio_features if f in profile]
        if not used_features:
            used_features = self.audio_features

        # Build one full raw-unit row: profile value where specified, dataset
        # median elsewhere, then scale it exactly like the real tracks were scaled.
        medians = self.df[self.audio_features].median()
        raw_row = medians.copy()
        for f, v in profile.items():
            if f in raw_row.index:
                raw_row[f] = v

        scaled_row = self.scaler.transform([raw_row[self.audio_features].to_numpy()])[0]
        scaled_series = pd.Series(scaled_row, index=self.audio_features)
        target_scaled = scaled_series[used_features].to_numpy()
        return target_scaled, used_features

    # ------------------------------------------------------------------ #
    # Strategy 3: free-text metadata matching (TF-IDF)
    # ------------------------------------------------------------------ #

    def recommend_by_query(self, query: str, n: int = 5) -> dict:
        if not query or not query.strip():
            return {"ok": False, "reason": "empty_query", "results": []}

        query_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        if sims.max() <= 0:
            return {"ok": False, "reason": "no_text_match", "results": []}

        results = self._rank(sims, exclude_idx=set(), n=n)
        results = self._attach_explanations_text(results, query)
        return {"ok": True, "results": results}

    # ------------------------------------------------------------------ #
    # Strategy 4: genre filter
    # ------------------------------------------------------------------ #

    def recommend_by_genre(self, genre: str, n: int = 5) -> dict:
        if not genre or not genre.strip():
            return {"ok": False, "reason": "empty_query", "results": []}
        key = genre.strip().lower()

        def has_genre(genres):
            if isinstance(genres, (list, tuple)):
                return any(key == g.lower() or key in g.lower() for g in genres)
            return key in str(genres).lower()

        mask = self.df["genres"].apply(has_genre) if "genres" in self.df.columns else self.df["track_genre"].str.lower().str.contains(key, na=False)
        matches = self.df[mask]
        if matches.empty:
            return {"ok": False, "reason": "unknown_genre", "results": []}

        top = matches.sort_values("popularity", ascending=False).head(n) if "popularity" in matches.columns else matches.head(n)
        results = []
        for idx, row in top.iterrows():
            results.append({
                "idx": idx,
                "score": row.get("popularity", 0) / 100.0,
                "explanation": f"Popular track tagged with genre '{genre}'.",
            })
        results = self._attach_row_info(results)
        return {"ok": True, "results": results}

    # ------------------------------------------------------------------ #
    # Strategy 5: hybrid = weighted blend of audio + text similarity
    # ------------------------------------------------------------------ #

    def recommend_hybrid(self, query: str = "", song_name: str = "", mood: str = "",
                          n: int = 5, audio_weight: float = 0.7, text_weight: float = 0.3,
                          feature_overrides: dict | None = None) -> dict:
        """
        Combine an audio-similarity signal with a text-similarity signal:
            final_score = audio_weight * audio_sim + text_weight * text_sim

        The audio signal comes from a reference song if one was given,
        otherwise from a mood profile if one was given, otherwise it is
        skipped (pure text search). Weights are configurable so the UI
        slider genuinely changes behaviour.
        """
        total = audio_weight + text_weight
        if total <= 0:
            audio_weight, text_weight = 0.7, 0.3
        else:
            audio_weight, text_weight = audio_weight / total, text_weight / total

        n_rows = len(self.df)
        audio_sim = np.zeros(n_rows)
        exclude = set()
        reference = None

        if song_name and song_name.strip():
            matches = self.find_song(song_name)
            if not matches.empty:
                ref_idx = matches["popularity"].idxmax() if "popularity" in matches.columns else matches.index[0]
                ref_vec = self._z_matrix[ref_idx].reshape(1, -1)
                audio_sim = self._friendly_similarity(cosine_similarity(ref_vec, self._z_matrix)[0])
                exclude = {ref_idx}
                reference = self._row_to_dict(self.df.loc[ref_idx])
        elif mood and mood in MOOD_PROFILES:
            target_scaled, used_features = self._profile_to_scaled_vector(
                {**MOOD_PROFILES[mood], **(feature_overrides or {})}
            )
            sub_matrix = self.df[[f"{f}_scaled" for f in used_features]].to_numpy()
            dist = np.linalg.norm(sub_matrix - target_scaled, axis=1)
            audio_sim = 1 - (dist / np.sqrt(len(used_features)))

        text_sim = np.zeros(n_rows)
        if query and query.strip():
            query_vec = self.tfidf_vectorizer.transform([query])
            text_sim = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        final_score = audio_weight * audio_sim + text_weight * text_sim
        if not np.any(final_score > 0):
            return {"ok": False, "reason": "no_match", "results": []}

        results = self._rank(final_score, exclude_idx=exclude, n=n)
        results = self._attach_explanations_hybrid(results, query, mood, reference)
        return {"ok": True, "reference": reference, "results": results}

    # ------------------------------------------------------------------ #
    # Shared ranking / formatting helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _friendly_similarity(cos_scores: np.ndarray) -> np.ndarray:
        """
        Map cosine similarity on z-score features (range roughly -1..1) onto
        a 0..1 "closeness" score that reads naturally as a percentage. Plain
        cosine on centered features can be negative (opposite audio
        profiles); (-1..1) -> (0..1) keeps ordering unchanged while making
        the displayed number ("73% similar") intuitive instead of showing
        e.g. a raw score of -0.2 for a poor match.
        """
        return (cos_scores + 1.0) / 2.0

    def _rank(self, scores: np.ndarray, exclude_idx: set, n: int) -> list[dict]:
        order = np.argsort(-scores)
        out = []
        for idx in order:
            if idx in exclude_idx:
                continue
            if scores[idx] <= 0:
                continue
            out.append({"idx": int(idx), "score": float(scores[idx])})
            if len(out) >= n:
                break
        return out

    def _row_to_dict(self, row: pd.Series) -> dict:
        genres = row.get("genres", [])
        return {
            "track_name": row.get("track_name", "Unknown"),
            "artists": row.get("artists", "Unknown"),
            "album_name": row.get("album_name", "Unknown"),
            "genre": genres[0] if isinstance(genres, list) and genres else row.get("track_genre", "Unknown"),
            "popularity": row.get("popularity", None),
        }

    def _attach_row_info(self, results: list[dict]) -> list[dict]:
        for r in results:
            r.update(self._row_to_dict(self.df.loc[r["idx"]]))
        return results

    def _attach_explanations_song(self, results: list[dict], ref_idx: int) -> list[dict]:
        ref = self.df.loc[ref_idx]
        for r in results:
            row = self.df.loc[r["idx"]]
            r["explanation"] = self._explain_audio_closeness(ref, row)
            r.update(self._row_to_dict(row))
        return results

    def _attach_explanations_mood(self, results: list[dict], mood: str, used_features: list[str]) -> list[dict]:
        for r in results:
            row = self.df.loc[r["idx"]]
            r["explanation"] = f"Matches the '{mood}' mood profile " \
                                f"({self._describe_features(row, used_features)})."
            r.update(self._row_to_dict(row))
        return results

    def _attach_explanations_text(self, results: list[dict], query: str) -> list[dict]:
        for r in results:
            row = self.df.loc[r["idx"]]
            r["explanation"] = f"Title, artist, album or genre text closely matches '{query}'."
            r.update(self._row_to_dict(row))
        return results

    def _attach_explanations_hybrid(self, results: list[dict], query: str, mood: str,
                                     reference: dict | None = None) -> list[dict]:
        for r in results:
            row = self.df.loc[r["idx"]]
            bits = []
            if reference:
                bits.append(f"has a similar audio profile to '{reference['track_name']}'")
            elif mood:
                bits.append(f"fits the '{mood}' mood")
            if query:
                bits.append(f"matches your text '{query}'")
            r["explanation"] = "Recommended because it " + " and ".join(bits) + "." if bits else "Strong overall match."
            r.update(self._row_to_dict(row))
        return results

    @staticmethod
    def _describe_features(row: pd.Series, features: list[str]) -> str:
        labels = {
            "danceability": "danceable", "energy": "energetic", "loudness": "loud",
            "speechiness": "speech-heavy", "acousticness": "acoustic",
            "instrumentalness": "instrumental", "liveness": "live-sounding",
            "valence": "upbeat", "tempo": "fast-tempo",
        }
        bits = []
        for f in features[:3]:
            val = row.get(f, None)
            if val is None:
                continue
            bits.append(labels.get(f, f))
        return ", ".join(bits) if bits else "similar audio profile"

    @staticmethod
    def _explain_audio_closeness(ref: pd.Series, row: pd.Series) -> str:
        # Pick the 2 features the pair is closest on for a human explanation
        # rather than dumping every numeric distance.
        feats = ["energy", "danceability", "valence", "acousticness", "tempo"]
        closeness = []
        for f in feats:
            if f in ref.index and f in row.index:
                diff = abs(ref[f] - row[f])
                closeness.append((f, diff))
        closeness.sort(key=lambda x: x[1])
        top = [f for f, _ in closeness[:2]]
        labels = {
            "energy": "energy", "danceability": "danceability", "valence": "mood/positivity",
            "acousticness": "acoustic character", "tempo": "tempo",
        }
        described = " and ".join(labels.get(f, f) for f in top)
        return f"This track has a similar {described} to your reference song."
