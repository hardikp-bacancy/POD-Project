# 🎵 Music Recommendation Chatbot

A conversational, domain-specific chatbot that recommends songs from natural-language requests — moods, activities, genres, artists, or a reference song — built on a real 114,000-track Spotify dataset with fully explainable, rule-based recommendation logic (no paid APIs, no LLM required).

## Overview

Type something like *"I want chill songs for studying"* or *"songs similar to Blinding Lights"* and the chatbot:

1. Detects what you're asking for (a mood, a genre, an artist, a specific song, or a general taste query).
2. Runs the matching recommendation strategy — content-based audio similarity, mood-to-feature matching, TF-IDF text search, or a hybrid blend.
3. Returns songs with track name, artist, album, genre, a similarity score, and a plain-English reason for each recommendation.

It is a prototype for a college/project discussion: functional, modular, explainable, and easy to extend.

## Features

- **Conversational UI** (Streamlit chat interface) with example prompts and adjustable settings.
- **Rule-based intent detection** — no LLM or API key needed; every decision is traceable in code.
- **Four recommendation strategies**: song-to-song similarity, mood-based matching, free-text metadata search, and a configurable hybrid blend of the two.
- **Mood vocabulary with synonym recognition** — "upbeat", "cheerful", "feel-good" all map to the same "happy" profile.
- **Sidebar controls**: number of recommendations, audio/text weight slider, genre filter, energy/danceability ranges, explicit-content toggle.
- **Explainability**: every recommendation includes a human-readable reason, not a raw similarity score.
- **Graceful error handling**: unknown songs/artists/genres, empty input, and invalid settings never crash the app.
- **Honest evaluation**: a documented test suite with 14 example queries, transparent about what can and can't be measured.

## Architecture

```
User
 ↓
Streamlit UI (app.py)
 ↓
Chatbot / Intent Detection (src/chatbot.py)
 ↓
Query Processing (mood synonyms, feature hints — src/utils.py)
 ↓
Recommendation Engine (src/recommender.py)
 ↓
Preprocessed Dataset (src/preprocessing.py + src/data_loader.py)
 ↓
Recommended Songs (with scores + explanations)
```

**Project structure:**

```
music-chatbot/
├── app.py                    # Streamlit UI — thin, no recommendation logic
├── requirements.txt
├── README.md
├── data/
│   ├── README.md             # dataset source & notes
│   └── spotify_tracks.parquet  # auto-downloaded on first run
├── src/
│   ├── data_loader.py        # download/load/validate the dataset
│   ├── preprocessing.py      # cleaning, dedup, scaling, TF-IDF
│   ├── recommender.py        # MusicRecommender: the 4 strategies + hybrid
│   ├── chatbot.py            # MusicChatbot: intent detection + NLP
│   └── utils.py              # mood profiles, synonyms, text helpers
├── notebooks/
│   └── exploration.ipynb     # Step-1 style dataset inspection, executed
└── tests/
    └── test_queries.py       # evaluation report (Section 12 of the brief)
```

The UI (`app.py`) never touches recommendation logic directly — it only calls `MusicChatbot.process_message(...)`, so the core system can be tested and reused without a browser (see `tests/test_queries.py` and the `.scratch/test_*.py` scripts used during development).

## Dataset

**Source:** [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) on Hugging Face (BSD licence) — 114,000 tracks across 114 genres.

| Column | Used for |
|---|---|
| `track_name`, `artists`, `album_name`, `track_genre` | display + text/TF-IDF matching |
| `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo` | content-based / mood similarity |
| `popularity` | ranking/tie-breaking, and choosing the "right" version when a title has duplicates |
| `explicit` | optional content filter |
| `key`, `mode`, `time_signature` | **not used** — categorical values where numeric distance is musically meaningless |

**Data quality, inspected before any modeling** (see `notebooks/exploration.ipynb`):
- ~16,000 songs are listed once per genre playlist they were scraped from (same song, multiple `track_genre` rows) → collapsed into one row per song with a merged `genres` list.
- 157 rows had `tempo == 0`, 1 had `duration_ms == 0` (extraction artifacts) → dropped.
- 3 rows were missing `track_name`/`artists` → dropped; missing `album_name`/`track_genre` → filled with `"Unknown"`.
- Genre labels turned out to be an **unreliable mood proxy** (e.g. the `happy` genre has below-median valence) — this is why moods map to audio features, not genre names.

## Recommendation Algorithm

### Why three different similarity metrics, not one

| Task | Metric | Why |
|---|---|---|
| Song → song ("similar to Blinding Lights") | **Cosine similarity** on z-score-standardized audio features | Compares the *shape* of a song's full audio profile. |
| Mood → songs ("chill songs") | **Weighted Euclidean distance to a target point** | A mood is a *destination* ("energy ≈ 0.3"), not a direction — cosine can't express "close to a specific value". |
| Text → songs (artist/genre/free text) | **Cosine similarity** on TF-IDF vectors | Standard, length-independent metric for sparse text vectors. |

### Feature scaling — two scalers, deliberately

- **MinMaxScaler** (0–1) on the 9 audio features, for mood matching. Mood profiles in `src/utils.py` are written in raw units (e.g. `tempo: 145.0`) and pushed through this same fitted scaler, so comparisons stay mathematically consistent.
- **StandardScaler** (z-score) on the same 9 features, for song-to-song cosine similarity. This matters: cosine similarity on non-negative MinMax features gives every song a similarity above ~0.85 regardless of how different they actually are (all vectors point into the same narrow cone). Centering the features first (mean 0) lets cosine actually discriminate — verified empirically in `notebooks/exploration.ipynb` (a genuinely different track scores ~0.38 vs. ~0.99 for a true near-neighbor).

### TF-IDF

`src/preprocessing.py` builds one text "document" per song from `artists + track_name + album_name + genres`, then fits a `TfidfVectorizer` (unigrams + bigrams, English stopwords removed) over the whole catalogue. A user's free-text query is vectorized with the same fitted vocabulary and compared via cosine similarity.

### Hybrid scoring

```python
final_score = audio_weight * audio_similarity + text_weight * text_similarity
```

Weights are configurable (sidebar slider in the UI, or `audio_weight=`/`text_weight=` parameters in code) and are auto-normalized to sum to 1. The audio signal comes from a reference song if one was named, otherwise a mood profile if one was detected, otherwise it's skipped (pure text search).

### Mood-to-feature mapping

`src/utils.py` defines `MOOD_PROFILES` (happy, sad, energetic, chill, romantic, workout, party, focus, sleep, angry) in **raw feature units**, converted through the fitted `MinMaxScaler` before comparison — not the [0,1]-literal values from a generic template, since real tempo/loudness ranges are dataset-specific. Values were chosen by reasoning about what each mood implies musically, then sanity-checked against this dataset's genre-level feature averages. `MOOD_SYNONYMS` maps ~50 everyday words ("joyful", "gym", "laid-back", "date night"...) onto these canonical moods, and `FEATURE_HINTS` lets words like "acoustic", "instrumental", "fast", "slow" nudge the profile even when combined with a mood word.

## Natural-Language Processing

`src/chatbot.py` is rule-based, on purpose: every routing decision is a readable regex or keyword lookup, not a black box, so the logic is fully explainable for a project discussion.

1. **Greeting / help** — exact keyword match.
2. **Similar-song** — patterns like `"similar to X"`, `"sounds like X"`, `"songs like X"` (deliberately *not* a bare `"like X"`, since "I **like** upbeat pop songs" is a preference statement, not a song reference — this was caught and fixed during testing).
3. **By-artist** — patterns like `"songs by X"`, `"more from X"`.
4. **By-genre** — only triggers if the extracted phrase is an actual genre in the dataset.
5. **By-mood** — any recognized mood synonym in the message.
6. **General query** — everything else falls through to hybrid text + audio search.

Example: *"I want relaxing acoustic music for studying"* → moods found: `chill`, `focus` (first one wins) → feature hint: `acousticness: 0.8` → hybrid search blending the chill mood profile (with acousticness boosted) against the full message text.

## Explainability

Every recommendation includes:
- Track name, artist, album, genre
- A match percentage (rescaled similarity/distance score, not a raw technical number)
- A one-line, plain-English reason (e.g. *"This track has a similar energy and danceability to your reference song."*)

## Installation

```bash
git clone <repository>
cd music-chatbot
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset downloads automatically on first run (~13 MB, no API key needed) and is cached in `data/`.

## Running the Application

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Example Queries

| You type | What happens |
|---|---|
| "Suggest some chill songs for studying." | mood match on `chill` |
| "I want energetic songs for a workout." | mood match on `energetic`/`workout` |
| "Recommend songs similar to Blinding Lights." | song-to-song audio similarity |
| "Give me some romantic songs." | mood match on `romantic` |
| "I like upbeat pop songs with high energy." | mood match on `happy` (text-driven, not song-reference) |
| "songs by Ed Sheeran" | artist lookup |
| "recommend some k-pop songs" | genre lookup |
| "Recommend songs similar to XYZ123" (unknown) | graceful *"I couldn't find that song..."* message |

## Evaluation

Run `PYTHONPATH=. python tests/test_queries.py` for a scripted report. This project has **no ground-truth relevance labels** (no user ever rated these songs against "chill for studying"), so we don't — and can't honestly — report precision/recall/accuracy. What the test suite verifies instead:

- The chatbot detects the *correct intent* for 14 manually written queries (drawn from the brief's own examples).
- It returns results when a request is satisfiable, and a graceful message (zero results, no crash) when it isn't (unknown song/artist, empty input).

Current result: **14/14 test cases behave as expected.** Judging whether the individual song recommendations are musically "good" remains a manual, subjective step — this is stated explicitly rather than dressed up as a metric.

## Limitations

- **Dataset coverage**: 114,000 tracks is large but not exhaustive — some songs/artists genuinely aren't present.
- **Mood interpretation is approximate**: moods are represented as target points in a 9-dimensional feature space; this is a reasonable heuristic, not a validated psychological model of what makes music "sad" or "happy".
- **Recommendation quality depends on dataset quality**: genre tags and audio features come from the source dataset as-is.
- **No real-time Spotify catalog integration**: this is a static snapshot, not a live API.
- **Natural-language understanding is intentionally lightweight**: regex/keyword-based, not a trained NLU model — it won't handle highly indirect or sarcastic phrasing.
- **Audio-only similarity can cross genres**: two songs can have near-identical tempo/energy/valence while sounding stylistically different to a human listener (e.g. a pop track and a hip-hop track with the same BPM and loudness). This is a real, explainable property of content-based filtering — the hybrid mode (blending in text/genre) mitigates it.

## Future Improvements

- Spotify Web API integration for a live, up-to-date catalog and 30-second audio previews.
- User accounts + preference history for personalized, session-persistent recommendations.
- Collaborative filtering (users who liked X also liked Y) once real user-interaction data exists.
- LLM-based intent extraction for more flexible, conversational phrasing.
- Feedback loop (thumbs up/down) to refine recommendations over time.
- Playlist generation and export.

## Presentation Prep

See below for a 3–5 minute demo script and answers to likely evaluator questions.

### Demo script (~4 minutes)

1. **(30s) Problem & domain.** "This is a domain-specific chatbot for music recommendation. Unlike a generic chatbot, every part of it — the NLP, the scoring, the vocabulary — is specialized for one task: understanding a listener's mood or taste and finding matching songs."
2. **(30s) Dataset.** "It's built on a real Spotify dataset — 114,000 tracks, 114 genres, with per-track audio features like energy, danceability, and valence. I inspected it first: found and fixed duplicate entries, invalid rows, and confirmed genre labels don't reliably predict mood — which shaped the whole design."
3. **(90s) Live demo.** Run 3–4 queries live: a mood query ("chill songs for studying"), a similar-song query ("songs similar to Blinding Lights"), an artist query, and an unknown-song query to show graceful error handling. Point out the match % and "why recommended" line each time.
4. **(60s) How it works.** "Three techniques: cosine similarity on standardized audio features for song-to-song matching, distance-to-target for moods, and TF-IDF cosine similarity for text. They're combined in a configurable hybrid score — shown live via the sidebar slider."
5. **(30s) Wrap-up.** "It's rule-based by design — fully explainable, no API key, runs entirely locally — with a documented evaluation and clear limitations."

### Likely questions & concise answers

**Q: Why cosine similarity instead of something else?**
A: It compares the *direction* of feature vectors, ignoring magnitude — the right notion of "similar shape" for audio profiles, and the standard metric for sparse TF-IDF text vectors.

**Q: Why normalize/scale the features first?**
A: Loudness ranges roughly −50 to +5 dB and tempo 0–240 BPM, while the other seven features are already 0–1. Unscaled, loudness alone would dominate every similarity calculation.

**Q: Why two different scalers (MinMax and StandardScaler)?**
A: MinMax keeps mood profiles as interpretable "target points" in bounded [0,1] space. StandardScaler centers audio features at zero so cosine similarity can actually discriminate between songs — cosine on all-positive MinMax vectors compresses everything above ~0.85 similarity, which I found and fixed during testing (documented in the notebook).

**Q: Why TF-IDF and not just keyword search?**
A: TF-IDF down-weights common words (e.g. "song", "music") and up-weights distinctive ones (an artist name, a genre), so matches reflect what's actually distinctive about the query.

**Q: Why combine multiple similarity signals?**
A: A pure-audio match can cross genres in ways that feel wrong to a listener (same tempo/energy, different vibe); a pure-text match ignores what the song actually sounds like. Blending both, with configurable weights, balances "sounds like this" against "is actually what you asked for."

**Q: How does it understand "I want something chill for studying"?**
A: "Chill" and "studying" both hit the synonym dictionary (→ `chill` and `focus` moods); the first one found becomes the primary mood profile, and the full sentence is also run through TF-IDF, so both the mood-feature match and any literal text match (e.g. an album called "Study Beats") contribute to the final ranking.

**Q: What doesn't it understand well?**
A: Indirect phrasing without a recognized keyword ("music for when I feel like the world is ending"), negation ("not too sad"), and comparative requests ("something more energetic than the last one") — the parser has no state between messages and no negation handling.

**Q: How would this become production-grade?**
A: Real user feedback signals (skips/likes) for collaborative filtering, a live catalog via the Spotify API, session/user accounts for personalization, and likely an LLM for intent parsing to handle the long tail of phrasing this rule-based parser misses.
