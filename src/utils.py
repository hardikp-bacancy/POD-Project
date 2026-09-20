"""
utils.py
--------
Shared vocabulary for the chatbot: mood -> audio-feature profiles, mood
synonyms, and small text-cleaning helpers used by more than one module.

Why these mood values (not the brief's example numbers)
---------------------------------------------------------
The brief's sample MOOD_PROFILES uses numbers like tempo=0.8, which only
makes sense if tempo were already scaled to [0, 1]. In the real dataset
tempo ranges 0-243 BPM, loudness ranges -49..+4 dB, etc. So profiles here
are written in the SAME RAW UNITS as the dataset columns; recommender.py
converts them through the fitted scaler before comparing to songs. This
keeps the numbers here readable ("tempo around 145 BPM") while staying
mathematically consistent with how real tracks are compared.

Values were chosen by reasoning about what each mood implies musically,
then sanity-checked against this dataset's genre-level feature averages
(see notebooks/exploration.ipynb) -- e.g. "sleep"/"ambient" genres really
do average low energy + high acousticness + high instrumentalness, which
matches the "chill"/"sad" profiles below.
"""

from __future__ import annotations

import re

# Each mood lists only the features that matter for it. Any feature left
# out is treated as "no preference" (recommender.py fills it with the
# dataset median so it doesn't bias the match).
MOOD_PROFILES: dict[str, dict[str, float]] = {
    "happy": {
        "valence": 0.85,       # musically "happy" = positive-sounding
        "energy": 0.7,
        "danceability": 0.7,
    },
    "sad": {
        "valence": 0.15,
        "energy": 0.25,
        "acousticness": 0.6,
    },
    "energetic": {
        "energy": 0.9,
        "danceability": 0.75,
        "tempo": 135.0,
    },
    "chill": {
        "energy": 0.3,
        "acousticness": 0.6,
        "valence": 0.5,
        "danceability": 0.5,
    },
    "romantic": {
        "valence": 0.6,
        "energy": 0.35,
        "acousticness": 0.5,
    },
    "workout": {
        "energy": 0.9,
        "danceability": 0.8,
        "tempo": 145.0,
    },
    "party": {
        "energy": 0.85,
        "danceability": 0.85,
        "valence": 0.7,
        "loudness": -4.0,
    },
    "focus": {
        # "focus"/"study" music: calm, few vocals/percussive surprises.
        "energy": 0.3,
        "instrumentalness": 0.5,
        "acousticness": 0.5,
        "speechiness": 0.05,
    },
    "sleep": {
        "energy": 0.15,
        "acousticness": 0.7,
        "instrumentalness": 0.5,
        "valence": 0.3,
        "tempo": 75.0,
    },
    "angry": {
        "energy": 0.9,
        "valence": 0.2,
        "loudness": -3.0,
    },
}

# Synonyms map free-text words the user might type onto one canonical mood
# key in MOOD_PROFILES. Keep this flat (word -> mood) so lookups are O(1).
MOOD_SYNONYMS: dict[str, str] = {
    # happy
    "happy": "happy", "joyful": "happy", "cheerful": "happy", "upbeat": "happy",
    "fun": "happy", "feel-good": "happy", "feelgood": "happy", "bright": "happy",
    # sad
    "sad": "sad", "emotional": "sad", "melancholic": "sad", "melancholy": "sad",
    "heartbroken": "sad", "gloomy": "sad", "blue": "sad", "down": "sad",
    # energetic
    "energetic": "energetic", "energy": "energetic", "high-energy": "energetic",
    "powerful": "energetic", "intense": "energetic", "pumped": "energetic",
    "hype": "energetic", "lively": "energetic",
    # chill
    "chill": "chill", "relaxing": "chill", "relaxed": "chill", "calm": "chill",
    "laid-back": "chill", "laidback": "chill", "mellow": "chill", "soothing": "chill",
    "peaceful": "chill", "easygoing": "chill",
    # romantic
    "romantic": "romantic", "romance": "romantic", "love": "romantic",
    "loving": "romantic", "date": "romantic", "date-night": "romantic",
    # workout
    "workout": "workout", "gym": "workout", "exercise": "workout",
    "running": "workout", "run": "workout", "cardio": "workout", "training": "workout",
    # party
    "party": "party", "dance": "party", "dancing": "party", "club": "party",
    "clubbing": "party", "celebration": "party",
    # focus / study
    "focus": "focus", "study": "focus", "studying": "focus", "work": "focus",
    "concentration": "focus", "concentrate": "focus", "reading": "focus",
    "instrumental": "focus",
    # sleep
    "sleep": "sleep", "sleepy": "sleep", "bedtime": "sleep", "night": "sleep",
    "lullaby": "sleep", "rest": "sleep",
    # angry
    "angry": "angry", "aggressive": "angry", "rage": "angry", "furious": "angry",
    "intense-anger": "angry",
}

# Words that describe a *direction* on one feature without naming a full
# mood. Chatbot.py uses these to nudge the profile even when a mood word is
# also present, e.g. "chill acoustic songs" -> chill profile + acousticness up.
FEATURE_HINTS: dict[str, tuple[str, float]] = {
    "acoustic": ("acousticness", 0.8),
    "unplugged": ("acousticness", 0.8),
    "instrumental": ("instrumentalness", 0.7),
    "loud": ("loudness", -3.0),
    "quiet": ("loudness", -25.0),
    "soft": ("loudness", -20.0),
    "fast": ("tempo", 150.0),
    "slow": ("tempo", 70.0),
    "danceable": ("danceability", 0.85),
    "danceability": ("danceability", 0.85),
}


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation, keeping word boundaries clean."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_moods_in_text(text: str) -> list[str]:
    """Return the canonical mood keys mentioned in free text (order-preserving, de-duplicated)."""
    words = normalize_text(text).split()
    found: list[str] = []
    for w in words:
        mood = MOOD_SYNONYMS.get(w)
        if mood and mood not in found:
            found.append(mood)
    return found


def find_feature_hints_in_text(text: str) -> dict[str, float]:
    """Return {feature: value} nudges for words like 'acoustic', 'fast', 'quiet'."""
    words = set(normalize_text(text).split())
    hints: dict[str, float] = {}
    for w in words:
        if w in FEATURE_HINTS:
            feat, val = FEATURE_HINTS[w]
            hints[feat] = val
    return hints
