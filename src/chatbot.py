"""
chatbot.py
----------
The conversational layer: turns a free-text user message into an intent +
extracted entities, calls the right MusicRecommender method, and formats a
friendly response. Intentionally rule-based (regex + keyword lookup) rather
than an LLM, so the whole pipeline is transparent and needs no API key --
exactly what the project brief asks for.

Supported intents
------------------
  similar_song   "songs similar to <song>", "recommend something like <song>"
  by_artist      "songs by <artist>", "more <artist>"
  by_genre       "some k-pop songs", "jazz recommendations"
  by_mood        "chill songs for studying", "I want something energetic"
  general_query  anything else with real content -> hybrid/text search
  help           "help", "what can you do"
  greeting       "hi", "hello"
  empty          blank input
"""

from __future__ import annotations

import re

from src.recommender import MusicRecommender
from src.utils import MOOD_PROFILES, find_feature_hints_in_text, find_moods_in_text, normalize_text

HELP_TEXT = (
    "I can recommend songs based on:\n"
    "- **Mood or activity** -> \"I want chill songs for studying\", \"energetic workout songs\"\n"
    "- **A specific song** -> \"songs similar to Blinding Lights\"\n"
    "- **An artist** -> \"songs by Ed Sheeran\"\n"
    "- **A genre** -> \"recommend some k-pop\"\n"
    "- **General taste** -> \"upbeat pop songs with high energy\"\n\n"
    "Try one of the examples above, or just tell me what you're in the mood for!"
)

GREETING_WORDS = {"hi", "hello", "hey", "yo", "sup", "hiya", "greetings"}
HELP_WORDS = {"help", "commands", "about", "what can you do", "how does this work"}

# Regex patterns for the "similar to <song>" and "by <artist>" intents.
# NOTE: deliberately does NOT include a bare "like X" pattern -- "I LIKE
# upbeat pop songs" is a preference statement, not a request for songs
# similar to a reference track named "upbeat pop songs". Every pattern
# below requires an unambiguous marker ("similar to", "sounds like", or a
# music noun immediately before "like") so ordinary "I like ..." sentences
# fall through to mood/general-query handling instead.
SIMILAR_PATTERNS = [
    r"similar to (?:the song )?[\"']?(.+?)[\"']?$",
    r"sounds? like [\"']?(.+?)[\"']?$",
    r"(?:songs?|tracks?|music|something|anything) (?:that are |that sound )?like [\"']?(.+?)[\"']?$",
    r"recommend(?:ations?)? like [\"']?(.+?)[\"']?$",
]
ARTIST_PATTERNS = [
    r"(?:songs?|music|tracks?) by [\"']?(.+?)[\"']?$",
    r"more (?:from|by) [\"']?(.+?)[\"']?$",
    r"artist(?:s)? like [\"']?(.+?)[\"']?$",
]
GENRE_PATTERNS = [
    r"some (.+?) (?:songs|music|tracks)$",
    r"(.+?) (?:genre|songs|music) recommendations?$",
    r"recommend (?:me )?(?:some )?(.+?)(?: songs| music)?$",
]


class MusicChatbot:
    def __init__(self, recommender: MusicRecommender):
        self.recommender = recommender

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def process_message(self, message: str, n: int = 5, audio_weight: float = 0.7,
                         text_weight: float = 0.3, filters: dict | None = None) -> dict:
        """
        Main entry point. Returns a dict:
            {"intent": str, "text": str, "results": list[dict], "reference": dict|None}
        `text` is a ready-to-display chat response; `results`/`reference` let
        the UI render structured song cards as well.
        """
        if message is None or not message.strip():
            return self._respond("empty", "Please type a message -- for example, "
                                            "\"I want some chill songs for studying\".")

        # Guard against an invalid recommendation count instead of letting it
        # silently produce zero (or an absurdly long) results list.
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 5
        if n < 1:
            return self._respond("invalid_input", "Please request at least 1 recommendation.")
        n = min(n, 50)  # sane upper bound so the UI never has to render hundreds of cards

        clean = message.strip()
        intent, entity = self._detect_intent(clean)

        try:
            if intent == "greeting":
                return self._respond(intent, "Hi! Tell me what you're in the mood for, "
                                              "or type 'help' to see what I can do.")
            if intent == "help":
                return self._respond(intent, HELP_TEXT)
            if intent == "similar_song":
                return self._handle_similar_song(entity, n)
            if intent == "by_artist":
                return self._handle_artist(entity, n)
            if intent == "by_genre":
                return self._handle_genre(entity, n)
            if intent == "by_mood":
                return self._handle_mood(clean, entity, n, audio_weight, text_weight, filters)
            return self._handle_general(clean, n, audio_weight, text_weight, filters)
        except Exception as exc:  # last-resort safety net -- the UI must never crash
            return self._respond("error", "Something went wrong while looking that up. "
                                           f"Please try rephrasing your request. ({exc})")

    # ------------------------------------------------------------------ #
    # Intent detection
    # ------------------------------------------------------------------ #

    def _detect_intent(self, message: str) -> tuple[str, str]:
        norm = normalize_text(message)

        if norm in GREETING_WORDS or norm.rstrip("!.") in GREETING_WORDS:
            return "greeting", ""
        if norm in HELP_WORDS or "what can you do" in norm or norm.startswith("help"):
            return "help", ""

        for pat in SIMILAR_PATTERNS:
            m = re.search(pat, norm)
            if m:
                return "similar_song", m.group(1).strip()

        for pat in ARTIST_PATTERNS:
            m = re.search(pat, norm)
            if m:
                return "by_artist", m.group(1).strip()

        moods = find_moods_in_text(message)
        if moods:
            return "by_mood", moods[0]

        for pat in GENRE_PATTERNS:
            m = re.search(pat, norm)
            if m:
                candidate = m.group(1).strip()
                # Only treat it as a genre if it actually names one we know;
                # otherwise fall through to general/text search.
                if self._looks_like_genre(candidate):
                    return "by_genre", candidate

        return "general_query", message

    def _looks_like_genre(self, candidate: str) -> bool:
        known_genres = set(self.recommender.df["track_genre"].dropna().str.lower().unique())
        candidate = candidate.strip().lower()
        if candidate in known_genres:
            return True
        return any(candidate == g or candidate in g.split("-") for g in known_genres)

    # ------------------------------------------------------------------ #
    # Intent handlers
    # ------------------------------------------------------------------ #

    def _handle_similar_song(self, song_name: str, n: int) -> dict:
        if not song_name:
            return self._respond("similar_song", "Which song would you like similar recommendations for?")
        res = self.recommender.recommend_by_song(song_name, n=n)
        if not res["ok"]:
            return self._respond(
                "similar_song",
                f"I couldn't find \"{song_name}\" in my dataset. "
                "Try another song, artist, genre, or mood.",
            )
        text = self._format_song_list(
            res["results"],
            header=f"Because you liked **{res['reference']['track_name']}** by "
                   f"{res['reference']['artists']}, here are some similar-sounding tracks:",
        )
        return {"intent": "similar_song", "text": text, "results": res["results"], "reference": res["reference"]}

    def _handle_artist(self, artist_name: str, n: int) -> dict:
        if not artist_name:
            return self._respond("by_artist", "Which artist are you interested in?")
        matches = self.recommender.find_artist(artist_name)
        if matches.empty:
            return self._respond(
                "by_artist",
                f"I couldn't find an artist called \"{artist_name}\" in my dataset. "
                "Try another song, artist, genre, or mood.",
            )
        top = matches.sort_values("popularity", ascending=False).head(n) if "popularity" in matches.columns else matches.head(n)
        results = []
        for idx, row in top.iterrows():
            results.append({
                "idx": idx, "score": (row.get("popularity", 0) or 0) / 100.0,
                "track_name": row["track_name"], "artists": row["artists"],
                "album_name": row.get("album_name", "Unknown"),
                "genre": row.get("track_genre", "Unknown"),
                "explanation": f"One of {row['artists']}'s popular tracks.",
            })
        text = self._format_song_list(results, header=f"Here are some popular tracks by **{artist_name}**:")
        return {"intent": "by_artist", "text": text, "results": results, "reference": None}

    def _handle_genre(self, genre: str, n: int) -> dict:
        res = self.recommender.recommend_by_genre(genre, n=n)
        if not res["ok"]:
            return self._respond(
                "by_genre",
                f"I couldn't find the genre \"{genre}\" in my dataset. "
                "Try another song, artist, genre, or mood.",
            )
        text = self._format_song_list(res["results"], header=f"Here are some popular **{genre}** tracks:")
        return {"intent": "by_genre", "text": text, "results": res["results"], "reference": None}

    def _handle_mood(self, original_message: str, mood: str, n: int,
                      audio_weight: float, text_weight: float, filters: dict | None) -> dict:
        overrides = find_feature_hints_in_text(original_message)
        res = self.recommender.recommend_hybrid(
            query=original_message, mood=mood, n=n * 3,  # over-fetch, then filter
            audio_weight=audio_weight, text_weight=text_weight, feature_overrides=overrides,
        )
        if not res["ok"]:
            return self._respond("by_mood", "I couldn't find songs matching that mood. Try another mood or describe it differently.")
        results = self._apply_filters(res["results"], filters)[:n]
        if not results:
            return self._respond("by_mood", "No songs matched your mood plus the current filters. Try loosening the sidebar filters.")
        text = self._format_song_list(results, header=f"Here are some songs that match your **{mood}** mood:")
        return {"intent": "by_mood", "text": text, "results": results, "reference": None}

    def _handle_general(self, message: str, n: int, audio_weight: float,
                         text_weight: float, filters: dict | None) -> dict:
        res = self.recommender.recommend_hybrid(query=message, n=n * 3, audio_weight=audio_weight, text_weight=text_weight)
        if not res["ok"]:
            return self._respond(
                "general_query",
                "I couldn't find anything matching that. Try mentioning a mood "
                "(chill, energetic, romantic...), a genre, an artist, or a song name.",
            )
        results = self._apply_filters(res["results"], filters)[:n]
        if not results:
            return self._respond("general_query", "No songs matched that plus the current filters. Try loosening the sidebar filters.")
        text = self._format_song_list(results, header="Here are some songs that match your request:")
        return {"intent": "general_query", "text": text, "results": results, "reference": None}

    # ------------------------------------------------------------------ #
    # Formatting / filtering helpers
    # ------------------------------------------------------------------ #

    def _apply_filters(self, results: list[dict], filters: dict | None) -> list[dict]:
        if not filters:
            return results
        df = self.recommender.df
        out = []
        for r in results:
            row = df.loc[r["idx"]]
            if filters.get("genre") and filters["genre"] != "Any":
                genres = row.get("genres", [row.get("track_genre", "")])
                if filters["genre"].lower() not in [g.lower() for g in genres]:
                    continue
            if filters.get("min_energy") is not None and row.get("energy", 0) < filters["min_energy"]:
                continue
            if filters.get("max_energy") is not None and row.get("energy", 1) > filters["max_energy"]:
                continue
            if filters.get("min_danceability") is not None and row.get("danceability", 0) < filters["min_danceability"]:
                continue
            if filters.get("max_danceability") is not None and row.get("danceability", 1) > filters["max_danceability"]:
                continue
            if filters.get("explicit_ok") is False and bool(row.get("explicit", False)):
                continue
            out.append(r)
        return out

    def _format_song_list(self, results: list[dict], header: str) -> str:
        lines = [header, ""]
        for i, r in enumerate(results, start=1):
            pct = int(round(r["score"] * 100))
            album = r.get("album_name") or "Unknown"
            genre = r.get("genre") or "Unknown"
            lines.append(
                f"**{i}. {r['track_name']}** — {r['artists']}\n"
                f"   Album: {album} | Genre: {genre} | Match: {pct}%\n"
                f"   _Why:_ {r['explanation']}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _respond(intent: str, text: str) -> dict:
        return {"intent": intent, "text": text, "results": [], "reference": None}
