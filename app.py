"""
app.py
------
Streamlit chat UI for the Music Recommendation Chatbot.

This file only wires up the interface: session state, sidebar controls,
and the chat window. All recommendation and NLP logic lives in src/, kept
separate so the core logic can be tested and reasoned about without a
browser (see .scratch/test_*.py and the run instructions in README.md).
"""

from __future__ import annotations

import streamlit as st

from src.chatbot import MusicChatbot
from src.data_loader import DatasetError, describe_dataset, load_raw_data
from src.preprocessing import preprocess_dataset
from src.recommender import MusicRecommender

st.set_page_config(page_title="Music Recommendation Chatbot", page_icon="\U0001F3B5", layout="wide")


# ---------------------------------------------------------------------- #
# Data / model loading (cached so it only runs once per session)
# ---------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading dataset and building the recommendation engine...")
def load_chatbot():
    """
    Load the dataset, preprocess it, and build the recommender + chatbot.
    Cached across reruns -- Streamlit reruns this script top-to-bottom on
    every interaction, so without caching we'd reload 80k+ songs on every click.
    """
    raw_df = load_raw_data()
    report = describe_dataset(raw_df)
    bundle = preprocess_dataset(raw_df)
    recommender = MusicRecommender(
        bundle["df"], bundle["scaler"], bundle["audio_features"],
        bundle["scaled_columns"], bundle["tfidf_matrix"], bundle["tfidf_vectorizer"],
        z_columns=bundle["z_columns"],
    )
    chatbot = MusicChatbot(recommender)
    genres = sorted(set(g for gs in bundle["df"]["genres"] for g in gs)) if "genres" in bundle["df"].columns else []
    has_explicit = "explicit" in bundle["df"].columns
    return chatbot, report, genres, has_explicit


# ---------------------------------------------------------------------- #
# App startup -- handle a missing/broken dataset gracefully (Section 10)
# ---------------------------------------------------------------------- #

try:
    chatbot, dataset_report, all_genres, has_explicit_col = load_chatbot()
except DatasetError as exc:
    st.error(
        "**Could not load the music dataset.**\n\n"
        f"{exc}\n\n"
        "Fix the data source and reload this page."
    )
    st.stop()


# ---------------------------------------------------------------------- #
# Sidebar controls
# ---------------------------------------------------------------------- #

with st.sidebar:
    st.header("Settings")

    n_recs = st.slider("Number of recommendations", min_value=1, max_value=15, value=5)

    st.subheader("Hybrid scoring weights")
    st.caption(
        "Controls the blend used for mood/general requests: "
        "`final_score = audio_weight * audio_similarity + text_weight * text_similarity`"
    )
    audio_weight = st.slider("Audio-feature weight", 0.0, 1.0, 0.7, 0.05)
    text_weight = round(1.0 - audio_weight, 2)
    st.caption(f"Text weight is automatically set to **{text_weight}** (weights sum to 1).")

    st.subheader("Filters")
    genre_filter = st.selectbox("Genre", options=["Any"] + all_genres, index=0)
    min_energy, max_energy = st.slider("Energy range", 0.0, 1.0, (0.0, 1.0), 0.05)
    min_dance, max_dance = st.slider("Danceability range", 0.0, 1.0, (0.0, 1.0), 0.05)
    explicit_ok = True
    if has_explicit_col:
        explicit_ok = st.checkbox("Allow explicit tracks", value=True)

    with st.expander("Dataset info"):
        st.text(dataset_report)

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

filters = {
    "genre": genre_filter,
    "min_energy": min_energy, "max_energy": max_energy,
    "min_danceability": min_dance, "max_danceability": max_dance,
    "explicit_ok": explicit_ok,
}


# ---------------------------------------------------------------------- #
# Header
# ---------------------------------------------------------------------- #

st.title("\U0001F3B5 Music Recommendation Chatbot")
st.caption("Tell me what you're in the mood for and I'll recommend some songs.")

EXAMPLES = [
    "Suggest some chill songs for studying.",
    "I want energetic songs for a workout.",
    "Recommend songs similar to Blinding Lights.",
    "Give me some romantic songs.",
    "I like upbeat pop songs with high energy.",
    "I want relaxing instrumental music.",
]

# ---------------------------------------------------------------------- #
# Chat state
# ---------------------------------------------------------------------- #

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "text": "Hi! I'm your music recommendation assistant. "
                                       "Tell me what you're in the mood for, or try one of the example prompts below.",
         "results": []}
    ]

with st.expander("Example queries (click to copy inspiration)"):
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        cols[i % 2].markdown(f"- _{ex}_")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("results"):
            for r in msg["results"]:
                with st.container(border=True):
                    pct = int(round(r["score"] * 100))
                    st.markdown(f"**{r['track_name']}** — {r['artists']}")
                    st.caption(
                        f"Album: {r.get('album_name', 'Unknown')} | "
                        f"Genre: {r.get('genre', 'Unknown')} | Match: {pct}%"
                    )
                    st.markdown(f"*Why recommended:* {r['explanation']}")


# ---------------------------------------------------------------------- #
# Chat input
# ---------------------------------------------------------------------- #

user_input = st.chat_input("Ask for song recommendations...")

if user_input is not None:
    st.session_state.messages.append({"role": "user", "text": user_input, "results": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Finding songs..."):
            response = chatbot.process_message(
                user_input, n=n_recs, audio_weight=audio_weight,
                text_weight=text_weight, filters=filters,
            )
        st.markdown(response["text"])
        for r in response["results"]:
            with st.container(border=True):
                pct = int(round(r["score"] * 100))
                st.markdown(f"**{r['track_name']}** — {r['artists']}")
                st.caption(
                    f"Album: {r.get('album_name', 'Unknown')} | "
                    f"Genre: {r.get('genre', 'Unknown')} | Match: {pct}%"
                )
                st.markdown(f"*Why recommended:* {r['explanation']}")

    st.session_state.messages.append(
        {"role": "assistant", "text": response["text"], "results": response["results"]}
    )
