"""
tests/test_queries.py
----------------------
A small, honest evaluation of the chatbot against manually written example
queries drawn directly from the project brief.

IMPORTANT -- what this is and isn't:
This dataset has no ground-truth "correct recommendations" for a query like
"I want chill songs for studying" (no user ever rated these songs against
that specific request). So we cannot report precision/recall/accuracy
numbers -- doing so would be fabricated. What we CAN check, and what this
script does check, is whether the system behaves correctly at the level we
can actually verify:
  - it recognises the intent we'd expect a human to recognise,
  - it returns a non-empty, non-crashing result for a satisfiable request,
  - it degrades gracefully (clear message, no exception) for an
    unsatisfiable one (unknown song / artist / genre, empty input).

Run with:  PYTHONPATH=. python tests/test_queries.py
"""

from __future__ import annotations

from src.chatbot import MusicChatbot
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_dataset
from src.recommender import MusicRecommender

# (query, expected_intent, expect_results) -- expect_results=False means we
# expect a graceful "couldn't find" style response with zero song results.
TEST_CASES = [
    ("Give me energetic workout songs.", "by_mood", True),
    ("Recommend relaxing songs.", "by_mood", True),
    ("I want happy pop music.", "by_mood", True),
    ("Songs similar to Blinding Lights.", "similar_song", True),
    ("Give me romantic songs.", "by_mood", True),
    ("I want acoustic music for studying.", "by_mood", True),
    ("Suggest some chill songs for studying.", "by_mood", True),
    ("I like upbeat pop songs with high energy.", "by_mood", True),
    ("songs by Ed Sheeran", "by_artist", True),
    ("recommend some k-pop songs", "by_genre", True),
    ("Songs similar to XYZ123NotARealSong", "similar_song", False),
    ("songs by NotARealArtistXYZ999", "by_artist", False),
    ("", "empty", False),
    ("help", "help", False),
]


def run_evaluation():
    raw = load_raw_data()
    bundle = preprocess_dataset(raw)
    recommender = MusicRecommender(
        bundle["df"], bundle["scaler"], bundle["audio_features"],
        bundle["scaled_columns"], bundle["tfidf_matrix"], bundle["tfidf_vectorizer"],
        z_columns=bundle["z_columns"],
    )
    chatbot = MusicChatbot(recommender)

    rows = []
    for query, expected_intent, expect_results in TEST_CASES:
        try:
            response = chatbot.process_message(query, n=5)
            crashed = False
        except Exception as exc:  # pragma: no cover - the whole point is this should never happen
            response = {"intent": "EXCEPTION", "results": []}
            crashed = True
            crash_msg = str(exc)

        intent_ok = response["intent"] == expected_intent
        results_ok = (len(response["results"]) > 0) == expect_results
        passed = intent_ok and results_ok and not crashed

        rows.append({
            "query": query or "(empty string)",
            "expected_intent": expected_intent,
            "actual_intent": response["intent"],
            "expected_results": "some" if expect_results else "none (graceful message)",
            "actual_n_results": len(response["results"]),
            "passed": passed,
        })

    return rows


def print_report(rows: list[dict]):
    print("=" * 100)
    print("PROTOTYPE EVALUATION -- manually defined test queries, not a ground-truth benchmark")
    print("=" * 100)
    n_passed = sum(r["passed"] for r in rows)
    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{status}] Query: {r['query']!r}")
        print(f"       Expected intent: {r['expected_intent']:<15} Actual intent: {r['actual_intent']}")
        print(f"       Expected results: {r['expected_results']:<28} Actual # results: {r['actual_n_results']}")
    print("\n" + "-" * 100)
    print(f"Summary: {n_passed}/{len(rows)} test cases behaved as expected "
          f"(correct intent detection + correct success/failure handling).")
    print(
        "\nNote: this counts whether the chatbot recognised the right kind of request and "
        "returned/withheld results appropriately -- it does NOT measure whether individual "
        "song recommendations are 'good' in a musical taste sense, since no ground-truth "
        "relevance labels exist for this dataset. Judging recommendation quality for the "
        "'passed' cases is a manual, subjective step (see README.md's Evaluation section)."
    )


if __name__ == "__main__":
    print_report(run_evaluation())
