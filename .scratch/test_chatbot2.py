from src.data_loader import load_raw_data
from src.preprocessing import preprocess_dataset
from src.recommender import MusicRecommender
from src.chatbot import MusicChatbot

raw = load_raw_data()
b = preprocess_dataset(raw)
rec = MusicRecommender(b['df'], b['scaler'], b['audio_features'], b['scaled_columns'], b['tfidf_matrix'], b['tfidf_vectorizer'], z_columns=b['z_columns'])
bot = MusicChatbot(rec)

queries = [
    "Recommend songs similar to Blinding Lights.",   # must still be similar_song
    "songs similar to shape of you",                  # lowercase
    "I want relaxing acoustic music",
    "give me some sad songs",
    "n=0 edge case",                                   # weird but shouldn't crash
]
for q in queries:
    r = bot.process_message(q, n=3)
    print("="*60, "\nUSER:", q, "\nINTENT:", r['intent'], "\n", r['text'][:200])

# invalid n values
for bad_n in [0, -5, 1000]:
    r = bot.process_message("chill songs", n=bad_n)
    print(f"\nn={bad_n} -> intent={r['intent']}, num_results={len(r['results'])}")
