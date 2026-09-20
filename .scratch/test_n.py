from src.data_loader import load_raw_data
from src.preprocessing import preprocess_dataset
from src.recommender import MusicRecommender
from src.chatbot import MusicChatbot

raw = load_raw_data()
b = preprocess_dataset(raw)
rec = MusicRecommender(b['df'], b['scaler'], b['audio_features'], b['scaled_columns'], b['tfidf_matrix'], b['tfidf_vectorizer'], z_columns=b['z_columns'])
bot = MusicChatbot(rec)
for bad_n in [0, -5, 1000, 3]:
    r = bot.process_message("chill songs", n=bad_n)
    print(f"n={bad_n} -> intent={r['intent']}, num_results={len(r['results'])}, text[:60]={r['text'][:60]!r}")
