from src.data_loader import load_raw_data
from src.preprocessing import preprocess_dataset
from src.recommender import MusicRecommender
from src.chatbot import MusicChatbot

raw = load_raw_data()
b = preprocess_dataset(raw)
rec = MusicRecommender(b['df'], b['scaler'], b['audio_features'], b['scaled_columns'], b['tfidf_matrix'], b['tfidf_vectorizer'], z_columns=b['z_columns'])
bot = MusicChatbot(rec)

queries = [
    "Suggest some chill songs for studying.",
    "I want energetic songs for a workout.",
    "Recommend songs similar to Blinding Lights.",
    "Give me some romantic songs.",
    "I like upbeat pop songs with high energy.",
    "I want relaxing instrumental music.",
    "Songs similar to Shape of You",
    "songs by Ed Sheeran",
    "recommend some k-pop songs",
    "hi",
    "help",
    "",
    "   ",
    "Recommend songs similar to XYZ123NoSuchSong",
    "songs by ZZZNoSuchArtist999",
]

for q in queries:
    r = bot.process_message(q, n=3)
    print("="*70)
    print("USER:", repr(q))
    print("INTENT:", r['intent'])
    print(r['text'][:500])
    print()
