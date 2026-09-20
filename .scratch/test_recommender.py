from src.data_loader import load_raw_data
from src.preprocessing import preprocess_dataset
from src.recommender import MusicRecommender

raw = load_raw_data()
b = preprocess_dataset(raw)
rec = MusicRecommender(b['df'], b['scaler'], b['audio_features'], b['scaled_columns'], b['tfidf_matrix'], b['tfidf_vectorizer'], z_columns=b['z_columns'])

def show(title, res):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    if not res['ok']:
        print("FAILED:", res['reason']); return
    if res.get('reference'):
        print("Reference:", res['reference']['track_name'], '-', res['reference']['artists'])
    for r in res['results']:
        print(f"  [{r['score']:.3f}] {r['track_name']} — {r['artists']} ({r.get('genre')})")
        print(f"          why: {r['explanation']}")

show("1) recommend_by_song('Blinding Lights')", rec.recommend_by_song("Blinding Lights", n=5))
show("2) recommend_by_song('NoSuchSongXYZ123')", rec.recommend_by_song("NoSuchSongXYZ123", n=5))
show("3) recommend_by_mood('chill')", rec.recommend_by_mood("chill", n=5))
show("4) recommend_by_mood('workout')", rec.recommend_by_mood("workout", n=5))
show("5) recommend_by_mood('unknownmood')", rec.recommend_by_mood("unknownmood", n=5))
show("6) recommend_by_query('acoustic love song piano')", rec.recommend_by_query("acoustic love song piano", n=5))
show("7) recommend_by_genre('k-pop')", rec.recommend_by_genre("k-pop", n=5))
show("8) recommend_hybrid(mood='chill', query='acoustic guitar')", rec.recommend_hybrid(mood="chill", query="acoustic guitar", n=5))
show("9) recommend_hybrid(song_name='Shape of You', query='pop')", rec.recommend_hybrid(song_name="Shape of You", query="pop", n=5))
show("10) empty query", rec.recommend_by_query("", n=5))
