from src.data_loader import load_raw_data
from src.preprocessing import preprocess_dataset
from src.recommender import MusicRecommender
from sklearn.metrics.pairwise import cosine_similarity

raw = load_raw_data()
b = preprocess_dataset(raw)
df = b['df']
rec = MusicRecommender(df, b['scaler'], b['audio_features'], b['scaled_columns'], b['tfidf_matrix'], b['tfidf_vectorizer'], z_columns=b['z_columns'])

ref_idx = df[df.track_name == "Blinding Lights"].sort_values('popularity', ascending=False).index[0]
# find a deliberately dissimilar track: slow, acoustic, low-energy classical/ambient
candidates = df[(df.track_genre.isin(['classical','ambient','sleep'])) & (df.energy < 0.1)]
print("dissimilar candidate pool size:", len(candidates))
c_idx = candidates.index[0]
print("dissimilar track:", df.loc[c_idx, ['track_name','artists','track_genre']].to_dict())

sim = cosine_similarity(rec._z_matrix[ref_idx].reshape(1,-1), rec._z_matrix[c_idx].reshape(1,-1))[0][0]
friendly = (sim+1)/2
print(f"raw z-cosine to dissimilar track: {sim:.3f}  friendly: {friendly:.3f}")

# also check overall score distribution across the full catalogue, not just top-5
import numpy as np
all_sims = cosine_similarity(rec._z_matrix[ref_idx].reshape(1,-1), rec._z_matrix)[0]
friendly_all = (all_sims+1)/2
print("\nfriendly score distribution across ALL songs vs Blinding Lights:")
print("  min:", friendly_all.min(), "p10:", np.percentile(friendly_all,10), "p50:", np.percentile(friendly_all,50), "p90:", np.percentile(friendly_all,90), "max (excl self):", np.sort(friendly_all)[-2])
