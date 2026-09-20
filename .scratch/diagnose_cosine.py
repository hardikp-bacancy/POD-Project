import numpy as np
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_dataset
from sklearn.metrics.pairwise import cosine_similarity

raw = load_raw_data()
b = preprocess_dataset(raw)
df = b['df']
feats = b['scaled_columns']
X = df[feats].to_numpy()

ref_idx = df[df.track_name == "Blinding Lights"].sort_values('popularity', ascending=False).index[0]
print("Reference:", df.loc[ref_idx, ['track_name','artists']+b['audio_features']].to_string())

# cosine on MinMax [0,1] features
ref = X[ref_idx].reshape(1,-1)
cos = cosine_similarity(ref, X)[0]
print("\ncosine sim stats: min", cos.min(), "max", cos.max(), "mean", cos.mean(), "std", cos.std())
print("cosine top5 idx:", np.argsort(-cos)[:6])

# euclidean distance -> closeness
dist = np.linalg.norm(X - X[ref_idx], axis=1)
closeness = 1 - dist / np.sqrt(len(feats))
print("\neuclid closeness stats: min", closeness.min(), "max", closeness.max(), "mean", closeness.mean(), "std", closeness.std())
order = np.argsort(-closeness)
order = [i for i in order if i != ref_idx][:5]
print("\nEuclidean nearest neighbors to Blinding Lights:")
print(df.loc[order, ['track_name','artists','track_genre']+b['audio_features']].to_string())
