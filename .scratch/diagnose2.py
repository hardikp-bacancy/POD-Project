import numpy as np
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_dataset, AUDIO_FEATURES
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

raw = load_raw_data()
b = preprocess_dataset(raw)
df = b['df']

Z = StandardScaler().fit_transform(df[b['audio_features']])
ref_idx = df[df.track_name == "Blinding Lights"].sort_values('popularity', ascending=False).index[0]
cos_z = cosine_similarity(Z[ref_idx].reshape(1,-1), Z)[0]
print("z-score cosine stats: min", cos_z.min(), "max", cos_z.max(), "mean", cos_z.mean(), "std", cos_z.std())
order = np.argsort(-cos_z)
order = [i for i in order if i != ref_idx][:5]
print(df.loc[order, ['track_name','artists','track_genre']].to_string())
