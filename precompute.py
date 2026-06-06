"""Offline pipeline -> data.parquet + sim_clip.npy + sim_text.npy.

Real run needs CBS buurten, PDOK aerial tiles, OpenCLIP and a sentence-transformer.
Without those (or with --mock) it writes random demo data so app.py runs immediately.

Real-mode sketch (fill in the TODOs, then call build(emb_clip, emb_text, df_cbs)):
  - buurten: geopandas.read_file(<PDOK Wijk- en buurtkaart>); centroid -> lat/lon
  - clip:    for each centroid fetch one PDOK aerial tile, OpenCLIP image embed
  - text:    SentenceTransformer(...).encode(descriptions)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA 
import umap

CBS_COLS = ["population", "income", "pct_green", "avg_age", "home_value", "density"]


def cosine(emb):
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return e @ e.T


def pca2(emb):
    return PCA(n_components=2).fit_transform(emb)

def umap2(emb):
    reducer = umap.UMAP()
    return reducer.fit_transform(emb)


def build(emb_clip, emb_text, df):
    """Attach PCA coords, save parquet + similarity matrices.

    The CBS 'dense vector' is just the standardized CBS stat columns -> its own space."""
    df = df.copy()
    emb_cbs = ((df[CBS_COLS] - df[CBS_COLS].mean()) / (df[CBS_COLS].std() + 1e-9)).values
    df[["clip_x", "clip_y"]] = pca2(emb_clip) # Change to umap2(emb_clip) for umap projections
    df[["text_x", "text_y"]] = pca2(emb_text)
    df[["cbs_x", "cbs_y"]] = pca2(emb_cbs)
    df.to_parquet("data.parquet")
    np.save("sim_clip.npy", cosine(emb_clip))
    np.save("sim_text.npy", cosine(emb_text))
    np.save("sim_cbs.npy", cosine(emb_cbs))
    print(f"wrote data.parquet ({len(df)} rows) + sim_clip/text/cbs.npy")


def mock(n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "code": [f"BU{i:04d}" for i in range(n)],
        "name": [f"Buurt {i}" for i in range(n)],
        "lat": 52.37 + rng.normal(0, 0.05, n), 
        "lon": 4.90 + rng.normal(0, 0.08, n),
    })
    for c in CBS_COLS:
        df[c] = rng.integers(10, 100, n)
    build(rng.normal(size=(n, 64)), rng.normal(size=(n, 64)), df)


if __name__ == "__main__":
    mock()
