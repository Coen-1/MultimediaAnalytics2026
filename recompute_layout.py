"""Re-derive the 2D UMAP layout + cluster labels for each modality straight from the
precomputed cosine-similarity matrices (no embeddings / heavy models needed).

For each space (clip / text / cbs):
  distance D = 1 - cosine_sim   (symmetric, zero diagonal)
  - UMAP(metric="precomputed") with tuned params -> tighter, better-separated blobs
  - KMeans on the resulting 2D coords -> balanced, contiguous cluster colours that
    match the blobs you actually see (precomputed agglomerative chained into one
    giant cluster + singletons here, so it is not used)

Writes <space>_x, <space>_y and <space>_cluster back into assets/data.parquet.
Run: python recompute_layout.py   (assets/data.parquet.bak holds the original)
"""
import numpy as np, pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ASSETS = "assets"
SPACES = ["clip", "text", "cbs"]
N_CLUSTERS = 8          # distinct colours per projection
N_NEIGHBORS = 30        # a bit more global structure than the default 15
MIN_DIST = 0.05         # tighter packing -> clearer separation between groups
SEED = 42


def layout(sim):
    d = 1.0 - sim                       # cosine sim -> distance
    np.fill_diagonal(d, 0.0)
    d[d < 0] = 0.0                      # guard tiny negatives from float error
    xy = umap.UMAP(metric="precomputed", n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
                   random_state=SEED).fit_transform(d)
    labels = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit_predict(
        StandardScaler().fit_transform(xy))
    return xy, labels


def main():
    df = pd.read_parquet(f"{ASSETS}/data.parquet")
    for s in SPACES:
        sim = np.load(f"{ASSETS}/sim_{s}.npy")
        xy, labels = layout(sim)
        df[f"{s}_x"], df[f"{s}_y"] = xy[:, 0], xy[:, 1]
        df[f"{s}_cluster"] = labels.astype(int)
        print(f"{s}: re-laid out {len(df)} buurten, {len(set(labels))} clusters")
    df.to_parquet(f"{ASSETS}/data.parquet")
    print("wrote assets/data.parquet with new *_x/*_y/*_cluster columns")


if __name__ == "__main__":
    main()
