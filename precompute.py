"""Offline pipeline -> data.parquet + sim_{clip,text,cbs}.npy.

Runs in MOCK mode by default: generates ~600 synthetic Dutch neighbourhoods
anchored to real city centres, organised around six archetypes. The CLIP and
text embeddings are drawn from a shared archetype latent plus per-space noise,
so the three projected spaces are visibly correlated but not identical — which
is what makes the cross-space comparison interesting in the demo.

Real run:
    - buurten:   geopandas.read_file(<PDOK Wijk- en buurtkaart>); centroid -> lat/lon
                 https://service.pdok.nl/cbs/wijkenbuurten/2023/atom/wijkenbuurten_2023.xml
    - CBS stats: CBS Open Data "Kerncijfers wijken en buurten" (Statline / OData);
                 join on `buurtcode`.
    - CLIP:      for each centroid fetch one PDOK Actueel_orthoHR aerial tile
                 (256x256 at z=15 covers ~1.2km), run open_clip (ViT-B/32 or the
                 RemoteCLIP / GeoCLIP variant). Stack to (N, D) np.float32.
    - Text:      neighbourhood descriptions from the Canvas dataset;
                 sentence-transformers e.g. "intfloat/multilingual-e5-base".

    Then: build(emb_clip, emb_text, df_cbs_with_latlon_and_text).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

CBS_COLS = ["population", "income", "pct_green", "avg_age", "home_value", "density"]
N_CLUSTERS = 6

# (name, CBS profile mean: pop, inc, green%, age, home_value_k, density)
ARCHETYPES = [
    ("urban_core",        [90, 72, 12, 36, 480, 95]),
    ("suburban",          [55, 68, 38, 44, 380, 50]),
    ("rural_farmland",    [15, 46, 78, 53, 280, 8]),
    ("coastal",           [35, 76, 52, 56, 540, 28]),
    ("industrial_port",   [48, 50, 14, 42, 220, 70]),
    ("woodland_village",  [22, 62, 68, 51, 430, 16]),
]

# (city, lat, lon, archetype weights[6], #neighbourhoods)
CITIES = [
    ("Amsterdam",  52.37, 4.90, [5, 2, 0, 0, 1, 0], 80),
    ("Rotterdam",  51.92, 4.48, [3, 2, 0, 0, 3, 0], 60),
    ("Den Haag",   52.08, 4.30, [3, 2, 0, 2, 0, 0], 45),
    ("Utrecht",    52.09, 5.12, [3, 3, 0, 0, 1, 1], 40),
    ("Eindhoven",  51.44, 5.48, [2, 3, 0, 0, 2, 0], 30),
    ("Groningen",  53.22, 6.57, [2, 2, 1, 0, 0, 1], 22),
    ("Tilburg",    51.56, 5.09, [2, 3, 0, 0, 1, 0], 22),
    ("Almere",     52.37, 5.21, [1, 4, 0, 0, 0, 1], 22),
    ("Breda",      51.59, 4.78, [2, 3, 0, 0, 0, 1], 20),
    ("Nijmegen",   51.81, 5.84, [2, 2, 0, 0, 0, 1], 20),
    ("Haarlem",    52.39, 4.64, [2, 3, 0, 1, 0, 0], 18),
    ("Arnhem",     51.98, 5.91, [2, 2, 0, 0, 1, 1], 18),
    ("Maastricht", 50.85, 5.69, [2, 2, 1, 0, 0, 1], 16),
    ("Apeldoorn",  52.21, 5.97, [1, 2, 1, 0, 0, 2], 16),
    ("Leeuwarden", 53.20, 5.78, [1, 2, 2, 0, 0, 1], 14),
    ("Zwolle",     52.51, 6.09, [1, 2, 1, 0, 0, 1], 14),
    ("Amersfoort", 52.16, 5.39, [1, 3, 0, 0, 0, 1], 14),
    ("Leiden",     52.16, 4.49, [2, 2, 0, 1, 0, 0], 16),
    ("Delft",      52.01, 4.36, [2, 2, 0, 0, 0, 0], 12),
    ("Zaanstad",   52.45, 4.81, [1, 2, 0, 0, 2, 0], 16),
    ("Zandvoort",  52.37, 4.53, [0, 1, 0, 3, 0, 0], 8),
    ("Vlissingen", 51.45, 3.58, [0, 1, 1, 2, 1, 0], 10),
    ("Texel",      53.05, 4.79, [0, 0, 2, 2, 0, 1], 6),
    ("Drenthe",    52.95, 6.62, [0, 0, 3, 0, 0, 2], 22),
    ("Zeeland",    51.50, 3.85, [0, 1, 2, 2, 0, 0], 20),
    ("Limburg",    51.10, 5.95, [0, 1, 2, 0, 0, 2], 18),
    ("Friesland",  53.10, 5.85, [0, 1, 3, 1, 0, 1], 26),
    ("Achterhoek", 52.00, 6.45, [0, 1, 3, 0, 0, 2], 22),
    ("Flevoland",  52.60, 5.50, [0, 1, 3, 0, 0, 1], 18),
    ("Brabant",    51.60, 5.30, [0, 2, 2, 0, 1, 1], 28),
]

DESCRIPTIONS = {
    "urban_core": [
        "Dense city-centre quarter with grachten, cafés and historic gables.",
        "High-rise inner-city block with shops, offices and young professionals.",
        "Pedestrianised core with narrow streets, museums and intense cycling traffic.",
    ],
    "suburban": [
        "Quiet post-war housing estate, rowhouses, gardens and a primary school.",
        "Family-oriented suburb with playgrounds, supermarkets and connected bike lanes.",
        "VINEX-style development of modern apartments, ample parking, schools nearby.",
    ],
    "rural_farmland": [
        "Open polder landscape dotted with farms, ditches and grazing cattle.",
        "Agricultural village ringed by maize fields and tractor-friendly roads.",
        "Tiny hamlet between wide pastures, the church spire visible from afar.",
    ],
    "coastal": [
        "Beachfront strip with dunes, boulevard restaurants and summer crowds.",
        "Seaside neighbourhood, salt-bitten houses, a small fishing harbour nearby.",
        "Wide beach, kite surfers and a row of pavilions in the dune valley.",
    ],
    "industrial_port": [
        "Container terminals, cranes and pipelines along a deep-water harbour.",
        "Industrial estate of warehouses, logistics hubs and freight rail sidings.",
        "Refineries and chemical plants with stacks visible across the river.",
    ],
    "woodland_village": [
        "Wooded village with sandy footpaths and villas hidden among the pines.",
        "Hilly Veluwe hamlet surrounded by heath and forest reserves.",
        "Tree-lined avenues, brick cottages, equestrian centres along the edges.",
    ],
}


def cosine(emb: np.ndarray) -> np.ndarray:
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return e @ e.T


def project_2d(emb: np.ndarray) -> np.ndarray:
    if HAVE_UMAP:
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(emb)
    return PCA(n_components=2).fit_transform(emb)


def make_correlated_embeddings(arch_idx: np.ndarray, n_arch: int,
                               dim: int = 64, shared: float = 0.7, seed: int = 0):
    """Build (clip, text) embeddings that share an archetype latent but differ.

    Each archetype has its own centre. Per-row latent = centre + jitter.
    Each space applies a random orthonormal rotation, then mixes with
    independent noise (1-shared). High `shared` -> spaces strongly agree.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=2.0, size=(n_arch, dim))
    latent = centres[arch_idx] + rng.normal(scale=0.6, size=(len(arch_idx), dim))

    def space(s: int) -> np.ndarray:
        r = np.random.default_rng(s)
        Q, _ = np.linalg.qr(r.normal(size=(dim, dim)))
        x = latent @ Q
        return (shared * x + (1 - shared) * r.normal(scale=2.0, size=x.shape)).astype(np.float32)

    return space(seed + 1), space(seed + 2)


def build(emb_clip: np.ndarray, emb_text: np.ndarray, df: pd.DataFrame) -> None:
    """Attach UMAP coords + cluster labels per space, save parquet + sim matrices."""
    df = df.copy()
    emb_cbs = ((df[CBS_COLS] - df[CBS_COLS].mean()) / (df[CBS_COLS].std() + 1e-9)).values.astype(np.float32)

    for space, emb in [("clip", emb_clip), ("text", emb_text), ("cbs", emb_cbs)]:
        xy = project_2d(emb)
        df[f"{space}_x"] = xy[:, 0]
        df[f"{space}_y"] = xy[:, 1]
        df[f"{space}_cluster"] = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10).fit(emb).labels_

    df.to_parquet("data.parquet")
    np.save("sim_clip.npy", cosine(emb_clip).astype(np.float32))
    np.save("sim_text.npy", cosine(emb_text).astype(np.float32))
    np.save("sim_cbs.npy",  cosine(emb_cbs).astype(np.float32))
    print(f"wrote data.parquet ({len(df)} rows) + sim_clip/text/cbs.npy"
          f"  (UMAP={'yes' if HAVE_UMAP else 'PCA fallback'})")


def mock() -> None:
    rng = np.random.default_rng(42)
    rows, arch_idx = [], []
    for city, lat, lon, weights, count in CITIES:
        w = np.array(weights, dtype=float)
        w /= w.sum()
        for j in range(count):
            ai = int(rng.choice(len(ARCHETYPES), p=w))
            name, profile = ARCHETYPES[ai]
            stats = np.clip(np.array(profile) + rng.normal(scale=np.array(profile) * 0.15), 1, None)
            spread = 0.04 if name in ("urban_core", "industrial_port") else 0.12
            rows.append({
                "code": f"{city[:3].upper()}{j:04d}",
                "name": f"{city} – {name.replace('_', ' ')} {j + 1}",
                "city": city,
                "archetype": name,
                "lat": lat + rng.normal(scale=spread * 0.6),
                "lon": lon + rng.normal(scale=spread),
                "description": rng.choice(DESCRIPTIONS[name]),
                **{c: float(v) for c, v in zip(CBS_COLS, stats)},
            })
            arch_idx.append(ai)

    df = pd.DataFrame(rows)
    emb_clip, emb_text = make_correlated_embeddings(np.array(arch_idx), n_arch=len(ARCHETYPES))
    build(emb_clip, emb_text, df)


if __name__ == "__main__":
    mock()
