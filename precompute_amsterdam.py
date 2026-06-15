"""Amsterdam-focused mock data -> data_ams.parquet + sim_ams_{clip,text,cbs}.npy.

About 85 real Amsterdam neighbourhoods (buurten) at approximate centroids, one
of eight archetypes per row. CBS values are sampled from realistic per-archetype
ranges; CLIP and text embeddings are sampled from a shared archetype latent
plus per-space noise so the three projection spaces are correlated but not
identical (which is what makes the cross-space comparison meaningful).

Real-data swap:
    df = geopandas.read_file("wijkenbuurten_2023_v1.gpkg", layer="buurten") \\
                  .query("gemeentecode == 'GM0363'")
    df = df.assign(lat=df.geometry.to_crs(4326).centroid.y,
                   lon=df.geometry.to_crs(4326).centroid.x)
    # join CBS Kerncijfers (population/income/...) on buurtcode
    # fetch one PDOK Actueel_orthoHR tile at z=16 per centroid, open_clip/RemoteCLIP -> emb_clip
    # SentenceTransformer("intfloat/multilingual-e5-base").encode(df.description) -> emb_text
    build(emb_clip, emb_text, df)
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

# archetype index -> (name, CBS profile mean)
# units: pop [head], income [CBS index 0-100], pct_green [0-100], age [yr],
#        home_value [k EUR], density [per km^2]
ARCHETYPES = [
    ("canal_belt",         [5000, 85, 8,  45, 750, 12000]),  # 0
    ("17c_historic",       [4000, 65, 12, 42, 580, 13000]),  # 1
    ("19c_belt",           [8000, 50, 15, 35, 450, 14000]),  # 2
    ("early_20c_garden",   [6000, 75, 35, 45, 580, 9000]),   # 3
    ("post_war_modernist", [7000, 38, 38, 42, 270, 7500]),   # 4
    ("modern_vinex",       [5000, 60, 22, 38, 490, 6500]),   # 5
    ("industrial_port",    [1000, 50, 12, 41, 250, 800]),    # 6
    ("rural_village",      [500,  55, 75, 52, 510, 200]),    # 7
]

# (buurt name, lat, lon, archetype_idx)
BUURTEN = [
    ("Jordaan",                            52.374, 4.881, 1),
    ("Westelijke Eilanden",                52.385, 4.892, 1),
    ("Haarlemmerbuurt",                    52.385, 4.890, 2),
    ("Burgwallen-Nieuwe Zijde",            52.374, 4.893, 1),
    ("Burgwallen-Oude Zijde",              52.373, 4.898, 1),
    ("Nieuwmarkt-Lastage",                 52.372, 4.901, 1),
    ("Grachtengordel-West",                52.371, 4.889, 0),
    ("Grachtengordel-Zuid",                52.366, 4.892, 0),
    ("Weteringschans",                     52.362, 4.890, 0),
    ("Leidsebuurt",                        52.363, 4.882, 0),
    ("Frederik Hendrikbuurt",              52.376, 4.873, 2),
    ("Da Costabuurt",                      52.367, 4.873, 2),
    ("Helmersbuurt",                       52.363, 4.871, 2),
    ("Overtoomse Sluis",                   52.361, 4.864, 2),
    ("Vondelbuurt",                        52.358, 4.866, 3),
    ("Museumkwartier",                     52.358, 4.879, 3),
    ("Stadionbuurt",                       52.341, 4.866, 3),
    ("Apollobuurt",                        52.343, 4.880, 3),
    ("Schinkelbuurt",                      52.349, 4.860, 2),
    ("Hoofddorppleinbuurt",                52.347, 4.853, 3),
    ("De Pijp - Noord",                    52.357, 4.892, 2),
    ("De Pijp - Centrum",                  52.354, 4.892, 2),
    ("De Pijp - Zuid",                     52.349, 4.892, 2),
    ("Rivierenbuurt - Noord",              52.348, 4.898, 3),
    ("Rivierenbuurt - Zuid",               52.342, 4.905, 3),
    ("Scheldebuurt",                       52.341, 4.892, 3),
    ("IJselbuurt",                         52.343, 4.902, 3),
    ("Buitenveldert-West",                 52.331, 4.876, 3),
    ("Buitenveldert-Oost",                 52.329, 4.890, 3),
    ("Zuidas",                             52.340, 4.873, 5),
    ("RAI",                                52.341, 4.890, 4),
    ("De Plantage",                        52.366, 4.913, 3),
    ("Weesperbuurt",                       52.362, 4.910, 2),
    ("Czaar Peterbuurt",                   52.366, 4.929, 2),
    ("Oostelijke Eilanden",                52.370, 4.927, 5),
    ("Oosterparkbuurt",                    52.359, 4.917, 2),
    ("Transvaalbuurt",                     52.354, 4.927, 2),
    ("Dapperbuurt",                        52.363, 4.928, 2),
    ("Indische Buurt - West",              52.366, 4.936, 2),
    ("Indische Buurt - Oost",              52.362, 4.946, 2),
    ("Oostpoort",                          52.359, 4.930, 5),
    ("Watergraafsmeer - Frankendael",      52.350, 4.928, 3),
    ("Watergraafsmeer - Middenmeer",       52.353, 4.932, 3),
    ("Watergraafsmeer - Don Bosco",        52.345, 4.945, 3),
    ("Watergraafsmeer - Betondorp",        52.345, 4.937, 3),
    ("Bijlmer-Centrum",                    52.317, 4.945, 4),
    ("Bijlmer-Oost (Holendrecht)",         52.296, 4.957, 4),
    ("Bijlmer - Geinwijk",                 52.302, 4.974, 4),
    ("Gaasperdam - Reigersbos",            52.300, 4.971, 4),
    ("Gaasperdam - Driemond",              52.300, 4.987, 7),
    ("Bullewijk",                          52.310, 4.945, 5),
    ("Amstel III",                         52.310, 4.940, 6),
    ("Venserpolder",                       52.328, 4.950, 4),
    ("IJburg - Steigereiland",             52.359, 5.011, 5),
    ("IJburg - Haveneiland West",          52.353, 5.000, 5),
    ("IJburg - Haveneiland Oost",          52.350, 5.013, 5),
    ("IJburg - Rieteilanden",              52.353, 5.020, 5),
    ("Zeeburgereiland",                    52.367, 4.972, 5),
    ("Houthavens",                         52.397, 4.880, 5),
    ("Spaarndammerbuurt",                  52.388, 4.870, 3),
    ("Westerpark",                         52.388, 4.879, 2),
    ("Staatsliedenbuurt",                  52.381, 4.872, 2),
    ("Bos en Lommer",                      52.380, 4.847, 4),
    ("Baarsjes - Noord",                   52.367, 4.852, 2),
    ("Baarsjes - Zuid",                    52.359, 4.852, 2),
    ("Slotermeer - Noord",                 52.387, 4.823, 4),
    ("Slotermeer - Zuid",                  52.377, 4.826, 4),
    ("Geuzenveld",                         52.388, 4.811, 4),
    ("Osdorp - Centrum",                   52.358, 4.812, 4),
    ("Osdorp - Zuid",                      52.349, 4.815, 4),
    ("Osdorp - De Aker",                   52.355, 4.795, 5),
    ("Nieuw-Sloten",                       52.342, 4.806, 5),
    ("Slotervaart",                        52.358, 4.825, 4),
    ("Oud-Noord - Volewijck",              52.396, 4.905, 2),
    ("Oud-Noord - Vogelbuurt",             52.395, 4.917, 4),
    ("Oud-Noord - IJplein",                52.387, 4.912, 4),
    ("Tuindorp Buiksloot",                 52.408, 4.927, 3),
    ("Tuindorp Oostzaan",                  52.408, 4.880, 3),
    ("Buikslotermeer",                     52.413, 4.927, 4),
    ("Banne Buiksloot",                    52.420, 4.918, 4),
    ("Nieuwendam-Noord",                   52.421, 4.943, 4),
    ("Durgerdam",                          52.392, 4.978, 7),
    ("Holysloot",                          52.430, 4.991, 7),
    ("Schellingwoude",                     52.388, 4.978, 7),
    ("Ransdorp",                           52.420, 4.984, 7),
    ("Westhaven",                          52.408, 4.785, 6),
    ("Coen-Vlothaven",                     52.413, 4.835, 6),
    ("Sloterdijk",                         52.388, 4.835, 6),
]

DESCRIPTIONS = {
    "canal_belt": [
        "Stately canal houses fronting the UNESCO grachtengordel, with cafés and houseboats at the quay.",
        "Quiet seventeenth-century canal stretch lined with merchant houses and elm trees.",
        "Iconic canal ring with corbel-step gables, museum boats and arched bridges every block.",
    ],
    "17c_historic": [
        "Tightly packed historic core with narrow streets, brown cafés and a steady tourist current.",
        "Warehouses converted to lofts, cobbled streets and bridges over the Brouwersgracht.",
        "Old shipping district, hofjes hidden behind a gate, working drawbridges and gull cries.",
    ],
    "19c_belt": [
        "Dense nineteenth-century cordon of three-storey brick terraces, market stalls and tram lines.",
        "Working-class belt turned hip, with bakeries, vintage shops and crammed cycling racks.",
        "Long avenues of narrow façades, gable stones, kebab and koffie on every corner.",
    ],
    "early_20c_garden": [
        "Berlage-era Plan Zuid blocks: tall brick housing, leafy avenues and tile-roofed schools.",
        "Quiet residential pockets with chestnut-lined streets, allotments and a village green.",
        "Inter-war garden suburb: brick villas, hedges, sport clubs and a small shopping cluster.",
    ],
    "post_war_modernist": [
        "Long modernist slabs of social housing set in green courts with playgrounds and bike sheds.",
        "Hexagonal Bijlmer blocks, raised walkways, parking decks and groundlevel multicultural shops.",
        "Stamp-style post-war estate: identical galerijflats, central lawn, prefab schools.",
    ],
    "modern_vinex": [
        "Brand-new waterfront blocks with rooftop terraces, glass façades and architect-designed bridges.",
        "VINEX-era neighbourhood: car-friendly streets, family homes, a supermarket and a primary school.",
        "Reclaimed island development: planted dykes, modern townhouses, kayak racks at the wharf.",
    ],
    "industrial_port": [
        "Container terminals, cranes and pipelines along a deep-water harbour with truck queues.",
        "Industrial estate of warehouses, logistics hubs, freight rail sidings and chain-link fencing.",
        "Refinery and chemical plants flanked by tanker berths; smokestacks against the polder horizon.",
    ],
    "rural_village": [
        "Tiny Waterland village of wooden houses around a dike, sheep grazing on the embankment.",
        "Single street of clapboard cottages, a church with a tilted spire, a windmill in the distance.",
        "Hamlet between drainage ditches and pastures, the IJsselmeer wind always present.",
    ],
}


def cosine(emb: np.ndarray) -> np.ndarray:
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return e @ e.T


def project_2d(emb: np.ndarray) -> np.ndarray:
    if HAVE_UMAP and len(emb) > 15:
        return umap.UMAP(n_neighbors=12, min_dist=0.1, random_state=0).fit_transform(emb)
    return PCA(n_components=2).fit_transform(emb)


def make_correlated_embeddings(arch_idx: np.ndarray, n_arch: int,
                               dim: int = 64, shared: float = 0.7, seed: int = 0):
    """(clip, text) embeddings that share an archetype latent but differ in noise."""
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
    df = df.copy()
    emb_cbs = ((df[CBS_COLS] - df[CBS_COLS].mean()) / (df[CBS_COLS].std() + 1e-9)).values.astype(np.float32)

    for space, emb in [("clip", emb_clip), ("text", emb_text), ("cbs", emb_cbs)]:
        xy = project_2d(emb)
        df[f"{space}_x"] = xy[:, 0]
        df[f"{space}_y"] = xy[:, 1]
        df[f"{space}_cluster"] = KMeans(n_clusters=N_CLUSTERS, random_state=0,
                                        n_init=10).fit(emb).labels_

    df.to_parquet("data_ams.parquet")
    np.save("sim_ams_clip.npy", cosine(emb_clip).astype(np.float32))
    np.save("sim_ams_text.npy", cosine(emb_text).astype(np.float32))
    np.save("sim_ams_cbs.npy",  cosine(emb_cbs).astype(np.float32))
    print(f"wrote data_ams.parquet ({len(df)} rows) + sim_ams_clip/text/cbs.npy"
          f"  (UMAP={'yes' if HAVE_UMAP else 'PCA fallback'})")


def mock() -> None:
    rng = np.random.default_rng(7)
    rows, arch_idx = [], []
    for k, (name, lat, lon, ai) in enumerate(BUURTEN):
        arch_name, profile = ARCHETYPES[ai]
        stats = np.clip(np.array(profile) + rng.normal(scale=np.array(profile) * 0.12),
                        1, None)
        rows.append({
            "code": f"AMS{k:04d}",
            "name": name,
            "archetype": arch_name,
            "lat": lat + rng.normal(scale=0.003),
            "lon": lon + rng.normal(scale=0.004),
            "description": rng.choice(DESCRIPTIONS[arch_name]),
            **{c: float(v) for c, v in zip(CBS_COLS, stats)},
        })
        arch_idx.append(ai)

    df = pd.DataFrame(rows)
    emb_clip, emb_text = make_correlated_embeddings(np.array(arch_idx), n_arch=len(ARCHETYPES))
    build(emb_clip, emb_text, df)


if __name__ == "__main__":
    mock()
