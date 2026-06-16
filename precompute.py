"""Offline pipeline -> data.parquet + sim_clip.npy + sim_text.npy + sim_cbs.npy.

`real()` builds genuine RemoteCLIP image embeddings aggregated per CBS buurt:
  - buurten: CBS Wijk- en buurtkaart polygons via PDOK WFS (filtered to GEMEENTE)
  - clip:    fetch the PDOK aerial tiles inside each buurt polygon, RemoteCLIP-embed
             each tile, mean-pool to one vector per buurt
Text + CBS stay mock for now. `mock()` writes random demo data so app.py runs without
any downloads (run with `--mock`).
"""

import io
import math
import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap

# CBS variables loaded per buurt: {column label: PDOK wijkenbuurten WFS field}.
# The WFS exposes ~150 fields — see readme.md "CBS WFS fields" for the full list to swap in.
CBS = {
    "population":      "aantalInwoners",
    "income":          "gemiddeldInkomenPerInwoner",
    "home_value":      "gemiddeldeWoningwaarde",
    "density":         "bevolkingsdichtheidInwonersPerKm2",
    "household_size":  "gemiddeldeHuishoudsgrootte",
    "pct_owner":       "percentageKoopwoningen",
    "pct_single_pers": "percentageEenpersoonshuishoudens",
    "pct_65plus":      "percentagePersonen65JaarEnOuder",
    "pct_dutch":       "percentageMetHerkomstlandNederland",
    "cars_per_hh":     "personenautosPerHuishouden",
}
CBS_COLS = list(CBS)   # downstream code (build/mock/app) still uses CBS_COLS

GEMEENTE = "Amsterdam"          # set to None for all of NL
ZOOM, MAX_TILES = 17, 16        # WMTS zoom (~190m/tile); cap tiles/buurt to bound cost
MODEL, CKPT = "ViT-L-14", "RemoteCLIP-ViT-L-14.pt"   # RemoteCLIP's largest checkpoint
TILE = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"
WFS = "https://service.pdok.nl/cbs/wijkenbuurten/2023/wfs/v1_0"


def cosine(emb):
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return e @ e.T


def pca2(emb):
    return PCA(n_components=2).fit_transform(emb)

def umap2(emb):  # cosine metric matches the cosine() similarity used for retrieval
    return umap.UMAP(metric="cosine", random_state=42).fit_transform(emb)


def build(emb_clip, emb_text, df):
    """Attach UMAP coords, save parquet + similarity matrices.

    Similarity (top-k) is always cosine in the full space; UMAP is only the 2D layout.
    The CBS 'dense vector' is just the standardized CBS stat columns -> its own space."""
    df = df.copy()
    c = df[CBS_COLS].apply(pd.to_numeric, errors="coerce").mask(lambda x: x < 0)
    df[CBS_COLS] = c              # persist cleaned values (NaN where CBS-suppressed) to parquet
    f = c.fillna(c.median())      # impute only for the embedding math
    emb_cbs = ((f - f.mean()) / (f.std() + 1e-9)).values
    df[["clip_x", "clip_y"]] = umap2(emb_clip)
    df[["text_x", "text_y"]] = umap2(emb_text)
    df[["cbs_x", "cbs_y"]] = umap2(emb_cbs)
    df.to_parquet("data.parquet")
    np.save("sim_clip.npy", cosine(emb_clip))
    np.save("sim_text.npy", cosine(emb_text))
    np.save("sim_cbs.npy", cosine(emb_cbs))
    print(f"wrote data.parquet ({len(df)} rows) + sim_clip/text/cbs.npy")


# --- text embeddings ----------------------------------------------------------

DESC = "data/gsv_analysis/descriptions_prompt_no_scaffold.jsonl"
# canonical 10m grid (EPSG:28992), 512m sub-patch -> tile centre. Origin from the
# gsv raster bbox (X:0-281600, Y:299520-627200); verified against the dataset's examples.
RASTER_TOP_Y, PATCH_M = 627200, 512


def text_emb(g, df):
    """Per-buurt Sentence-BERT vector: embed each tile's MLLM description, spatial-join the
    tile centre to its buurt, mean-pool. Buurten with no described tile get the column mean."""
    import geopandas as gpd
    from sentence_transformers import SentenceTransformer
    t = pd.read_json(DESC, lines=True)[["patch_row", "patch_col", "mllm_output"]]
    x = t.patch_col * PATCH_M + PATCH_M / 2
    y = RASTER_TOP_Y - (t.patch_row * PATCH_M + PATCH_M / 2)
    pts = gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(x, y), crs=28992).to_crs(4326)
    code = g.rename(columns={"buurtcode": "code"})[["code", "geometry"]]
    j = gpd.sjoin(pts, code, predicate="within")          # keep tiles inside a modelled buurt
    print(f"text: {len(j)} described tiles in {j.code.nunique()}/{len(df)} buurten")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode(j.mllm_output.tolist(), show_progress_bar=True)
    per = pd.DataFrame(vec, index=j.code).groupby(level=0).mean()  # mean-pool to buurt level
    per = per.reindex(df.code)
    return per.fillna(per.mean()).values


# --- real CLIP pipeline -------------------------------------------------------

def buurten():
    import geopandas as gpd
    from urllib.parse import quote
    flt = ""
    if GEMEENTE:  # this WFS honours the OGC XML filter, not CQL_FILTER
        xml = (f"<Filter><PropertyIsEqualTo><PropertyName>gemeentenaam</PropertyName>"
               f"<Literal>{GEMEENTE}</Literal></PropertyIsEqualTo></Filter>")
        flt = f"&filter={quote(xml)}"
    url = (f"{WFS}?service=WFS&version=2.0.0&request=GetFeature"
           f"&typeName=wijkenbuurten:buurten&outputFormat=json&count=20000{flt}")
    return gpd.read_file(url).to_crs(4326)  # buurtcode, buurtnaam, geometry


def deg2tile(lat, lon, z):
    n = 2 ** z
    return ((lon + 180) / 360 * n,
            (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def tile_center(x, y, z):
    n = 2 ** z
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n)))),
            (x + 0.5) / n * 360 - 180)


def tiles_in(poly):
    """XYZ tiles at ZOOM whose centre falls inside polygon, capped at MAX_TILES."""
    from shapely.geometry import Point
    lon0, lat0, lon1, lat1 = poly.bounds
    x0, x1 = int(deg2tile(lat0, lon0, ZOOM)[0]), int(deg2tile(lat0, lon1, ZOOM)[0])
    y0, y1 = int(deg2tile(lat1, lon0, ZOOM)[1]), int(deg2tile(lat0, lon0, ZOOM)[1])
    hits = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
            if poly.contains(Point(*tile_center(x, y, ZOOM)[::-1]))]
    if not hits:  # buurt smaller than one tile -> fall back to its centre tile
        c = poly.representative_point()
        hits = [tuple(int(v) for v in deg2tile(c.y, c.x, ZOOM))]
    if len(hits) > MAX_TILES:
        hits = [hits[i] for i in np.linspace(0, len(hits) - 1, MAX_TILES).astype(int)]
    return hits


def fetch_tile(x, y):
    import requests
    from PIL import Image
    os.makedirs("tiles", exist_ok=True)
    path = f"tiles/{ZOOM}_{x}_{y}.jpg"
    if not os.path.exists(path):
        r = requests.get(TILE.format(z=ZOOM, x=x, y=y), timeout=30)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    return Image.open(path).convert("RGB")


def encode_buurt(poly, model, preprocess):
    import torch
    tiles = tiles_in(poly)
    imgs = []
    for x, y in tiles:
        try:
            img = fetch_tile(x, y)
            if np.asarray(img).std() < 5:
                continue  # blank/white tile = no aerial coverage
            imgs.append(preprocess(img))
        except Exception:
            pass  # skip fetch/decode errors

    if not imgs:
        return None
    dev = next(model.parameters()).device
    with torch.no_grad():
        emb = model.encode_image(torch.stack(imgs).to(dev)).cpu().numpy()
    return emb.mean(0)


def real():
    import open_clip
    import torch
    from huggingface_hub import hf_hub_download
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model.load_state_dict(torch.load(
        hf_hub_download("chendelong/RemoteCLIP", CKPT), map_location="cpu"))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    from tqdm import tqdm
    g = buurten()
    print(f"{len(g)} buurten in {GEMEENTE or 'NL'}; embedding...")
    rows, embs = [], []
    for _, b in tqdm(g.iterrows(), total=len(g)):
        e = encode_buurt(b.geometry, model, preprocess)
        if e is None:
            continue
        p = b.geometry.representative_point()
        rows.append({"code": b.buurtcode, "name": b.buurtnaam, "lat": p.y, "lon": p.x,
                     **{k: b[v] for k, v in CBS.items()}})  # real CBS stats from the WFS
        embs.append(e)
    df = pd.DataFrame(rows)
    clip = np.vstack(embs)
    build(clip, text_emb(g, df), df)  # real Sentence-BERT text space
    g[g.buurtcode.isin(df.code)][["buurtcode", "geometry"]].rename(
        columns={"buurtcode": "code"}).to_file("buurten.geojson", driver="GeoJSON")  # polygons for the map


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
    mock() if "--mock" in sys.argv else real()
