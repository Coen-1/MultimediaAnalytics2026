"""Offline pipeline -> data.parquet + emb_clip.npy + emb_text.npy + emb_cbs.npy.

`real()` builds genuine RemoteCLIP image embeddings aggregated per CBS buurt:
  - buurten: CBS Wijk- en buurtkaart polygons via PDOK WFS (filtered to GEMEENTE)
  - clip:    fetch the PDOK aerial tiles inside each buurt polygon, RemoteCLIP-embed
             each tile, mean-pool to one vector per buurt
Text + CBS stay mock for now. `mock()` writes random demo data so app.py runs without
any downloads (run with `--mock`).
"""

import io
import json
import math
import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap

# CBS variables loaded per buurt -> {our column name: PDOK wijkenbuurten WFS field}.
# The full column catalogue (Dutch field, our name, description, spider label, unit) lives
# in cbs_columns.json; edit that file to add/relabel measures. app.py reads the same file.
CBS_META = json.load(open(os.path.join(os.path.dirname(__file__), "assets", "cbs_columns.json")))
CBS = {e["name"]: e["dutch"] for e in CBS_META}   # all numeric measures, not just a curated few
CBS_COLS = list(CBS)   # downstream code (build/mock/app) still uses CBS_COLS

GEMEENTE = "Amsterdam"          # None = all of NL; set to a name like "Amsterdam" to scope down
ZOOM, MAX_TILES = 17, 16        # WMTS zoom (~190m/tile); cap tiles/buurt to bound cost
WORKERS, BATCH = 64, 256        # parallel tile fetches; CLIP images per GPU batch
MODEL, CKPT = "ViT-L-14", "RemoteCLIP-ViT-L-14.pt"   # RemoteCLIP's largest checkpoint
TILE = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"
WFS = "https://service.pdok.nl/cbs/wijkenbuurten/2023/wfs/v1_0"
OUT = os.path.join(os.path.dirname(__file__), "assets")   # write straight to where app.py reads


def out(name):
    return os.path.join(OUT, name)


def check_overwrite(*names):
    exist = [n for n in names if os.path.exists(out(n))]
    if exist and input(f"overwrite {', '.join(exist)} in {OUT}/? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")


def unit(emb):  # row-normalized embeddings; cosine(a,b) == unit(a) @ unit(b).T
    return (emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def pca2(emb):
    return PCA(n_components=2).fit_transform(emb)

def umap2(emb):  # cosine metric matches the cosine() similarity used for retrieval;
    # n_neighbors/min_dist tuned for tighter, better-separated neighbourhood groups in the 2D layout
    return umap.UMAP(metric="cosine", n_neighbors=30, min_dist=0.05, random_state=42).fit_transform(emb)


def build(emb_clip, emb_text, df):
    """Attach UMAP coords, save parquet + similarity matrices.

    Similarity (top-k) is always cosine in the full space; UMAP is only the 2D layout.
    The CBS 'dense vector' is just the standardized CBS stat columns -> its own space."""
    df = df.copy()
    c = df[CBS_COLS].apply(pd.to_numeric, errors="coerce").mask(lambda x: x < 0)
    df[CBS_COLS] = c              # persist cleaned values (NaN where CBS-suppressed) to parquet
    f = c.fillna(c.median()).fillna(0)   # impute; columns empty for every buurt have no median -> neutral 0
    emb_cbs = ((f - f.mean()) / (f.std() + 1e-9)).values
    xy = np.hstack([umap2(emb_clip), umap2(emb_text), umap2(emb_cbs)])   # one concat avoids frame fragmentation
    df = pd.concat([df, pd.DataFrame(xy, index=df.index,
                   columns=["clip_x", "clip_y", "text_x", "text_y", "cbs_x", "cbs_y"])], axis=1)
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(out("data.parquet"))
    np.save(out("emb_clip.npy"), unit(emb_clip))   # store normalized embeddings, not the N×N matrix
    np.save(out("emb_text.npy"), unit(emb_text))
    np.save(out("emb_cbs.npy"), unit(emb_cbs))
    print(f"wrote {OUT}/data.parquet ({len(df)} rows) + emb_clip/text/cbs.npy")


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
    PAGE, parts, start = 1000, [], 0   # GeoServer caps GetFeature at 1000 -> page with startIndex
    while True:
        url = (f"{WFS}?service=WFS&version=2.0.0&request=GetFeature&typeName=wijkenbuurten:buurten"
               f"&outputFormat=json&count={PAGE}&startIndex={start}{flt}")
        page = gpd.read_file(url)
        parts.append(page)
        start += len(page)
        if len(page) < PAGE:           # last (short) page reached
            break
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    print(f"fetched {len(g)} buurten from WFS")
    return g.to_crs(4326)  # buurtcode, buurtnaam, geometry


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


def fetch_all(tiles):
    """Warm the tiles/ disk cache concurrently (PDOK WMTS tolerates parallelism)."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    def get(xy):
        try:
            fetch_tile(*xy)
        except Exception:
            pass  # same silent-skip as the embed pass
    with ThreadPoolExecutor(WORKERS) as ex:
        list(tqdm(ex.map(get, tiles), total=len(tiles), desc="fetch"))


def embed_tiles(tiles, model, preprocess):
    """RemoteCLIP-embed every (cached) tile in fixed BATCH chunks -> {(x, y): vec}."""
    import torch
    from tqdm import tqdm
    dev = next(model.parameters()).device
    out = {}
    for i in tqdm(range(0, len(tiles), BATCH), desc="embed"):
        chunk, imgs = [], []
        for xy in tiles[i:i + BATCH]:
            try:
                img = fetch_tile(*xy)
                if np.asarray(img).std() < 5:
                    continue  # blank/white tile = no aerial coverage
                chunk.append(xy)
                imgs.append(preprocess(img))
            except Exception:
                pass  # skip fetch/decode errors
        if not imgs:
            continue
        with torch.no_grad():
            vecs = model.encode_image(torch.stack(imgs).to(dev)).cpu().numpy()
        out.update(zip(chunk, vecs))
    return out


def real():
    check_overwrite("data.parquet", "emb_clip.npy", "emb_text.npy", "emb_cbs.npy", "buurten.geojson")
    import open_clip
    import torch
    from huggingface_hub import hf_hub_download
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model.load_state_dict(torch.load(
        hf_hub_download("chendelong/RemoteCLIP", CKPT), map_location="cpu"))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    g = buurten()
    print(f"{len(g)} buurten in {GEMEENTE or 'NL'}; embedding...")
    btiles = {b.buurtcode: tiles_in(b.geometry) for _, b in g.iterrows()}
    uniq = sorted({t for ts in btiles.values() for t in ts})
    fetch_all(uniq)                            # parallel-fetch every tile once
    vec = embed_tiles(uniq, model, preprocess) # then big-batch embed them
    rows, embs = [], []
    for _, b in g.iterrows():
        es = [vec[t] for t in btiles[b.buurtcode] if t in vec]
        if not es:
            continue  # all tiles blank/failed -> skip (== old encode_buurt None)
        p = b.geometry.representative_point()
        rows.append({"code": b.buurtcode, "name": b.buurtnaam, "lat": p.y, "lon": p.x,
                     **{k: b[v] for k, v in CBS.items()}})  # real CBS stats from the WFS
        embs.append(np.mean(es, axis=0))       # mean-pool tiles -> one vector per buurt
    df = pd.DataFrame(rows)
    clip = np.vstack(embs)
    build(clip, text_emb(g, df), df)  # real Sentence-BERT text space
    g[g.buurtcode.isin(df.code)][["buurtcode", "geometry"]].rename(
        columns={"buurtcode": "code"}).to_file(out("buurten.geojson"), driver="GeoJSON")  # polygons for the map


def mock(n=200):
    check_overwrite("data.parquet", "emb_clip.npy", "emb_text.npy", "emb_cbs.npy")
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
