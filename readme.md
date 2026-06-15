# NL Neighbourhood Embedding Explorer

Interactive dashboard for comparing Dutch neighbourhoods across three embedding
spaces (CLIP aerial imagery, neighbourhood text descriptions, CBS socio-economic
vectors), with the map of the Netherlands as the shared anchor.

There are three apps in this repo:

| app | scope | port | data files |
| --- | --- | --- | --- |
| `app.py` | NL-wide, ~600 neighbourhoods, 30 cities (3-view explorer) | `:8050` | `data.parquet`, `sim_{clip,text,cbs}.npy` |
| `app_amsterdam.py` | Amsterdam-only, ~85 buurten + search + comparison panel | `:8051` | `data_ams.parquet`, `sim_ams_{clip,text,cbs}.npy` |
| `app_nl.py` | **NL platform**, ~1850 neighbourhoods, choropleth + click-through stats + similarity heatmap | `:8052` | `data_nl.parquet`, `sim_nl_*.npy`, `assets/nl_cells.geojson` |

## Quick start

```bash
.venv/bin/python -m pip install -r requirements.txt

# NL-wide 3-view explorer
.venv/bin/python precompute.py            # writes data.parquet + sim_*.npy
.venv/bin/python app.py                   # http://127.0.0.1:8050

# Amsterdam deep-dive (separate terminal)
.venv/bin/python precompute_amsterdam.py  # writes data_ams.parquet + sim_ams_*.npy
.venv/bin/python app_amsterdam.py         # http://127.0.0.1:8051

# NL choropleth platform (separate terminal)
.venv/bin/python precompute_nl.py         # writes data_nl.parquet + sim_nl_*.npy + assets/nl_cells.geojson
.venv/bin/python app_nl.py                # http://127.0.0.1:8052
```

## NL platform (`app_nl.py`) — features

A choropleth of ~1850 Dutch neighbourhoods over PDOK aerial tiles, linked to the
three embedding views (CLIP aerial / text descriptions / CBS).

- **Click-through statistics** — header buttons (housing price, income,
  population, avg age, household size, green %, planned development) recolour the
  *map and all three embedding scatters* by that variable.
- **Find-similar-by CLIP / Descriptions / CBS** — pick a neighbourhood, choose a
  space, and the map becomes a cosine-**similarity heatmap** with the top-k most
  similar highlighted (k slider). The scatters switch to cluster colouring with
  the selection (red) + top-k (blue) emphasised.
- **Organic borders** — neighbourhood areas are *perturbed Voronoi cells*
  (deterministic per-edge noise so shared borders stay glued) so they look like
  real boundaries, not ruler-straight lines. Toggle-able.
- **Live hover** — hovering any area updates the spider plot (selected vs hovered
  overlay) via a lightweight callback that never redraws the map, so it stays
  responsive.
- **Random location** button for exploration; **fill-opacity** slider.

Performance note: the polygon geometry is written once to
`assets/nl_cells.geojson` (≈1.1 MB) and referenced by **URL**, so interaction
callbacks only ship per-neighbourhood value arrays, never the geometry. The
dense `sim_nl_*.npy` matrices (1850² × 4 B ≈ 14 MB each) are fine at this scale;
for the full ~14k buurten swap them for an ANN index (FAISS/hnswlib) over the
embedding matrix.

Mock mode generates ~600 synthetic neighbourhoods anchored to ~30 real Dutch
city centres, organised around six archetypes (urban core, suburban, rural
farmland, coastal, industrial port, woodland village). Embeddings are sampled
from a shared archetype latent + per-space noise so the three views are
correlated but not identical — the demo cluster structure is intentional.

## What the UI does

- **Map (centre)** — PDOK aerial tiles, one dot per neighbourhood. Click a dot
  to select it.
- **Three UMAP scatters (right)** — CLIP, text, CBS. Click a point to select.
- **Spider + table + description (left)** — the selected neighbourhood's CBS
  profile, archetype, and textual description.
- **Controls**:
  - `Highlight top-k by`: which similarity space ranks the top-15 neighbours
    that get enlarged in every view.
  - `Colour dots by clusters of`: which K-means partition drives the base
    colour (or `archetype` for the ground-truth labels).
  - `zoom map to selection on click`: when set, clicking in a scatter pans/
    zooms the map to that neighbourhood. Pan/zoom you do yourself is
    preserved between clicks (via `uirevision`).

Red = selected, large dot = top-k similar, base colour = cluster/archetype.

## Amsterdam app — extra features

On top of what the NL version does:

- **Address / buurt search** in the header. Tries an exact buurt-name match
  first, then Nominatim (bounded to Amsterdam) for general addresses (e.g.
  "Vondelpark", "Dam 1", "Java-eiland"), snaps the result to the nearest buurt
  centroid. Falls back to substring match if Nominatim is unavailable.
- **Adjustable top-k** (slider, 3–25). The list shows each neighbour's name,
  archetype, cosine score and a small score-bar. Clicking an item sets it as
  the *comparison target* (blue dot), keeping the primary selection (red).
- **"Why are they similar?" panel** when a comparison is active:
  - Per-space cosine score with a same-cluster / different-cluster indicator
  - Paired CBS bar chart (normalised) so you can see at a glance which
    features pull the two together and which push them apart
  - Side-by-side descriptions
  - Spider plot overlays the comparison's CBS profile on the selected one
- **Zoom in the embedding scatters**: drag draws a box-zoom, scroll-wheel
  zooms, double-click resets. Lets you drill into a cluster to see what
  actually drives the distances.

## Architecture

```
precompute.py  ──►  data.parquet           (lat/lon, CBS cols, archetype,
                                            description, {clip,text,cbs}_x/y,
                                            {clip,text,cbs}_cluster)
               ──►  sim_clip.npy           (NxN cosine sim of CLIP embeddings)
               ──►  sim_text.npy           (NxN cosine sim of text embeddings)
               ──►  sim_cbs.npy            (NxN cosine sim of CBS vectors)

app.py         ──►  Dash app, single click-callback updates a dcc.Store,
                    a render callback rebuilds six figures from the store
                    + control values. Scatters use Scattergl, the map uses
                    Scattermap with PDOK Actueel_orthoHR raster tiles.
```

## Going from mock → real data

Three datasets to swap in, all keyed by buurtcode:

### 1. CBS Wijk- en buurtkaart (geometry + statistics)

```python
import geopandas as gpd
# Latest atom feed of the WBK shapefile
url = "https://service.pdok.nl/cbs/wijkenbuurten/2023/atom/wijkenbuurten_2023.xml"
buurten = gpd.read_file("wijkenbuurten_2023_v1.gpkg", layer="buurten")
buurten = buurten[buurten.geometry.is_valid]
buurten["lat"] = buurten.geometry.to_crs(4326).centroid.y
buurten["lon"] = buurten.geometry.to_crs(4326).centroid.x
```

The same dataset carries the CBS Kerncijfers columns (population, income, etc.)
joined on `buurtcode`. Map them to the `CBS_COLS` names in `precompute.py`.

### 2. PDOK aerial imagery → CLIP embeddings

For each buurt centroid, fetch a single 256x256 tile from the PDOK WMTS
service at zoom 15 (≈ 1.2 km on a side — covers a typical buurt):

```python
import math, requests
from PIL import Image
from io import BytesIO

WMTS = ("https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/"
        "Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg")

def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat))
                          + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y

def fetch_tile(lat, lon, z=15):
    x, y = lonlat_to_tile(lon, lat, z)
    return Image.open(BytesIO(requests.get(WMTS.format(z=z, x=x, y=y)).content))
```

Embed with **open_clip** ViT-B/32 (fast, generic) or — better for aerial imagery —
**RemoteCLIP** ([huggingface.co/chendelong/RemoteCLIP](https://huggingface.co/chendelong/RemoteCLIP))
or **GeoCLIP**. RemoteCLIP was fine-tuned on RS image–text pairs and matches
the "finetuned for earth observation" line in the project description.

```python
import open_clip, torch
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai")
model.eval().to("cuda")
with torch.no_grad():
    batch = torch.stack([preprocess(img) for img in images]).to("cuda")
    emb_clip = model.encode_image(batch).cpu().numpy()
```

### 3. Neighbourhood descriptions → text embeddings

The Canvas dataset of descriptions, joined on buurtcode. For multilingual
Dutch text use `intfloat/multilingual-e5-base` (or `paraphrase-multilingual-
MiniLM-L12-v2` if you need lighter).

```python
from sentence_transformers import SentenceTransformer
emb_text = SentenceTransformer("intfloat/multilingual-e5-base").encode(
    ["passage: " + d for d in df["description"]], show_progress_bar=True)
```

### Putting it together

```python
from precompute import build, CBS_COLS
df = buurten_df_with_latlon_and_cbs_and_description  # one row per buurt
build(emb_clip, emb_text, df[["code", "name", "city", "archetype",
                              "lat", "lon", "description", *CBS_COLS]])
```

`build()` standardises CBS columns to make the CBS vector, runs UMAP (or PCA
fallback) on each space, computes K-means cluster labels, and writes the same
`data.parquet` + `sim_*.npy` files. `app.py` doesn't need changes.

## Performance notes

- Scattergl for the embedding plots — comfortable to ~50k points.
- Similarity matrices stored as float32 (`N²·4` bytes). 10k neighbourhoods is
  ~400 MB; if you push to all 13k Dutch buurten, swap the dense matrix for an
  on-the-fly cosine call against the embedding matrix.
- UMAP fit on ~10k×768 takes ~1 minute on CPU. Cache the parquet.

## Files

| file | purpose |
| --- | --- |
| `precompute.py` | NL mock data generator + `build()` for real embeddings |
| `app.py` | NL-wide Dash dashboard |
| `precompute_amsterdam.py` | Amsterdam buurten mock + `build()` for real Ams pipeline |
| `app_amsterdam.py` | Amsterdam Dash dashboard (search + top-k + why-similar + zoom) |
| `precompute_nl.py` | NL platform mock (~1850 nbhds) + organic polygons + `build()` |
| `app_nl.py` | NL choropleth platform (click-through stats + similarity heatmap) |
| `assets/nl_cells.geojson` | organic neighbourhood polygons, served by URL |
| `data*.parquet` | one row per neighbourhood, coords + CBS cols |
| `sim_*.npy` | NxN cosine similarity per space |
| `requirements.txt` | runtime deps |
