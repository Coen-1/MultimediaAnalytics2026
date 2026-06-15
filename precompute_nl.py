"""NL-wide mock platform data.

Outputs:
    data_nl.parquet            one row per neighbourhood (centroid, stats,
                               descriptions, UMAP coords, cluster labels)
    sim_nl_{clip,text,cbs}.npy NxN cosine similarity per space
    assets/nl_cells.geojson    one organic polygon per neighbourhood, served
                               statically by Dash and referenced by URL so the
                               interaction callbacks never reship geometry

The polygons are *perturbed Voronoi cells*: we tessellate the centroids, then
displace each edge with deterministic per-edge noise so neighbouring cells stay
glued (shared borders match exactly) while the borders look organic rather than
ruler-straight.

Swap to real data: replace `mock()` with a loader that reads the CBS Wijk- en
buurtkaart (geometry + Kerncijfers), encodes PDOK aerial tiles with a CLIP-family
model and the descriptions with a sentence transformer, then calls
`build(emb_clip, emb_text, df, polygons)`.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

# numeric statistics that form the CBS dense vector (order matters)
CBS_COLS = ["population", "income", "avg_age", "household_size",
            "home_value", "green_pct", "density"]
# extra clickable statistic that is shown but not part of the embedding vector
EXTRA_COLS = ["planned_units"]
N_CLUSTERS = 6

DEV_CATEGORIES = ["None", "Residential", "Commercial", "Mixed-use", "Infrastructure"]

# archetype -> profile means for CBS_COLS, then development tendency (weights over
# DEV_CATEGORIES) and a planned-units scale.
ARCHETYPES = [
    # name                pop   inc  age  hh   home grn dens   dev-weights              units
    ("urban_core",       [22000, 34, 38, 1.7, 480, 8,  9000], [2, 4, 3, 4, 1], 900),
    ("historic_center",  [9000,  31, 41, 1.6, 560, 10, 11000], [3, 2, 3, 3, 1], 400),
    ("suburban",         [12000, 33, 42, 2.4, 360, 32, 3500], [3, 4, 1, 1, 1], 500),
    ("modern_vinex",     [15000, 36, 36, 2.6, 430, 24, 4200], [1, 5, 1, 3, 2], 1300),
    ("rural_farmland",   [1800,  30, 46, 2.5, 320, 72, 250], [6, 1, 0, 0, 1], 60),
    ("coastal",          [6000,  35, 48, 2.0, 540, 40, 1500], [3, 2, 2, 2, 1], 250),
    ("industrial_port",  [2500,  28, 42, 2.1, 250, 14, 700], [2, 1, 4, 1, 4], 700),
    ("woodland_village", [3500,  37, 50, 2.3, 470, 66, 600], [4, 2, 1, 1, 1], 150),
]

# (city, lat, lon, archetype weights over the 8 archetypes, #neighbourhoods)
CITIES = [
    ("Amsterdam",  52.37, 4.90, [6, 4, 2, 1, 0, 0, 2, 0], 150),
    ("Rotterdam",  51.92, 4.48, [5, 2, 2, 2, 0, 0, 4, 0], 120),
    ("Den Haag",   52.08, 4.30, [5, 2, 3, 1, 0, 2, 1, 0], 100),
    ("Utrecht",    52.09, 5.12, [5, 3, 3, 2, 0, 0, 1, 1], 90),
    ("Eindhoven",  51.44, 5.48, [3, 1, 3, 2, 0, 0, 2, 1], 70),
    ("Groningen",  53.22, 6.57, [3, 2, 2, 1, 1, 0, 1, 2], 55),
    ("Tilburg",    51.56, 5.09, [3, 1, 3, 2, 0, 0, 1, 1], 50),
    ("Almere",     52.37, 5.21, [1, 0, 2, 6, 0, 0, 1, 1], 55),
    ("Breda",      51.59, 4.78, [3, 1, 3, 2, 0, 0, 0, 2], 45),
    ("Nijmegen",   51.81, 5.84, [3, 1, 2, 1, 0, 0, 1, 2], 45),
    ("Haarlem",    52.39, 4.64, [3, 3, 3, 1, 0, 1, 0, 1], 40),
    ("Arnhem",     51.98, 5.91, [3, 1, 2, 1, 0, 0, 1, 2], 40),
    ("Maastricht", 50.85, 5.69, [3, 2, 2, 1, 0, 0, 1, 2], 40),
    ("Apeldoorn",  52.21, 5.97, [1, 1, 2, 1, 1, 0, 0, 4], 40),
    ("Zaanstad",   52.45, 4.81, [2, 1, 2, 2, 0, 0, 3, 0], 40),
    ("Amersfoort", 52.16, 5.39, [2, 1, 3, 3, 0, 0, 0, 1], 38),
    ("Zwolle",     52.51, 6.09, [2, 1, 2, 2, 1, 0, 1, 2], 35),
    ("Leiden",     52.16, 4.49, [3, 3, 2, 1, 0, 1, 0, 1], 35),
    ("Dordrecht",  51.81, 4.67, [3, 2, 2, 1, 0, 0, 2, 1], 32),
    ("Enschede",   52.22, 6.89, [2, 1, 3, 1, 1, 0, 1, 2], 35),
    ("Delft",      52.01, 4.36, [3, 2, 2, 1, 0, 0, 1, 1], 28),
    ("Alkmaar",    52.63, 4.75, [2, 2, 2, 2, 1, 1, 0, 1], 28),
    ("Leeuwarden", 53.20, 5.79, [2, 1, 2, 1, 2, 0, 1, 2], 30),
    ("Venlo",      51.37, 6.17, [2, 1, 2, 1, 1, 0, 2, 1], 26),
    ("Deventer",   52.25, 6.16, [2, 1, 2, 1, 1, 0, 1, 2], 26),
    ("Hilversum",  52.22, 5.18, [2, 2, 3, 1, 0, 0, 0, 3], 26),
    ("Helmond",    51.48, 5.66, [2, 1, 3, 2, 0, 0, 1, 1], 24),
    ("Hengelo",    52.27, 6.79, [2, 1, 2, 1, 1, 0, 2, 1], 22),
    ("Zoetermeer", 52.06, 4.49, [1, 0, 2, 5, 0, 0, 0, 1], 30),
    ("Emmen",      52.78, 6.90, [1, 0, 2, 1, 3, 0, 1, 2], 28),
    ("Den Bosch",  51.70, 5.30, [3, 2, 2, 2, 0, 0, 1, 1], 32),
    ("Zandvoort",  52.37, 4.53, [1, 1, 1, 0, 0, 5, 0, 0], 14),
    ("Vlissingen", 51.45, 3.57, [1, 1, 1, 1, 1, 4, 2, 0], 18),
    ("Texel",      53.05, 4.80, [0, 0, 1, 0, 3, 3, 0, 2], 12),
    ("Drenthe",    52.90, 6.55, [0, 0, 1, 0, 6, 0, 0, 3], 45),
    ("Zeeland",    51.50, 3.85, [0, 1, 1, 0, 4, 3, 0, 1], 42),
    ("Friesland",  53.05, 5.80, [0, 1, 1, 1, 6, 1, 0, 2], 55),
    ("Achterhoek", 52.00, 6.40, [0, 1, 2, 0, 5, 0, 0, 3], 48),
    ("Flevopolder",52.55, 5.55, [0, 0, 1, 2, 6, 0, 1, 1], 42),
    ("Veluwe",     52.20, 5.80, [0, 1, 1, 0, 3, 0, 0, 6], 45),
    ("Westland",   52.00, 4.20, [0, 1, 1, 2, 3, 1, 2, 0], 36),
    ("Limburg-Z",  50.90, 5.95, [0, 1, 2, 0, 4, 0, 1, 3], 40),
]

DESCRIPTIONS = {
    "urban_core": [
        "Dense inner-city quarter with apartment blocks, tram lines, shops and a young, mobile population.",
        "Mixed-use core of offices, cafés and high-rise housing; busy day and night.",
        "Compact urban district, narrow streets, intense cycling traffic and little greenery.",
    ],
    "historic_center": [
        "Old town of canal houses and stepped gables, cobbled lanes and a steady tourist current.",
        "Protected historic centre with churches, hofjes and converted warehouses.",
        "Monumental core, drawbridges and brown cafés along the water.",
    ],
    "suburban": [
        "Quiet residential suburb of rowhouses with gardens, schools and a local shopping street.",
        "Post-war family neighbourhood, tree-lined streets, playgrounds and ample parking.",
        "Calm commuter belt of brick terraces and allotments near the ring road.",
    ],
    "modern_vinex": [
        "Newly built VINEX estate of family homes, wide avenues, a supermarket and a primary school.",
        "Reclaimed-land development with modern townhouses, planted dykes and cycle highways.",
        "Architect-designed waterfront blocks, rooftop terraces and fresh landscaping.",
    ],
    "rural_farmland": [
        "Open polder of farms, ditches and grazing cattle, a church spire on the horizon.",
        "Agricultural hamlet ringed by maize fields and tractor roads.",
        "Scattered farmsteads between wide pastures under a big sky.",
    ],
    "coastal": [
        "Beachfront strip with dunes, a boulevard of restaurants and summer crowds.",
        "Seaside village, salt-bitten houses and a small fishing harbour.",
        "Wide beach, kite surfers and dune-valley pavilions.",
    ],
    "industrial_port": [
        "Container terminals, cranes and pipelines along a deep-water harbour.",
        "Industrial estate of warehouses, logistics hubs and freight rail sidings.",
        "Refineries and tank farms behind chain-link fencing by the river.",
    ],
    "woodland_village": [
        "Wooded village of villas hidden among pines, sandy footpaths and heath nearby.",
        "Leafy Veluwe community, brick cottages and equestrian centres.",
        "Green, affluent enclave of detached houses behind tall hedges.",
    ],
}


# -------------------- embeddings --------------------

def cosine(emb):
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return (e @ e.T).astype(np.float32)


def project_2d(emb):
    if HAVE_UMAP and len(emb) > 15:
        return umap.UMAP(n_neighbors=15, min_dist=0.12, random_state=0).fit_transform(emb)
    return PCA(n_components=2).fit_transform(emb)


def correlated_embeddings(arch_idx, n_arch, dim=64, shared=0.7, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=2.0, size=(n_arch, dim))
    latent = centres[arch_idx] + rng.normal(scale=0.6, size=(len(arch_idx), dim))

    def space(s):
        r = np.random.default_rng(s)
        Q, _ = np.linalg.qr(r.normal(size=(dim, dim)))
        x = latent @ Q
        return (shared * x + (1 - shared) * r.normal(scale=2.0, size=x.shape)).astype(np.float32)

    return space(seed + 1), space(seed + 2)


# -------------------- organic polygons --------------------

def _voronoi_finite_polygons_2d(vor, radius=None):
    new_regions, new_vertices = [], vor.vertices.tolist()
    centre = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2
    ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges.setdefault(p1, []).append((p2, v1, v2))
        ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        verts = vor.regions[region]
        if all(v >= 0 for v in verts):
            new_regions.append(verts)
            continue
        new_region = [v for v in verts if v >= 0]
        for p2, v1, v2 in ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            mid = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(mid - centre, n)) * n
            far = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        order = np.argsort(np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0]))
        new_regions.append(list(np.array(new_region)[order]))
    return new_regions, np.asarray(new_vertices)


def _clip_to_rect(poly, rect):
    xmin, ymin, xmax, ymax = rect

    def isect(a, b, axis, val):
        d = b[axis] - a[axis]
        t = 0.0 if d == 0 else (val - a[axis]) / d
        return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]

    def clip(pts, inside, axis, val):
        out = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            ina, inb = inside(a), inside(b)
            if ina and inb:
                out.append(b)
            elif ina and not inb:
                out.append(isect(a, b, axis, val))
            elif not ina and inb:
                out.append(isect(a, b, axis, val)); out.append(b)
        return out

    pts = poly
    pts = clip(pts, lambda p: p[0] >= xmin, 0, xmin)
    if pts: pts = clip(pts, lambda p: p[0] <= xmax, 0, xmax)
    if pts: pts = clip(pts, lambda p: p[1] >= ymin, 1, ymin)
    if pts: pts = clip(pts, lambda p: p[1] <= ymax, 1, ymax)
    return pts


def _edge_midpoints(a, b, n_sub=3, amp=0.10):
    """Deterministic wavy midpoints for edge a-b.

    Keyed on the *unordered* endpoints so the two cells sharing this edge get the
    identical points (borders stay glued). Points are generated for the sorted
    orientation and reversed if the caller traverses the other way.
    """
    ka = (round(a[0], 5), round(a[1], 5))
    kb = (round(b[0], 5), round(b[1], 5))
    flip = ka > kb
    p, q = (kb, ka) if flip else (ka, kb)
    seed = (hash((p, q)) & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)
    p = np.array(p); q = np.array(q)
    d = q - p
    length = np.hypot(*d)
    if length < 1e-4:
        return []
    perp = np.array([-d[1], d[0]]) / length
    pts = []
    for i in range(1, n_sub + 1):
        t = i / (n_sub + 1)
        base = p + t * d
        # taper towards the endpoints so corners stay sharp and shared
        taper = np.sin(np.pi * t)
        offset = rng.normal(scale=amp) * length * taper
        pts.append((base + perp * offset).tolist())
    if flip:
        pts.reverse()
    return pts


def organic_cells(points, pad=0.06):
    vor = Voronoi(points)
    regions, vertices = _voronoi_finite_polygons_2d(vor)
    rect = (points[:, 0].min() - pad, points[:, 1].min() - pad,
            points[:, 0].max() + pad, points[:, 1].max() + pad)
    cells = []
    for reg in regions:
        poly = [vertices[v].tolist() for v in reg]
        poly = _clip_to_rect(poly, rect)
        if len(poly) < 3:
            cells.append([])
            continue
        ring = []
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            ring.append([round(a[0], 5), round(a[1], 5)])
            for m in _edge_midpoints(a, b):
                ring.append([round(m[0], 5), round(m[1], 5)])
        ring.append(ring[0])
        cells.append(ring)
    return cells


def write_geojson(cells, names, path="assets/nl_cells.geojson"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    feats = []
    for i, ring in enumerate(cells):
        if not ring:
            continue
        feats.append({"type": "Feature", "id": i,
                      "properties": {"fid": i, "name": names[i]},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, separators=(",", ":"))
    return path


# -------------------- build --------------------

def build(emb_clip, emb_text, df, cells):
    df = df.copy()
    emb_cbs = ((df[CBS_COLS] - df[CBS_COLS].mean()) /
               (df[CBS_COLS].std() + 1e-9)).values.astype(np.float32)

    for space, emb in [("clip", emb_clip), ("text", emb_text), ("cbs", emb_cbs)]:
        xy = project_2d(emb)
        df[f"{space}_x"] = xy[:, 0]
        df[f"{space}_y"] = xy[:, 1]
        df[f"{space}_cluster"] = KMeans(n_clusters=N_CLUSTERS, random_state=0,
                                        n_init=10).fit(emb).labels_

    df.to_parquet("data_nl.parquet")
    np.save("sim_nl_clip.npy", cosine(emb_clip))
    np.save("sim_nl_text.npy", cosine(emb_text))
    np.save("sim_nl_cbs.npy", cosine(emb_cbs))
    write_geojson(cells, df["name"].tolist())
    print(f"wrote data_nl.parquet ({len(df)} rows) + sim_nl_*.npy + "
          f"assets/nl_cells.geojson  (UMAP={'yes' if HAVE_UMAP else 'PCA'})")


def mock():
    rng = np.random.default_rng(11)
    rows, arch_idx = [], []
    for city, lat, lon, weights, count in CITIES:
        w = np.array(weights, float)
        w = w / w.sum()
        spread = 0.05 if city in ("Amsterdam", "Rotterdam", "Den Haag", "Utrecht") else 0.11
        for j in range(count):
            ai = int(rng.choice(len(ARCHETYPES), p=w))
            name, profile = ARCHETYPES[ai][0], ARCHETYPES[ai][1]
            dev_w = np.array(ARCHETYPES[ai][2], float); dev_w /= dev_w.sum()
            unit_scale = ARCHETYPES[ai][3]
            stats = np.clip(np.array(profile, float) *
                            (1 + rng.normal(scale=0.16, size=len(profile))), 1, None)
            dev_cat = DEV_CATEGORIES[int(rng.choice(len(DEV_CATEGORIES), p=dev_w))]
            units = 0 if dev_cat == "None" else float(max(0, rng.normal(unit_scale, unit_scale * 0.4)))
            rows.append({
                "code": f"{city[:3].upper()}{j:04d}",
                "name": f"{city} – {name.replace('_', ' ')} {j + 1}",
                "city": city,
                "archetype": name,
                "lat": lat + rng.normal(scale=spread * 0.6),
                "lon": lon + rng.normal(scale=spread),
                "description": rng.choice(DESCRIPTIONS[name]),
                "development": dev_cat,
                "planned_units": round(units),
                **{c: float(round(v, 2)) for c, v in zip(CBS_COLS, stats)},
            })
            arch_idx.append(ai)

    df = pd.DataFrame(rows)
    pts = df[["lon", "lat"]].values
    cells = organic_cells(pts)
    emb_clip, emb_text = correlated_embeddings(np.array(arch_idx), len(ARCHETYPES))
    build(emb_clip, emb_text, df, cells)


if __name__ == "__main__":
    mock()
