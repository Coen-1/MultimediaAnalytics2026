"""Amsterdam neighbourhood embedding explorer.

Adds on top of the NL version:
  - address / buurt search (Nominatim, with substring fallback)
  - adjustable top-k slider, clickable top-k list with per-row sim score
  - "why are they similar?" comparison panel:
      * cosine score per space + cluster-match indicator
      * paired CBS bar chart (normalised)
      * side-by-side descriptions
  - box-zoom + scroll-zoom in the embedding scatters

Two selection states:
  state.i  primary selection (drives top-k, spider, info, map highlight)
  state.j  comparison target (set by clicking a top-k item; cleared on any new
           primary selection)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

df = pd.read_parquet("data_ams.parquet").reset_index(drop=True)
SIM = {"clip": np.load("sim_ams_clip.npy"),
       "text": np.load("sim_ams_text.npy"),
       "cbs":  np.load("sim_ams_cbs.npy")}
CBS = ["population", "income", "pct_green", "avg_age", "home_value", "density"]
NORM = (df[CBS] - df[CBS].min()) / (df[CBS].max() - df[CBS].min() + 1e-9)
N = len(df)

PDOK_TILE = ("https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/"
             "Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg")
MAP_STYLE = {
    "version": 8,
    "sources": {"p": {"type": "raster", "tiles": [PDOK_TILE], "tileSize": 256,
                      "attribution": "PDOK Luchtfoto"}},
    "layers": [{"id": "p", "type": "raster", "source": "p"}],
}

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759",
           "#b07aa1", "#76b7b2", "#edc949", "#9c755f"]
SELECTED = "#e6194b"
COMPARE = "#2470bb"
AMS_CENTRE = dict(lat=52.358, lon=4.910)
AMS_ZOOM = 11.2


# -------------------- neighbourhood cells (Voronoi) --------------------
# We only store centroids, so we synthesise a neighbourhood *area* per buurt as
# the Voronoi cell of its centroid, clipped to the Amsterdam bounding box. These
# are drawn as faint outlines (so the map reads as a neighbourhood map) and the
# selected / comparison cells are filled — far easier to see than a single dot.

from scipy.spatial import Voronoi  # noqa: E402


def _voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions into finite polygons (2D recipe)."""
    new_regions, new_vertices = [], vor.vertices.tolist()
    centre = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        verts = vor.regions[region]
        if all(v >= 0 for v in verts):
            new_regions.append(verts)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in verts if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centre, n)) * n
            far = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        order = np.argsort(np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0]))
        new_regions.append(list(np.array(new_region)[order]))
    return new_regions, np.asarray(new_vertices)


def _clip_to_rect(poly, rect):
    """Sutherland-Hodgman clip of a polygon to an axis-aligned rectangle."""
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
                out.append(isect(a, b, axis, val))
                out.append(b)
        return out

    pts = poly
    pts = clip(pts, lambda p: p[0] >= xmin, 0, xmin)
    if pts: pts = clip(pts, lambda p: p[0] <= xmax, 0, xmax)
    if pts: pts = clip(pts, lambda p: p[1] >= ymin, 1, ymin)
    if pts: pts = clip(pts, lambda p: p[1] <= ymax, 1, ymax)
    return pts


def _build_cells():
    pts = df[["lon", "lat"]].values  # x=lon, y=lat
    vor = Voronoi(pts)
    regions, vertices = _voronoi_finite_polygons_2d(vor)
    pad = 0.012
    rect = (pts[:, 0].min() - pad, pts[:, 1].min() - pad,
            pts[:, 0].max() + pad, pts[:, 1].max() + pad)
    cells = []
    for reg in regions:
        poly = [vertices[v].tolist() for v in reg]
        clipped = _clip_to_rect(poly, rect)
        if clipped:
            clipped.append(clipped[0])  # close ring
        cells.append(clipped)
    return cells


CELLS = _build_cells()  # CELLS[k] is a closed [lon,lat] ring for df row k


def cells_geojson(indices):
    feats = []
    for i in indices:
        ring = CELLS[i]
        if not ring:
            continue
        feats.append({"type": "Feature", "id": int(i),
                      "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": {"name": str(df.iloc[i]["name"])}})
    return {"type": "FeatureCollection", "features": feats}


ALL_CELLS_GEOJSON = cells_geojson(range(N))


# -------------------- figures --------------------

def base_colours(color_by: str) -> np.ndarray:
    labels = (df["archetype"].astype("category").cat.codes.values
              if color_by == "archetype"
              else df[f"{color_by}_cluster"].values)
    return np.array([PALETTE[int(l) % len(PALETTE)] for l in labels], dtype=object)


def topk_indices(i: int, sim_key: str, k: int) -> np.ndarray:
    return np.argsort(SIM[sim_key][i])[::-1][1:k + 1]


def marker_arrays(i: int, j: int | None, sim_key: str, color_by: str, k: int):
    c = base_colours(color_by).copy()
    s = np.full(N, 5.5)
    o = np.full(N, 0.55)
    if sim_key != "none":
        hot = topk_indices(i, sim_key, k)
        s[hot] = 11
        o[hot] = 1.0
    if j is not None and 0 <= j < N and j != i:
        c[j] = COMPARE
        s[j] = 14
        o[j] = 1.0
    c[i] = SELECTED
    s[i] = 17
    o[i] = 1.0
    return c, s, o


def scatter(space: str, i: int, j: int | None, sim_key: str, color_by: str, k: int) -> go.Figure:
    c, s, o = marker_arrays(i, j, sim_key, color_by, k)
    fig = go.Figure(go.Scattergl(
        x=df[f"{space}_x"], y=df[f"{space}_y"], mode="markers",
        marker=dict(color=c, size=s, opacity=o, line=dict(width=0)),
        text=df["name"] + "  (" + df["archetype"] + ")",
        hoverinfo="text",
        customdata=df.index.values,
    ))
    fig.update_layout(
        title=dict(text=f"{space.upper()} — UMAP  ·  drag to box-zoom · scroll · 2x-click resets",
                   x=0.02, y=0.97, font=dict(size=10, color="#444")),
        margin=dict(l=4, r=4, t=22, b=4),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#fafafa", paper_bgcolor="#ffffff",
        showlegend=False, dragmode="zoom", uirevision=f"scatter-{space}",
    )
    return fig


def map_layers(i: int, j: int | None):
    """Overlay layers: faint cell outlines everywhere + filled selected/compare cells."""
    layers = [
        # all neighbourhood boundaries, subtle
        {"sourcetype": "geojson", "source": ALL_CELLS_GEOJSON, "type": "line",
         "color": "rgba(255,255,255,0.45)", "line": {"width": 1}},
        # selected cell: fill + crisp outline
        {"sourcetype": "geojson", "source": cells_geojson([i]), "type": "fill",
         "color": SELECTED, "opacity": 0.42},
        {"sourcetype": "geojson", "source": cells_geojson([i]), "type": "line",
         "color": "#ffffff", "line": {"width": 4}},
        {"sourcetype": "geojson", "source": cells_geojson([i]), "type": "line",
         "color": SELECTED, "line": {"width": 2}},
    ]
    if j is not None and 0 <= j < N and j != i:
        layers += [
            {"sourcetype": "geojson", "source": cells_geojson([j]), "type": "fill",
             "color": COMPARE, "opacity": 0.25},
            {"sourcetype": "geojson", "source": cells_geojson([j]), "type": "line",
             "color": COMPARE, "line": {"width": 2.5}},
        ]
    return layers


def mapfig(i: int, j: int | None, sim_key: str, color_by: str, k: int,
           follow: bool, zoom_to_sel: bool) -> go.Figure:
    c, s, o = marker_arrays(i, j, sim_key, color_by, k)
    s = s * 0.6 + 4  # a touch larger -> easier to click
    fig = go.Figure(go.Scattermap(
        lat=df["lat"], lon=df["lon"], mode="markers",
        marker=dict(color=c, size=s, opacity=o),
        text=df["name"] + "  ·  " + df["archetype"].str.replace("_", " "),
        hoverinfo="text",
        hoverlabel=dict(font=dict(size=13), bgcolor="white"),
        customdata=df.index.values,
    ))
    if follow and zoom_to_sel:
        centre = dict(lat=df.lat.iloc[i], lon=df.lon.iloc[i])
        zoom, uirev = 14, f"sel-{i}"
    else:
        centre, zoom, uirev = AMS_CENTRE, AMS_ZOOM, "stable"
    fig.update_layout(
        map=dict(style=MAP_STYLE, center=centre, zoom=zoom, layers=map_layers(i, j)),
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False, uirevision=uirev,
    )
    return fig


def spider(i: int, j: int | None) -> go.Figure:
    vi = NORM.iloc[i].tolist()
    traces = [go.Scatterpolar(
        r=vi + [vi[0]], theta=CBS + [CBS[0]],
        fill="toself", line=dict(color=SELECTED),
        fillcolor="rgba(230,25,75,0.25)", name="selected",
    )]
    if j is not None and j != i:
        vj = NORM.iloc[j].tolist()
        traces.append(go.Scatterpolar(
            r=vj + [vj[0]], theta=CBS + [CBS[0]],
            fill="toself", line=dict(color=COMPARE),
            fillcolor="rgba(36,112,187,0.20)", name="compare",
        ))
    fig = go.Figure(traces)
    fig.update_layout(
        polar=dict(bgcolor="#fafafa",
                   radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                   angularaxis=dict(tickfont=dict(size=10))),
        margin=dict(l=40, r=40, t=10, b=10), height=230, paper_bgcolor="#ffffff",
        legend=dict(orientation="h", x=0.5, y=-0.05, xanchor="center", font=dict(size=10)),
        showlegend=(j is not None and j != i),
    )
    return fig


def cbs_bars(i: int, j: int) -> go.Figure:
    """Paired CBS values (normalised) for the selected vs comparison buurt."""
    fig = go.Figure([
        go.Bar(y=CBS, x=NORM.iloc[i].values, orientation="h",
               name=df.iloc[i]["name"], marker_color=SELECTED,
               hovertemplate=f"{df.iloc[i]['name']}<br>%{{y}} = %{{x:.2f}}<extra></extra>"),
        go.Bar(y=CBS, x=NORM.iloc[j].values, orientation="h",
               name=df.iloc[j]["name"], marker_color=COMPARE,
               hovertemplate=f"{df.iloc[j]['name']}<br>%{{y}} = %{{x:.2f}}<extra></extra>"),
    ])
    fig.update_layout(
        barmode="group", height=210, margin=dict(l=86, r=10, t=28, b=10),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, y=1.18, xanchor="center", font=dict(size=10)),
        title=dict(text="CBS values, normalised 0–1", x=0.02, font=dict(size=11)),
        xaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="#eee"),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        font=dict(size=10),
    )
    return fig


# -------------------- search --------------------

def geocode(q: str):
    """Return (row_index, status_message) or (None, error_message)."""
    if not q or not q.strip():
        return None, ""
    s = q.strip()
    exact = df[df["name"].str.lower() == s.lower()]
    if len(exact):
        return int(exact.index[0]), f"matched buurt: {exact.iloc[0]['name']}"
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{s}, Amsterdam, Netherlands",
                    "format": "json", "limit": 1, "countrycodes": "nl",
                    "viewbox": "4.74,52.42,5.07,52.28", "bounded": 1},
            headers={"User-Agent": "ma-dashboard-prototype/0.1"},
            timeout=5,
        )
        hits = r.json()
        if hits:
            lat, lon = float(hits[0]["lat"]), float(hits[0]["lon"])
            d2 = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
            i = int(d2.idxmin())
            place = hits[0]["display_name"].split(",")[0]
            return i, f"snapped from “{place}” → {df.iloc[i]['name']}"
    except Exception:
        pass
    sub = df[df["name"].str.lower().str.contains(s.lower(), regex=False)]
    if len(sub):
        return int(sub.index[0]), f"matched buurt: {sub.iloc[0]['name']}"
    return None, "no match"


# -------------------- layout --------------------

PANEL = {"padding": "10px 14px", "borderBottom": "1px solid #eee"}
LABEL = {"fontSize": "11px", "color": "#666", "marginBottom": "4px",
         "textTransform": "uppercase", "letterSpacing": "0.04em"}

app = Dash(__name__, title="Amsterdam Embeddings")
app.layout = html.Div(
    style={
        "display": "grid",
        "gridTemplateColumns": "minmax(320px, 360px) 1fr minmax(360px, 440px)",
        "gridTemplateRows": "48px 1fr",
        "height": "100vh",
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "fontSize": "13px", "color": "#222",
    },
    children=[
        # header
        html.Div(style={
            "gridColumn": "1 / -1", "background": "#1f2937", "color": "white",
            "display": "flex", "alignItems": "center", "padding": "0 14px",
            "gap": "16px",
        }, children=[
            html.Div("Amsterdam Embedding Explorer",
                     style={"fontWeight": 600, "fontSize": "14px", "flexShrink": 0}),
            dcc.Input(id="search", type="text", debounce=True, n_submit=0,
                      placeholder="Search address or buurt (Enter to submit)…",
                      style={"flex": 1, "padding": "6px 10px", "fontSize": "13px",
                             "border": "1px solid #4b5563", "borderRadius": 4,
                             "background": "#374151", "color": "white"}),
            html.Div(id="search-status",
                     style={"fontSize": "11px", "color": "#9ca3af",
                            "minWidth": "180px", "textAlign": "right"}),
        ]),

        # LEFT (scrollable)
        html.Div(style={"borderRight": "1px solid #eee", "overflowY": "auto"},
                 children=[
                     html.Div(id="info", style=PANEL),
                     html.Div(style=PANEL, children=[
                         html.Div("CBS profile", style=LABEL),
                         dcc.Graph(id="spider", config={"displayModeBar": False},
                                   style={"height": "230px"}),
                     ]),
                     html.Div(style=PANEL, children=[
                         html.Div("Description", style=LABEL),
                         html.Div(id="description",
                                  style={"fontSize": "12px", "color": "#444",
                                         "fontStyle": "italic", "lineHeight": "1.4"}),
                     ]),
                     html.Div(style=PANEL, children=[
                         html.Div([
                             html.Span("Top-k similar by ", style=LABEL),
                             html.Span(id="topk-sim-label",
                                       style={"fontSize": "11px", "color": "#222",
                                              "fontWeight": 600,
                                              "letterSpacing": "0.04em"}),
                         ], style={"display": "flex", "alignItems": "baseline",
                                   "gap": "4px", "marginBottom": "4px"}),
                         dcc.Slider(id="k", min=3, max=25, step=1, value=10,
                                    marks={3: "3", 10: "10", 25: "25"},
                                    tooltip={"placement": "bottom",
                                             "always_visible": False}),
                         html.Div(id="topk-list", style={"marginTop": "10px"}),
                     ]),
                     html.Div(id="compare-panel",
                              style={"padding": "10px 14px",
                                     "background": "#f9fafb",
                                     "borderTop": "1px solid #eee",
                                     "borderBottom": "1px solid #eee"}),
                     html.Div(style=PANEL, children=[
                         html.Div("Highlight top-k by", style=LABEL),
                         dcc.RadioItems(id="simkey", value="clip", inline=True,
                                        options=[{"label": " CLIP", "value": "clip"},
                                                 {"label": " Text", "value": "text"},
                                                 {"label": " CBS",  "value": "cbs"},
                                                 {"label": " none", "value": "none"}],
                                        labelStyle={"marginRight": "10px"}),
                         html.Div("Colour dots by clusters of",
                                  style={**LABEL, "marginTop": "10px"}),
                         dcc.RadioItems(id="colorby", value="archetype", inline=True,
                                        options=[{"label": " CLIP",      "value": "clip"},
                                                 {"label": " Text",      "value": "text"},
                                                 {"label": " CBS",       "value": "cbs"},
                                                 {"label": " archetype", "value": "archetype"}],
                                        labelStyle={"marginRight": "10px"}),
                         dcc.Checklist(id="follow", value=["f"],
                                       options=[{"label": " zoom map to selection on click",
                                                 "value": "f"}],
                                       style={"marginTop": "8px"}),
                     ]),
                     html.Div(style={**PANEL, "fontSize": "11px", "color": "#888"},
                              children=[
                                  "Red = selected · Blue = comparison · large dots = top-k. ",
                                  "Click a top-k item to compare; click the map or any scatter to change the selection. ",
                                  "Scatters: drag for box-zoom; scroll wheel zooms; double-click resets.",
                              ]),
                 ]),

        # CENTRE
        html.Div(style={"position": "relative"}, children=[
            dcc.Graph(id="map", style={"height": "100%"},
                      config={"displayModeBar": False, "scrollZoom": True}),
        ]),

        # RIGHT
        html.Div(style={"display": "flex", "flexDirection": "column",
                        "borderLeft": "1px solid #eee"},
                 children=[
                     dcc.Graph(id="clip", style={"flex": 1, "minHeight": 0},
                               config={"displayModeBar": False, "scrollZoom": True,
                                       "doubleClick": "reset"}),
                     dcc.Graph(id="text", style={"flex": 1, "minHeight": 0},
                               config={"displayModeBar": False, "scrollZoom": True,
                                       "doubleClick": "reset"}),
                     dcc.Graph(id="cbs",  style={"flex": 1, "minHeight": 0},
                               config={"displayModeBar": False, "scrollZoom": True,
                                       "doubleClick": "reset"}),
                 ]),

        dcc.Store(id="state",
                  data={"i": 0, "j": None, "trigger": "init", "msg": ""}),
    ],
)


# -------------------- helpers for callbacks --------------------

def _pick(click):
    if not click: return None
    pt = click["points"][0]
    return int(pt.get("customdata", pt.get("pointIndex", 0)))


def info_block(row):
    return html.Div([
        html.Div(row["name"], style={"fontWeight": 600, "fontSize": "14px"}),
        html.Div(row["archetype"].replace("_", " "),
                 style={"fontSize": "11px", "color": "#666", "marginTop": "2px"}),
        html.Table(style={"width": "100%", "fontSize": "12px",
                          "marginTop": "8px", "borderCollapse": "collapse"},
                   children=[
                       html.Tr([html.Td(c, style={"color": "#666", "padding": "2px 0"}),
                                html.Td(f"{row[c]:.1f}",
                                        style={"textAlign": "right", "padding": "2px 0",
                                               "fontVariantNumeric": "tabular-nums"})])
                       for c in CBS
                   ]),
    ])


def topk_item(idx: int, score: float, primary_i: int, compare_j: int | None):
    row = df.iloc[idx]
    bar_pct = max(2, int(round(score * 100)))
    is_compare = compare_j is not None and idx == compare_j
    return html.Div(
        id={"type": "topk-item", "index": int(idx)},
        n_clicks=0,
        style={
            "padding": "6px 10px", "marginBottom": "4px",
            "border": "1px solid " + (COMPARE if is_compare else "#eee"),
            "borderRadius": 4, "cursor": "pointer",
            "background": "#eff5fb" if is_compare else "#fff",
        },
        children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between",
                            "alignItems": "baseline", "gap": "6px"},
                     children=[
                         html.Span(row["name"], style={"fontSize": "12px",
                                                       "fontWeight": 500,
                                                       "color": "#222"}),
                         html.Span(f"{score:.2f}",
                                   style={"fontSize": "11px", "color": "#666",
                                          "fontVariantNumeric": "tabular-nums"}),
                     ]),
            html.Div(style={"display": "flex", "justifyContent": "space-between",
                            "alignItems": "center", "marginTop": "2px",
                            "fontSize": "10px", "color": "#888"},
                     children=[
                         html.Span(row["archetype"].replace("_", " ")),
                         html.Div(style={"width": "60px", "height": "4px",
                                         "background": "#eee", "borderRadius": 2,
                                         "overflow": "hidden"},
                                  children=[html.Div(style={
                                      "width": f"{bar_pct}%", "height": "100%",
                                      "background": "#4e79a7"})]),
                     ]),
        ],
    )


def compare_panel(i: int, j: int | None):
    if j is None or j == i:
        return html.Div("Click a top-k item to compare it with the selected buurt.",
                        style={"color": "#888", "fontSize": "11px",
                               "fontStyle": "italic"})
    rs, rc = df.iloc[i], df.iloc[j]
    scores = [(k_, float(SIM[k_][i, j]),
               bool(df.iloc[i][f"{k_}_cluster"] == df.iloc[j][f"{k_}_cluster"]))
              for k_ in ["clip", "text", "cbs"]]

    score_table = html.Table(style={"width": "100%", "fontSize": "11px",
                                    "marginTop": "6px", "borderCollapse": "collapse"},
                             children=[
        html.Tr([
            html.Td(k_.upper(), style={"width": "40px", "color": "#666"}),
            html.Td(html.Div(style={"background": "#eee", "height": "8px",
                                    "borderRadius": 2, "overflow": "hidden"},
                             children=html.Div(style={
                                 "width": f"{max(0, sc) * 100:.0f}%",
                                 "background": "#4e79a7", "height": "100%"}))),
            html.Td(f"{sc:.2f}",
                    style={"width": "44px", "textAlign": "right",
                           "fontVariantNumeric": "tabular-nums", "color": "#222"}),
            html.Td("● same cluster" if match else "○ different cluster",
                    style={"width": "130px", "fontSize": "10px",
                           "color": "#2e7d32" if match else "#999",
                           "paddingLeft": "8px"}),
        ]) for k_, sc, match in scores
    ])

    desc_grid = html.Div(style={"display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "10px", "marginTop": "10px",
                                "fontSize": "11px"},
                         children=[
        html.Div([html.Div(rs["name"], style={"fontWeight": 600, "color": SELECTED}),
                  html.Div(f"“{rs['description']}”",
                           style={"fontStyle": "italic", "color": "#444",
                                  "marginTop": "2px"})]),
        html.Div([html.Div(rc["name"], style={"fontWeight": 600, "color": COMPARE}),
                  html.Div(f"“{rc['description']}”",
                           style={"fontStyle": "italic", "color": "#444",
                                  "marginTop": "2px"})]),
    ])

    return html.Div([
        html.Div("Why are they similar?", style=LABEL),
        score_table,
        dcc.Graph(figure=cbs_bars(i, j),
                  config={"displayModeBar": False},
                  style={"height": "210px", "marginTop": "8px"}),
        desc_grid,
    ])


# -------------------- callbacks --------------------

@app.callback(
    Output("state", "data"),
    Input("clip", "clickData"), Input("text", "clickData"),
    Input("cbs",  "clickData"), Input("map",  "clickData"),
    Input("search", "n_submit"),
    Input({"type": "topk-item", "index": ALL}, "n_clicks"),
    State("search", "value"), State("state", "data"),
    prevent_initial_call=True,
)
def on_event(c_clip, c_text, c_cbs, c_map, _search_n, tk_clicks, q, st):
    src = ctx.triggered_id
    if isinstance(src, dict) and src.get("type") == "topk-item":
        if not tk_clicks or not any(tk_clicks):
            return no_update
        return {**st, "j": int(src["index"]), "msg": ""}
    if src == "search":
        i, msg = geocode(q)
        if i is None:
            return {**st, "msg": msg}
        return {"i": int(i), "j": None, "trigger": "search", "msg": msg}
    if src in ("clip", "text", "cbs", "map"):
        cd = {"clip": c_clip, "text": c_text, "cbs": c_cbs, "map": c_map}[src]
        i = _pick(cd)
        if i is None:
            return no_update
        return {"i": int(i), "j": None, "trigger": src, "msg": ""}
    return no_update


@app.callback(
    Output("clip", "figure"), Output("text", "figure"), Output("cbs", "figure"),
    Output("map", "figure"), Output("spider", "figure"),
    Output("info", "children"), Output("description", "children"),
    Output("topk-list", "children"), Output("compare-panel", "children"),
    Output("topk-sim-label", "children"), Output("search-status", "children"),
    Input("state", "data"),
    Input("simkey", "value"), Input("colorby", "value"),
    Input("follow", "value"), Input("k", "value"),
)
def render(state, simkey, colorby, follow, k):
    i = int(state.get("i", 0))
    j = state.get("j")
    j = int(j) if j is not None else None
    k = int(k or 10)
    follow_on = "f" in (follow or [])
    zoom_to_sel = state.get("trigger") in ("clip", "text", "cbs", "search")
    row = df.iloc[i]
    sim_for_list = simkey if simkey != "none" else "clip"
    hot = topk_indices(i, sim_for_list, k)
    items = [topk_item(int(idx), float(SIM[sim_for_list][i, int(idx)]), i, j)
             for idx in hot]

    return (
        scatter("clip", i, j, simkey, colorby, k),
        scatter("text", i, j, simkey, colorby, k),
        scatter("cbs",  i, j, simkey, colorby, k),
        mapfig(i, j, simkey, colorby, k, follow_on, zoom_to_sel),
        spider(i, j),
        info_block(row),
        f"“{row['description']}”",
        items,
        compare_panel(i, j),
        sim_for_list.upper(),
        state.get("msg", ""),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8051)
