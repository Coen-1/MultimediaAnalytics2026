import json, math, os
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.io as pio
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, ctx, exceptions
import requests
from shapely.geometry import shape, Point

pio.templates.default = "plotly_white"

df = pd.read_parquet("assets/data.parquet")
SIM = {"clip": np.load("assets/sim_clip.npy"), "text": np.load("assets/sim_text.npy"), "cbs": np.load("assets/sim_cbs.npy")}
with open("assets/buurten.geojson") as f:
    AREAS = json.load(f)
POLYGONS = []
for feature in AREAS["features"]:
    name = feature["properties"]["code"]
    idx = df[df["code"] == name].index[0]
    poly = shape(feature["geometry"])
    POLYGONS.append((name, idx, poly))
CBS = ["population", "income", "home_value", "density", "household_size",
       "pct_owner", "pct_single_pers", "pct_65plus", "pct_dutch", "cars_per_hh"]
LBL = {"population": "Residents", "income": "Income", "home_value": "Home value", "density": "Density",
       "household_size": "Household size", "pct_owner": "% owners", "pct_single_pers": "% single-person",
       "pct_65plus": "% 65+", "pct_dutch": "% Dutch", "cars_per_hh": "Cars/household"}  # spider axis labels
DESC = {"population": "Number of residents", "income": "Avg. income per resident", "home_value": "Average home value",
        "density": "Residents per km²", "household_size": "Average household size", "pct_owner": "% owner-occupied homes",
        "pct_single_pers": "% single-person households", "pct_65plus": "% residents 65 and older",
        "pct_dutch": "% Dutch-origin residents", "cars_per_hh": "Cars per household"}  # 3-4 word column descriptions
SPACE = {"clip": "Visual similarity (aerial imagery)", "text": "Description similarity (text)",
         "cbs": "Statistical similarity (CBS data)"}  # readable embedding-plot titles
OPTS = [{"label": DESC[c], "value": c} for c in CBS]               # readable column chips / dropdown
QOPTS = [{"label": f"{DESC[c]}  ·  {c}", "value": c} for c in CBS]  # query dropdown also shows the queryable column name
NORM = (df[CBS] - df[CBS].min()) / (df[CBS].max() - df[CBS].min() + 1e-9)
NORM_MEDIAN = NORM.median()   # per-indicator median of the available values; used to fill CBS-suppressed gaps
K = 10
TILE_TYPES = {"satellite": {"tiles": "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg",
                            "attribution": "Luchtfoto © Kadaster / Beeldmateriaal.nl"},
              "streetview": {"tiles": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", 
                             "attribution": "© OpenStreetMap contributors"}}
EXTRA_AREA_HIGHLIGHT = 20
ACCENT, SEL = "#4c78a8", "#e45756"
CLIP_C, TEXT_C, CBS_C = "#4c78a8", "#59a14f", "#9c6ade"
COMBINED_SCALE = [
    [0.00, "#d9eeec"],
    [0.35, "#8bcac7"],
    [0.65, "#3c9ba2"],
    [0.85, "#176779"],
    [1.00, "#082f49"],
]
FACET_BORDER = {1: "#f2c14e", 2: "#f78154", 3: "#7b2cbf"}
SPACE_SHORT = {"clip": "CLIP", "text": "Text", "cbs": "CBS"}
SURFACE_SCALE = {
    "clip": [[0, "#f4f6f8"], [0.35, "#dce5ee"], [0.7, "#94afca"], [1, CLIP_C]],
    "text": [[0, "#f4f6f8"], [0.35, "#e0eadf"], [0.7, "#9bc394"], [1, TEXT_C]],
    "cbs": [[0, "#f4f6f8"], [0.35, "#e8e0f1"], [0.7, "#c1a7df"], [1, CBS_C]],
}
UMAP_CORNERS = ("#a9bfd2", "#ddb29d", "#a9c6b2", "#c1afd2")
SPIDER_C_OUT = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
                '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
SPIDER_C_IN = ['rgba(122,131,250,0.25)', 'rgba(241,110,88,0.25)', 'rgba(38,211,165,0.25)',
               'rgba(183,122,250,0.25)', 'rgba(255,175,114,0.25)', 'rgba(59,217,244,0.25)',
               'rgba(255,124,162,0.25)', 'rgba(192,235,147,0.25)', 'rgba(255,166,255,0.25)', 'rgba(254,210,107,0.25)']
MODS = (("clip", CLIP_C), ("text", TEXT_C), ("cbs", CBS_C))
K_MAX = 25            # upper bound of the "similar shown" slider
DOT_OFFSET_PX = 5.5   # separation between modality dots at a shared neighbourhood centre
FONT = "Inter, system-ui, sans-serif"
PAGE = {"display": "flex", "gap": "12px", "padding": "0 12px 12px", "boxSizing": "border-box",
        "background": "#f5f6f8", "fontFamily": FONT}
CARD = {"background": "#fff", "borderRadius": "10px", "padding": "8px",
        "boxShadow": "0 1px 3px rgba(0,0,0,.12)", "overflow": "hidden"}
OVERLAY = {"position": "fixed", "inset": 0, "background": "rgba(0,0,0,.35)", "zIndex": 1000,
           "display": "flex", "alignItems": "center", "justifyContent": "center"}  # intro popup backdrop
BTN = {"background": ACCENT, "color": "#fff", "border": "none", "borderRadius": "8px", "padding": "0 14px",
       "cursor": "pointer", "fontWeight": "600", "fontFamily": FONT}  # primary button
CLR_BTN = {"background": SEL, "color": "#fff", "border": "none", "borderRadius": "8px", "padding": "0 14px",
       "cursor": "pointer", "fontWeight": "600", "fontFamily": FONT}  # primary button
LABEL = {"fontSize": "11px", "fontWeight": "700", "letterSpacing": ".04em", "textTransform": "uppercase",
         "color": "#8a8f98", "fontFamily": FONT, "margin": "2px 0"}  # small section header


def _fit(lat, lon):  # one-time center+zoom that frames every point
    c = dict(lat=float((lat.min() + lat.max()) / 2), lon=float((lon.min() + lon.max()) / 2))
    span = max(lat.max() - lat.min(), (lon.max() - lon.min()) * math.cos(math.radians(c["lat"])))
    return c, math.log2(360 / max(span, 1e-6)) - 1.2          # -1.2 = padding fudge
CENTER, ZOOM = _fit(df.lat, df.lon)

def avg_sim(F, space):  # mean cosine similarity of every buurt to the focus set F (>=1 row), in `space`
    return SIM[space][list(F)].mean(axis=0)

def topk_by(scores, filter, exclude, k):  # k filtered buurten with the highest score, excluding the focus set
    ex = set(exclude)
    return sorted((j for j in filter if j not in ex), key=lambda j: scores[j], reverse=True)[:k]

def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def _hex(rgb):
    return "#" + "".join(f"{max(0, min(255, round(v))):02x}" for v in rgb)

def _bivariate_colors(space, rows):
    """A soft four-corner colour field tied to the two displayed UMAP coordinates."""
    x = df[f"{space}_x"].to_numpy(float)
    y = df[f"{space}_y"].to_numpy(float)
    x0, x1 = np.quantile(x, [0.02, 0.98])
    y0, y1 = np.quantile(y, [0.02, 0.98])
    xn = np.clip((x - x0) / max(x1 - x0, 1e-9), 0, 1)
    yn = np.clip((y - y0) / max(y1 - y0, 1e-9), 0, 1)
    bl, br, tl, tr = (np.array(_rgb(c), dtype=float) for c in UMAP_CORNERS)
    out = []
    for idx in rows:
        bottom = bl * (1 - xn[idx]) + br * xn[idx]
        top = tl * (1 - xn[idx]) + tr * xn[idx]
        out.append(_hex(bottom * (1 - yn[idx]) + top * yn[idx]))
    return out

def _relative_percentiles(scores, rows):
    values = pd.Series([scores[idx] for idx in rows], index=rows)
    return values.rank(method="average", pct=True).to_dict()

def _facet_dot_offsets(count):
    if count == 1:
        return [(0.0, 0.0)]
    if count == 2:
        return [(-0.75, 0.0), (0.75, 0.0)]
    return [(0.0, 0.85), (-0.72, -0.45), (0.72, -0.45)]

def fmt(v):
    if isinstance(v, str):
        return v
    return "—" if pd.isna(v) else round(float(v), 1)

def geocode(q):
    """Return (row_index, status_message) or (None, error_message)."""
    if not q or not q.strip():
        return [], ""
    s = q.strip()
    exact = df[df["name"].str.lower() == s.lower()]
    if len(exact):
        return [int(exact.index[0])], ""
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
            pt = Point(lon, lat)
            for name, idx, poly in POLYGONS:
                if poly.contains(pt):
                    return [idx], ""
            
    except Exception:
        pass
    return [], "no match"

def scatter(space, i, filter, k=K):
    i = list(i or [])
    filter = list(filter or [])
    hot = topk_by(avg_sim(i, space), filter, i, k) if len(i) else []

    f = go.Figure(go.Scattergl(
        x=df[f"{space}_x"].iloc[filter],
        y=df[f"{space}_y"].iloc[filter],
        mode="markers",
        marker=dict(
            color=_bivariate_colors(space, filter),
            size=6.5,
            opacity=0.72,
            line=dict(width=0.6, color="rgba(255,255,255,.72)")
        ),
        customdata=filter,
        text=df["name"].iloc[filter],
        hoverinfo="text"
    ))

    # Similar points use a soft modality accent with a white edge instead of black rings.
    if hot:
        f.add_trace(go.Scattergl(
            x=df[f"{space}_x"].iloc[hot],
            y=df[f"{space}_y"].iloc[hot],
            mode="markers",
            marker=dict(
                size=10.5,
                color=dict(MODS)[space],
                opacity=0.88,
                line=dict(width=1.3, color="white")
            ),
            customdata=hot,
            text=[f"{df.at[idx, 'name']}<br>Top-{k} {SPACE_SHORT[space]} match" for idx in hot],
            hoverinfo="text"
        ))

    if i:
        f.add_trace(go.Scattergl(
            x=df[f"{space}_x"].iloc[i],
            y=df[f"{space}_y"].iloc[i],
            mode="markers",
            marker=dict(size=15, color=SEL, line=dict(width=2, color="white")),
            customdata=i,
            text=df["name"].iloc[i],
            hoverinfo="text"
        ))

    axis = dict(visible=False)

    return f.update_layout(
        title=SPACE[space],
        title_font=dict(size=13, color="#39434d"),
        clickmode="event+select",
        showlegend=False,
        xaxis=axis,
        yaxis=axis,
        margin=dict(l=6, r=6, t=26, b=6),
        dragmode="pan",
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

def mapfig(i, filter, map_style, k=K, view=None, color_mode="overlap"):
    """Build an overlap-count map or a continuous single-modality similarity surface."""
    i = list(i or [])
    filter = list(filter or [])
    k = int(k or K)
    zoom = float((view or {}).get("zoom", ZOOM))
    has_areas = os.path.exists("assets/buurten.geojson")

    if i:
        scores = {space: avg_sim(i, space) for space, _ in MODS}
        top = {space: topk_by(scores[space], filter, i, k) for space, _ in MODS}
    else:
        scores = {space: None for space, _ in MODS}
        top = {space: [] for space, _ in MODS}

    ranks = {
        space: {idx: rank for rank, idx in enumerate(nodes, start=1)}
        for space, nodes in top.items()
    }
    percentiles = {
        space: _relative_percentiles(scores[space], filter) if i else {}
        for space, _ in MODS
    }
    top_sets = {space: set(nodes) for space, nodes in top.items()}
    memberships = {
        idx: tuple(space for space, _ in MODS if idx in top_sets[space])
        for idx in set().union(*top_sets.values())
    } if i else {}
    combined_strength = {
        idx: float(np.mean([percentiles[space][idx] for space, _ in MODS]))
        for idx in memberships
    }
    if combined_strength:
        strength_lo, strength_hi = min(combined_strength.values()), max(combined_strength.values())
        display_strength = {
            idx: (value - strength_lo) / max(strength_hi - strength_lo, 1e-9)
            for idx, value in combined_strength.items()
        }
    else:
        display_strength = {}

    f = go.Figure()

    # Transparent base keeps every polygon clickable without visually filling the map.
    if has_areas:
        f.add_choroplethmap(
            geojson="/assets/buurten.geojson",
            featureidkey="properties.code",
            locations=df["code"].iloc[filter],
            customdata=filter,
            text=df["name"].iloc[filter],
            z=[0.0] * len(filter),
            zmin=0,
            zmax=1,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            showscale=False,
            marker=dict(opacity=0.01, line=dict(width=0)),
            selected=dict(marker=dict(opacity=0.01)),
            unselected=dict(marker=dict(opacity=0.01)),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        )

    if has_areas and i and color_mode in dict(MODS):
        # Relative percentile is easier to read across modalities whose raw cosine ranges differ.
        pct = percentiles[color_mode]
        hover = [
            f"{df.at[idx, 'name']}<br>{SPACE_SHORT[color_mode]} similarity: "
            f"{scores[color_mode][idx]:.3f}<br>Relative percentile: {pct[idx]:.0%}"
            for idx in filter
        ]
        f.add_choroplethmap(
            geojson="/assets/buurten.geojson",
            featureidkey="properties.code",
            locations=df["code"].iloc[filter],
            customdata=filter,
            text=hover,
            z=[pct[idx] for idx in filter],
            zmin=0,
            zmax=1,
            colorscale=SURFACE_SCALE[color_mode],
            showscale=True,
            colorbar=dict(
                title=dict(text=f"{SPACE_SHORT[color_mode]} similarity: low → high", side="top"),
                orientation="h",
                thickness=8,
                len=0.31,
                x=0.80,
                xanchor="center",
                y=0.975,
                yanchor="top",
                ticks="",
                showticklabels=False,
                bgcolor="rgba(255,255,255,.82)",
                outlinewidth=0
            ),
            marker=dict(opacity=0.58, line=dict(width=0.35, color="rgba(255,255,255,.55)")),
            selected=dict(marker=dict(opacity=0.58)),
            unselected=dict(marker=dict(opacity=0.58)),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        )
    elif has_areas and i:
        relevant_nodes = sorted(memberships)
        combined_hover = []
        for idx in relevant_nodes:
            spaces = memberships[idx]
            reasons = " + ".join(SPACE_SHORT[space] for space in spaces)
            detail = "<br>".join(
                f"{SPACE_SHORT[space]}: rank {ranks[space][idx]} · {scores[space][idx]:.3f}"
                for space in spaces
            )
            combined_hover.append(
                f"{df.at[idx, 'name']}<br>Combined relative similarity: "
                f"{combined_strength[idx]:.0%}<br>{len(spaces)} matching facet"
                f"{'s' if len(spaces) > 1 else ''}: {reasons}<br>{detail}"
            )
        f.add_choroplethmap(
            geojson="/assets/buurten.geojson",
            featureidkey="properties.code",
            locations=df["code"].iloc[relevant_nodes],
            customdata=relevant_nodes,
            text=combined_hover,
            z=[display_strength[idx] for idx in relevant_nodes],
            zmin=0,
            zmax=1,
            colorscale=COMBINED_SCALE,
            showscale=True,
            colorbar=dict(
                title=dict(text="Combined similarity<br>(darker = stronger)", side="top"),
                orientation="h",
                thickness=8,
                len=0.34,
                x=0.80,
                xanchor="center",
                y=0.975,
                yanchor="top",
                tickvals=[0, 1],
                ticktext=["Lower top-k", "Higher top-k"],
                tickfont=dict(size=9),
                bgcolor="rgba(255,255,255,.88)",
                outlinewidth=0
            ),
            marker=dict(opacity=0.78, line=dict(width=0.4, color="rgba(255,255,255,.7)")),
            selected=dict(marker=dict(opacity=0.78)),
            unselected=dict(marker=dict(opacity=0.78)),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        )

        # Transparent overlays encode the number of top-k facets using unmistakable border colours.
        for count in (1, 2, 3):
            nodes = sorted(idx for idx, spaces in memberships.items() if len(spaces) == count)
            if not nodes:
                continue
            f.add_choroplethmap(
                geojson="/assets/buurten.geojson",
                featureidkey="properties.code",
                locations=df["code"].iloc[nodes],
                customdata=nodes,
                text=[combined_hover[relevant_nodes.index(idx)] for idx in nodes],
                z=[1.0] * len(nodes),
                zmin=0,
                zmax=1,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker=dict(
                    opacity=1,
                    line=dict(width=1.2 + count * 0.75, color=FACET_BORDER[count])
                ),
                selected=dict(marker=dict(opacity=1)),
                unselected=dict(marker=dict(opacity=1)),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False
            )

        for count in (1, 2, 3):
            f.add_scattermap(
                lat=[None],
                lon=[None],
                mode="markers",
                marker=dict(color=FACET_BORDER[count], size=6 + count * 2, opacity=1),
                name=f"Outline · {count} top-k facet{'s' if count > 1 else ''}",
                legendrank=10 + count,
                showlegend=True
            )

    # The selected neighbourhood stays unambiguous in every colouring mode.
    if has_areas and i:
        f.add_choroplethmap(
            geojson="/assets/buurten.geojson",
            featureidkey="properties.code",
            locations=df["code"].iloc[i],
            customdata=i,
            text=df["name"].iloc[i],
            z=[1.0] * len(i),
            zmin=0,
            zmax=1,
            colorscale=[[0, SEL], [1, SEL]],
            showscale=False,
            marker=dict(opacity=0.46, line=dict(width=2.1, color=SEL)),
            selected=dict(marker=dict(opacity=0.46)),
            unselected=dict(marker=dict(opacity=0.46)),
            hovertemplate="%{text}<br>Selected<extra></extra>",
            showlegend=False
        )

    # Separate small dots around a shared centre make every matching modality legible.
    dot_data = {
        space: {"lat": [], "lon": [], "text": [], "customdata": []}
        for space, _ in MODS
    }
    lon_per_px = 1.40625 / (2 ** zoom)
    for idx, spaces in memberships.items():
        offsets = _facet_dot_offsets(len(spaces))
        lat_per_px = lon_per_px * math.cos(math.radians(float(df.at[idx, "lat"])))
        for space, (dx, dy) in zip(spaces, offsets):
            dot_data[space]["lat"].append(float(df.at[idx, "lat"]) + dy * DOT_OFFSET_PX * lat_per_px)
            dot_data[space]["lon"].append(float(df.at[idx, "lon"]) + dx * DOT_OFFSET_PX * lon_per_px)
            dot_data[space]["customdata"].append(idx)
            dot_data[space]["text"].append(
                f"{df.at[idx, 'name']}<br>{SPACE_SHORT[space]} match · "
                f"rank {ranks[space][idx]}<br>Similarity: {scores[space][idx]:.3f}"
            )

    halo_lat = sum((dot_data[space]["lat"] for space, _ in MODS), [])
    halo_lon = sum((dot_data[space]["lon"] for space, _ in MODS), [])
    if halo_lat:
        f.add_scattermap(
            lat=halo_lat,
            lon=halo_lon,
            mode="markers",
            hoverinfo="skip",
            showlegend=False,
            selectedpoints=[],
            unselected=dict(marker=dict(opacity=1)),
            marker=dict(color="white", size=13, opacity=1)
        )

    for rank, (space, color) in enumerate(MODS, start=30):
        dots = dot_data[space]
        if not dots["lat"]:
            continue
        f.add_scattermap(
            lat=dots["lat"],
            lon=dots["lon"],
            mode="markers",
            customdata=dots["customdata"],
            text=dots["text"],
            hovertemplate="%{text}<extra></extra>",
            selectedpoints=[],
            unselected=dict(marker=dict(opacity=1)),
            marker=dict(color=color, size=9, opacity=1),
            name=f"Dot · {SPACE_SHORT[space]}",
            legendrank=rank,
            showlegend=True
        )

    if i:
        d = df.iloc[i]
        f.add_scattermap(
            lat=d.lat,
            lon=d.lon,
            mode="markers",
            customdata=i,
            text=d.name,
            hovertemplate="%{text}<br>Selected<extra></extra>",
            selectedpoints=[],
            unselected=dict(marker=dict(opacity=1)),
            marker=dict(color="white", size=19, opacity=1),
            showlegend=False
        )
        f.add_scattermap(
            lat=d.lat,
            lon=d.lon,
            mode="markers",
            customdata=i,
            text=d.name,
            hovertemplate="%{text}<br>Selected<extra></extra>",
            selectedpoints=[],
            unselected=dict(marker=dict(opacity=1)),
            marker=dict(color=SEL, size=14, opacity=1),
            name="Selected",
            legendrank=40,
            showlegend=True
        )

    filter_ids = df["code"].iloc[filter].tolist()
    style = {
        "version": 8,
        "sources": {
            "p": {
                "type": "raster",
                "tiles": [TILE_TYPES[map_style]["tiles"]],
                "tileSize": 256,
                "attribution": TILE_TYPES[map_style]["attribution"]
            }
        },
        "layers": [{"id": "p", "type": "raster", "source": "p"}]
    }

    if has_areas:
        style["sources"]["b"] = {"type": "geojson", "data": "/assets/buurten.geojson"}
        style["layers"] += [
            {
                "id": "b-case",
                "type": "line",
                "source": "b",
                "paint": {
                    "line-color": "#111820",
                    "line-width": 1.8,
                    "line-opacity": 0.28,
                    "line-blur": 0.3
                },
                "filter": ["in", ["get", "code"], ["literal", filter_ids]]
            },
            {
                "id": "b",
                "type": "line",
                "source": "b",
                "paint": {
                    "line-color": "#f8fafc",
                    "line-width": 0.75,
                    "line-opacity": 0.74
                },
                "filter": ["in", ["get", "code"], ["literal", filter_ids]]
            }
        ]
        if len(filter) < EXTRA_AREA_HIGHLIGHT:
            style["layers"].append({
                "id": "b-filter",
                "type": "line",
                "source": "b",
                "paint": {"line-color": ACCENT, "line-width": 2.2, "line-opacity": 0.82},
                "filter": ["in", ["get", "code"], ["literal", filter_ids]]
            })

    return f.update_layout(
        map=dict(style=style, center=CENTER, zoom=ZOOM, domain=dict(x=[0, 1], y=[0, 1])),
        uirevision="keep",
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode="event",
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,.88)",
            bordercolor="rgba(75,85,99,.18)",
            borderwidth=1,
            font=dict(size=10, color="#364152"),
            itemclick=False,
            itemdoubleclick=False
        )
    )

def spider(i, cols):
    theta = [LBL[c] for c in cols]
    closed = theta + [theta[0]]
    f = go.Figure()
    if len(i) == 0 or len(i) > len(SPIDER_C_OUT):
        fig_title = "No area selected" if len(i) == 0 else "Cannot show this many areas"
        f.add_trace(go.Scatterpolar(r=[0]*(len(cols)+1), theta=closed, fill="toself",
                                    line_color="rgba(0,0,0,0)", fillcolor="rgba(0,0,0,0)"))
    else:
        fig_title = "Comparing each area"
        any_missing = False
        for ci, item in enumerate(i):
            raw = NORM.iloc[item][cols]
            missing = raw.isna()                       # CBS suppresses values for small/private areas
            vals = raw.fillna(NORM_MEDIAN[cols])        # impute to the median so the polygon stays complete
            radius = vals.tolist() + [vals.iloc[0]]
            f.add_trace(go.Scatterpolar(r=radius, theta=closed, fill="toself", mode="lines",
                                        line_color=SPIDER_C_OUT[ci], fillcolor=SPIDER_C_IN[ci], name=df.name[item]))
            if missing.any():  # flag imputed vertices with a hollow marker so missing data stays honest
                any_missing = True
                f.add_trace(go.Scatterpolar(
                    r=[vals[c] for c in cols if missing[c]], theta=[LBL[c] for c in cols if missing[c]],
                    mode="markers", marker=dict(symbol="circle-open", size=9, color=SPIDER_C_OUT[ci],
                                                line=dict(width=2, color=SPIDER_C_OUT[ci])),
                    hovertemplate="%{theta}: not reported (drawn at median)<extra></extra>", showlegend=False))
        if any_missing:
            fig_title = "Comparing each area  (○ = not reported)"

    return f.update_layout(title=fig_title, title_font_size=13, title_x=0.5,
                           polar=dict(radialaxis=dict(visible=True, range=[0, 1]),
                                      angularaxis=dict(tickfont=dict(size=10))),
                           margin=dict(l=55, r=55, t=34, b=24), showlegend=False)

def table(i, cols):
    columns = [{"name": "Indicator", "id": "field"}] + [{"name": fmt(df["name"][idx]), "id": f"area_{idx}"} for idx in i]
    rows = [{"field": DESC[k], **{f"area_{idx}": fmt(df[k][idx]) for idx in i}} for k in cols]
    return rows, columns

app = Dash(__name__, external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"])
app.layout = html.Div(style={"background": "#f5f6f8", "height": "100vh"}, children=[
    dcc.Store(id="current_selection", data=[]),
    dcc.Store(id="map_click"),   # {points, shift} from a map click, filled clientside so we know if shift was held
    dcc.Store(id="shift_capture_ready"),
    dcc.Store(id="current_columns", data=CBS),
    dcc.Store(id="current_filter", data=df.index.to_list()),
    dcc.Store(id="map_view"),   # {zoom} read from the live map, so the modality dots keep a constant on-screen spacing
    dcc.Store(id="query_text_selection_start", data=0),
    dcc.Store(id="query_text_selection_end", data=0),
    dcc.Store(id="intro_seen", storage_type="local"),   # remembers if the popup was shown before (per browser)
    html.Div(id="intro", style={**OVERLAY, "display": "none"}, children=[
        html.Div(style={**CARD, "maxWidth": "470px", "padding": "20px 24px", "fontFamily": FONT, "lineHeight": "1.5"}, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                html.Span("Welcome to the Neighbourhood embedding explorer", style={"fontWeight": "700", "fontSize": "16px"}),
                html.Button("×", id="intro_close", style={"border": "none", "background": "none",
                            "fontSize": "24px", "lineHeight": "1", "cursor": "pointer"})]),
            html.P("This tool lets you explore the neighbourhoods of Amsterdam and see which ones are alike and why."),
            html.P("Every dot is a neighbourhood. The map makes it easy to see where they are. The three plots on the right group "
                   "them by how similar they are in different ways: by what they look like from above using CLIP, by how they get described in "
                   "words by using neighbourhood descriptions, and by their CBS statistics."),
            html.P("Click a neighbourhood, on the map or in any plot, to select it. The data of the selected neighbourhood shows up in the table "
                   "and the spider chart, and the ones most like it in the different facets light up on the map and in the plots."),
            html.P("Use Map colour to switch between top-k overlap and a single-facet similarity surface. In overlap mode only top-k-relevant "
                   "areas are filled: darker means stronger combined similarity. Yellow, orange, and purple outlines mark matches in one, two, "
                   "or three top-k facets. The solid blue, green, and purple centre dots identify CLIP, text, and CBS."),
            html.P("If you want to compare a few neighbourhoods, hold shift and click on them in either the map or the plots. The table and spider chart then show them next to each other."),
            html.P("You can also filter with a query and choose which indicators to show, both on the left."),
            html.P("Click the i in the top right whenever you want to read this again.",
                   style={"color": "#888", "fontSize": "13px"})])]),
    html.Div(style={"padding": "10px 16px", "fontFamily": FONT, "display": "flex",
                    "alignItems": "center", "justifyContent": "space-between"}, children=[
        html.Span("Neighbourhood embedding explorer", style={"fontWeight": "700", "fontSize": "16px"}),
        html.Button("i", id="info_button", style={"width": "26px", "height": "26px", "borderRadius": "50%",
                    "border": "1px solid #ccc", "background": "#fff", "cursor": "pointer", "fontStyle": "italic",
                    "fontWeight": "700"})]),
    html.Div(style={**PAGE, "height": "calc(100vh - 44px)"}, children=[
        html.Div(style={"width": "25%", **CARD, "display": "flex", "flexDirection": "column", "gap": "8px"}, children=[
            html.Div("Filter", style=LABEL),
            html.Div("​", id="query_message", style={"color": "black", "fontSize": "14px", "fontFamily": FONT}),
            html.Div(style={"display": "flex", "flexDirection": "row", "gap": "6px"}, children=[
                dcc.Textarea("", id="query_text", rows=1, placeholder="Enter queries here: ",
                             style={"resize": "none", "flex": "1", "fontFamily": FONT,
                                    "border": "1px solid #d0d4da", "borderRadius": "8px", "padding": "6px 8px"}),
                dcc.Button("Filter", id="filter_button", style=BTN),
                dcc.Button("Clear", id="clear_filter_button", style=CLR_BTN)
            ]),
            dcc.Dropdown(id="query_columns", options=QOPTS, value=None,
                         placeholder="Use the dropdown to insert attributes in the query"),
            html.Hr(style={"width": "100%", "border": "none", "borderTop": "1px solid #e3e6ea", "margin": "4px 0"}),
            html.Div("Indicators", style=LABEL),
            dcc.Dropdown(OPTS, CBS, id="column_select", closeOnSelect=False, multi=True, clearable=True,
                         placeholder="At least one column must be selected"),
            html.Div("Similar shown per facet (K)", style=LABEL),
            dcc.Slider(1, K_MAX, 1, value=K, id="k_slider", marks={1: "1", K_MAX: str(K_MAX)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                html.Div("Selected areas", style=LABEL),
                dcc.Button("Clear", id="clear_button", style={"border": "1px solid #d0d4da", "background": "#fff",
                           "color": "#555", "borderRadius": "8px", "padding": "0 10px", "cursor": "pointer",
                           "fontSize": "12px", "fontFamily": FONT})]),
            html.Div(style={"flex": "1", "minHeight": "0", "overflowY": "auto"}, children=[  # table scrolls, spider stays put
                dash_table.DataTable(id="table",
                    columns=[{"name": "Indicator", "id": "field"}],
                    style_as_list_view=True,
                    style_header={"background": "#f0f2f5", "fontWeight": "600", "border": "none",
                                  "fontFamily": FONT, "padding": "6px 10px"},
                    style_cell={"padding": "6px 10px", "border": "none", "fontFamily": FONT, "fontSize": "13px",
                                "textAlign": "left", "whiteSpace": "normal", "fontVariantNumeric": "tabular-nums"},
                    style_cell_conditional=[{"if": {"column_id": "field"}, "minWidth": "150px", "width": "45%"}],
                    style_data_conditional=[{"if": {"row_index": "odd"}, "background": "#fafbfc"}],
                    style_table={"overflowX": "auto"})]),
            dcc.Graph(id="spider", style={"height": "340px", "flexShrink": 0})]),
        html.Div(style={"width": "45%", **CARD, "display": "flex", "flexDirection": "column", "position": "relative"}, children=[
            html.Div("​", id="search_result"),
            dcc.Input(id="search", type="text", debounce=True, n_submit=0,
                    placeholder="Search address or buurt (Enter to submit)…",
                    style={"padding": "6px 10px", "fontSize": "13px",
                           "border": "1px solid #d5dae1", "borderRadius": 7,
                           "background": "#fff", "color": "#26313d"}),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px", "padding": "6px 2px 4px",
                            "fontFamily": FONT, "fontSize": "12px", "color": "#68727d"}, children=[
                html.Span("Map colour", style={"fontWeight": "700", "whiteSpace": "nowrap"}),
                dcc.RadioItems(
                    options=[
                        {"label": "Overlap", "value": "overlap"},
                        {"label": "CLIP", "value": "clip"},
                        {"label": "Text", "value": "text"},
                        {"label": "CBS", "value": "cbs"}
                    ],
                    value="overlap",
                    id="map_color_mode",
                    inline=True,
                    labelStyle={"marginRight": "10px", "cursor": "pointer", "whiteSpace": "nowrap"},
                    inputStyle={"marginRight": "4px", "accentColor": ACCENT}
                )
            ]),
            dcc.Graph(id="map", style={"height": "100%"},
                      config={"displayModeBar": False, "scrollZoom": True}),
            html.Div(dcc.RadioItems({"satellite": "Satellite map", "streetview": "Streetview map"}, 'satellite',
                                    id="map_style", inline=True,
                                    labelStyle={"marginRight": "12px", "cursor": "pointer"},
                                    inputStyle={"marginRight": "5px"}),
                     style={"position": "absolute", "bottom": "12px", "left": "12px", "zIndex": 1,
                            "background": "rgba(255,255,255,.9)", "padding": "6px 10px", "borderRadius": "8px",
                            "boxShadow": "0 1px 3px rgba(0,0,0,.2)", "fontFamily": FONT, "fontSize": "13px"})
        ]),
        html.Div(style={"width": "30%", **CARD, "display": "flex", "flexDirection": "column", "gap": "4px", "position":"relative"}, children=[
            dcc.Graph(id="clip", style={"flex": 1}, config={"scrollZoom": True}),
            dcc.Graph(id="text", style={"flex": 1}, config={"scrollZoom": True}),
            dcc.Graph(id="cbs", style={"flex": 1}, config={"scrollZoom": True})]),
    ]),
])


# TODO: some weird stuff happens when switching selection between plots
# - both picking a single point and lasso/box overwrite selection -> therefore are fine
# - but shift+clicking to add a point in a plot remembers last selection in that specific plot
# - not global last selection so that can lead to weird behaviour
# keep track of most recently selected item
@app.callback(Output("current_selection", "data"),
              Input("clip", "selectedData"),
              Input("text", "selectedData"),
              Input("cbs", "selectedData"),
              Input("map_click", "data"),
              State("current_selection", "data"),
              prevent_initial_call=True)
def update_current_selection(clip, text, cbs, map_click, current):
    # the map selects on a plain click anywhere inside a buurt; shift+click adds/removes for comparing.
    # map_click carries the clicked area(s) + whether shift was held, so we accumulate the selection in
    # Python rather than relying on Plotly keeping its own selection across figure redraws.
    if ctx.triggered_id == "map_click":
        clicked = list(dict.fromkeys(
            int(p["customdata"])
            for p in (map_click or {}).get("points", [])
            if p.get("customdata") is not None
        ))
        if not clicked:
            raise exceptions.PreventUpdate
        if not map_click.get("shift"):
            return list(dict.fromkeys(clicked))
        sel = list(current)
        for idx in clicked:  # shift+click toggles each clicked area in/out of the current selection
            sel = [c for c in sel if c != idx] if idx in sel else sel + [idx]
        return sel

    cd = {"clip": clip, "text": text, "cbs": cbs}.get(ctx.triggered_id)
    if cd is None:
        raise exceptions.PreventUpdate
    pts = cd["points"]

    # for some reason box and lasso select trigger the callback twice
    # the second time with an empty box / lasso -> so no update to prevent overwrite
    if "range" not in cd.keys() and "lassoPoints" not in cd.keys() and len(pts) == 0:
        raise exceptions.PreventUpdate

    i = [int(pt["customdata"]) for pt in pts if "customdata" in pt]
    return list(dict.fromkeys(i))  # dedupe, just in case a point shows up twice

# Install modifier-key capture immediately, before the first map interaction.
app.clientside_callback(
    """
    function(selection) {
        if (!window.__shiftCapInit) {
            document.addEventListener("mousedown", e => { window.__lastShift = e.shiftKey; }, true);
            window.__shiftCapInit = true;
        }
        return true;
    }
    """,
    Output("shift_capture_ready", "data"),
    Input("current_selection", "data"))

# Report ordinary map clicks to Python together with the captured shift state. Using clickData avoids
# Plotly's own selection/dimming state; selection and deselection are handled solely by our toggle logic.
app.clientside_callback(
    """
    function(clickData) {
        if (!window.__shiftCapInit) {
            document.addEventListener("mousedown", e => { window.__lastShift = e.shiftKey; }, true);
            window.__shiftCapInit = true;
        }
        if (!clickData || !clickData.points || !clickData.points.length) return window.dash_clientside.no_update;
        return {points: clickData.points, shift: !!window.__lastShift};
    }
    """,
    Output("map_click", "data"),
    Input("map", "clickData"),
    prevent_initial_call=True)

# clear button resets the selection back to nothing
@app.callback(Output("current_selection", "data", allow_duplicate=True),
              Input("clear_button", "n_clicks"),
              prevent_initial_call=True)
def clear_selection(_):
    return []

# keep track of selected columns
@app.callback(Output("current_columns", "data"),
              Input("column_select", "value"),
              prevent_initial_call=True)
def update_current_columns(value):
    if not value:
        raise exceptions.PreventUpdate
    return value

# use search to select area
@app.callback(Output("current_selection", "data", allow_duplicate=True),
              Output("search_result", "children"),
              Input("search", "value"),
              prevent_initial_call=True)
def search_location(adress):
    return geocode(adress)

# when selection or filter changes update all figures
# run sequentially: update when all figures are ready
@app.callback(Output("clip", "figure"),
              Output("text", "figure"),
              Output("cbs", "figure"),
              Output("map", "figure"),
              Output("spider", "figure"),
              Output("table", "data"),
              Output("table", "columns"),
              Input("current_selection", "data"),
              Input("current_filter", "data"),
              Input("k_slider", "value"),
              State("current_columns", "data"),
              State("map_style", "value"),
              State("map_view", "data"),
              State("map_color_mode", "value"))
def update_figure_selections(i, filter, k, cols, map_style, view, color_mode):
    return scatter("clip", i, filter, k), scatter("text", i, filter, k), scatter("cbs", i, filter, k), \
            mapfig(i, filter, map_style, k, view, color_mode), spider(i, cols), *table(i, cols)

# when selected columns change, update spiderplot and table
@app.callback(Output("spider", "figure", allow_duplicate=True),
              Output("table", "data", allow_duplicate=True),
              Output("table", "columns", allow_duplicate=True),
              Input("current_columns", "data"),
              State("current_selection", "data"),
              prevent_initial_call=True)
def update_figure_columns(cols, i):
    return spider(i, cols), *table(i, cols)

# when map style or map colouring changes update map
@app.callback(Output("map", "figure", allow_duplicate=True),
              Input("map_style", "value"),
              Input("map_color_mode", "value"),
              State("current_selection", "data"),
              State("current_filter", "data"),
              State("k_slider", "value"),
              State("map_view", "data"),
              prevent_initial_call=True)
def update_map_appearance(map_style, color_mode, i, filter, k, view):
    return mapfig(i, filter, map_style, k, view, color_mode)

# redraw the map when the zoom level changes so the offset modality dots keep a constant on-screen spacing.
@app.callback(Output("map", "figure", allow_duplicate=True),
              Input("map_view", "data"),
              State("current_selection", "data"),
              State("current_filter", "data"),
              State("k_slider", "value"),
              State("map_style", "value"),
              State("map_color_mode", "value"),
              prevent_initial_call=True)
def update_map_view(view, i, filter, k, map_style, color_mode):
    return mapfig(i, filter, map_style, k, view, color_mode)

# read the live maplibre zoom on each pan/zoom; bucket it so we only redraw when the zoom really changes.
app.clientside_callback(
    """
    function(relayout) {
        var ds = window.dash_clientside;
        var gd = document.querySelector('#map .js-plotly-plot');
        var sp = gd && gd._fullLayout && gd._fullLayout.map && gd._fullLayout.map._subplot;
        var m = sp && (sp.map || sp._map);
        if (!m || !m.getZoom) return ds.no_update;
        var zb = Math.round(m.getZoom() * 4) / 4;
        if (window.__lastZoomBucket === zb) return ds.no_update;
        window.__lastZoomBucket = zb;
        return {zoom: zb};
    }
    """,
    Output("map_view", "data"),
    Input("map", "relayoutData"),
    prevent_initial_call=True)

# show the intro popup on first visit + whenever the "i" button is clicked, hide it on the cross
@app.callback(Output("intro", "style"),
              Input("intro_seen", "data"),
              Input("info_button", "n_clicks"),
              Input("intro_close", "n_clicks"))
def toggle_intro(seen, info, close):
    if ctx.triggered_id == "intro_close":
        return {**OVERLAY, "display": "none"}
    if ctx.triggered_id == "info_button":
        return {**OVERLAY, "display": "flex"}
    return {**OVERLAY, "display": "none" if seen else "flex"}  # initial load: only show if never seen

# remember that the popup has been shown so it stays closed on later visits
@app.callback(Output("intro_seen", "data"),
              Input("intro_close", "n_clicks"),
              prevent_initial_call=True)
def mark_intro_seen(_):
    return True

# track most recent selection positions in textarea to enable inserting in the middle
# - does not keep track of changing selection with arrow keys until n_blur is triggered
app.clientside_callback(
    """
    function(n_clicks, n_blurs) {
        const textarea = document.querySelector("textarea#query_text")
        if (!textarea) return [null, null];
        return [textarea.selectionStart, textarea.selectionEnd];
    }
    """,
    Output("query_text_selection_start", "data"),
    Output("query_text_selection_end", "data"),
    Input("query_text", "n_clicks"),
    Input("query_text", "n_blur"),
    prevent_initial_call=True)

# when an item in the query dropdown is clicked, replace most recent selection with
# correct column key and update selection positions
@app.callback(
    Output("query_text", "value"),
    Output("query_text_selection_start", "data", allow_duplicate=True),
    Output("query_text_selection_end", "data", allow_duplicate=True),
    Output("query_columns", "value"),
    Input("query_columns", "value"),
    State("query_text", "value"),
    State("query_text_selection_start", "data"),
    State("query_text_selection_end", "data"),
    prevent_initial_call=True)
def replace_query_section(col_name, text, start, end):
    col_name = col_name or ""
    s = text[:start] + col_name + text[end:]
    return s, start+len(col_name), start+len(col_name), None

# when button is clicked, use current textarea to query dataframe and return valid indices
@app.callback(
    Output("current_filter", "data"),
    Output("current_selection", "data", allow_duplicate=True),
    Output("query_message", "children"),
    Output("query_message", "style"),
    Input("filter_button", "n_clicks"),
    State("query_text", "value"),
    State("current_selection", "data"),
    prevent_initial_call=True)
def filter_dataframe(n_clicks, query_text, i):
    try:
        filter = df.query(query_text).index.to_list() if query_text else df.index.to_list()
        if len(filter) == 0:
            return no_update, no_update, "This query has no valid items", {"color": "red"}
        if len(filter) == len(df):
            return filter, i, "​", {"color": "black"}
        i = [idx for idx in i if idx in filter]
        return filter, i, f"Found {len(filter)} items", {"color": "black"}
        
    except:
        return no_update, no_update, "This is not a valid query", {"color": "red"}
    
@app.callback(Output("query_text", "value", allow_duplicate=True),
              Output("current_filter", "data", allow_duplicate=True),
              Input("clear_filter_button", "n_clicks"),
              prevent_initial_call=True)
def clear_filter(n_clicks):
    return "", df.index.to_list()

if __name__ == "__main__":
    app.run(debug=True)
