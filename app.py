import json, math, os
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.io as pio
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, ctx, exceptions
import requests
from shapely.geometry import shape, Point

pio.templates.default = "plotly_white"

df = pd.read_parquet("assets/data.parquet")
EMB = {"clip": np.load("assets/emb_clip.npy"), "text": np.load("assets/emb_text.npy"), "cbs": np.load("assets/emb_cbs.npy")}  # unit-normalized; avg_sim computes cosine on the fly
with open("assets/buurten.geojson") as f:
    AREAS = json.load(f)
POLYGONS = []
for feature in AREAS["features"]:
    name = feature["properties"]["code"]
    idx = df[df["code"] == name].index[0]
    poly = shape(feature["geometry"])
    POLYGONS.append((name, idx, poly))
# CBS column catalogue lives in cbs_columns.json (Dutch field, our name, description,
# spider label, unit). CBS = every measure present in the data; CBS_DEFAULT = the curated
# starter set shown in the spider/table before the user picks more from the dropdown.
_CBS_META = json.load(open(os.path.join(os.path.dirname(__file__), "assets", "cbs_columns.json")))
_CBS_META = [e for e in _CBS_META if e["name"] in df.columns]  # tolerate data with fewer columns
CBS = [e["name"] for e in _CBS_META]
CBS_DEFAULT = [e["name"] for e in _CBS_META if e.get("default")] or CBS[:10]
LBL = {e["name"]: e["spider"] for e in _CBS_META}        # short spider axis labels
DESC = {e["name"]: e["description"] for e in _CBS_META}   # readable column descriptions
UNIT = {e["name"]: e["unit"] for e in _CBS_META}          # unit per column
SPACE = {"clip": "Visual similarity (aerial imagery)", "text": "Description similarity (text)",
         "cbs": "Statistical similarity (CBS data)"}  # readable embedding-plot titles
OPTS = [{"label": f"{DESC[c]} ({UNIT[c]})", "value": c} for c in CBS]   # readable column chips / dropdown
QOPTS = [{"label": f"{DESC[c]}  ·  {c}", "value": c} for c in CBS]      # query dropdown also shows the queryable column name
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
SPIDER_C_OUT = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
                '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
SPIDER_C_IN = ['rgba(122,131,250,0.25)', 'rgba(241,110,88,0.25)', 'rgba(38,211,165,0.25)',
               'rgba(183,122,250,0.25)', 'rgba(255,175,114,0.25)', 'rgba(59,217,244,0.25)',
               'rgba(255,124,162,0.25)', 'rgba(192,235,147,0.25)', 'rgba(255,166,255,0.25)', 'rgba(254,210,107,0.25)']
MODS = (("clip", CLIP_C), ("text", TEXT_C), ("cbs", CBS_C))
K_MAX = 25            # upper bound of the "similar shown" slider
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
# clamp panning/zoom-out to just beyond the buurt polygons so the map stays on Amsterdam
_pb = [p.bounds for _, _, p in POLYGONS]   # each: (lon_min, lat_min, lon_max, lat_max)
_PAD = 0.08
BOUNDS = dict(west=min(b[0] for b in _pb) - _PAD, south=min(b[1] for b in _pb) - _PAD,
              east=max(b[2] for b in _pb) + _PAD, north=max(b[3] for b in _pb) + _PAD)

def avg_sim(F, space):  # mean cosine similarity of every buurt to the focus set F (>=1 row), in `space`
    E = EMB[space]                       # rows are unit-normalized, so E @ mean(E[F]) == old SIM[F].mean(0)
    return E @ E[list(F)].mean(axis=0)

def topk_by(scores, filter, exclude, k):  # k filtered buurten with the highest score, excluding the focus set
    ex = set(exclude)
    return sorted((j for j in filter if j not in ex), key=lambda j: scores[j], reverse=True)[:k]

def _relative_percentiles(scores, rows):
    values = pd.Series([scores[idx] for idx in rows], index=rows)
    return values.rank(method="average", pct=True).to_dict()

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

def scatter(space, i, filter, k=K, hover=True):
    i = list(i or [])
    filter = list(filter or [])
    hot = topk_by(avg_sim(i, space), filter, i, k) if len(i) else []

    f = go.Figure(go.Scattergl(
        x=df[f"{space}_x"].iloc[filter],
        y=df[f"{space}_y"].iloc[filter],
        mode="markers",
        marker=dict(
            color="#b3bdc7",
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
        hovermode=("closest" if hover else False),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

def mapfig(i, filter, map_style, k=K, view=None, color_mode=None, topk_only=True):
    """Build a similarity surface coloured by the average over the selected embeddings."""
    i = list(i or [])
    filter = list(filter or [])
    k = int(k or K)
    has_areas = os.path.exists("assets/buurten.geojson")

    if i:
        scores = {space: avg_sim(i, space) for space, _ in MODS}
    else:
        scores = {space: None for space, _ in MODS}

    percentiles = {
        space: _relative_percentiles(scores[space], filter) if i else {}
        for space, _ in MODS
    }

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

    # Colour by the average relative percentile across whichever embeddings are checked.
    selected_spaces = [space for space, _ in MODS if space in (color_mode or [])]
    if has_areas and i and selected_spaces:
        # Combined score = average relative percentile across the checked facets (percentile is easier to
        # read across modalities whose raw cosine ranges differ). Computed for every filtered neighbourhood.
        avg_pct = {
            idx: float(np.mean([percentiles[space][idx] for space in selected_spaces]))
            for idx in filter
        }
        # Top-K toggle on: keep exactly the K highest-scoring areas, ranked by the combined score;
        # off: colour every neighbourhood.
        if topk_only:
            ranked = topk_by(avg_pct, filter, i, k)
            rank = {idx: r + 1 for r, idx in enumerate(ranked)}
            surface_idx = ranked
        else:
            surface_idx = filter
        label = " + ".join(SPACE_SHORT[space] for space in selected_spaces)
        hover_text = [
            f"{df.at[idx, 'name']}<br>" +
            (f"{label} similarity: {avg_pct[idx]:.0%}<br>Rank: #{rank[idx]} of {k}" if topk_only
             else f"{label} similarity: {avg_pct[idx]:.0%}")
            for idx in surface_idx
        ]
        f.add_choroplethmap(
            geojson="/assets/buurten.geojson",
            featureidkey="properties.code",
            locations=df["code"].iloc[surface_idx],
            customdata=surface_idx,
            text=hover_text,
            z=[avg_pct[idx] for idx in surface_idx],
            zmin=0,
            zmax=1,
            colorscale=SURFACE_SCALE[selected_spaces[0]] if len(selected_spaces) == 1 else COMBINED_SCALE,
            showscale=True,
            colorbar=dict(
                title=dict(text=f"{label} similarity: low → high", side="top"),
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
                    "line-color": "#a855f7" if map_style == "streetview" else "#facc15",
                    "line-width": 1.4,
                    "line-opacity": 0.9
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
        map=dict(style=style, center=CENTER, zoom=ZOOM, bounds=BOUNDS, domain=dict(x=[0, 1], y=[0, 1])),
        uirevision="keep",
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode="event",
        hovermode="closest",
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

def spider(i, cols, hover=True):
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
                           # keep the gridline rings but hide the 0.2/0.4/… numbers (radius = normalised magnitude)
                           polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, ticks=""),
                                      angularaxis=dict(tickfont=dict(size=10))),
                           margin=dict(l=55, r=55, t=34, b=24), showlegend=False,
                           hovermode=("closest" if hover else False))

def table(i, cols):
    columns = [{"name": "Indicator", "id": "field"}] + [{"name": fmt(df["name"][idx]), "id": f"area_{idx}"} for idx in i]
    rows = [{"field": f"{DESC[k]} ({UNIT[k]})", **{f"area_{idx}": fmt(df[k][idx]) for idx in i}} for k in cols]
    return rows, columns

app = Dash(__name__, external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"])
app.layout = html.Div(style={"background": "#f5f6f8", "height": "100vh"}, children=[
    dcc.Store(id="current_selection", data=[]),
    dcc.Store(id="map_click"),   # {points, shift} from a map click, filled clientside so we know if shift was held
    dcc.Store(id="shift_capture_ready"),
    dcc.Store(id="current_columns", data=CBS_DEFAULT),
    dcc.Store(id="current_filter", data=df.index.to_list()),
    dcc.Store(id="map_view"),   # {zoom} read from the live map, so the modality dots keep a constant on-screen spacing
    dcc.Store(id="query_text_selection_start", data=0),
    dcc.Store(id="query_text_selection_end", data=0),
    dcc.Store(id="intro_seen", storage_type="local"),   # remembers if the popup was shown before (per browser)
    html.Div(id="intro", style={**OVERLAY, "display": "none"}, children=[
        html.Div(style={**CARD, "width": "min(92vw, 640px)", "maxHeight": "90vh", "overflowY": "auto",
                        "padding": "20px 24px", "fontFamily": FONT, "lineHeight": "1.5"}, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                html.Span("Welcome to the Neighbourhood embedding explorer", style={"fontWeight": "700", "fontSize": "16px"}),
                html.Button("×", id="intro_close", style={"border": "none", "background": "none",
                            "fontSize": "24px", "lineHeight": "1", "cursor": "pointer"})]),
            html.P("This tool helps you compare the neighbourhoods of Amsterdam. You pick one, and it shows which other "
                   "neighbourhoods are alike, and why they are alike."),
            html.P("The map in the middle shows every neighbourhood as its real area. The three plots on the right show the same "
                   "neighbourhoods as dots, each plot grouping them by one kind of similarity: how they look from above (CLIP), how "
                   "they are described in words (the neighbourhood descriptions), and their CBS statistics."),
            html.P("Click a neighbourhood, on the map or in any plot, to select it. Its numbers appear in the table and the spider "
                   "chart, and the ones most like it light up everywhere. You can also type an address or a neighbourhood name in the "
                   "search box above the map to jump straight to it."),
            html.P("When something is selected, the map shades the other neighbourhoods by how similar they are: darker means a closer "
                   "match. The Map colour boxes above the map choose which kind of similarity is used (CLIP, text, CBS, or a mix), the "
                   "Similar shown slider on the left sets how many neighbours are highlighted, and turning off Top-K only shades every "
                   "neighbourhood instead of just the closest ones."),
            html.P("To compare a few neighbourhoods, hold shift and click them; " 
                   "the table and spider chart then show them side by side, and Clear empties the selection. "
                   "On the left you can choose which indicators to show in the table and spider chart. "
                   "Additionally, the query panel can be used to filter which neighborhoods to show. "
                   "To use it, a valid filter must be entered (e.g. 'population >= 5000 | density >20000'). "
                   "The dropdown menu below can be used to find and insert attributes into the query text. "),
            html.P("The spider chart is relative: each axis runs from the lowest value in the city at the centre to the highest at the "
                   "edge, so you read the shape rather than exact numbers (the table has those). A hollow circle marks a value CBS did "
                   "not report, drawn in the middle so the shape stays whole."),
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
            html.Div(" ", id="query_message", style={"color": "black", "fontSize": "14px", "fontFamily": FONT}),
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
            html.Div("Similar shown per facet (K)", style=LABEL),
            dcc.Slider(1, K_MAX, 1, value=K, id="k_slider", marks={1: "1", K_MAX: str(K_MAX)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Hr(style={"width": "100%", "border": "none", "borderTop": "1px solid #e3e6ea", "margin": "4px 0"}),
            html.Div("Indicators", style=LABEL),
            dcc.Dropdown(OPTS, CBS_DEFAULT, id="column_select", closeOnSelect=False, multi=True, clearable=True,
                         placeholder="At least one column must be selected"),
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                html.Div("Selected areas", style=LABEL),
                html.Div(style={"display": "flex", "gap": "6px"}, children=[
                    dcc.Button("Clear", id="clear_button", style={"border": "1px solid #d0d4da", "background": "#fff",
                               "color": "#555", "borderRadius": "8px", "padding": "0 10px", "cursor": "pointer",
                               "fontSize": "12px", "fontFamily": FONT})])]),
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
                dcc.Checklist(
                    options=[
                        {"label": "CLIP", "value": "clip"},
                        {"label": "Text", "value": "text"},
                        {"label": "CBS", "value": "cbs"}
                    ],
                    value=["clip", "text", "cbs"],
                    id="map_color_mode",
                    inline=True,
                    labelStyle={"marginRight": "10px", "cursor": "pointer", "whiteSpace": "nowrap"},
                    inputStyle={"marginRight": "4px", "accentColor": ACCENT}
                ),
                dcc.Checklist(options=[{"label": "Top-K only", "value": "on"}], value=["on"], id="topk_toggle",
                              inline=True, style={"marginLeft": "auto", "whiteSpace": "nowrap"},
                              labelStyle={"cursor": "pointer"}, inputStyle={"marginRight": "4px", "accentColor": ACCENT})
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
    sel, msg = geocode(adress)
    return (sel if sel else no_update), msg   # keep the current selection on an empty/failed search

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
              Input("topk_toggle", "value"),
              State("current_columns", "data"),
              State("map_style", "value"),
              State("map_view", "data"),
              State("map_color_mode", "value"))
def update_figure_selections(i, filter, k, topk_toggle, cols, map_style, view, color_mode):
    topk = "on" in (topk_toggle or [])
    return scatter("clip", i, filter, k), scatter("text", i, filter, k), scatter("cbs", i, filter, k), \
            mapfig(i, filter, map_style, k, view, color_mode, topk), spider(i, cols), *table(i, cols)

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
              State("topk_toggle", "value"),
              prevent_initial_call=True)
def update_map_appearance(map_style, color_mode, i, filter, k, view, topk_toggle):
    return mapfig(i, filter, map_style, k, view, color_mode, "on" in (topk_toggle or []))

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
              Output("query_message", "children", allow_duplicate=True),
              Output("query_message", "style", allow_duplicate=True),
              Input("clear_filter_button", "n_clicks"),
              prevent_initial_call=True)
def clear_filter(n_clicks):
    return "", df.index.to_list(), "", {"color": "black"}

if __name__ == "__main__":
    app.run(debug=True)
