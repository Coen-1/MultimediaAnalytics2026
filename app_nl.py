"""NL multimedia-analytics platform.

A choropleth of ~1800 Dutch neighbourhoods over PDOK aerial tiles, linked to three
embedding views (CLIP aerial / text descriptions / CBS statistics).

Two ways to colour the country:
  * Click a STATISTIC  -> choropleth + scatters encode that variable
                          (housing price, age, household size, development ...).
  * Pick a NEIGHBOURHOOD and a SIMILARITY space -> the map becomes a similarity
    heatmap to that neighbourhood, with the top-k most similar highlighted.

Hovering any area updates the spider plot live (a lightweight callback that never
touches the map, so it stays responsive). Borders are toggle-able, and a random
button drops you somewhere new for exploration.

Geometry is served once as a static GeoJSON asset and referenced by URL, so the
interaction callbacks only ever ship per-neighbourhood value arrays.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

df = pd.read_parquet("data_nl.parquet").reset_index(drop=True)
SIM = {"clip": np.load("sim_nl_clip.npy"),
       "text": np.load("sim_nl_text.npy"),
       "cbs":  np.load("sim_nl_cbs.npy")}
N = len(df)

CBS_COLS = ["population", "income", "avg_age", "household_size",
            "home_value", "green_pct", "density"]
NORM = (df[CBS_COLS] - df[CBS_COLS].min()) / (df[CBS_COLS].max() - df[CBS_COLS].min() + 1e-9)

GEOJSON_URL = "/assets/nl_cells.geojson"
LOCATIONS = df.index.tolist()

PRETTY = {"population": "Population", "income": "Income (k€)", "avg_age": "Avg age",
          "household_size": "Household size", "home_value": "Housing price (k€)",
          "green_pct": "Green %", "density": "Density (/km²)",
          "planned_units": "Planned dev (units)"}
STAT_KEYS = ["home_value", "income", "population", "avg_age",
             "household_size", "green_pct", "planned_units"]
SPACES = [("clip", "CLIP · aerial"), ("text", "Descriptions"), ("cbs", "CBS stats")]

PDOK_TILE = ("https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/"
             "Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg")
MAP_STYLE = {"version": 8,
             "sources": {"p": {"type": "raster", "tiles": [PDOK_TILE], "tileSize": 256}},
             "layers": [{"id": "p", "type": "raster", "source": "p"}]}
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759",
           "#b07aa1", "#76b7b2", "#edc949", "#9c755f"]
SELECTED = "#e6194b"
NL_CENTER = dict(lat=52.15, lon=5.45)
NL_ZOOM = 6.6


# -------------------- map --------------------

def topk(i, space, k):
    return np.argsort(SIM[space][i])[::-1][1:k + 1]


def border_layer(on):
    if not on:
        return []
    return [{"sourcetype": "geojson", "source": GEOJSON_URL, "type": "line",
             "color": "rgba(255,255,255,0.55)", "line": {"width": 0.5}}]


def map_figure(state, borders_on, opacity, k):
    color, i = state["color"], state["i"]
    if color["kind"] == "stat":
        key = color["key"]
        z = df[key].values
        colorscale, cbar = "Plasma", PRETTY[key]
        zmin, zmax = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    else:  # similarity
        space = color["space"]
        if i is None:
            z = df[f"{space}_cluster"].values
            colorscale, cbar = "Rainbow", "cluster"
            zmin, zmax = 0, df[f"{space}_cluster"].max()
        else:
            z = SIM[space][i]
            colorscale, cbar = "YlOrRd", "cosine sim"
            zmin, zmax = float(np.percentile(z, 5)), 1.0

    fig = go.Figure(go.Choroplethmap(
        geojson=GEOJSON_URL, locations=LOCATIONS, z=z, featureidkey="id",
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        marker=dict(opacity=opacity, line=dict(width=0)),
        colorbar=dict(title=dict(text=cbar, side="right"), thickness=12, len=0.55,
                      x=0.99, xanchor="right", bgcolor="rgba(255,255,255,0.6)"),
        text=df["name"], customdata=df.index.values,
        hovertemplate="%{text}<extra></extra>",
    ))

    # highlight selected + top-k as a marker overlay
    if i is not None:
        idx = [i]
        sizes = [16]
        colors = [SELECTED]
        if color["kind"] == "sim":
            hot = list(topk(i, color["space"], k))
            idx += hot
            sizes += [8] * len(hot)
            colors += ["#1f78ff"] * len(hot)
        fig.add_trace(go.Scattermap(
            lat=df.lat.iloc[idx], lon=df.lon.iloc[idx], mode="markers",
            marker=dict(size=sizes, color=colors),
            text=df.name.iloc[idx], hoverinfo="text", showlegend=False,
        ))

    center, zoom, uirev = NL_CENTER, NL_ZOOM, "nl"
    if i is not None and state.get("recenter"):
        center, zoom, uirev = dict(lat=df.lat.iloc[i], lon=df.lon.iloc[i]), 11, f"sel-{i}"
    fig.update_layout(
        map=dict(style=MAP_STYLE, center=center, zoom=zoom, layers=border_layer(borders_on)),
        margin=dict(l=0, r=0, t=0, b=0), uirevision=uirev,
    )
    return fig


# -------------------- scatters --------------------

def scatter(space, state, k):
    color, i = state["color"], state["i"]
    if color["kind"] == "stat":
        key = color["key"]
        marker = dict(color=df[key].values, colorscale="Plasma", size=5,
                      opacity=0.8, showscale=False,
                      cmin=float(np.percentile(df[key], 2)),
                      cmax=float(np.percentile(df[key], 98)))
    else:
        labels = df[f"{space}_cluster"].values
        marker = dict(color=[PALETTE[int(l) % len(PALETTE)] for l in labels],
                      size=5, opacity=0.7)
    fig = go.Figure(go.Scattergl(
        x=df[f"{space}_x"], y=df[f"{space}_y"], mode="markers", marker=marker,
        text=df["name"] + " · " + df["archetype"], hoverinfo="text",
        customdata=df.index.values,
    ))
    # emphasise selection + top-k
    if i is not None:
        hot = list(topk(i, space, k)) if color["kind"] == "sim" else []
        if hot:
            fig.add_trace(go.Scattergl(
                x=df[f"{space}_x"].iloc[hot], y=df[f"{space}_y"].iloc[hot],
                mode="markers", marker=dict(size=9, color="#1f78ff",
                                            line=dict(width=1, color="white")),
                hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scattergl(
            x=[df[f"{space}_x"].iloc[i]], y=[df[f"{space}_y"].iloc[i]],
            mode="markers", marker=dict(size=15, color=SELECTED,
                                        line=dict(width=2, color="white")),
            hoverinfo="skip", showlegend=False))
    label = dict(clip="CLIP · aerial", text="Descriptions", cbs="CBS stats")[space]
    fig.update_layout(
        title=dict(text=label, x=0.03, y=0.96, font=dict(size=11, color="#333")),
        margin=dict(l=3, r=3, t=20, b=3), xaxis=dict(visible=False),
        yaxis=dict(visible=False), plot_bgcolor="#fafafa", paper_bgcolor="#fff",
        showlegend=False, dragmode="zoom", uirevision=f"sc-{space}",
    )
    return fig


def spider(hover_i, sel_i):
    traces = []
    if sel_i is not None:
        v = NORM.iloc[sel_i].tolist()
        traces.append(go.Scatterpolar(r=v + [v[0]], theta=CBS_COLS + [CBS_COLS[0]],
                                      fill="toself", name="selected",
                                      line=dict(color=SELECTED),
                                      fillcolor="rgba(230,25,75,0.22)"))
    if hover_i is not None and hover_i != sel_i:
        v = NORM.iloc[hover_i].tolist()
        traces.append(go.Scatterpolar(r=v + [v[0]], theta=CBS_COLS + [CBS_COLS[0]],
                                      fill="toself", name="hovered",
                                      line=dict(color="#1f78ff"),
                                      fillcolor="rgba(31,120,255,0.18)"))
    if not traces:
        traces.append(go.Scatterpolar(r=[0] * (len(CBS_COLS) + 1),
                                      theta=CBS_COLS + [CBS_COLS[0]]))
    fig = go.Figure(traces)
    fig.update_layout(
        polar=dict(bgcolor="#fafafa",
                   radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                   angularaxis=dict(tickfont=dict(size=9))),
        margin=dict(l=30, r=30, t=8, b=8), height=215, paper_bgcolor="#fff",
        showlegend=len(traces) > 1,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center", font=dict(size=9)),
    )
    return fig


def info_block(i):
    if i is None:
        return html.Div("Click a neighbourhood on the map or a point in an "
                        "embedding view to inspect it.",
                        style={"color": "#888", "fontStyle": "italic", "fontSize": "12px"})
    r = df.iloc[i]
    fields = [("city", r["city"]), ("archetype", r["archetype"].replace("_", " ")),
              ("development", f"{r['development']} ({int(r['planned_units'])} units)")]
    stat_rows = [("Housing price", f"{r['home_value']:.0f} k€"),
                 ("Income", f"{r['income']:.0f} k€"),
                 ("Population", f"{r['population']:.0f}"),
                 ("Avg age", f"{r['avg_age']:.0f}"),
                 ("Household size", f"{r['household_size']:.1f}"),
                 ("Green", f"{r['green_pct']:.0f} %")]
    return html.Div([
        html.Div(r["name"], style={"fontWeight": 600, "fontSize": "13px"}),
        html.Div(" · ".join(f"{v}" for _, v in fields),
                 style={"fontSize": "11px", "color": "#666", "margin": "2px 0 6px"}),
        html.Table(style={"width": "100%", "fontSize": "12px", "borderCollapse": "collapse"},
                   children=html.Tbody([html.Tr([
                       html.Td(label, style={"color": "#666", "padding": "1px 0"}),
                       html.Td(val, style={"textAlign": "right",
                                           "fontVariantNumeric": "tabular-nums"})])
                       for label, val in stat_rows])),
        html.Div(f"“{r['description']}”",
                 style={"fontSize": "11px", "fontStyle": "italic", "color": "#444",
                        "marginTop": "8px", "lineHeight": "1.35"}),
    ])


# -------------------- layout --------------------

def btn(label, _id, accent=False):
    return html.Button(label, id=_id, n_clicks=0, style={
        "padding": "5px 9px", "margin": "2px", "fontSize": "11px",
        "border": "1px solid " + ("#1f2937" if accent else "#ccc"),
        "borderRadius": "4px", "cursor": "pointer",
        "background": "#1f2937" if accent else "#fff",
        "color": "#fff" if accent else "#222"})


CTRL_LABEL = {"fontSize": "10px", "color": "#9ca3af", "textTransform": "uppercase",
              "letterSpacing": "0.04em", "margin": "0 6px 0 4px", "whiteSpace": "nowrap"}

app = Dash(__name__, title="NL Embedding Platform")
app.layout = html.Div(style={
    "display": "grid", "gridTemplateRows": "auto 1fr", "height": "100vh",
    "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "fontSize": "13px", "color": "#222"},
    children=[
        # header / control bar
        html.Div(style={"background": "#111827", "color": "#fff", "padding": "7px 12px",
                        "display": "flex", "flexWrap": "wrap", "alignItems": "center",
                        "gap": "4px"},
                 children=[
                     html.Div("🇳🇱 NL Embedding Platform",
                              style={"fontWeight": 600, "marginRight": "12px"}),
                     html.Span("Colour by", style=CTRL_LABEL),
                     *[btn(PRETTY[k], {"type": "stat", "key": k}) for k in STAT_KEYS],
                     html.Span("Similar by", style={**CTRL_LABEL, "marginLeft": "10px"}),
                     *[btn(lab, {"type": "space", "key": s}) for s, lab in SPACES],
                     btn("🎲 Random", "random", accent=True),
                 ]),

        # body
        html.Div(style={"display": "grid",
                        "gridTemplateColumns": "minmax(280px,320px) 1fr minmax(300px,380px)",
                        "minHeight": 0},
                 children=[
                     # left
                     html.Div(style={"borderRight": "1px solid #eee", "overflowY": "auto",
                                     "padding": "10px 12px"},
                              children=[
                                  html.Div(id="caption",
                                           style={"fontSize": "12px", "fontWeight": 600,
                                                  "color": "#1f2937", "marginBottom": "8px"}),
                                  html.Div("Live CBS profile (hover a neighbourhood)",
                                           style=CTRL_LABEL),
                                  dcc.Graph(id="spider", config={"displayModeBar": False},
                                            style={"height": "215px"}),
                                  html.Hr(style={"border": "none", "borderTop": "1px solid #eee"}),
                                  html.Div(id="info"),
                              ]),
                     # center map
                     html.Div(style={"position": "relative"}, children=[
                         dcc.Graph(id="map", style={"height": "100%"},
                                   config={"displayModeBar": False, "scrollZoom": True}),
                         html.Div(style={
                             "position": "absolute", "bottom": "8px", "left": "8px",
                             "background": "rgba(255,255,255,0.92)", "padding": "6px 8px",
                             "borderRadius": "5px", "fontSize": "11px",
                             "display": "flex", "alignItems": "center", "gap": "10px"},
                             children=[
                                 dcc.Checklist(id="borders",
                                               options=[{"label": " borders", "value": "b"}],
                                               value=["b"], style={"display": "inline-block"}),
                                 html.Span("fill", style={"color": "#666"}),
                                 html.Div(dcc.Slider(id="opacity", min=0.25, max=1, step=0.05,
                                                     value=0.65, marks=None,
                                                     tooltip={"placement": "top"}),
                                          style={"width": "90px"}),
                                 html.Span("top-k", style={"color": "#666"}),
                                 html.Div(dcc.Slider(id="k", min=5, max=30, step=1, value=12,
                                                     marks=None,
                                                     tooltip={"placement": "top"}),
                                          style={"width": "90px"}),
                             ]),
                     ]),
                     # right scatters
                     html.Div(style={"display": "flex", "flexDirection": "column",
                                     "borderLeft": "1px solid #eee"},
                              children=[
                                  dcc.Graph(id="clip", style={"flex": 1, "minHeight": 0},
                                            config={"displayModeBar": False, "scrollZoom": True}),
                                  dcc.Graph(id="text", style={"flex": 1, "minHeight": 0},
                                            config={"displayModeBar": False, "scrollZoom": True}),
                                  dcc.Graph(id="cbs", style={"flex": 1, "minHeight": 0},
                                            config={"displayModeBar": False, "scrollZoom": True}),
                              ]),
                 ]),

        dcc.Store(id="state", data={"i": None, "color": {"kind": "stat", "key": "home_value"},
                                    "recenter": False}),
    ])


# -------------------- callbacks --------------------

def _pick(cd):
    if not cd:
        return None
    pt = cd["points"][0]
    return int(pt.get("customdata", pt.get("location", pt.get("pointIndex", 0))))


@app.callback(
    Output("state", "data"),
    Input("map", "clickData"),
    Input("clip", "clickData"), Input("text", "clickData"), Input("cbs", "clickData"),
    Input({"type": "stat", "key": ALL}, "n_clicks"),
    Input({"type": "space", "key": ALL}, "n_clicks"),
    Input("random", "n_clicks"),
    State("state", "data"),
    prevent_initial_call=True,
)
def on_event(c_map, c_clip, c_text, c_cbs, _stat, _space, _rand, st):
    trig = ctx.triggered_id
    st = dict(st)
    st["recenter"] = False
    if isinstance(trig, dict) and trig.get("type") == "stat":
        st["color"] = {"kind": "stat", "key": trig["key"]}
    elif isinstance(trig, dict) and trig.get("type") == "space":
        st["color"] = {"kind": "sim", "space": trig["key"]}
    elif trig == "random":
        st["i"] = int(np.random.randint(N))
        st["recenter"] = True
    elif trig in ("map", "clip", "text", "cbs"):
        i = _pick({"map": c_map, "clip": c_clip, "text": c_text, "cbs": c_cbs}[trig])
        if i is None:
            return no_update
        st["i"] = i
    else:
        return no_update
    return st


@app.callback(
    Output("map", "figure"), Output("clip", "figure"), Output("text", "figure"),
    Output("cbs", "figure"), Output("info", "children"), Output("caption", "children"),
    Input("state", "data"), Input("borders", "value"),
    Input("opacity", "value"), Input("k", "value"),
)
def render(state, borders, opacity, k):
    k = int(k or 12)
    borders_on = "b" in (borders or [])
    color, i = state["color"], state["i"]
    if color["kind"] == "stat":
        cap = f"Map coloured by: {PRETTY[color['key']]}"
    else:
        sp = dict(SPACES)[color["space"]]
        cap = (f"Similarity heatmap · {sp}"
               if i is not None else f"Pick a neighbourhood to rank by {sp}")
    return (map_figure(state, borders_on, opacity, k),
            scatter("clip", state, k), scatter("text", state, k),
            scatter("cbs", state, k), info_block(i), cap)


@app.callback(
    Output("spider", "figure"),
    Input("map", "hoverData"), Input("state", "data"),
)
def hover_spider(hover, state):
    hi = _pick(hover) if hover else None
    return spider(hi, state.get("i"))


if __name__ == "__main__":
    app.run(debug=True, port=8052)
