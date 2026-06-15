"""Dash app: NL neighbourhood embedding explorer.

Three-column responsive layout:
  - LEFT:   selection profile (spider + CBS table) and controls
  - CENTRE: map of NL with PDOK aerial tiles
  - RIGHT:  three UMAP scatters (CLIP / text / CBS)

State flow:
  Any click (map or scatter) updates `state.i` -> all six panels re-render.
  `state.zoom_to` only becomes true when the trigger is a scatter click
  (the user pointed *somewhere* and wants the map to follow).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

# --- data ---
df = pd.read_parquet("data.parquet").reset_index(drop=True)
SIM = {"clip": np.load("sim_clip.npy"),
       "text": np.load("sim_text.npy"),
       "cbs":  np.load("sim_cbs.npy")}
CBS = ["population", "income", "pct_green", "avg_age", "home_value", "density"]
NORM = (df[CBS] - df[CBS].min()) / (df[CBS].max() - df[CBS].min() + 1e-9)
N, K = len(df), 15

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
SELECTED_COLOR = "#e6194b"
NL_CENTER = dict(lat=52.2, lon=5.3)
NL_ZOOM = 6.4


def topk(i: int, sim_key: str, k: int = K) -> np.ndarray:
    return np.argsort(SIM[sim_key][i])[::-1][1:k + 1]


def base_colors(color_by: str) -> np.ndarray:
    labels = (df["archetype"].astype("category").cat.codes.values
              if color_by == "archetype"
              else df[f"{color_by}_cluster"].values)
    return np.array([PALETTE[int(l) % len(PALETTE)] for l in labels], dtype=object)


def marker_arrays(i: int, sim_key: str, color_by: str):
    c = base_colors(color_by).copy()
    s = np.full(N, 5.5)
    o = np.full(N, 0.55)
    if sim_key != "none":
        hot = topk(i, sim_key)
        s[hot] = 11
        o[hot] = 1.0
    c[i] = SELECTED_COLOR
    s[i] = 17
    o[i] = 1.0
    return c, s, o


def scatter(space: str, i: int, sim_key: str, color_by: str) -> go.Figure:
    c, s, o = marker_arrays(i, sim_key, color_by)
    fig = go.Figure(go.Scattergl(
        x=df[f"{space}_x"], y=df[f"{space}_y"],
        mode="markers",
        marker=dict(color=c, size=s, opacity=o, line=dict(width=0)),
        text=df["name"] + "  (" + df["archetype"] + ")",
        hoverinfo="text",
        customdata=df.index.values,
    ))
    fig.update_layout(
        title=dict(text=f"{space.upper()} — UMAP", x=0.02, y=0.97,
                   font=dict(size=12, color="#333")),
        margin=dict(l=4, r=4, t=22, b=4),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="#fafafa", paper_bgcolor="#ffffff",
        showlegend=False, dragmode="pan", uirevision="scatter",
    )
    return fig


def mapfig(i: int, sim_key: str, color_by: str, follow: bool, zoom_to_sel: bool) -> go.Figure:
    c, s, o = marker_arrays(i, sim_key, color_by)
    s = s * 0.55 + 3  # smaller on the map
    fig = go.Figure(go.Scattermap(
        lat=df["lat"], lon=df["lon"], mode="markers",
        marker=dict(color=c, size=s, opacity=o),
        text=df["name"], hoverinfo="text",
        customdata=df.index.values,
    ))
    if follow and zoom_to_sel:
        centre, zoom = dict(lat=df.lat.iloc[i], lon=df.lon.iloc[i]), 12
        uirev = f"sel-{i}"
    else:
        centre, zoom, uirev = NL_CENTER, NL_ZOOM, "stable"
    fig.update_layout(
        map=dict(style=MAP_STYLE, center=centre, zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        uirevision=uirev,
    )
    return fig


def spider(i: int) -> go.Figure:
    vals = NORM.iloc[i].tolist()
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=CBS + [CBS[0]],
        fill="toself", line=dict(color=SELECTED_COLOR),
        fillcolor="rgba(230,25,75,0.25)",
    ))
    fig.update_layout(
        polar=dict(bgcolor="#fafafa",
                   radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                   angularaxis=dict(tickfont=dict(size=10))),
        margin=dict(l=40, r=40, t=10, b=10), height=230, showlegend=False,
        paper_bgcolor="#ffffff",
    )
    return fig


# --- layout ---

PANEL = {"padding": "10px 14px", "borderBottom": "1px solid #eee"}
LABEL = {"fontSize": "11px", "color": "#666", "marginBottom": "4px",
         "textTransform": "uppercase", "letterSpacing": "0.04em"}

app = Dash(__name__, title="NL Neighbourhood Embeddings")
app.layout = html.Div(
    style={
        "display": "grid",
        "gridTemplateColumns": "minmax(290px, 320px) 1fr minmax(340px, 420px)",
        "gridTemplateRows": "100vh",
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "fontSize": "13px",
        "color": "#222",
    },
    children=[
        # LEFT
        html.Div(style={"display": "flex", "flexDirection": "column",
                        "borderRight": "1px solid #eee", "overflow": "auto"},
                 children=[
                     html.Div("NL Neighbourhood Embeddings",
                              style={"padding": "12px 14px", "fontWeight": 600,
                                     "background": "#1f2937", "color": "white",
                                     "fontSize": "14px"}),
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
                         html.Div("Highlight top-k by", style=LABEL),
                         dcc.RadioItems(id="simkey", value="clip", inline=True,
                                        options=[{"label": " CLIP", "value": "clip"},
                                                 {"label": " Text", "value": "text"},
                                                 {"label": " CBS",  "value": "cbs"},
                                                 {"label": " none", "value": "none"}],
                                        labelStyle={"marginRight": "10px"}),
                         html.Div("Colour dots by clusters of", style={**LABEL, "marginTop": "10px"}),
                         dcc.RadioItems(id="colorby", value="archetype", inline=True,
                                        options=[{"label": " CLIP",      "value": "clip"},
                                                 {"label": " Text",      "value": "text"},
                                                 {"label": " CBS",       "value": "cbs"},
                                                 {"label": " archetype", "value": "archetype"}],
                                        labelStyle={"marginRight": "10px"}),
                         dcc.Checklist(id="follow", value=["f"],
                                       options=[{"label": " zoom map to selection on click",
                                                 "value": "f"}],
                                       style={"marginTop": "10px"}),
                     ]),
                     html.Div(style={**PANEL, "fontSize": "11px", "color": "#888"},
                              children=[
                                  "Red = selected · large = top-k similar · ",
                                  "click anywhere to drive the views.",
                              ]),
                 ]),

        # CENTRE
        html.Div(style={"position": "relative"}, children=[
            dcc.Graph(id="map", style={"height": "100vh"},
                      config={"displayModeBar": False, "scrollZoom": True}),
        ]),

        # RIGHT
        html.Div(style={"display": "flex", "flexDirection": "column",
                        "borderLeft": "1px solid #eee"},
                 children=[
                     dcc.Graph(id="clip", style={"flex": 1, "minHeight": 0},
                               config={"displayModeBar": False}),
                     dcc.Graph(id="text", style={"flex": 1, "minHeight": 0},
                               config={"displayModeBar": False}),
                     dcc.Graph(id="cbs",  style={"flex": 1, "minHeight": 0},
                               config={"displayModeBar": False}),
                 ]),

        dcc.Store(id="state", data={"i": 0, "trigger": "init"}),
    ],
)


def _pick(click_data) -> int | None:
    if not click_data: return None
    pt = click_data["points"][0]
    if "customdata" in pt:
        return int(pt["customdata"])
    return int(pt.get("pointIndex", 0))


@app.callback(
    Output("state", "data"),
    Input("clip", "clickData"), Input("text", "clickData"),
    Input("cbs",  "clickData"), Input("map",  "clickData"),
    prevent_initial_call=True,
)
def on_click(c_clip, c_text, c_cbs, c_map):
    src = ctx.triggered_id
    i = _pick({"clip": c_clip, "text": c_text, "cbs": c_cbs, "map": c_map}[src])
    if i is None: return no_update
    return {"i": i, "trigger": src}


@app.callback(
    Output("clip", "figure"), Output("text", "figure"), Output("cbs", "figure"),
    Output("map", "figure"), Output("spider", "figure"),
    Output("info", "children"), Output("description", "children"),
    Input("state", "data"),
    Input("simkey", "value"), Input("colorby", "value"), Input("follow", "value"),
)
def render(state, simkey, colorby, follow):
    i = int(state["i"])
    row = df.iloc[i]
    follow_on = "f" in (follow or [])
    zoom_to_sel = state.get("trigger") in ("clip", "text", "cbs")

    info = html.Div([
        html.Div(row["name"], style={"fontWeight": 600, "fontSize": "13px"}),
        html.Div(f"{row['city']} · {row['archetype'].replace('_', ' ')}",
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
    return (
        scatter("clip", i, simkey, colorby),
        scatter("text", i, simkey, colorby),
        scatter("cbs",  i, simkey, colorby),
        mapfig(i, simkey, colorby, follow_on, zoom_to_sel),
        spider(i),
        info,
        f"“{row['description']}”",
    )


if __name__ == "__main__":
    app.run(debug=True)
