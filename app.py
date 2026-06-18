import json, math, os
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.io as pio
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, ctx, exceptions

pio.templates.default = "plotly_white"

df = pd.read_parquet("data.parquet")
SIM = {"clip": np.load("sim_clip.npy"), "text": np.load("sim_text.npy"), "cbs": np.load("sim_cbs.npy")}
CBS = ["population", "income", "home_value", "density", "household_size",
       "pct_owner", "pct_single_pers", "pct_65plus", "pct_dutch", "cars_per_hh"]
LBL = {"population": "pop", "income": "income", "home_value": "home €", "density": "density",
       "household_size": "hh size", "pct_owner": "% owner", "pct_single_pers": "% single",
       "pct_65plus": "% 65+", "pct_dutch": "% dutch", "cars_per_hh": "cars/hh"}  # short spider axis labels
NORM = (df[CBS] - df[CBS].min()) / (df[CBS].max() - df[CBS].min() + 1e-9)
K = 10
TILE = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"
STYLE = {"version": 8, "sources": {"p": {"type": "raster", "tiles": [TILE], "tileSize": 256}},
         "layers": [{"id": "p", "type": "raster", "source": "p"}]}
if os.path.exists("buurten.geojson"):  # draw the real CBS buurt boundaries over the aerial photo
    STYLE["sources"]["b"] = {"type": "geojson", "data": json.load(open("buurten.geojson"))}
    STYLE["layers"] += [  # cased line: dark halo under a bright line so borders read over any imagery
        {"id": "b-case", "type": "line", "source": "b",
         "paint": {"line-color": "#1a1a1a", "line-width": 3, "line-opacity": 0.45,
                   "line-blur": 0.5}},
        {"id": "b", "type": "line", "source": "b",
         "paint": {"line-color": "#ffd54f", "line-width": 1.4, "line-opacity": 0.9}}]

ACCENT, SEL, SIM_C = "#1e88e5", "#e53935", "#fb8c00"   # blue / red / orange
CLIP_C, TEXT_C = "#1e88e5", "#2e7d32"                   # map: blue=CLIP, green=text
FONT = "Inter, system-ui, sans-serif"
PAGE = {"display": "flex", "gap": "12px", "padding": "0 12px 12px", "boxSizing": "border-box",
        "background": "#f5f6f8", "fontFamily": FONT}
CARD = {"background": "#fff", "borderRadius": "10px", "padding": "8px",
        "boxShadow": "0 1px 3px rgba(0,0,0,.12)", "overflow": "hidden"}


def _fit(lat, lon):  # one-time center+zoom that frames every point
    c = dict(lat=float((lat.min() + lat.max()) / 2), lon=float((lon.min() + lon.max()) / 2))
    span = max(lat.max() - lat.min(), (lon.max() - lon.min()) * math.cos(math.radians(c["lat"])))
    return c, math.log2(360 / max(span, 1e-6)) - 1.2          # -1.2 = padding fudge
CENTER, ZOOM = _fit(df.lat, df.lon)


def topk(i, sim):  # K most similar rows to i (excluding itself)
    return np.argsort(sim[i])[::-1][1:K + 1]


def colors(i, hot, base="#cfd8dc", hotc=SIM_C):
    c = np.full(len(df), base, dtype=object); s = np.full(len(df), 6)
    c[hot] = hotc; s[hot] = 9; c[i] = SEL; s[i] = 16
    return c, s


def scatter(space, i):
    c, s = colors(i, topk(i, SIM[space]))
    f = go.Figure(go.Scattergl(x=df[f"{space}_x"], y=df[f"{space}_y"], mode="markers",
                               marker=dict(color=c, size=s, line=dict(width=0.5, color="rgba(0,0,0,.25)")),
                               customdata=np.arange(len(df)), text=df["name"], hoverinfo="text"))
    axis = dict(visible=False)  # UMAP coords are arbitrary -> hide axes
    return f.update_layout(title=f"{space} embedding (UMAP)", title_font_size=13,
                           xaxis=axis, yaxis=axis, margin=dict(l=6, r=6, t=26, b=6))


def mapfig(i):
    ct, cl = set(topk(i, SIM["text"])), set(topk(i, SIM["clip"]))
    groups = [("Other", "rgba(70,70,70,.55)", 6, set(range(len(df))) - cl - ct - {i}),
              ("CLIP-similar", CLIP_C, 13, cl - ct - {i}),
              ("Text-similar", TEXT_C, 13, ct - {i}),
              ("Selected", SEL, 20, {i})]
    f = go.Figure()
    for name, color, size, idx in groups:
        idx = sorted(idx)
        if not idx:
            continue
        d = df.iloc[idx]
        if name != "Other":   # white halo so the dot reads on the aerial photo
            f.add_scattermap(lat=d.lat, lon=d.lon, mode="markers", hoverinfo="skip",
                             showlegend=False, customdata=idx, marker=dict(color="white", size=size + 5))
        f.add_scattermap(lat=d.lat, lon=d.lon, mode="markers", name=name, text=d.name,
                         customdata=idx, hovertemplate="%{text}<extra></extra>",
                         marker=dict(color=color, size=size))
    return f.update_layout(map=dict(style=STYLE, center=CENTER, zoom=ZOOM), uirevision="keep",
                           margin=dict(l=0, r=0, t=0, b=0),
                           legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,.85)",
                                       bordercolor="#ddd", borderwidth=1, font=dict(size=11)))


def spider(i, cols):
    theta = [LBL[c] for c in cols]
    f = go.Figure(go.Scatterpolar(r=NORM.iloc[i].tolist() + [NORM.iloc[i, 0]],
                                  theta=theta + [theta[0]], fill="toself",
                                  line_color=ACCENT, fillcolor="rgba(30,136,229,.25)"))
    return f.update_layout(title=df.name[i], title_font_size=13, title_x=0.5,
                           polar=dict(radialaxis=dict(visible=True, range=[0, 1]),
                                      angularaxis=dict(tickfont=dict(size=10))),
                           margin=dict(l=55, r=55, t=34, b=24))


app = Dash(__name__)
app.layout = html.Div(style={"background": "#f5f6f8", "height": "100vh"}, children=[
    dcc.Store(id="current_selection", data=0),
    dcc.Store(id="current_columns", data=CBS),
    html.Div("Neighbourhood embedding explorer", style={"padding": "10px 16px",
             "fontWeight": "700", "fontSize": "16px", "fontFamily": FONT}),
    html.Div(style={**PAGE, "height": "calc(100vh - 44px)"}, children=[
        html.Div(style={"width": "25%", **CARD, "display": "flex", "flexDirection": "column"}, children=[
            dcc.Dropdown(LBL, CBS, id="column_select", closeOnSelect=False, multi=True, clearable=True,
                         placeholder="At least one column must be selected"),
            dash_table.DataTable(id="table",
                columns=[{"name": "Field", "id": "field"}, {"name": "Value", "id": "value"}],
                style_as_list_view=True,
                style_header={"background": "#f0f2f5", "fontWeight": "600", "border": "none",
                              "fontFamily": FONT, "padding": "6px 10px"},
                style_cell={"padding": "6px 10px", "border": "none", "fontFamily": FONT,
                            "fontSize": "13px", "textAlign": "left"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "background": "#fafbfc"},
                                        {"if": {"column_id": "value"},
                                         "fontVariantNumeric": "tabular-nums"}]),
            dcc.Graph(id="spider", style={"height": "340px", "flexShrink": 0})]),
        html.Div(style={"width": "45%", **CARD}, children=[
            dcc.Graph(id="map", style={"height": "100%"})]),
        html.Div(style={"width": "30%", **CARD, "display": "flex", "flexDirection": "column"}, children=[
            dcc.Graph(id="clip", style={"flex": 1}),
            dcc.Graph(id="text", style={"flex": 1}),
            dcc.Graph(id="cbs", style={"flex": 1})]),
    ]),
])


def fmt(v):
    if isinstance(v, str):
        return v
    return "—" if pd.isna(v) else round(float(v), 1)


@app.callback(Output("current_selection", "data"),
              Input("clip", "clickData"), 
              Input("text", "clickData"), 
              Input("cbs", "clickData"),
              Input("map", "clickData"),
              prevent_initial_call=True)
def update_current_selection(*clicks):
    cd = dict(zip(["clip", "text", "cbs", "map"], clicks)).get(ctx.triggered_id)
    pt = cd["points"][0] if cd else {}
    # map traces are subsets -> use customdata; Scattergl clicks omit it -> pointIndex (single ordered trace)
    i = int(pt["customdata"]) if pt.get("customdata") is not None else int(pt.get("pointIndex", 0))
    return i

@app.callback(Output("current_columns", "data"),
              Input("column_select", "value"),
              prevent_initial_call=True)
def update_current_columns(value):
    if not value:
        raise exceptions.PreventUpdate
    return value

# Run all sequentially: update when all figures are ready
@app.callback(Output("clip", "figure"), 
              Output("text", "figure"), 
              Output("cbs", "figure"),
              Output("map", "figure"), 
              Output("spider", "figure"), 
              Output("table", "data"),
              Input("current_selection", "data"),
              State("current_columns", "data"))
def update_figure_selections(i, cols):
    row = [{"field": k, "value": fmt(df[k][i])} for k in ["name"] + cols]
    return scatter("clip", i), scatter("text", i), scatter("cbs", i), mapfig(i), spider(i, cols), row

@app.callback(Output("spider", "figure", allow_duplicate=True),
              Output("table", "data", allow_duplicate=True),
              Input("current_columns", "data"),
              State("current_selection", "data"),
              prevent_initial_call=True)
def update_figure_columns(cols, i):
    row = [{"field": k, "value": fmt(df[k][i])} for k in ["name"] + cols]
    return spider(i, cols), row

'''
# Run all parallel: updates each figure when ready
@app.callback(Output("clip", "figure"),
              Input("current_selection", "data"))
def update_clip_figure(i):
    return scatter("clip", i)

@app.callback(Output("text", "figure"),
              Input("current_selection", "data"))
def update_text_figure(i):
    return scatter("text", i)

@app.callback(Output("cbs", "figure"),
              Input("current_selection", "data"))
def update_cbs_figure(i):
    return scatter("cbs", i)

@app.callback(Output("map", "figure"),
              Input("current_selection", "data"))
def update_map(i):
    return mapfig(i)

@app.callback(Output("spider", "figure"),
              Input("current_selection", "data"),
              Input("current_columns", "data"))
def update_spider_figure(i, cols):
    return spider(i, cols)

@app.callback(Output("table", "data"),
              Input("current_selection", "data"),
              Input("current_columns", "data"))
def update_cbs_table(i, cols):
    row = [{"field": k, "value": fmt(df[k][i])} for k in ["name"] + cols]
    return row
'''

if __name__ == "__main__":
    app.run(debug=True)
