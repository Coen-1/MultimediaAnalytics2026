import numpy as np, pandas as pd, plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, ctx

df = pd.read_parquet("data.parquet")
SIM = {"clip": np.load("sim_clip.npy"), "text": np.load("sim_text.npy"), "cbs": np.load("sim_cbs.npy")}
CBS = ["population", "income", "pct_green", "avg_age", "home_value", "density"]
NORM = (df[CBS] - df[CBS].min()) / (df[CBS].max() - df[CBS].min() + 1e-9)
K = 10
TILE = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"
STYLE = {"version": 8, "sources": {"p": {"type": "raster", "tiles": [TILE], "tileSize": 256}},
         "layers": [{"id": "p", "type": "raster", "source": "p"}]}


def topk(i, sim):  # K most similar rows to i (excluding itself)
    return np.argsort(sim[i])[::-1][1:K + 1]


def colors(i, hot, base="#bbb", hotc="orange"):
    c = np.full(len(df), base, dtype=object); s = np.full(len(df), 6)
    c[hot] = hotc; s[hot] = 9; c[i] = "red"; s[i] = 16
    return c, s


def scatter(space, i):
    c, s = colors(i, topk(i, SIM[space]))
    f = go.Figure(go.Scattergl(x=df[f"{space}_x"], y=df[f"{space}_y"], mode="markers",
                               marker=dict(color=c, size=s), text=df["name"], hoverinfo="text"))
    return f.update_layout(title=f"{space} embedding (PCA)", margin=dict(l=0, r=0, t=30, b=0))


def mapfig(i):
    c, s = colors(i, topk(i, SIM["clip"]), base="white", hotc="deepskyblue")
    c[topk(i, SIM["text"])] = "lime"; c[i] = "red"  # blue=clip-similar, green=text-similar
    f = go.Figure(go.Scattermap(lat=df["lat"], lon=df["lon"], mode="markers",
                                marker=dict(color=c, size=s), text=df["name"], hoverinfo="text"))
    return f.update_layout(map=dict(style=STYLE, center=dict(lat=df.lat[i], lon=df.lon[i]), zoom=11),
                           margin=dict(l=0, r=0, t=0, b=0))


def spider(i):
    f = go.Figure(go.Scatterpolar(r=NORM.iloc[i].tolist() + [NORM.iloc[i, 0]],
                                  theta=CBS + [CBS[0]], fill="toself"))
    return f.update_layout(title=df.name[i], polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                           margin=dict(l=30, r=30, t=30, b=30))


app = Dash(__name__)
app.layout = html.Div(style={"display": "flex", "height": "100vh"}, children=[
    html.Div(style={"width": "25%"}, children=[
        dash_table.DataTable(id="table"),
        dcc.Graph(id="spider")]),
    html.Div(style={"width": "45%"}, children=[dcc.Graph(id="map", style={"height": "100%"})]),
    html.Div(style={"width": "30%"}, children=[
        dcc.Graph(id="clip", style={"height": "33vh"}),
        dcc.Graph(id="text", style={"height": "33vh"}),
        dcc.Graph(id="cbs", style={"height": "33vh"})]),
])


@app.callback(Output("clip", "figure"), Output("text", "figure"), Output("cbs", "figure"),
              Output("map", "figure"), Output("spider", "figure"), Output("table", "data"),
              Input("clip", "clickData"), Input("text", "clickData"), Input("cbs", "clickData"),
              Input("map", "clickData"))
def update(*clicks):
    cd = dict(zip(["clip", "text", "cbs", "map"], clicks)).get(ctx.triggered_id)
    i = cd["points"][0]["pointIndex"] if cd else 0
    row = [{"field": k, "value": df[k][i]} for k in ["name"] + CBS]
    return scatter("clip", i), scatter("text", i), scatter("cbs", i), mapfig(i), spider(i), row


if __name__ == "__main__":
    app.run(debug=True)
