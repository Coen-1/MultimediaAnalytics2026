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
EXTRA_AREA_HIGHLIGHT = 20
TILE = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"
STYLE = {"version": 8, "sources": {"p": {"type": "raster", "tiles": [TILE], "tileSize": 256}},
         "layers": [{"id": "p", "type": "raster", "source": "p"}]}
if os.path.exists("assets/buurten.geojson"):  # draw the real CBS buurt boundaries over the aerial photo
    STYLE["sources"]["b"] = {"type": "geojson", "data": "/assets/buurten.geojson"}
    STYLE["layers"] += [  # cased line: dark halo under a bright line so borders read over any imagery
        {"id": "b-case", "type": "line", "source": "b",
         "paint": {"line-color": "#1a1a1a", "line-width": 3, "line-opacity": 0.45,
                   "line-blur": 0.5}},
        {"id": "b", "type": "line", "source": "b",
         "paint": {"line-color": "#ffd54f", "line-width": 1.4, "line-opacity": 0.9}}]

ACCENT, SEL, SIM_C = "#1e88e5", "#e53935", "#fb8c00"   # blue / red / orange
CLIP_C, TEXT_C = "#1e88e5", "#2e7d32"                   # map: blue=CLIP, green=text
SPIDER_C_OUT = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
                '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
SPIDER_C_IN = ['rgba(122,131,250,0.25)', 'rgba(241,110,88,0.25)', 'rgba(38,211,165,0.25)', 
               'rgba(183,122,250,0.25)', 'rgba(255,175,114,0.25)', 'rgba(59,217,244,0.25)', 
               'rgba(255,124,162,0.25)', 'rgba(192,235,147,0.25)', 'rgba(255,166,255,0.25)', 'rgba(254,210,107,0.25)']
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

def topk(i, sim, filter):  # K most similar rows to i (excluding itself)
    if len(i) == 1:
        return [filter[j] for j in np.argsort(sim[i[0]][filter])[::-1][1:K + 1]]
    return []

def colors(i, hot, base="#cfd8dc", hotc=SIM_C):
    c = np.full(len(df), base, dtype=object); s = np.full(len(df), 6)
    c[hot] = hotc; s[hot] = 9; c[i] = SEL; s[i] = 16
    return c, s

def fmt(v):
    if isinstance(v, str):
        return v
    return "—" if pd.isna(v) else round(float(v), 1)

def scatter(space, i, filter):
    c, s = colors(i, topk(i, SIM[space], filter))
    f = go.Figure(go.Scattergl(x=df[f"{space}_x"].iloc[filter], y=df[f"{space}_y"].iloc[filter], mode="markers",
                               marker=dict(color=c[filter], size=s[filter], line=dict(width=0.5, color="rgba(0,0,0,.25)")),
                               customdata=filter, text=df["name"].iloc[filter], hoverinfo="text"))
    axis = dict(visible=False)  # UMAP coords are arbitrary -> hide axes
    return f.update_layout(title=f"{space} embedding (UMAP)", title_font_size=13, clickmode="event+select",
                           xaxis=axis, yaxis=axis, margin=dict(l=6, r=6, t=26, b=6))

def mapfig(i, filter):
    ct, cl = topk(i, SIM["text"], filter), topk(i, SIM["clip"], filter)
 
    f = go.Figure()
    c = np.full(len(df), "rgba(70,70,70,.55)", dtype=object)
    s = np.full(len(df), 6)
    c[cl] = CLIP_C; s[cl] = 13; c[ct] = TEXT_C; s[ct] = 13; c[i] = SEL; s[i] = 20
    highlight = cl + ct + i

    d = df.iloc[highlight]
    f.add_scattermap(lat=d.lat, lon=d.lon, mode='markers', hoverinfo="skip", showlegend=False, 
                     selectedpoints=[], unselected=dict(marker=dict(opacity=1)),
                     marker=dict(color="white", size=s[highlight] + 5))
    d = df.iloc[filter]
    f.add_scattermap(lat=d.lat, lon=d.lon, mode='markers', text=d.name, customdata=filter, 
                     hovertemplate="%{text}<extra></extra>", unselected=dict(marker=dict(opacity=1)),
                     marker=dict(color=c[filter], size=s[filter]), showlegend=False)
    
    # required to add a legend, no points are plotted
    legend_items = [("CLIP-similar", CLIP_C, 13), ("Text-similar", TEXT_C, 13), ("Selected", SEL, 20)]
    for item in legend_items:    
        f.add_scattermap(lat=[None], lon=[None], mode="markers", name=item[0], 
                         showlegend=True, marker=dict(color=item[1], size=item[2]))
    
    style = STYLE.copy()
    if os.path.exists("assets/buurten.geojson"):
        style["layers"] = STYLE["layers"].copy()

        # If threshold is reached add extra highlight to areas
        if len(filter) < EXTRA_AREA_HIGHLIGHT:
            style["layers"] += [{"id": "b2", "type": "line", "source": "b",
            "paint": {"line-color": "#30fd29", "line-width": 5, "line-opacity": 0.9}}]
        
        # Find buurtcodes from df -> only show outlines of areas in filter_ids
        filter_ids = df["code"].iloc[filter]
        maplibre_filter = ["in", ["get", "code"], ["literal", filter_ids]]
        style["layers"] = [
            {**layer, "filter": maplibre_filter} if layer["id"] in ("b", "b-case", "b2") else layer
            for layer in style["layers"]
        ]

    return f.update_layout(map=dict(style=style, center=CENTER, zoom=ZOOM), uirevision="keep",
                           margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select",
                           legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,.85)",
                                       bordercolor="#ddd", borderwidth=1, font=dict(size=11)))

def spider(i, cols):
    theta = [LBL[c] for c in cols]
    f = go.Figure()
    if len(i) == 0:
        radius = [0]*(len(cols)+1)
        fig_title = "No area selected"
        f.add_trace(go.Scatterpolar(r=radius, theta=theta + [theta[0]], fill="toself", 
                                    line_color="rgba(0,0,0,0)", fillcolor="rgba(0,0,0,0)"))
    elif len(i) > len(SPIDER_C_OUT):
        radius = [0]*(len(cols)+1)
        fig_title = "Cannot show this many areas"
        f.add_trace(go.Scatterpolar(r=radius, theta=theta + [theta[0]], fill="toself", 
                                    line_color="rgba(0,0,0,0)", fillcolor="rgba(0,0,0,0)"))
    else:
        fig_title = "Comparing each area"
        for ci, item in enumerate(i):
            color_out = SPIDER_C_OUT[ci]
            color_in = SPIDER_C_IN[ci]
            radius = NORM.iloc[item][cols].tolist() + [NORM.iloc[item][cols[0]]]
            title = df.name[item]
            f.add_trace(go.Scatterpolar(r=radius, theta=theta + [theta[0]], fill="toself",
                                  line_color=color_out, fillcolor=color_in, name=title))
    
    return f.update_layout(title=fig_title, title_font_size=13, title_x=0.5,
                           polar=dict(radialaxis=dict(visible=True, range=[0, 1]),
                                      angularaxis=dict(tickfont=dict(size=10))),
                           margin=dict(l=55, r=55, t=34, b=24), showlegend=False)

def table(i, cols):
    columns = [{"name": "Area", "id": "field"}] + [{"name": fmt(df["name"][idx]), "id": f"area_{idx}"} for idx in i]
    rows = [{"field": k, **{f"area_{idx}": fmt(df[k][idx]) for idx in i}} for k in cols]
    return rows, columns

app = Dash(__name__)
app.layout = html.Div(style={"background": "#f5f6f8", "height": "100vh"}, children=[
    dcc.Store(id="current_selection", data=[]),
    dcc.Store(id="current_columns", data=CBS),
    dcc.Store(id="current_filter", data=df.index.to_list()),
    dcc.Store(id="query_text_selection_start", data=0),
    dcc.Store(id="query_text_selection_end", data=0),
    html.Div("Neighbourhood embedding explorer", style={"padding": "10px 16px",
             "fontWeight": "700", "fontSize": "16px", "fontFamily": FONT}),
    html.Div(style={**PAGE, "height": "calc(100vh - 44px)"}, children=[
        html.Div(style={"width": "25%", **CARD, "display": "flex", "flexDirection": "column"}, children=[
            dcc.Dropdown(CBS, CBS, id="column_select", closeOnSelect=False, multi=True, clearable=True,
                         placeholder="At least one column must be selected"),
            html.Hr(),
            html.Div("​", id="query_message", style={"color": "black", "fontsize": "14px", "fontFamily": FONT}),
            html.Div(style={"display": "flex", "flexdirection": "row"}, children=[
                dcc.Textarea("", id="query_text", rows=1, style={"resize": "none"}),
                dcc.Button("Filter", id="filter_button")
            ]),
            dcc.Dropdown(id="query_columns", options=CBS, value=None),
            dash_table.DataTable(id="table",
                columns=[{"name": "Field", "id": "field"}],
                style_as_list_view=True,
                style_header={"background": "#f0f2f5", "fontWeight": "600", "border": "none",
                              "fontFamily": FONT, "padding": "6px 10px"},
                style_cell={"padding": "6px 10px", "border": "none", "fontFamily": FONT,
                            "fontSize": "13px", "textAlign": "left"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "background": "#fafbfc"},
                                        {"if": {"column_id": "value"},
                                         "fontVariantNumeric": "tabular-nums"}],
                style_table={'overflowX': 'scroll'}),
            dcc.Graph(id="spider", style={"height": "340px", "flexShrink": 0})]),
        html.Div(style={"width": "45%", **CARD}, children=[
            dcc.Graph(id="map", style={"height": "100%"})]),
        html.Div(style={"width": "30%", **CARD, "display": "flex", "flexDirection": "column"}, children=[
            dcc.Graph(id="clip", style={"flex": 1}),
            dcc.Graph(id="text", style={"flex": 1}),
            dcc.Graph(id="cbs", style={"flex": 1})]),
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
              Input("map", "selectedData"),
              prevent_initial_call=True)
def update_current_selection(*clicks):
    cd = dict(zip(["clip", "text", "cbs", "map"], clicks)).get(ctx.triggered_id)
    if cd is None:
        raise exceptions.PreventUpdate
    pts = cd["points"]

    # for some reason box and lasso select trigger the callback twice 
    # the second time with an empty box / lasso -> so no update to prevent overwrite
    if "range" not in cd.keys() and "lassoPoints" not in cd.keys() and len(pts) == 0:
        raise exceptions.PreventUpdate
    
    i = [int(pt["customdata"]) for pt in pts if "customdata" in pt]
    return i

# keep track of selected columns
@app.callback(Output("current_columns", "data"),
              Input("column_select", "value"),
              prevent_initial_call=True)
def update_current_columns(value):
    if not value:
        raise exceptions.PreventUpdate
    return value

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
              State("current_columns", "data"))
def update_figure_selections(i, filter, cols):
    return scatter("clip", i, filter), scatter("text", i, filter), scatter("cbs", i, filter), \
            mapfig(i, filter), spider(i, cols), *table(i, cols)

# when selected columns change, update spiderplot and table
@app.callback(Output("spider", "figure", allow_duplicate=True),
              Output("table", "data", allow_duplicate=True),
              Input("current_columns", "data"),
              State("current_selection", "data"),
              prevent_initial_call=True)
def update_figure_columns(cols, i):
    return spider(i, cols), table(i, cols)

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
    return s, start+len(col_name), start+len(col_name), ""

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

if __name__ == "__main__":
    app.run(debug=True)
