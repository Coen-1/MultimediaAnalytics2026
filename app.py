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
K = 10
TILE_TYPES = {"satellite": {"tiles": "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg",
                            "attribution": "Luchtfoto © Kadaster / Beeldmateriaal.nl"},
              "streetview": {"tiles": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", 
                             "attribution": "© OpenStreetMap contributors"}}
EXTRA_AREA_HIGHLIGHT = 20
ACCENT, SEL, SIM_C = "#1e88e5", "#e53935", "#fb8c00"   # blue / red / orange
CLIP_C, TEXT_C, CBS_C = "#1e88e5", "#2e7d32", "#b621aa"                  # map: blue=CLIP, green=text, 
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
OVERLAY = {"position": "fixed", "inset": 0, "background": "rgba(0,0,0,.35)", "zIndex": 1000,
           "display": "flex", "alignItems": "center", "justifyContent": "center"}  # intro popup backdrop
BTN = {"background": ACCENT, "color": "#fff", "border": "none", "borderRadius": "8px", "padding": "0 14px",
       "cursor": "pointer", "fontWeight": "600", "fontFamily": FONT}  # primary button
LABEL = {"fontSize": "11px", "fontWeight": "700", "letterSpacing": ".04em", "textTransform": "uppercase",
         "color": "#8a8f98", "fontFamily": FONT, "margin": "2px 0"}  # small section header


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

def geocode(q):
    """Return (row_index, status_message) or (None, error_message)."""
    if not q or not q.strip():
        return [], ""
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
            pt = Point(lon, lat)
            for name, idx, poly in POLYGONS:
                if poly.contains(pt):
                    return [idx], name
            
    except Exception:
        pass
    return [], "no match"

def scatter(space, i, filter):
    c, s = colors(i, topk(i, SIM[space], filter))
    f = go.Figure(go.Scattergl(x=df[f"{space}_x"].iloc[filter], y=df[f"{space}_y"].iloc[filter], mode="markers",
                               marker=dict(color=c[filter], size=s[filter], line=dict(width=0.5, color="rgba(0,0,0,.25)")),
                               customdata=filter, text=df["name"].iloc[filter], hoverinfo="text"))
    axis = dict(visible=False)  # UMAP coords are arbitrary -> hide axes
    return f.update_layout(title=SPACE[space], title_font_size=13, clickmode="event+select",
                           xaxis=axis, yaxis=axis, margin=dict(l=6, r=6, t=26, b=6), dragmode="pan")

def mapfig(i, filter, map_style):
    ct, cl, cb = topk(i, SIM["text"], filter), topk(i, SIM["clip"], filter), topk(i, SIM["cbs"], filter)
 
    f = go.Figure()
    c = np.full(len(df), "rgba(70,70,70,.55)", dtype=object)
    s = np.full(len(df), 6)
    c[cb] = CBS_C; s[cb] = 13; c[cl] = CLIP_C; s[cl] = 13; c[ct] = TEXT_C; s[ct] = 13; c[i] = SEL; s[i] = 20
    highlight = cl + ct + cb + i

    d = df.iloc[highlight]
    f.add_scattermap(lat=d.lat, lon=d.lon, mode='markers', hoverinfo="skip", showlegend=False, 
                     selectedpoints=[], unselected=dict(marker=dict(opacity=1)),
                     marker=dict(color="white", size=s[highlight] + 5))
    d = df.iloc[filter]
    f.add_scattermap(lat=d.lat, lon=d.lon, mode='markers', text=d.name, customdata=filter, 
                     hovertemplate="%{text}<extra></extra>", unselected=dict(marker=dict(opacity=1)),
                     marker=dict(color=c[filter], size=s[filter]), showlegend=False)
    
    # required to add a legend, no points are plotted
    legend_items = [("CBS-similar", CBS_C, 13), ("CLIP-similar", CLIP_C, 13), ("Text-similar", TEXT_C, 13), ("Selected", SEL, 20)]
    for item in legend_items:    
        f.add_scattermap(lat=[None], lon=[None], mode="markers", name=item[0], 
                         showlegend=True, marker=dict(color=item[1], size=item[2]))
    
    # find buurtcodes from df -> use filter to only show outlines of areas in filter_ids
    filter_ids = df["code"].iloc[filter]

    # build style from scratch, easier than copying & editing most properties
    style = {"version": 8, "sources": {"p": {"type": "raster", "tiles": [TILE_TYPES[map_style]["tiles"]], 
                                             "tileSize": 256, "attribution": TILE_TYPES[map_style]["attribution"]}},
            "layers": [{"id": "p", "type": "raster", "source": "p"}]}
    if os.path.exists("assets/buurten.geojson"):  # draw the real CBS buurt boundaries over the aerial photo
        style["sources"]["b"] = {"type": "geojson", "data": "/assets/buurten.geojson"}
        style["layers"] += [  # cased line: dark halo under a bright line so borders read over any imagery
            {"id": "b-case", "type": "line", "source": "b",
            "paint": {"line-color": "#1a1a1a", "line-width": 3, "line-opacity": 0.45,
                    "line-blur": 0.5}, "filter":["in", ["get", "code"], ["literal", filter_ids]]},
            {"id": "b", "type": "line", "source": "b",
            "paint": {"line-color": "#ffd54f", "line-width": 1.4, "line-opacity": 0.9},
            "filter":["in", ["get", "code"], ["literal", filter_ids]]}]
        
        if len(filter) < EXTRA_AREA_HIGHLIGHT:
            style["layers"] += [{"id": "b2", "type": "line", "source": "b",
            "paint": {"line-color": "#30fd29", "line-width": 5, "line-opacity": 0.9},
            "filter":["in", ["get", "code"], ["literal", filter_ids]]}] 
    
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
    columns = [{"name": "Indicator", "id": "field"}] + [{"name": fmt(df["name"][idx]), "id": f"area_{idx}"} for idx in i]
    rows = [{"field": DESC[k], **{f"area_{idx}": fmt(df[k][idx]) for idx in i}} for k in cols]
    return rows, columns

app = Dash(__name__, external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"])
app.layout = html.Div(style={"background": "#f5f6f8", "height": "100vh"}, children=[
    dcc.Store(id="current_selection", data=[]),
    dcc.Store(id="current_columns", data=CBS),
    dcc.Store(id="current_filter", data=df.index.to_list()),
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
                dcc.Button("Filter", id="filter_button", style=BTN)
            ]),
            dcc.Dropdown(id="query_columns", options=QOPTS, value=None,
                         placeholder="Use the dropdown to insert attributes in the query"),
            html.Hr(style={"width": "100%", "border": "none", "borderTop": "1px solid #e3e6ea", "margin": "4px 0"}),
            html.Div("Indicators", style=LABEL),
            dcc.Dropdown(OPTS, CBS, id="column_select", closeOnSelect=False, multi=True, clearable=True,
                         placeholder="At least one column must be selected"),
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
                           "border": "1px solid #4b5563", "borderRadius": 4,
                           "background": "#374151", "color": "white"}),
            dcc.Graph(id="map", style={"height": "100%"}),
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
              State("current_columns", "data"),
              State("map_style", "value"))
def update_figure_selections(i, filter, cols, map_style):
    return scatter("clip", i, filter), scatter("text", i, filter), scatter("cbs", i, filter), \
            mapfig(i, filter, map_style), spider(i, cols), *table(i, cols)

# when selected columns change, update spiderplot and table
@app.callback(Output("spider", "figure", allow_duplicate=True),
              Output("table", "data", allow_duplicate=True),
              Output("table", "columns", allow_duplicate=True),
              Input("current_columns", "data"),
              State("current_selection", "data"),
              prevent_initial_call=True)
def update_figure_columns(cols, i):
    return spider(i, cols), *table(i, cols)

# when map style changes update map
@app.callback(Output("map", "figure", allow_duplicate=True),
              Input("map_style", "value"),
              State("current_selection", "data"),
              State("current_filter", "data"),
              prevent_initial_call=True)
def update_map_style(map_style, i, filter):
    return mapfig(i, filter, map_style)

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

if __name__ == "__main__":
    app.run(debug=True)
