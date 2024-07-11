# importing the packages
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
import plotly.io as pio
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.graph_objs._figure import Figure
from dash.exceptions import PreventUpdate


# initializing the app
app = Dash(
    name=__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# load and transform data
## data
df = pd.read_csv('globalterrorismdb_0718dist_cleaned.csv', encoding='ansi')
df.city = df.city.fillna("Unknown")
## years
YEARS = df['year'].unique()
YEARS.sort()

## app setting
### ui
SIDEBAR_WIDTH = 0.2
SIDEBAR_WIDTH_TXT = f"{SIDEBAR_WIDTH*100:.0f}%"
MAINBODY_WIDTH = 1 - SIDEBAR_WIDTH
MAINBODY_WIDTH_TXT = f"{MAINBODY_WIDTH*100:.0f}%"
pio.templates.default = 'plotly_white'
def convert_hex_to_rgba(hex_color, transparency=0.5):
    return f'rgba({int(hex_color[1:3], 16)}, {int(hex_color[3:5], 16)}, {int(hex_color[5:7], 16)}, {transparency})'


## data
region = df.region.unique().astype(str)
country = df.country.unique().astype(str)
city = df.city.unique().astype(str)
region.sort()
country.sort()
city.sort()

def make_plot(df):
    attack = df[['year', 'casualty', 'success']].groupby(
        'year'
    ).agg(
        {
            'year':'count',
            'casualty': ['sum', 'mean'],
            'success': 'mean'
        }
    )
    attack['success'] *= 100
    # flatten columns
    attack.columns = 'number_attacks number_casualties average_casualties success_rate'.split()
    # # Making the figure with area plots
    line_attack_and_ratio = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[5, 3, 3], vertical_spacing=0.02)
    # Attacks and Casualties
    fig_att_cas = px.area(
        data_frame=attack,
        x=attack.index,
        y=['number_attacks', 'number_casualties']
    )
    
    colors = ['#636EFA', '#EF553B']  # Assigning colors manually
    for i, d in enumerate(fig_att_cas.data):
        d.name = ['Terrorist Attack', 'Casualties'][i]  # Setting trace names
        d.line.color = colors[i]  # Setting line color
        d.fillcolor = convert_hex_to_rgba(colors[i])  # Adding transparency to fill color
        line_attack_and_ratio.add_trace(go.Scatter(x=d.x, y=d.y, mode='lines', name=d.name, fill='tozeroy', line=dict(color=d.line.color), fillcolor=d.fillcolor), row=1, col=1)
    
    # Average casualties per attack
    fig_cas_per_att = px.area(
        data_frame=attack,
        x=attack.index,
        y='average_casualties'
    )
    fig_cas_per_att.data[0].name = 'Average casualties per terrorist attack'
    fig_cas_per_att.data[0].line.color = '#FFA15A'  # Setting line color
    fig_cas_per_att.data[0].fillcolor = convert_hex_to_rgba(fig_cas_per_att.data[0].line.color)  # Adding transparency to fill color
    line_attack_and_ratio.add_trace(go.Scatter(x=fig_cas_per_att.data[0].x, y=fig_cas_per_att.data[0].y, mode='lines', name=fig_cas_per_att.data[0].name, fill='tozeroy', line=dict(color=fig_cas_per_att.data[0].line.color), fillcolor=fig_cas_per_att.data[0].fillcolor), row=2, col=1)
    
    # Success rate
    fig_success = px.area(data_frame=attack, x=attack.index, y='success_rate')
    fig_success.data[0].name = 'Success rate'
    fig_success.data[0].line.color = '#FF6692'  # Setting line color
    fig_success.data[0].fillcolor = convert_hex_to_rgba(fig_success.data[0].line.color)  # Adding transparency to fill color
    line_attack_and_ratio.add_trace(go.Scatter(x=fig_success.data[0].x, y=fig_success.data[0].y, mode='lines', name=fig_success.data[0].name, fill='tozeroy', line=dict(color=fig_success.data[0].line.color), fillcolor=fig_success.data[0].fillcolor), row=3, col=1)
    
    # Updating axes and layout
    line_attack_and_ratio.update_yaxes(title_text="Count", row=1, col=1)
    line_attack_and_ratio.update_yaxes(title_text="Ratio", row=2, col=1)
    line_attack_and_ratio.update_yaxes(title_text="Success rate", row=3, col=1)
    line_attack_and_ratio.update_xaxes(title_text="Year", row=3, col=1)
    line_attack_and_ratio.update_layout(
        title_text="Terrorist Attacks and Casualties Across Time",
        # width=1150,
        height=500,
        showlegend=True  # Ensure legends are shown
    )
    line_attack_and_ratio.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.5,
        xanchor="left",
        x=0
    ))
    return line_attack_and_ratio

## app layout
app.layout = html.Div(
    id="page",
    className="container-fluid bg-dark text-light",
    style={
        "padding": "75px",
    },
    children=[
        html.Div(
            id="title",
            children=[html.H1(children="An EDA tool for Global Terrorism")],
        ),
        html.Div(
            id="header",
            className="row",
            children=[
                html.Div(
                    className="col",
                    children=[
                        html.Div(
                            className="row text-dark",
                            children=[
                                html.Div(
                                    className="container-fuild",
                                    children=[
                                        html.H3("Region", className="text-light"),
                                        dcc.Dropdown(
                                            options=region, 
                                            placeholder="Select a Region",
                                            id='region',
                                            multi=True
                                        ),
                                    ]
                                ), # put the dropdowns here
                                html.Div(
                                    className="col-md-4",
                                    children=[
                                        html.H3("Country", className="text-light"),
                                        dcc.Dropdown(
                                            options=country, 
                                            placeholder="Select a Country",
                                            id='country',
                                            multi=True
                                        ),
                                    ]
                                ),
                                html.Div(
                                    className="col-md-4",
                                    children=[
                                        html.H3("City", className="text-light"),
                                        dcc.Dropdown(
                                            options=city, 
                                            placeholder="Select a City",
                                            id='city',
                                            multi=True
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ]
                )
            ]
        ),
        html.Div(
            className="col-md-3",
            style={"margin-top":"5px"},
            children=[
                html.Button(
                    "Select All Regions",
                    id="select-all-region",
                    className="btn btn-primary",
                    n_clicks=0
                ), 
            ]
        ), 
        html.Div(
            id="row1",
            className="row",
            style={
                "margin-top": "20px",
                "padding-bottom": "20px",
            },
            children=[
                html.Div(
                    id="left_col_row1",
                    className="col-md-7",
                    children=[
                        html.H3("Evolution of terrorists attacks over time"),
                        html.P(children="The following graph show the trend of the number of terrorists attack, casualities and success rate."),
                        dcc.Graph(id='trend', figure=make_plot(df))
                    ]
                ),
                html.Div(
                    id="mid_col_row1",
                    className="col-md-1",
                    style={
                        "width": "2%",
                    },
                ),
                html.Div(
                    id="right_col_row1",
                    className="col",
                    children=[
                        html.H3("Map Projection of Terrorist Attacks"),
                        html.P(
                            id="slider-text",
                            children=f"Use the slider to select the Year: {min(YEARS)}.",
                        ),
                        dcc.Slider(
                            id="year-slider",
                            min=min(YEARS),
                            max=max(YEARS),
                            value=min(YEARS),
                            step=1,
                            marks={
                                str(y): {"label": str(y)} for y in YEARS[::5]
                            }
                        ),
                        dcc.Graph(id="map"),
                        html.Br(),
                    ]
                ),
            ]
        ),
        html.Div(
            id="row2",
            className="row",
            children=[
                html.Div(
                    id="left_col_row2",
                    className="col-md-5",
                    children=[
                        html.H3("Terrorist Attacks per Country"),
                        html.P("Hierarchical representation of the number of terrorist attacks."),
                        dcc.Graph(id="sunburst")
                    ]
                ),
                html.Div(
                    id="mid_col_row2",
                    className="col-md-1",
                    style={
                        "width": "2%",
                    },
                ),
                html.Div(
                    id="right_col_row2",
                    className="col",
                    children=[
                        html.H3("Terrorist Attacks & Casualty per City*"),
                        html.P("*Displaying top 15 cities in selected regions/countries"),
                        html.Div(
                            className="row",
                            children=[
                                html.Div(className="col-6", children=[dcc.Graph(id="bar_city")]),
                                html.Div(className="col-6", children=[dcc.Graph(id="bar_city_success")])
                            ]
                        ),
                        html.Br()
                    ]
                ),
            ]
        ),
    ]
)

@callback(
    Output("bar_city", "figure"),
    Output("bar_city_success", "figure"),
    Input("region", "value"),
    Input("country", "value"),
    Input("city", "value"),
)
def plot_bar_city(region, country, city):
    if not (region or country or city):
        tmp_df = df
    else:
            if region:
                tmp_df = df[df['region'].isin(region)]
            if country:
                tmp_df = df[df['country'].isin(country)]
            if city:
                tmp_df = df[df['city'].isin(city)]
    top_cities = tmp_df['city'].value_counts(ascending=True).reset_index()
    top_n_cities = top_cities.nlargest(n=15, columns='count')
    city_order = top_n_cities['city']
    top_n_cities['city'] = pd.Categorical(top_n_cities['city'], categories=city_order, ordered=True)
    # bar plot of top cities
    bar_city = px.bar(
        data_frame=top_n_cities,
        y='city',
        x='count',
        labels={
            'city': 'City',
            'count': 'Count',
        }
    )
    # casualty
    success = tmp_df[['city', 'casualty']].groupby(
        'city'
    ).mean()
    success = success.reset_index()
    success['city'] = pd.Categorical(success['city'], categories=city_order, ordered=True)
    success = success.sort_values('city', ascending=True)
    bar_city_success = px.bar(
        data_frame=success,
        y='city',
        x='casualty',
        labels={
            'city': 'City',
            'casualty': 'Avg. Casualty'
        },
        log_x=True
    )
    return bar_city, bar_city_success


@callback(
    Output("country", "options"),
    Input("region", "value")
)
def update_countries(selected_values):
    return sorted(df[
        df['region'].isin(selected_values)
    ]["country"].unique())

@callback(
    Output("city", "options"),
    Input("country", "value")
)
def update_cities(selected_values):
    if not selected_values:
        return city
    return sorted(df[
        df['country'].isin(selected_values)
    ]["city"].unique())

@callback(
    Output("region", "value"),
    Input("select-all-region", "n_clicks")
)
def select_all_region(sel_region):
    if not sel_region: # if no button is cliked
        raise PreventUpdate
    return region

@callback(
    Output("trend", "figure"),
    Input("region", "value"),
    Input("country", "value"),
    Input("city", "value")
)
def update_lineplot(region, country, city):
    if not (region or country or city):
        tmp_df = df
    else:
            if region:
                tmp_df = df[df['region'].isin(region)]
            if country:
                tmp_df = df[df['country'].isin(country)]
            if city:
                tmp_df = df[df['city'].isin(city)]
    return make_plot(tmp_df)

@callback(
    Output("slider-text", "children"),
    Output("map", "figure"),
    Input("year-slider", "value")
)
def plot_map(year):
    txt = f"Use the slider to select the Year: {year}"
    df_geo = df.dropna(subset='casualty').query(f"year=={year}")
    df_geo['sqrt_casualty'] = np.sqrt(df_geo.casualty)
    geo = px.scatter_geo(
        data_frame=df_geo,
        lon='longitude',
        lat='latitude',
        size='sqrt_casualty',
        color='casualty',
        hover_name='city',
        projection="orthographic"
    )
    return txt, geo

@callback(
    Output("sunburst", "figure"),
    Input("region", "value"),
    Input("country", "value")
)
def plot_sunburst(region, country):
    if not (region or country):
        tmp_df = df
    else:
            if region:
                tmp_df = df[df['region'].isin(region)]
            if country:
                tmp_df = df[df['country'].isin(country)]
    sun_region_country = px.sunburst(
        data_frame=tmp_df['region country'.split()].dropna(),  
        path='region country'.split()
    )
    return sun_region_country

if __name__ == "__main__":
    app.run_server(debug=True)