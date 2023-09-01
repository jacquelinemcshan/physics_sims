import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, callback, ctx
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

masses_in_col_size=3


mass_1_input= html.Div([html.H3(children='Mass 1'), 
    html.Div([
        dbc.Row(html.H4(children="Mass (kg)")),
        dbc.Row(dcc.Input(id='m1_mass', type='number', min=1, max=500, step=0.1, value=60))]),
    html.Div([
        dbc.Row(html.H4(children="Initial Position (m)")),
        dbc.Row([
            dbc.Col(dbc.Label("x direction"), width=masses_in_col_size),
            dbc.Col(dbc.Label("y direction"), width=masses_in_col_size),
            dbc.Col(dbc.Label("z direction"), width=masses_in_col_size)
            ]),
        dbc.Row([
            dbc.Col(dcc.Input(id='m1_x_pos', type='number', min=-50, max=50, step=0.01, value=5), width=masses_in_col_size),
            dbc.Col(dcc.Input(id='m1_y_pos', type='number', min=-50, max=50, step=0.01, value=-4), width=masses_in_col_size),
            dbc.Col(dcc.Input(id='m1_z_pos', type='number', min=-50, max=50, step=0.01, value=0), width=masses_in_col_size)
            ])
    ]),
    html.Div([
        dbc.Row(html.H4(children="Initial Velocity (m/s)")),
        dbc.Row([
            dbc.Col(dbc.Label("x direction"), width=masses_in_col_size),
            dbc.Col(dbc.Label("y direction"), width=masses_in_col_size),
            dbc.Col(dbc.Label("z direction"), width=masses_in_col_size)
            ]),
    dbc.Row([
        dbc.Col(dcc.Input(id='m1_x_vel', type='number', min=-5, max=5, step=0.000001, value=-0.00001),width=masses_in_col_size),
        dbc.Col(dcc.Input(id='m1_y_vel', type='number', min=-5, max=5, step=0.000001, value=0.00001), width=masses_in_col_size),
        dbc.Col(dcc.Input(id='m1_z_vel', type='number', min=-5, max=5, step=0.000001, value=0.00007), width=masses_in_col_size),
        ])
    ])
])
         

mass_2_input= html.Div([
    html.H3(children='Mass 2'), 
    html.Div([
        dbc.Row(html.H4(children="Mass (kg)")),
        dbc.Row(dcc.Input(id='m2_mass', type='number', min=1, max=500, step=0.1, value=300))
        ]),
    html.Div([
        dbc.Row(html.H4(children="Initial Position (m)")),
        dbc.Row([
            dbc.Col(dbc.Label("x direction"), width=masses_in_col_size),
            dbc.Col(dbc.Label("y direction"), width=masses_in_col_size),
            dbc.Col(dbc.Label("z direction"), width=masses_in_col_size)
                ])
            ]),
dbc.Row([
    dbc.Col(dcc.Input(id='m2_x_pos', type='number', min=-50, max=50, step=0.01, value=-3), width=masses_in_col_size),
    dbc.Col(dcc.Input(id='m2_y_pos', type='number', min=-50, max=50, step=0.01, value=7), width=masses_in_col_size),
    dbc.Col(dcc.Input(id='m2_z_pos', type='number', min=-50, max=50, step=0.01, value=-3), width=masses_in_col_size),
    ]),
dbc.Row(html.H4(children="Initial Velocity (m/s)")),
dbc.Row([
    dbc.Col(dbc.Label("x direction"), width=masses_in_col_size),
    dbc.Col(dbc.Label("y direction"), width=masses_in_col_size),
    dbc.Col(dbc.Label("z direction"), width=masses_in_col_size)
    ]),
dbc.Row([
    dbc.Col(dcc.Input(id='m2_x_vel', type='number', min=-5, max=5, step=0.000001, value=0), width=masses_in_col_size),
    dbc.Col(dcc.Input(id='m2_y_vel', type='number', min=-5, max=5, step=0.000001, value=0), width=masses_in_col_size),
    dbc.Col(dcc.Input(id='m2_z_vel', type='number', min=-5, max=5, step=0.000001, value=0), width=masses_in_col_size),
    ])])

mass_3_input= html.Div([html.H3(children='Mass 3'), 
dbc.Row(html.H4(children="Mass (kg)")),
dbc.Row(dcc.Input(id='m3_mass', type='number', min=1, max=500, step=0.1, value=80)),
dbc.Row(html.H4(children="Initial Position (m)")),
dbc.Row([
    dbc.Col(dbc.Label("x direction"), width=masses_in_col_size),
    dbc.Col(dbc.Label("y direction"), width=masses_in_col_size),
    dbc.Col(dbc.Label("z direction"), width=masses_in_col_size)
    ]),
dbc.Row([
    dbc.Col(dcc.Input(id='m3_x_pos', type='number', min=-50, max=50, step=0.01, value=-10), width=masses_in_col_size),
    dbc.Col(dcc.Input(id='m3_y_pos', type='number', min=-50, max=50, step=0.01, value=9), width=masses_in_col_size),
    dbc.Col(dcc.Input(id='m3_z_pos', type='number', min=-50, max=50, step=0.01, value=-5), width=masses_in_col_size),
    ]),
dbc.Row(html.H4(children="Initial Velocity (m/s)")),
dbc.Row([
    dbc.Col(dbc.Label("x direction"), width=masses_in_col_size),
    dbc.Col(dbc.Label("y direction"), width=masses_in_col_size),
    dbc.Col(dbc.Label("z direction"), width=masses_in_col_size)
    ]),
dbc.Row([
    dbc.Col(dcc.Input(id='m3_x_vel', type='number', min=-5, max=5, step=0.000001, value=-0.00009),width=masses_in_col_size),
    dbc.Col(dcc.Input(id='m3_y_vel', type='number', min=-5, max=5, step=0.000001, value=0.00001), width=masses_in_col_size),
    dbc.Col(dcc.Input(id='m3_z_vel', type='number', min=-5, max=5, step=0.000001, value=0), width=masses_in_col_size),
    ])])
                  
graph_layout=html.Div([
    html.H3(children='Graph:'),
        html.Button("Generate Graph", id="graph-button", n_clicks=0),
        html.Div(id="user_inputs"),
        html.Div(id="pendulum-graph")
        ]
        )


ui_layout=dbc.Card(
    dbc.CardBody([mass_1_input,mass_2_input,mass_3_input]))




app.layout = dbc.Container(
    [
        html.H1("3 Body Gravitional Problem"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(ui_layout, width=4),
                dbc.Col(graph_layout, width=8),
            ],
            align="center",
        ),
    ],
    fluid=True,
)
#http://dash-bootstrap-components.opensource.faculty.ai/examples/iris/

G=6.674**(-11)

if __name__ == '__main__':
    app.run(debug=True)