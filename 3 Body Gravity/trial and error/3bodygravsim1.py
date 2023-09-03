import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp
from datetime import timedelta

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

@app.callback(Output('user_inputs', 'children'),
              [Input( 'm1_mass', 'value'), Input( 'm2_mass', 'value'),Input( 'm3_mass', 'value'),
               Input( 'm1_x_pos', 'value'), Input( 'm1_y_pos', 'value'),Input( 'm1_z_pos', 'value'),
               Input( 'm1_x_vel', 'value'), Input( 'm1_y_vel', 'value'),Input( 'm1_z_vel', 'value'), 
               Input( 'm2_x_pos', 'value'), Input( 'm2_y_pos', 'value'),Input( 'm2_z_pos', 'value'),
               Input( 'm2_x_vel', 'value'), Input( 'm2_y_vel', 'value'),Input( 'm2_z_vel', 'value'),
               Input( 'm3_x_pos', 'value'), Input( 'm3_y_pos', 'value'),Input( 'm3_z_pos', 'value'),
               Input( 'm3_x_vel', 'value'), Input( 'm3_y_vel', 'value'),Input( 'm3_z_vel', 'value'), 
               Input("graph-button", "n_clicks")])
def fetch_data_from_user_input(m1_mass, m2_mass, m3_mass, m1_x_pos, m1_y_pos, m1_z_pos,m1_x_vel, m1_y_vel, m1_z_vel,
                              m2_x_pos, m2_y_pos, m2_z_pos,m2_x_vel, m2_y_vel, m2_z_vel, 
                              m3_x_pos, m3_y_pos, m3_z_pos,m3_x_vel, m3_y_vel, m3_z_vel, n):
        
        ma,mb,mc=m1_mass, m2_mass, m3_mass

        xa_ini, ya_ini, za_ini= m1_x_pos, m1_y_pos, m1_z_pos
        v_xa_ini, v_ya_ini, v_za_ini=m1_x_vel, m1_y_vel, m1_z_vel
        
        xb_ini, yb_ini, zb_ini=m2_x_pos, m2_y_pos, m2_z_pos
        v_xb_ini, v_yb_ini, v_zb_ini=m2_x_vel, m2_y_vel, m2_z_vel
        
        xc_ini, yc_ini, zc_ini=m3_x_pos, m3_y_pos, m3_z_pos,
        v_xc_ini, v_yc_ini, v_zc_ini=m3_x_vel, m3_y_vel, m3_z_vel

        def test_eq(t, y):
             xa, rhoxa,ya, rhoya, za, rhoza, xb, rhoxb,yb, rhoyb, zb, rhozb, xc, rhoxc, yc, rhoyc, zc, rhozc =y
             
             rab=np.sqrt((xa-xb)**2+(ya-yb)**2+(za-zb)**2)
             rac= np.sqrt((xa-xc)**2+(ya-yc)**2+(za-zc)**2)
            
             rba=np.sqrt((xb-xa)**2+(yb-ya)**2+(zb-za)**2)
             rbc=np.sqrt((xb-xc)**2+(yb-yc)**2+(zb-zc)**2)
             
             rca=np.sqrt((xc-xa)**2+(yc-ya)**2+(zc-za)**2)
             rcb=np.sqrt((xc-xb)**2+(yc-yb)**2+(zc-zb)**2) 
             
             Mat=np.identity(9)
             invMat=inv(Mat)  
             
             f1, f2 = [-G*((mb*(xa-xb)/rab**3)+(mc*(xa-xc)/rac**3))],[-G*((mb*(ya-yb)/rab**3)+(mc*(ya-yc)/rac**3))]
             f3 = [-G*((mb*(za-zb)/rab**3)+(mc*(za-zc)/rac**3))]
             
             f4, f5 = [-G*((ma*(xb-xa)/rba**3)+(mc*(xb-xc)/rbc**3))],[-G*((mb*(yb-ya)/rba**3)+(mc*(yb-yc)/rbc**3))]
             f6 = [-G*((mb*(zb-za)/rba**3)+(mc*(zb-zc)/rbc**3))]
             
             f7, f8 = [-G*((ma*(xc-xa)/rca**3)+(mb*(xc-xb)/rcb**3))],[-G*((ma*(yc-ya)/rca**3)+(mb*(yc-yb)/rcb**3))]
             f9 = [-G*((ma*(zc-za)/rca**3)+(mb*(zc-zb)/rcb**3))]
             
             f=np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9]) 
             
             invMatf=invMat.dot(f)
             
             dxa,dya,dza=rhoxa,rhoya,rhoza
             drhoxa,drhoya,drhoza=invMatf[0], invMatf[1], invMatf[2] 
             
             dxb,dyb,dzb=rhoxb,rhoyb,rhozb 
             drhoxb,drhoyb,drhozb=invMatf[3], invMatf[4], invMatf[5] 
             
             dxc,dyc,dzc=rhoxc,rhoyc,rhozc 
             drhoxc,drhoyc,drhozc=invMatf[6], invMatf[7], invMatf[8]
             
             return (dxa, drhoxa, dya, drhoya, dza, drhoza,  dxb, drhoxb, dyb, drhoyb, dzb, drhozb, 
            dxc, drhoxc, dyc, drhoyc, dzc, drhozc)
        
        def grav_body_solver(*args):
            y0=[xa_ini,v_xa_ini, ya_ini,v_ya_ini, za_ini, v_za_ini,xb_ini,v_xb_ini, yb_ini,v_yb_ini, zb_ini, v_zb_ini,
                xc_ini,v_xc_ini, yc_ini,v_yc_ini, zc_ini, v_zc_ini]
            tmax=5*10**5
            dt=0.1
            t = np.arange(0, tmax, dt)
            
            func=solve_ivp(test_eq, [0,tmax], y0, method='DOP853', dense_output=False, t_eval=t)
            return(func)
        
       
        solved_func=grav_body_solver()
        
        step=500
        
        xa, ya, za=solved_func.y[0][::step], solved_func.y[2][::step], solved_func.y[4][::step]
        xb, yb, zb=solved_func.y[6][::step], solved_func.y[8][::step], solved_func.y[10][::step]
        xc, yc, zc=solved_func.y[12][::step], solved_func.y[14][::step],solved_func.y[16][::step]
        
        time=solved_func.t[::step]

        if ctx.triggered_id == "graph-button":
             fig = fig = go.Figure(frames=[go.Frame(data=[go.Scatter3d(x=[xa[k]], y=[ya[k]], z=[za[k]],
                    mode='markers', marker=dict(size=3),
                    name='mass 1'), go.Scatter3d(x=[xb[k]], y=[yb[k]], z=[zb[k]],
                    mode='markers', marker=dict(size=3),
                    name='mass 2'), go.Scatter3d(x=[xc[k]], y=[yc[k]], z=[zc[k]],
                    mode='markers', marker=dict(size=3), name='mass 3')],
                                                    name=str(k))
                                                    for k in range(len(time))])
                                                    
             fig.add_trace(go.Scatter3d(x=xa, y=ya, z=za,
                    mode='markers', marker=dict(size=3), name='mass 1'))
             
             fig.add_trace( go.Scatter3d(x=xb, y=yb, z=zb,
                    mode='markers', marker=dict(size=3),
                    name='mass 2'))
             
             fig.add_trace(go.Scatter3d(x=xc, y=yc, z=zc,
                    mode='markers', marker=dict(size=3), name='mass 3'))
             
             def frame_args(duration):
                  return {
                       "frame": {"duration": duration},
                         "mode": "immediate",
                           "fromcurrent": True,
                           "transition": {"duration": duration, "easing": "linear"},
                           }
             sliders = [
                  {
                        "pad": {"b": 10, "t": 60},
                        "len": 0.9,
                         "x": 0.1,
                        "y": 0,
                        "steps": [
                        {
                                "args": [[f.name], frame_args(0)],
                                "label": "{:0>8}".format(str(timedelta(seconds=time[k]))),
                                "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
                ]

             fig.update_layout(
                   title='3 Body Gravitational Problem',
                     width=600,
                     height=600,
                     scene=dict(
                          aspectmode='cube'),
                    updatemenus = [
                    {
                     "buttons": [
                        {
                         "args": [None, frame_args(0)],
                         "label": "&#9654;", # play symbol
                         "method": "animate",
                        },
                        {
                         "args": [[None], frame_args(0)],
                         "label": "&#9724;", # pause symbol
                         "method": "animate",
                        },
                     ],
                     "direction": "left",
                     "pad": {"r": 10, "t": 70},
                     "type": "buttons",
                     "x": 0.1,
                     "y": 0,
                    }
                 ],
                sliders=sliders
                )

             fig.show()
             return dcc.Graph(figure=fig)




    

if __name__ == '__main__':
    app.run(debug=True)