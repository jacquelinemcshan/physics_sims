import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp
from datetime import timedelta

import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, callback, ctx, State
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

masses_in_col_size=3


mass_1_input= dbc.AccordionItem([ 
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
], title= "Mass 1")
         

mass_2_input= dbc.AccordionItem([ 
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
    ])], title="Mass 2")

mass_3_input= dbc.AccordionItem([ 
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
    ])], title="Mass 3")
                  
graph_layout= dbc.Card([dbc.CardHeader(html.H3(children="Graph of the Three Body Problem")),
    dbc.CardBody([
    html.Div(dbc.Button("Generate Graph", className="me-1", id="graph-button", n_clicks=0)),
        html.Div(id="graph_gen")])], class_name="min-vh-90 flex-grow-1"
        )

ui_layout=dbc.Card([dbc.CardHeader(html.H3(children="Variables")),
    dbc.CardBody(dbc.Accordion([mass_1_input,mass_2_input,mass_3_input], always_open=True, class_name="flex-grow-1 overflow-scroll mb-3", style={"maxHeight": "75vh"}))])




app.layout = dbc.Container(
    [
        html.H1("3 Body Gravitional Problem", className="pt-2"),
        dbc.Row(
            [
                dbc.Col(ui_layout, width=4, align="start"),
                dbc.Col(graph_layout, width=7, align="start"),
            ],
            align="center"
        ),
     
     ],
    fluid=True, class_name="mb-3", style={"padding-bottom": "140px"}
   
)
#http://dash-bootstrap-components.opensource.faculty.ai/examples/iris/

G=6.674**(-11)

@app.callback(Output('graph_gen', 'children'),
              [Input( 'm1_mass', 'value'), Input( 'm2_mass', 'value'),Input( 'm3_mass', 'value'),
               Input( 'm1_x_pos', 'value'), Input( 'm1_y_pos', 'value'),Input( 'm1_z_pos', 'value'),
               Input( 'm1_x_vel', 'value'), Input( 'm1_y_vel', 'value'),Input( 'm1_z_vel', 'value'), 
               Input( 'm2_x_pos', 'value'), Input( 'm2_y_pos', 'value'),Input( 'm2_z_pos', 'value'),
               Input( 'm2_x_vel', 'value'), Input( 'm2_y_vel', 'value'),Input( 'm2_z_vel', 'value'),
               Input( 'm3_x_pos', 'value'), Input( 'm3_y_pos', 'value'),Input( 'm3_z_pos', 'value'),
               Input( 'm3_x_vel', 'value'), Input( 'm3_y_vel', 'value'),Input( 'm3_z_vel', 'value'), 
               Input("graph-button", "n_clicks")])
def run_sim(m1_mass, m2_mass, m3_mass, m1_x_pos, m1_y_pos, m1_z_pos,m1_x_vel, m1_y_vel, m1_z_vel,
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
            tmax=5*10**5/2
            dt=0.1
            t = np.arange(0, tmax, dt)
            
            func=solve_ivp(test_eq, [0,tmax], y0, method='DOP853', dense_output=False, t_eval=t)
            return(func)
        
       
        solved_func=grav_body_solver()
        
        step=5000
        
        xa, ya, za=solved_func.y[0][::step], solved_func.y[2][::step], solved_func.y[4][::step]
        xb, yb, zb=solved_func.y[6][::step], solved_func.y[8][::step], solved_func.y[10][::step]
        xc, yc, zc=solved_func.y[12][::step], solved_func.y[14][::step],solved_func.y[16][::step]
        
        time=solved_func.t[::step]
        
        
        x_max=max([max(xa), max(xb), max(xc)])
        x_min=min([min(xa), min(xb), min(xc)])
        y_max=max([max(ya), max(yb), max(yc)])
        y_min=min([min(ya), min(yb), min(yc)])
        z_max=max([max(za), max(zb), max(zc)]) 
        z_min=min([min(za), min(zb), min(zc)])
        time_max=max(time)
        if ctx.triggered_id == "graph-button":
           
           fig = go.Figure(data=[go.Scatter3d(x=xa, y=ya, z=za, 
                                              text = time,
                                              hovertemplate ='<i>x</i>: %{x:.3f} m <br>'+
                                              '<i>y</i>: %{y:.3f} m <br>'+
                                              '<i>z</i>: %{z:.3f} m <br>'+ 
                                              '<i>time</i>: %{text:f} s',
                        mode='markers',  name= "Mass 1", marker=dict(colorscale = 'Purp', 
                            cmin = 1000, color = time, cmax = time_max,
                         colorbar=dict(orientation='v',x=0.8, xref="container", title="Mass 1 <br> Over Time <br> (s)"
    ),
                            size=3), showlegend=False),
                        go.Scatter3d(x=xb, y=yb, z=zb,  
                                       text = time,
                                              hovertemplate ='<i>x</i>: %{x:.3f} m <br>'+
                                              '<i>y</i>: %{y:.3f} m <br>'+
                                              '<i>z</i>: %{z:.3f} m <br>'+ 
                                              '<i>time</i>: %{text:f} s',
                        mode='markers', name= "Mass 2", marker=dict(colorscale = 'Burg', 
                                cmin = 1000, color = time, cmax = time_max,
                        colorbar=dict(orientation='v',x=0.9, xref="container", title="Mass 2 <br> Over Time <br> (s)"
    ),
                                                              size=3), showlegend=False),
           go.Scatter3d(x=xc, y=yc, z=zc, 
                        text = time,
                          hovertemplate ='<i>x</i>: %{x:.3f} m <br>'+
                                              '<i>y</i>: %{y:.3f} m <br>'+
                                              '<i>z</i>: %{z:.3f} m <br>'+ 
                                              '<i>time</i>: %{text:f} s',
                                              
                  mode='markers', name= "Mass 3", marker=dict(colorscale = 'Tealgrn', 
                            cmin = 1000, color = time, cmax = time_max,
                    colorbar=dict(orientation='v',x=1, xref="container", title="Mass 3 <br> Over Time <br> (s)"
    ),        
                            size=3), showlegend=False)])
           
           frames=[go.Frame(data=[go.Scatter3d(x=[xa[k]], y=[ya[k]], z=[za[k]],
                                              hovertemplate ='<i>x</i>: %{x:.3f} m <br>'+
                                              '<i>y</i>: %{y:.3f} m <br>'+
                                              '<i>z</i>: %{z:.3f} m <br>',
                    mode='markers', marker=dict(size=5, colorscale = 'Purp', 
                            cmin = 1000, color = [time[k]], cmax =time_max, line=dict(width=2,
                                        color='black')),
                    name='Mass 1'), 
                    
                    go.Scatter3d(x=[xb[k]], y=[yb[k]], z=[zb[k]],
                                   hovertemplate ='<i>x</i>: %{x:.3f} m <br>'+ 
                                     '<i>y</i>: %{y:.3f} m <br>'+
                                     '<i>z</i>: %{z:.3f} m <br>',
                    mode='markers', marker=dict(size=5,colorscale = 'Burg', 
                            cmin = 1000, color = [time[k]], cmax = time_max, line=dict(width=2,
                                        color='black')),
                    name='Mass 2'), 
                    go.Scatter3d(x=[xc[k]], y=[yc[k]], z=[zc[k]],
                                   hovertemplate ='<i>x</i>: %{x:.3f} m <br>'+
                                              '<i>y</i>: %{y:.3f} m <br>'+
                                              '<i>z</i>: %{z:.3f} m <br>',
                    mode='markers', marker=dict(size=5,
                    colorscale = 'Tealgrn', 
                            cmin = 1000, color = [time[k]], cmax = time_max, line=dict(width=2,
                                        color='black')), name='Mass 3')],
                                                    name=str(k))
                                                    for k in range(0,len(time),5)]
           fig.update(frames=frames)
           def frame_args(duration):
                  return {
                       "frame": {"duration": duration},
                         "mode": "immediate",
                           "fromcurrent": True,
                           "transition": {"duration": duration, "easing": "linear"},
                           }
           sliders = [
                    {
                        "pad": {"b": 10, "t": 30},
                        "len": 0.9,
                         "x": 0.1,
                        "y": 0,
                        "steps": [
                        {
                                "args": [[f.name], frame_args(5)],
                                "label": "{} s".format(time[k]),
                                "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                         ],
                     }
                    ]

           fig.update_layout(title='Position Graph of the Three Masses', title_font_size=23, title_x=0.3,title_xanchor='center', 
            title_y=0.87, title_yanchor='bottom',
                     width=900,
                     height=800,
                     autosize=True,
                     scene=dict(
                          aspectmode='cube',
                          xaxis_title='x position (m)',
                          yaxis_title='y position (m)',
                              zaxis_title='z position (m)',
                              xaxis=dict(range=[x_min-1, x_max+1], autorange=False),
                               yaxis=dict(range=[y_min-1, y_max+1], autorange=False),
                               zaxis=dict(range=[z_min-1, z_max+1], autorange=False)),
                              updatemenus = [
                    {
                     "buttons": [
                        {
                         "args": [None, frame_args(0)],
                         "label": "&#9654;", # play symbol
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
                sliders=sliders)
           
           fig.show()
           return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run(debug=True)
   
