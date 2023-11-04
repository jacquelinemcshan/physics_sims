import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp

import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, callback, ctx
import dash_bootstrap_components as dbc

g=9.8
time_step=5 #for animation frames

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

g=9.8
time_step=5 #for animation frames

mass_1_input= dbc.AccordionItem([ 
    html.Div([
        dbc.Row(html.H4(children="Mass")),
        dbc.Row(dcc.Slider(value=8,min=0, max=24, step=0.1, id='mass1', 
                                    marks={i: '{} kg'.format(i) for i in range(0,24,4)},
                 tooltip={"placement": "top", "always_visible": True}))]),
    html.Div([
        dbc.Row(html.H4(children="String Length")),
        dbc.Row([dcc.Slider( id="length1", value=16, min=0, max=24, step=0.1, 
                                    marks={i: '{} m'.format(i) for i in range(0,24,4)}, 
                                    tooltip={"placement": "top", "always_visible": True}
                            )
            ]),
    ]),
    html.Div([
        dbc.Row(html.H4(children="Starting Angle")),
        dbc.Row([dcc.Slider(value=np.rad2deg(4*np.pi/9), min=np.rad2deg(-np.pi), max=np.rad2deg(np.pi), step=0.01, 
                    marks={np.rad2deg(-np.pi): '-180°',np.rad2deg(-2*np.pi/3): '-120°', np.rad2deg(-np.pi/3): '-60°',
                      0: '0°',
                     np.rad2deg(np.pi/3):'60°', np.rad2deg(2*np.pi/3):'120°'},  
                    tooltip={"placement": "top", "always_visible": True},id='ini1')]),
    ])
], title= "Mass 1")
         
mass_2_input= dbc.AccordionItem([ 
    html.Div([
        dbc.Row(html.H4(children="Mass")),
        dbc.Row(dcc.Slider(value=6,min=0, max=24, step=0.1, id='mass2', 
                                    marks={i: '{} kg'.format(i) for i in range(0,24,4)},
                 tooltip={"placement": "top", "always_visible": True}))]),
    html.Div([
        dbc.Row(html.H4(children="String Length")),
        dbc.Row([dcc.Slider( id="length2", value=12, min=0, max=24, step=0.1, 
                                    marks={i: '{} m'.format(i) for i in range(0,24,4)}, 
                                    tooltip={"placement": "top", "always_visible": True}
                            )
            ]),
    ]),
    html.Div([
        dbc.Row(html.H4(children="Starting Angle")),
        dbc.Row([dcc.Slider(value=np.rad2deg(3*np.pi/10), min=np.rad2deg(-np.pi), max=np.rad2deg(np.pi), step=0.01, 
                    marks={np.rad2deg(-np.pi): '-180°',np.rad2deg(-2*np.pi/3): '-120°', np.rad2deg(-np.pi/3): '-60°',
                      0: '0°',
                     np.rad2deg(np.pi/3):'60°', np.rad2deg(2*np.pi/3):'120°'},  
                    tooltip={"placement": "top", "always_visible": True},id='ini2')]),
    ])
], title= "Mass 2")


mass_3_input= dbc.AccordionItem([ 
    html.Div([
        dbc.Row(html.H4(children="Mass")),
        dbc.Row(dcc.Slider(value=4,min=0, max=24, step=0.1, id='mass3', 
                                    marks={i: '{} kg'.format(i) for i in range(0,24,4)},
                 tooltip={"placement": "top", "always_visible": True}))]),
    html.Div([
        dbc.Row(html.H4(children="String Length")),
        dbc.Row([dcc.Slider( id="length3", value=6, min=0, max=24, step=0.1, 
                                    marks={i: '{} m'.format(i) for i in range(0,24,4)}, 
                                    tooltip={"placement": "top", "always_visible": True}
                            )
            ]),
    ]),
    html.Div([
        dbc.Row(html.H4(children="Starting Angle")),
        dbc.Row([dcc.Slider(value=np.rad2deg(4*np.pi/9), min=np.rad2deg(-np.pi), max=np.rad2deg(np.pi), step=0.01, 
                    marks={np.rad2deg(-np.pi): '-180°',np.rad2deg(-2*np.pi/3): '-120°', np.rad2deg(-np.pi/3): '-60°',
                      0: '0°',
                     np.rad2deg(np.pi/3):'60°', np.rad2deg(2*np.pi/3):'120°'},  
                    tooltip={"placement": "top", "always_visible": True},id='ini3')]),
    ])
], title= "Mass 3")

                  
graph_layout= dbc.Card([dbc.CardHeader(html.H3(children="Graph of the Triple Pendulum")),
    dbc.CardBody([
    html.Div(
        dbc.Stack([dbc.Button("Calculate", className="me-1", id="calc-button-pend", n_clicks=0),
                   dbc.Button("Graph", className="me-1", id="graph-button-pend", n_clicks=0),
                   dbc.Button("Explanation and Code", className="me-1 mx-auto", id="code-button", href="https://github.com/jacquelinemcshan/physics_sims/wiki/Triple-Pendulum", external_link=True)], 
                   direction="horizontal", gap=2)),
        html.Div(id="graph_gen_pend")], )], class_name="min-vh-90 flex-grow-1", 
        )
ui_layout=dbc.Card([dbc.CardHeader(html.H3(children="Variables")),
    dbc.CardBody(dbc.Accordion([mass_1_input,mass_2_input,mass_3_input], always_open=True, class_name="flex-grow-1 overflow-scroll mb-3", style={"maxHeight": "75vh"}))])


app.layout = dbc.Container(
    [ dcc.Store(id='memory_pend'),
        html.H1('Triple Pendulum', className="pt-2"),
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


@callback(Output('memory_pend', 'data'),
              [Input( 'mass1', 'value'), Input( 'mass2', 'value'),Input( 'mass3', 'value'),
               Input( 'length1', 'value'), Input( 'length2', 'value'),Input( 'length3', 'value'),
               Input( 'ini1', 'value'), Input( 'ini2', 'value'),Input( 'ini3', 'value'), 
               Input("calc-button-pend", "n_clicks")])

def fetch_data_from_user_input(input_value, input_value2, input_value3, input_value4,
                               input_value5, input_value6, input_value7, input_value8, input_value9,n):
    
    if ctx.triggered_id == "calc-button-pend":
    
    
        m1,m2,m3=input_value,input_value2, input_value3
        L1,L2,L3=input_value4,input_value5, input_value6
        ini1,ini2,ini3=np.deg2rad(input_value7),np.deg2rad(input_value8), np.deg2rad(input_value9)
        M1=m1+m2+m3
        M2=m2+m3
        L=L1+L2+L3

    
    
        def der_eq(t, y):
            theta1, omega1, theta2, omega2, theta3, omega3=y

            c1, c2, c3 = np.cos(theta1-theta2), np.cos(theta2-theta3), np.cos(theta1-theta3)
            s1, s2, s3 = np.sin(theta1-theta2), np.sin(theta2-theta3), np.sin(theta1-theta3)
       
            Mat=np.array([[M1*L1, M2*L2*c1, m3*L3*c3], [M2*L1*c1, M2*L2, m3*L3*c2], [m3*L1*c3, m3*L2*c2, m3*L3]])
            invMat=inv(Mat)
        
            f=np.array([[-M1*g*np.sin(theta2)-M2*L2*(omega2**2)*s1-m3*L3*(omega3**2)*s3],
               [-M2*g*np.sin(theta2)+M2*L1*(omega1**2)*s1-m3*L3*(omega3**2)*s2],
               [m3*g*np.sin(theta3)+m3*L1*(omega1**2)*s3-m3*L3*(omega3**2)*s2]])
        
            invMatf=invMat.dot(f)
        
            dtheta1=omega1
            domega1=invMatf[0]
            dtheta2=omega2
            domega2=invMatf[1]
            dtheta3=omega3
            domega3=invMatf[2]
        
            return dtheta1, domega1, dtheta2, domega2,  dtheta3, domega3
        def pendulum_solver(*args):
            tmax= 10.5
            dt=0.005
            t = np.arange(0, tmax, dt)
            y0=[ini1, 0,ini2, 0, ini3, 0]
        
            func=solve_ivp(der_eq, [0,tmax], y0, method='Radau', t_eval=t)
            return func
        
        func=pendulum_solver()
        theta1, theta2, theta3 = func.y[0,:], func.y[2], func.y[4]
        x1=L1*np.sin(theta1)
        y1=-L1*np.cos(theta1)
        x2=x1+L2*np.sin(theta2)
        y2=y1-L2*np.cos(theta2)
        x3=x2+L3*np.sin(theta3)
        y3=y2-L3*np.cos(theta3)
        positions=[theta1,theta2,theta3,x1,y1,x2,y2,x3,y3, L]
        return(positions)
        
@app.callback(Output('graph_gen_pend', 'children'),
              [Input('memory_pend', 'data'), Input("graph-button-pend", "n_clicks")], prevent_initial_call=True)      
def graph(data_input,n): 
        
        if ctx.triggered_id == "graph-button-pend":
             x1, x2, x3=data_input[3], data_input[5], data_input[7]
             y1, y2, y3=data_input[4], data_input[6], data_input[8]
             L=data_input[9]
             
             fig = go.Figure(data=[go.Scatter(
                x=x1, y=y1, mode="lines",name='Mass 1', line=dict(color='blue'),showlegend=False),
                    go.Scatter(x=x2, y=y2,  name='Mass 2',mode="lines", line=dict(color='red'),showlegend=False),
                    go.Scatter(x=x3, y=y3, name='Mass 3', mode='lines', line=dict(color='green'), showlegend=False),
                    go.Scatter(x=[0, y1, None, y1,y2, None,y2, y3],
                               y=[0, y1, None, y1,y2, None,y2, y3], 
                               mode="lines", showlegend=False),

                    go.Scatter(x=[None], y=[None], mode="markers", name="Mass 1", 
                               marker=dict(size=5, color="blue")),
                    go.Scatter(x=[None], y=[None], mode="markers", name="Mass 2", 
                               marker=dict(size=5, color="red")),
                    go.Scatter(x=[None], y=[None], mode="markers", name="Mass 3", 
                               marker=dict(size=5, color="green"))
  
                               ],
                               layout=go.Layout(xaxis=dict(range=[-L-0.5, L+0.5], autorange=False, zeroline=False),
                                                yaxis=dict(range=[-L-0.5, L+0.5], autorange=False, zeroline=False),
                  title_text='Position Graph of a Triple Pendulum', hovermode="closest", 
                  
                  updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", 
                                                                  args=[None, {"frame": {"duration": 90, "redraw": False},}])])],
                  ),
                  frames=[go.Frame(
                       data=[go.Scatter(x=[0, x1[k], None,x1[k], x2[k], None, x2[k], x3[k]],
                                 y=[0, y1[k], None, y1[k], y2[k], None,y2[k], y3[k]], mode="lines", 
                                 line=dict(color="black", width=1)),
                                go.Scatter(x=[x1[k]],y=[y1[k]],mode="markers",name='Mass 1',
                                             marker=dict(color="blue", size=5)), 
                                go.Scatter(x=[x2[k]],  y=[y2[k]], name='Mass 2',mode="markers", marker=dict(color="red", size=5),),
                                go.Scatter(x=[x3[k]], y=[y3[k]], name='Mass 3',mode="markers",marker=dict(color="green", size=5),)], 
                                name=str(k), 
                                layout=go.Layout(annotations=[dict(xref="x domain", yref="y domain", x=1, y=1,
                                showarrow=False, align='right',
                                text='<b>'+'Time:'+ ' ' + "%.2f" % round(k*0.005, 2)+' '+ 'seconds''</b>'),])
                                  )
                                  for k in range(0, len(x1), time_step)
                                    ],)
             fig.update_layout(
                               title_font_size=23, title_x=0.48,title_xanchor='center', 
                               title_y=0.87, title_yanchor='bottom')
             fig.update_xaxes(title_text='Position (m)')
             fig.update_yaxes(title_text='Position (m)')
             
             return dcc.Graph(figure=fig, style={"display": "flex", 'height': '50vw'})
    

if __name__ == '__main__':
    app.run(debug=True)