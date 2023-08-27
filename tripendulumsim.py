import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, callback, ctx

g=9.8

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([  
    html.H1(children='Triple Pendulum'),
    
    dcc.Slider(min=0, max=20, step=0.1, value=3, id='mass1'
               ),
    html.Div(id="mass1Val"),
     dcc.Slider(value=3,
            min=0, max=20, step=0.1, id="mass2",
    ),
    html.Div(id="mass2Val"),
     dcc.Slider(value=3,
            min=0, max=20, step=0.1, id="mass3",
    ),
    html.Div(id="mass3Val"),
    dcc.Slider(
            id="length1", value=3,
            min=0, max=20, step=0.1,
    ),
    html.Div(id="length1Val"),
    dcc.Slider(
            id="length2", value=3,
            min=0, max=20, step=0.1,
    ),
     html.Div(id="length2Val"),
     dcc.Slider(
            id="length3", value=3,
            min=0, max=20, step=0.1,
    ),
    dcc.Slider(0, 20, 0.1,
               value=3,  marks=None, 
               tooltip={"placement": "bottom", "always_visible": True},
               id='ini1'
    ),
    html.Div(id="ini1Val"),
    dcc.Slider(0, 20, 0.1,
               value=2,  marks=None, 
               tooltip={"placement": "bottom", "always_visible": True},
               id='ini2'
    ),
    html.Div(id="ini2Val"),
    dcc.Slider(-np.pi, np.pi, 0.01,
               value=1,  marks=None, 
               tooltip={"placement": "bottom", "always_visible": True},
               id='ini3'
    ),
    html.Div(id="ini3Val"),
     html.Button("Run", id="graph-button", n_clicks=0),
     html.Div(id="pendulum-graph"),
])

@app.callback(
    [Output("mass1Val", "children")],
    [Input('mass1', "value")]
)
def useInput1(m1):
    return [m1]

@callback(
    Output("mass2Val", "children"),
    Input('mass2', "value")
)
def useInput2(mass2):
    m2=mass2
    return m2

@callback(
    [Output("mass3Val", "children")],
    [Input('mass3', "value")]
)
def useInput3(mass3):
    m3=mass3
    return m3
@callback(
    [Output("length1Val", "children")],
    [Input('length1', "value")]
)
def useInput4(length1):
    L1=length1
    return L1

@callback(
    Output("length2Val", "children"),
    Input('length2', "value")
)
def useInput5(length2):
    L2=length2
    return L2
@callback(
    Output("length3Val", "children"),
    Input('length3', "value")
)
def useInput6(length3):
    L3=length3
    return L3

@callback(
    Output("ini1Val", "children"),
    Input('ini1', "value")
)
def useInput7(ini1):
    return ini1

@callback(
    Output("ini2Val", "children"),
    Input('ini2', "value")
)
def useInput8(ini2):
    return ini2
@callback(
    Output("in31Val", "children"),
    Input('ini3', "value")
)
def useInput9(ini3):
    return ini3


def der_eq(t, y):
    
    theta1, omega1, theta2, omega2, theta3, omega3=y
    M1=m1+m2+m3
    M2=m1+m3
    
  
    
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
    tmax= 10
    dt=0.001
    t = np.arange(0, tmax, dt)
    
    
    func=solve_ivp(der_eq, [0,tmax], y0, method='Radau',dense_output=True, t_eval=t)
    
    return(func)




def pendulum_graph(*args):    
    func=pendulum_solver()
    
    theta1, theta2, theta3 = func.y[0,:], func.y[2], func.y[4]
    
    x1=L1*np.sin(theta1)
    y1=-L1*np.cos(theta1)
    x2=x1+L2*np.sin(theta2)
    y2=y1-L2*np.cos(theta2)
    x3=x2+L3*np.sin(theta3)
    y3=y2-L3*np.cos(theta3)

    positions=[theta1,theta2,theta3,x1,y1,x2,y2,x3,y3]
    return(positions)

graph_inputs=pendulum_graph()
x1, x2, x3=graph_inputs[4], graph_inputs[6], graph_inputs[8]
y1, y2, y3=graph_inputs[5], graph_inputs[7], graph_inputs[9]
    


@callback(
    Output("pendulum-graph", "children"),
    Input("graph-button", "n_clicks"), 
    prevent_initial_call=True,
    )
def run_graph(n):
    if ctx.triggered_id == "graph-button":
        fig = go.Figure(
    data=[go.Scatter(go.Scatter(x=x1, y=y1,
                    mode='lines',
                    name='mass 1')),
          go.Scatter(go.Scatter(x=x2, y=y2,
                    mode='lines',
                    name='mass 2')),
          go.Scatter(go.Scatter(x=x3, y=y3,
                    mode='lines',
                    name='mass 3')),
          go.Scatter(
            x=[0, y1, None, y1,y2, None,y2, y3],
            y=[0, y1, None, y1,y2, None,y2, y3],
            mode="lines", showlegend=False),
          
          ],
    layout=go.Layout(xaxis=dict(range=[-L-0.5, L+0.5], autorange=False, zeroline=False),
        yaxis=dict(range=[-L-0.5, L+0.5], autorange=False, zeroline=False),
        
        title_text='Animated Triple Pendulum', hovermode="closest",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, {"frame": {"duration": 90, "redraw": False},}])])]),
    
    frames=[go.Frame(
        data=[go.Scatter(
            x=[0, x1[k], None,x1[k], x2[k], None, x2[k], x3[k]],
            y=[0, y1[k], None, y1[k], y2[k], None,y2[k], y3[k]],
            mode="lines",
            line=dict(color="black", width=1)),
            go.Scatter(
            x=[x1[k]],
            y=[y1[k]],
            mode="markers",name='mass 1',
            marker=dict(color="blue", size=5)), go.Scatter(
            x=[x2[k]],
            y=[y2[k]],
            mode="markers",
            marker=dict(color="red", size=5),), go.Scatter(
            x=[x3[k]],
            y=[y3[k]],
            mode="markers",
            marker=dict(color="green", size=5),)
            ])

        for k in range(0, len(x1), 3)]
    )
    fig.update_layout(height=600, width=600)
    fig.show()
    return dcc.Graph(figure=fig)
if __name__ == '__main__':
    app.run(debug=True)