import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, callback

g=9.8

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([  
    html.H1(children='Triple Pendulum'),
    
    dcc.Slider(0, 20, 0.1,
               value=3,
               id='mass1'
    ),
    dcc.Slider(0, 20, 0.1,
               value=2,
               id='mass2'
    ),
    dcc.Slider(0, 20, 0.1,
               value=1,
               id='mass3'
    ),
    dcc.Slider(0, 20, 0.1,
               value=3,
               id='length1'
    ),
    dcc.Slider(0, 20, 0.1,
               value=2,
               id='length2'
    ),
    dcc.Slider(0, 20, 0.1,
               value=1,
               id='length3'
    ),
    dcc.Slider(0, 20, 0.1,
               value=3,
               id='ini1'
    ),
    dcc.Slider(0, 20, 0.1,
               value=2,
               id='ini2'
    ),
    dcc.Slider(-np.pi, np.pi, 0.01,
               value=1,
               id='ini3'
    ),

    
    dcc.Graph(id="graph")
])

def der_eq(t, y):
    
    theta1, omega1, theta2, omega2, theta3, omega3=y
    m1=Input('mass1', 'value')
    m2=Input('mass2', 'value')
    m3=Input('mass3', 'value')
    
    L1=Input('length1', 'value')
    L2=Input('length2', 'value')
    L3=Input('length3', 'value')
    
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
    y0=[Input('ini1', 'value'), 0,Input('ini1', 'value'), 0, Input('ini3', 'value'), 0]
    
    func=solve_ivp(der_eq, [0,tmax], y0, method='Radau',dense_output=True, t_eval=t)
    
    return(func)

def pendulum_graph(*args):
    
    L1, L2, L3=Input('length1', 'value'), Input('length2', 'value'), Input('length3', 'value')
    L=L1+L2+L3
    
    func=pendulum_solver()
    
    theta1, theta2, theta3 = func.y[0,:], func.y[2], func.y[4]
    
    x1=L1*np.sin(theta1)
    y1=-L1*np.cos(theta1)
    x2=x1+L2*np.sin(theta2)
    y2=y1-L2*np.cos(theta2)
    x3=x2+L3*np.sin(theta3)
    y3=y2-L3*np.cos(theta3)
    
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
            x=[0, x1, None, x1,x2, None, x2, x3],
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

        for k in range(0, len(func.t), 3)]
    )
    return(fig)


if __name__ == '__main__':
    app.run(debug=True)