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
    
    dcc.Slider(value=3,
    min=0, max=20, step=0.1, id='mass1',  marks=None,
               tooltip={"placement": "bottom", "always_visible": True}
               ),
    
     dcc.Slider(value=2,
            min=0, max=20, step=0.1, id="mass2", marks=None,
            tooltip={"placement": "bottom", "always_visible": True}
    ),
    
     dcc.Slider(value=1,
            min=0, max=20, step=0.1, id="mass3", marks=None,
            tooltip={"placement": "bottom", "always_visible": True}
    ),
   
    dcc.Slider(
            id="length1", value=3,
            min=0, max=20, step=0.1, marks=None,
            tooltip={"placement": "bottom", "always_visible": True}
    ),
    
    dcc.Slider(
            id="length2", value=2,
            min=0, max=20, step=0.1, marks=None,
            tooltip={"placement": "bottom", "always_visible": True}
    ),
     
     dcc.Slider(
            id="length3", value=1,
            min=0, max=20, step=0.1, marks=None,
            tooltip={"placement": "bottom", "always_visible": True}
    ),
    dcc.Slider(value=1, min=-np.pi, max=np.pi, step=0.001, marks=None, 
               tooltip={"placement": "bottom", "always_visible": True},
               id='ini1'
    ),
    
    dcc.Slider(value=1, min=-np.pi, max=np.pi, step=0.001,  marks=None, 
               tooltip={"placement": "bottom", "always_visible": True},
               id='ini2'
    ),
    
    dcc.Slider(value=1, min=-np.pi, max=np.pi, step=0.001,  marks=None, 
               tooltip={"placement": "bottom", "always_visible": True},
               id='ini3'
    ),
    
    html.Button("Run", id="graph-button", n_clicks=0),
    html.Div(id="user_inputs"),
    html.Div(id="pendulum-graph"),
])


@app.callback(Output('user_inputs', 'children'),
              [Input( 'mass1', 'value'), Input( 'mass2', 'value'),Input( 'mass3', 'value'),
               Input( 'length1', 'value'), Input( 'length2', 'value'),Input( 'length3', 'value'),
               Input( 'ini1', 'value'), Input( 'ini2', 'value'),Input( 'ini3', 'value'), 
               Input("graph-button", "n_clicks")])
def fetch_data_from_user_input(input_value, input_value2, input_value3, input_value4,
                               input_value5, input_value6, input_value7, input_value8, input_value9,n):
    
    
    m1,m2,m3=input_value,input_value2, input_value3
    L1,L2,L3=input_value4,input_value5, input_value6
    ini1,ini2,ini3=input_value7,input_value8, input_value9
    M1=m1+m2+m3
    M2=m1+m2
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
        tmax= 10
        dt=0.001
        t = np.arange(0, tmax, dt)
        y0=[ini1, 0,ini2, 0, ini3, 0]
        
        func=solve_ivp(der_eq, [0,tmax], y0, method='Radau',dense_output=True, t_eval=t)
        return(func)
    
    def pendulum_graph(*args): 
        func=pendulum_solver()
        theta1, theta2, theta3 = func.y[0,:], func.y[2], func.y[4]
        time=func.y
        
        x1=L1*np.sin(theta1)
        y1=-L1*np.cos(theta1)
        x2=x1+L2*np.sin(theta2)
        y2=y1-L2*np.cos(theta2)
        x3=x2+L3*np.sin(theta3)
        y3=y2-L3*np.cos(theta3)
        positions=[theta1,theta2,theta3,x1,y1,x2,y2,x3,y3,time]
        return(positions)
        
    position=pendulum_graph()
    x1, x2, x3=position[3], position[5], position[7]
    y1, y2, y3=position[4], position[6], position[8]
    time=position[9]
    time_step=2
    
    
    if ctx.triggered_id == "graph-button":
        
        fig = go.Figure(frames=[go.Frame(data=[go.Scatter(x=[x1[k]],y=[y1[k]],mode="markers",name='mass 1',
                                             marker=dict(color="blue", size=5)), 
                                go.Scatter(x=[x3[k]], y=[y3[k]], name='mass 3',mode="markers",marker=dict(color="green", size=5),),
                                go.Scatter(x=[x2[k]],  y=[y2[k]], name='mass 2',mode="markers", marker=dict(color="red", size=5),),
                                go.Scatter(x=[0, x1[k], None, x1[k], x2[k],None, x2[k],x3[k]],y=[0, y1[k], None, y1[k], y2[k],None, y2[k],y3[k]], mode="lines", 
                                 line=dict(color="black", width=1), showlegend=False)
                                  ],
                                         name=str(k) 
                                         # you need to name the frame for the animation to behave properly
                                         )
            for k in range(0,len(x1),time_step)],

            layout=go.Layout(xaxis=dict(range=[-L-0.5, L+0.5], autorange=False, zeroline=False),
                yaxis=dict(range=[-L-0.5, L+0.5], autorange=False, zeroline=False)))
                
        # Add data to be displayed before animation starts
        fig.add_trace(go.Scatter(
            x=x1, y=y1,mode="markers",name='mass 1',
                                             marker=dict(color="blue", size=5)))
        fig.add_trace(go.Scatter(x=x2, y=y2, name='mass 2',mode="markers", marker=dict(color="red", size=5)))
        fig.add_trace(go.Scatter(x=x3, y=y3,  name='mass 3',mode="markers",marker=dict(color="green", size=5)))
        fig.add_trace(go.Scatter(x=[0, x1, None, x1, x2, None, x2,x3],y=[0, y1, None, y1, y2,None, y2,y3], mode="lines", 
                                 line=dict(color="black", width=1), showlegend=False))
    

        sliders = [
             {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name],{"frame": {"duration": 90, "redraw": False},}],
                        "label": f"{round(k*0.001*time_step,3)} seconds",
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                    ],
                }
            ]

                # Layout
        fig.update_layout(
            title='Triple Pendulum',
            width=600,
            height=600,
            updatemenus = [
                {
                    "buttons": [
                        {
                        "args": [None, {"frame": {"duration": 90, "redraw": False},}],
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
             sliders=sliders
            )

        fig.show()
        return dcc.Graph(figure=fig)
    
#https://community.plotly.com/search?q=show%20time%20animation%20%23python%20order%3Alatest
if __name__ == '__main__':
    app.run(debug=True)