import numpy as np
import itertools
from itertools import product
from itertools import chain
from numpy.linalg import inv
from scipy.integrate import solve_ivp
import pandas as pd

G=6.674**(-11)

mass_list=np.arange(1,250,14).tolist()
pos_list=np.arange(-10, 10, 2).tolist()

mass_lists_combos=list(product(mass_list,mass_list,mass_list))
pos_lists_combos=list(product(pos_list,pos_list,pos_list))


initial_value_combos=list(product(mass_lists_combos,pos_lists_combos,pos_lists_combos,pos_lists_combos))

def grav_body_solver(i):
   
   ma, mb, mc =initial_value_combos[i][0][0], initial_value_combos[i][0][1],initial_value_combos[i][0][2]
   xa_ini, ya_ini, za_ini=initial_value_combos[i][1][0], initial_value_combos[i][1][1],initial_value_combos[i][1][2]
   xb_ini, yb_ini, zb_ini=initial_value_combos[i][2][0], initial_value_combos[i][2][1],initial_value_combos[i][2][2]
   xc_ini, yc_ini, zc_ini=initial_value_combos[i][3][0], initial_value_combos[i][3][1],initial_value_combos[i][3][2]
   
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
        
        
   y0=[xa_ini,0,ya_ini,0, za_ini, 0,xb_ini,0, yb_ini,0, zb_ini, 0, xc_ini, 0, yc_ini,0, zc_ini, 0]
   tmax=(5*10**5)/2
   dt=0.1
   ts = np.arange(0, tmax, dt)
            
   func=solve_ivp(test_eq, [0,tmax], y0, method='DOP853', dense_output=False, t_eval=ts)

   xa, ya, za=func.y[0], func.y[2], func.y[4]
   xb, yb, zb=func.y[6], func.y[8], func.y[10]
   xc, yc, zc=func.y[12], func.y[14], func.y[16]

   values=[ma, mb, mc, xa, ya, za,xb, yb, zb, xc, yc, zc]   

   return(values)

grav_body_vals=[]

for i in range(0,len(initial_value_combos)):
     val=grav_body_solver(i)
     grav_body_vals.append(val)

y=np.asarray(grav_body_vals)
y.tofile('grav_body_data.csv', sep = ',')   
        