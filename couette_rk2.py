# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 23:11:28 2019

@author:
"""

import numpy as np
import matplotlib.pyplot as plt



''' Analytical solution of Couette flow '''
def u_exact(U,h,nu,y,t):
    u = U * (1-y / h)
    for n in range(1,100000):
        u -= 2 * U / np.pi * (1/n) *  np.e **(-n**2*np.pi**2*nu*t/h**2)*np.sin(n*np.pi*y/h)
    return u

''' Parameters '''
h = 1   # gap between plate
U = 1   # velocity of under wall
nu = 0.0005 # viscosity coefficient

nx = 51 # number of mesh

y = h * np.linspace(0,1,nx) # y coordinates
dy = h / (nx-1) # mesh size
dt = 0.02   # timestep
nend = 20000    # number of iteration

''' Initial conditions '''
u = np.zeros(nx)
un = np.zeros(nx)
u1 = np.zeros(nx)

u[0] = U
u1[0] = U    
un[0] = U

''' 2nd order Runge-Kutta '''
for nt in range(nend):
    u1[1:nx-1] = u[1:nx-1] + 0.5 * dt * nu * (u[0:nx-2] - 2 * u[1:nx-1] + u[2:nx]) / dy**2 

#    for j in range(1,nx-1):
#        u1[j] = u[j] + 0.5* dt * nu * (u[j-1] - 2 * u[j] + u[j+1]) / dy**2 

    un[1:nx-1] = u[1:nx-1] + dt * nu * (u1[0:nx-2] - 2 * u1[1:nx-1] + u1[2:nx]) / dy**2 
#    for j in range(1,nx-1):
#        un[j] = u[j] + dt * nu * (u1[j-1] - 2 * u1[j] + u1[j+1]) / dy**2 
       
    u = un.copy()

    ''' Plot '''
    if np.mod(nt + 3000, 4000) == 0 and nt < 5000 and nt > 0:
        ue = u_exact(U,h,nu,y,dt*nt)
        
        plt.plot(ue,y, color = 'black', label = "Analytical solution")
        plt.plot(un,y,"o", color = 'None', markeredgecolor='black', label = "Numerical solution")


    elif np.mod(nt + 3000, 4000) == 0 and nt > 0: 
        ue = u_exact(U,h,nu,y,dt*nt)
        
        plt.plot(ue,y, color = 'black')
        plt.plot(un,y,"o", color = 'None', markeredgecolor='black')

plt.xlabel("u")
plt.ylabel("y")
plt.legend()
plt.show()