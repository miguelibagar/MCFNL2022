# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:24:57 2022

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt

from solver.FDTD2D import *



## -- Grid
tEnd = 5

xIni = -5
xEnd = 5
N = int(101)

ss = np.linspace(xIni,xEnd,N)

fdtd = FDTD2D(ss)

## -- Condiciones iniciales
media = (xIni + xEnd)/2.0
sigma= (xIni-xEnd)/10.0

fdtd.E = np.exp( - (np.power(fdtd.Xgrid,2) + np.power(fdtd.Ygrid,2)) / \
                (2.0*sigma**2))
fdtd.Hx = fdtd.XGrid_Hx*0.0
fdtd.Hy = fdtd.XGrid_Hy*0.0

## -- Algoritmo
t = 0.0
while t<tEnd:
    t = fdtd.step(t)

plt.contourf(fdtd.Xgrid,fdtd.Ygrid,fdtd.E)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(fdtd.Xgrid,fdtd.Ygrid,fdtd.E, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()