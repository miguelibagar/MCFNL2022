import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as an

from solver.FDTD2D import *



## -- Grid
tEnd = 15

xIni = -5
xEnd = 5
N = int(101)

ss = np.linspace(xIni,xEnd,N)

fdtd = FDTD2D(ss)

## -- Condiciones iniciales
sigma= (xIni-xEnd)/10.0

fdtd.E = np.exp( - (np.power(fdtd.Xgrid,2) + np.power(fdtd.Ygrid,2)) / \
                (2.0*sigma**2))
fdtd.Hx = fdtd.XGrid_Hx*0.0
fdtd.Hy = fdtd.XGrid_Hy*0.0

## -- Algoritmo
t = 0.0
while t<tEnd:
    t = fdtd.step(t)
    
##################################
## Animacion
##################################

t = 0.0
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111)

fdtd.E = np.exp( - (np.power(fdtd.Xgrid,2) + np.power(fdtd.Ygrid,2)) / \
                (2.0*sigma**2))
fdtd.Hx = fdtd.XGrid_Hx*0.0
fdtd.Hy = fdtd.XGrid_Hy*0.0

def update(i):
    ax.clear()
    
    global t

    t = fdtd.step(t)       
    
    wframe = plt.contourf(fdtd.Xgrid, fdtd.Ygrid, fdtd.E,cmap=cm.seismic)
    
    return wframe

ani = an.FuncAnimation(fig, update,
                              frames=range(int(tEnd/fdtd.Dt)),
                              interval=5)


plt.show()
print("END")