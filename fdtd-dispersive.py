
import numpy as np
import matplotlib.pyplot as plt


#mu = 4*pi*1e-7
#eps0 = 8.8541878128e-12
# Voy a trabajar en unidades naturales
mu = 4*3.14159*1e-7
c = 3e8

## - Parametrizacion de la permitividad
cVec = np.array([5.987e-1 +4.195e3j,-2.211e-1+2.680e-1j,-4.240+7.324e2j])
aVec = np.array([-2.502e-2-8.626e-3j,-2.021-9.407e-1j,-1.467e1-1.338j])

## -- Grid
tEnd = 1.0

xIni = 0
xEnd = 10
N = int(101)

grid = np.linspace(xIni,xEnd,N)
dualGrid = (grid[:-1] + grid[1:])/2.0 # Tiene un punto menos que el primal

## - Condicion inicial (gaussiana)
media = (xIni + xEnd)/2.0
sigma= (xIni-xEnd)/10.0

Eini = np.exp( - np.power(grid - media,2) / (2.0*sigma**2))

plt.plot(grid,Eini)
plt.show()

## -- Espaciados
Dx = grid[1] - grid[0]
Dt = 0.8*Dx/c # Para que sea dimensionalmente correcto

## -- Parametros necesarios para resolver
kVec = 2 / (1-0.5*Dt*aVec)
bVec = 8.85e-12*Dt*(cVec / (1-0.5*Dt*aVec))

## -- Algoritmo
# Primero inicializamos
Eold = Eini
Jold = np.zeros([N,len(cVec)])
Hold = dualGrid*0.0

Enew = Eold*0.0
Jnew = Jold*0.0
Hnew = Hold*0.0

# Aqui pongo condiciones de contorno de campo nulo
# El campo electrico tiene la cosa de que llega hasta los extremos
Enew[0] = 0.0
Enew[-1] = 0.0

t=0.0
while t < tEnd:
    suma1 = np.real(np.sum(Jold*kVec,axis=1))
    suma2 = np.sum(np.real(bVec))
    
    Enew[-1:1] = Eold[-1:1] + (Dt/suma2)*((Hold[1:] - Hold[:-1])/Dx - suma1[-1:1])
    
    Jnew = Jold*kVec + bVec*((Enew-Eold)/Dt)
    
    Hnew[:] = -(Dt/(mu*Dx)) * (Enew[1:] - Enew[:-1]) + Hold[:]
    
    Eold[:] = Enew[:]
    Hold[:] = Hnew[:]
    
    Eold[0] = 0.0
    Eold[-1] = 0.0
    
    t+=Dt


# grid[a:b] coge el intervalo [a,b). El -1 es el ultimo (empiezo en 0)
# Luego grid[:-1] coge desde el primero hasta el penultimo y
# grid[1:] coge desde el segundo hasta el ultimo

plt.plot(grid,Enew)
plt.plot(dualGrid,Hnew)
plt.grid()
plt.show()