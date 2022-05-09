import numpy as np
import matplotlib.pyplot as plt


#mu = 4*pi*1e-7
#eps0 = 8.8541878128e-12
# Voy a trabajar en unidades naturales
mu = 1
eps = 1
c = 1

## - Parametrizacion de la permitividad
cVec = np.array([5.987e-1 +4.195e3j,-2.211e-1+2.680e-1j,-4.240+7.324e2j])
aVec = np.array([-2.502e-2-8.626e-3j,-2.021-9.407e-1j,-1.467e1-1.338j])

#cVec = np.array([5.987e-1 +4.195e3j])
#aVec = np.array([-2.502e-2-8.626e-3j])

## -- Grid
tEnd = 5

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
kVec = (1+0.5*Dt*aVec) / (1-0.5*Dt*aVec)
k1Vec = 2 / (1-0.5*Dt*aVec)
bVec = eps*Dt*(cVec / (1-0.5*Dt*aVec))

## -- Algoritmo
# Primero inicializamos
Eold = Eini*(1+0j)
Jold = np.zeros([N,len(cVec)])*(0+0j)
Hold = dualGrid*(0+0j)

Enew = Eold*(0+0j)
Jnew = Jold*(0+0j)
Hnew = Hold*(0+0j)

suma2 = 100+ np.sum(np.real(bVec))

limite = 80

t=0.0
while t < tEnd:
    suma1 = np.real(np.sum(Jold*k1Vec,axis=1))
            
    Enew[1:limite] = Eold[1:limite] - (Dt/(eps*Dx))*(Hold[1:limite]-Hold[0:(limite-1)])
    #Enew[1:3] = Eold[1:3] - (Dt/(eps*Dx))*(Hold[1:3]-Hold[0:2])
    
    Enew[limite:-1] = Eold[limite:-1] + (Dt/suma2)*((Hold[limite:] - Hold[(limite-1):-1])/Dx - suma1[limite:-1])
    #Enew[3:-1] = Eold[3:-1] + (Dt/suma2)*((Hold[3:] - Hold[2:-1])/Dx - suma1[3:-1])
    
    for p in range(len(cVec)):
        Jnew[:,p] = Jold[:,p]*kVec[p] + (bVec[p]/Dt)*(Enew-Eold)
    
    Hnew[:] = -(Dt/(mu*Dx)) * (Enew[1:] - Enew[:-1]) + Hold[:]
    
    Eold[:] = Enew[:]
    Jold[:] = Jnew[:]
    Hold[:] = Hnew[:]
    
    Eold[0] = 0+0j
    Eold[-1] = 0+0j
    
    t+=Dt


# grid[a:b] coge el intervalo [a,b). El -1 es el ultimo (empiezo en 0)
# Luego grid[:-1] coge desde el primero hasta el penultimo y
# grid[1:] coge desde el segundo hasta el ultimo

plt.close()
plt.figure()
plt.plot(grid,np.real(Enew))
plt.plot(dualGrid,np.real(Hnew))
plt.legend(['Electrico','Magnetico'])
plt.xlim(0,11)
plt.ylim(-3,3)
rectangle = plt.Rectangle((limite/xEnd,-3.0),4,6,fc='blue',ec="blue")
plt.gca().add_patch(rectangle)
plt.grid()