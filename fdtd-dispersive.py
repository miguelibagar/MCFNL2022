import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#mu = 4*pi*1e-7
#eps0 = 8.8541878128e-12
# Voy a trabajar en unidades naturales
mu = 1
eps = 1
c = 1

## - Parametrizacion de la permitividad
# cVec = np.array([5.987e-1 +4.195e3j,-2.211e-1+2.680e-1j,-4.240+7.324e2j])
#aVec = np.array([-2.502e-2-8.626e-3j,-2.021-9.407e-1j,-1.467e1-1.338j])
aVec = np.array([-20-30j,-20-10j,-10-10j])*1e2

cVec = np.array([-10-30j,-20-10j,20-10j])*1e4

#cVec = np.array([5.987e-1 +4.195e3j])
#aVec = np.array([-2.502e-2-8.626e-3j])

## -- Grid
tEnd = 40

xIni = 0
xEnd = 10
N = int(101)

grid = np.linspace(xIni,xEnd,N)
dualGrid = (grid[:-1] + grid[1:])/2.0 # Tiene un punto menos que el primal

## - Condicion inicial (gaussiana)
media = (xIni + xEnd)/3.0
sigma= (xIni-xEnd)/50.0

Eini = np.exp( - np.power(grid - media,2) / (2.0*sigma**2))

plt.plot(grid,Eini)
plt.show()

## -- Espaciados
Dx = grid[1] - grid[0]
Dt = 0.80*Dx/c # Para que sea dimensionalmente correcto

## -- Parametros necesarios para resolver
vTest = 1-0.5*Dt*aVec
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


Edata, Hdata = [],[]
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111)

suma2 = 200 + np.sum(np.real(bVec))

limite = 40

acum = 0
suma1 = np.zeros(N)*(0+0j)

t=0.0
while t < tEnd:
    for i in range(N):
        for p in range(len(cVec)):
            acum = acum + Jold[i,p]*(1+kVec[p])
        suma1[i] = np.real(acum)
        acum = 0

    #Enew[1:limite+1] = Eold[1:limite+1] - (Dt/(eps*Dx))*(Hold[1:limite+1]-Hold[0:(limite)])
    #Enew[1:3] = Eold[1:3] - (Dt/(eps*Dx))*(Hold[1:3]-Hold[0:2])
    
    Enew[1:-1] = Eold[1:-1] + (Dt/suma2)*((Hold[1:] - Hold[:-1])/Dx - suma1[1:-1])
    #Enew[limite:-1] = Eold[limite:-1] + (Dt/suma2)*((Hold[limite:] - Hold[(limite-1):-1])/Dx - suma1[limite:-1])
    #Enew[3:-1] = Eold[3:-1] + (Dt/suma2)*((Hold[3:] - Hold[2:-1])/Dx - suma1[3:-1])
    
    for p in range(len(cVec)):
        Jnew[:,p] = Jold[:,p]*kVec[p] + (bVec[p]/Dt)*(Enew-Eold)
    
    Hnew[:] = -(Dt/(mu*Dx)) * (Enew[1:] - Enew[:-1]) + Hold[:]
    
    EnewR = np.real(Enew)
    HnewR = np.real(Hnew)
    
    Edata.append(EnewR.copy())
    Hdata.append(HnewR.copy())
    
    Eold = Enew[:]
    Jold = Jnew[:]
    Hold = Hnew[:]
    
    Eold[0] = 0+0j
    Eold[-1] = 0+0j
    
    
    
    t+=Dt


# grid[a:b] coge el intervalo [a,b). El -1 es el ultimo (empiezo en 0)
# Luego grid[:-1] coge desde el primero hasta el penultimo y
# grid[1:] coge desde el segundo hasta el ultimo

# plt.close()
plt.figure()
plt.plot(grid,np.real(Enew))
plt.plot(dualGrid,np.real(Hnew))
plt.legend(['Electrico','Magnetico'])
#plt.xlim(0,11)
#plt.ylim(-3,3)
rectangle = plt.Rectangle((limite/xEnd,-3.0),4,6,fc='blue',ec="blue")
plt.gca().add_patch(rectangle)
plt.grid()





def animate(i):
    ax.clear()
    ax.set(xlim=(0, 10), ylim=(-10, 10))

    ax.plot(grid,Edata[i],'o' ,color = 'darkblue',markersize=2)
    ax.plot(dualGrid,Hdata[i],'o',color = 'red',markersize=2)
    rectangle = plt.Rectangle((8,-10), 2, 40, fc='blue',ec="blue")
    plt.gca().add_patch(rectangle)
    plt.grid(True)
    




anim = animation.FuncAnimation(fig, animate,frames=len(Edata),interval=20,blit=False)
#anim.save('caca.mp4', writer = 'ffmpeg', fps = 30)


#anim.save('basic_animation.gif', fps=30, extra_args=['-vcodec', 'libx264'])
#anim.save('caca.gif')
                            
plt.show()

# grid[a:b] coge el intervalo [a,b). El -1 es el ultimo (empiezo en 0)
# Luego grid[:-1] coge desde el primero hasta el penultimo y
# grid[1:] coge desde el segundo hasta el ultimo

