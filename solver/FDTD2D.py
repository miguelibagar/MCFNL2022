
import numpy as np


class FDTD2D:
    def __init__(self,ss):
        ss_cross = (ss[:-1]+ss[1:])/2.0
        
        self.Xgrid,self.Ygrid = np.meshgrid(ss,ss)
        self.XGrid_Hy,self.YGrid_Hy = np.meshgrid(ss_cross,ss)
        self.XGrid_Hx,self.YGrid_Hx = np.meshgrid(ss,ss_cross)
        
        c0 = 1.0
        
        self.Dx = ss[1] - ss[0]
        self.Dt = 0.8*self.Dx/c0 
        
        self.E = 0.0*self.Xgrid
        self.Hx = 0.0*self.XGrid_Hx
        self.Hy = 0.0*self.XGrid_Hy
        
        
    def step(self, t):
        E = self.E
        Hx = self.Hx
        Hy = self.Hy

        E[1:-1,1:-1] = E[1:-1,1:-1] +\
            (self.Dt/(self.Dx)) * (Hy[1:-1,1:] - Hy[1:-1,:-1]) -\
            (self.Dt/(self.Dx)) * (Hx[1:,1:-1] - Hx[:-1,1:-1]) 
            
        Hx[:,:] = Hx[:,:] - (self.Dt/(self.Dx)) * (E[1:,:] - E[:-1,:])
        Hy[:,:] = Hy[:,:] + (self.Dt/(self.Dx)) * (E[:,1:] - E[:,:-1])
        
        return t + self.Dt


