import numpy as np
import scipy.linalg as LA

r"""
This code calculates the magnitude of fermionic hopping correlator between any two sites of a finite lattice
"""

def correlator(x1, y1, x2, y2, Nx, Ny, t, mu, beta):

    # fermionic hopping correlator between sites (x1, y1) and (x2, y2)
    # finite lattice size: Nx * Ny
    # inverse temeprature beta 

    def tb2D(Nx, Ny, t, mu):
    
        Ham = np.zeros((Nx*Ny, Nx*Ny))
        for n in range(Nx):
            for m in range(Ny-1):
                Ham[m+Ny*n, m+1+Ny*n] = -t 
                Ham[m+1+Ny*n, m+Ny*n] = -t 
        for m in range(Ny):
            for n in range(Nx-1):
                Ham[m+Ny*n, m+Ny*(n+1)] = -t 
                Ham[m+Ny*(n+1), m+Ny*n] = -t

        for ms in range(Nx*Ny):
            Ham[ms, ms] = -mu

        return Ham

    def vecorth(x, y, Ny):
        return Ny*x + y

    def corr(x1, y1, x2, y2, W, V, beta):

        sv1 = vecorth(x1, y1, Ny)
        sv2 = vecorth(x2, y2, Ny)
        s=0
        for i in range(len(W)):
            s = s + (V[sv1, i]*V[sv2, i])/(1+np.exp(beta*W[i]))
        return s
   
    HamT = tb2D(Nx, Ny, t, mu)

    W, V = LA.eig(HamT)            

    val = corr(x1, y1, x2, y2, W, V, beta)

    return val


Nx = 3
Ny = 2
t = 1
beta = 0.2
mu = 0


x1, y1, x2, y2 = 2, 0, 2, 1  # horizontal
x11, y11, x22, y22 = 0, 1, 1, 1 # vertical



c = correlator(x1, y1, x2, y2, Nx, Ny, t, mu, beta)
d = correlator(x11, y11, x22,y22, Nx, Ny, t, mu, beta)

print(c, d)
