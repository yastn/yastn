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
        for n in range(Ny):
            for m in range(Nx-1):
                Ham[m+Nx*n, m+1+Nx*n] = -t 
                Ham[m+1+Nx*n, m+Nx*n] = -t 
        for m in range(Nx):
            for n in range(Ny-1):
                Ham[m+Nx*n, m+Nx*(n+1)] = -t 
                Ham[m+Nx*(n+1), m+Nx*n] = -t

        for ms in range(Nx*Ny):
            Ham[ms, ms] = -mu

        return Ham

    def vecorth(y, x, Nx):
        return Nx*y + x

    def corr(x1, y1, x2, y2, W, V, beta):

        sv1 = vecorth(y1, x1, Nx)
        sv2 = vecorth(y2, x2, Nx)
        s=0
        for i in range(len(W)):
            s = s + (V[sv1, i]*V[sv2, i])/(1+np.exp(beta*W[i]))
        return s
   
    HamT = tb2D(Nx, Ny, t, mu)

    W, V = LA.eig(HamT)           

    val = corr(x1, y1, x2, y2, W, V, beta)

    return val


Nx = 5
Ny = 5
t = 1
beta = 1
mu = 0
x1, y1, x2, y2 = 2, 1, 2, 2

c = correlator(x1, y1, x2, y2, Nx, Ny, t, mu, beta)
print(c)
