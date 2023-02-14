import numpy as np
import scipy.linalg as LA

r"""
This code calculates the magnitude of fermionic hopping correlator between any two sites of a finite lattice
"""

def correlator(x1, y1, x2, y2, Nx, Ny, t, mu, beta):

    # fermionic hopping correlator between sites (x1, y1) and (x2, y2)
    # finite lattice size: Nx * Ny
    # inverse temperature beta 

    def tb2D(Nx, Ny, t, mu):
    
        Ham = np.zeros((Nx*Ny, Nx*Ny))
        for m in range(Nx):
            for n in range(Ny-1):
                Ham[n+Ny*m, n+1+Ny*m] = -t 
                Ham[n+1+Ny*m, n+Ny*m] = -t 
        for m in range(Nx-1):
            for n in range(Ny):
                Ham[n+Ny*m, n+Ny*(m+1)] = -t 
                Ham[n+Ny*(m+1), n+Ny*m] = -t

        for ms in range(Nx*Ny):
            Ham[ms, ms] = -mu

        return Ham

    def vecorth(y, x, Ny):
        return Ny*x + y

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


Nx = 2
Ny = 2
t = 1
beta = 0.1
mu = 0
x1, y1, x2, y2 = 1, 0, 1, 1
x11, y11, x21, y21 = 0, 1, 1, 1

c1 = correlator(x1, y1, x2, y2, Nx, Ny, t, mu, beta)
c2 = correlator(x11, y11, x21, y21, Nx, Ny, t, mu, beta)

print(c1, c2)



