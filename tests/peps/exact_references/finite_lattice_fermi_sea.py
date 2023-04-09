import numpy as np
import scipy.linalg as LA

r"""
This code calculates the magnitude of hopping correlator between any two sites of a 
finite 2D tight-binding lattice in two dimensions with nearest neighbor hopping and some 
chemical potential. Used for testing finite PEPS calculations for 2D spinless 
fermi sea and Fermi-Hubbard models.
"""

def tb2D(Nx, Ny, t, mu):
    """ Generates 2D tight binding Hamiltonian of size Nx x Ny with given tunneling strength t and 
    chemical potential mu """
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


def correlator(site1, site2, Nx, Ny, t, mu, beta):
    """
    fermionic hopping correlator between sites (x1, y1) and (x2, y2)
    finite lattice size: Nx * Ny inverse temeprature beta 
    """

    x1,y1 = site1
    x2,y2 = site2

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

Nx, Ny = 3, 2
t = 1
beta = 0.1
mu = 0
site1, site2 = (2, 0), (2, 1)
site3, site4 = (0, 1), (1, 1)

c1 = correlator(site1, site2, Nx, Ny, t, mu, beta)
c2 = correlator(site3, site4, Nx, Ny, t, mu, beta)

print("Correlation between for spinless fermi sea at inverse temperature beta", beta, "between", site1, "and", site2, "is", c1)
print("Correlation between for spinless fermi sea at inverse temperature beta", beta, "between", site3, "and", site4, "is", c2)
