import numpy as np
import settings_transport_vectorization_full as settings
from settings_transport_vectorization_full import backend
from settings_transport_vectorization_full import Tensor
from tens.utils import ncon

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
K = np.array([[0, -1], [1, 0]])
Y = 1j * np.array([[0, -1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])

# STATE
def thermal_state(NL, io, ww, temp):
    """
    Generate vectorization of a thermal state according Fermi-Dirac distribution.
    Output is of MPS form with nr_aux=0 and nr_phys=4.

    Parameters
    ----------
    NL: int
        Number of states in a lead with occupancies acc. to Fermi-Dirac distribution.
        
    io: list of size NS
        List of occupancies in the impurity.
    
    temp: list
        Temperature on sites, zero for the impurity

    ww: list
        List of energies in a chain. Impurity energies are ignored - io used instead.
    
    """
    NS = len(io)
    N = 2*NL+NS
    psi = yamps.mps.Mps(N, nr_phys=1)
    val = np.zeros((1,4,1))
    val[0,0,0] = 1.
    for n in range(N):  # empty tensors
        psi.A[n] = Tensor.Tensor(settings=settings)
        if n<NL or n>NL+NS-1:
            wk = ww[n]
            tp = temp[n]
            p = ( np.exp(wk/tp)-1. )/( np.exp(wk/tp)+1. ) if tp > 1e-6 else 1.*np.sign(wk)
        else:
            p = 1.-2.*io[n-NL]
        val[0,-1,0] = p
        psi.A[n].set_block(val=val)
    return psi
    
# HAMILTONIAN
def H_M_psi_1im_mixed(L, v, mu=0, w0=1, wS=0, gamma=0, ordered=False, dV=False, tempL=False, tempR=False, dt=False, AdagA=False):
    vert = 0 # 0 - IZXY 
    N = 2 * L + 1 # total number of sites
    muL = -0.5 * mu
    muR = +0.5 * mu
    kk = np.arange(1, L + 1, 1)
    ww = 2 * w0 * np.cos(kk * np.pi / (L + 1))
    vk = np.sqrt(2 / (L + 1)) * v * np.sin(kk * np.pi / (L + 1))
    LSR = np.concatenate((-1 + 0 * ww,np.array([2]),+1 + 0 * ww)) # -1,0,1 = L,S,R
    wk = np.concatenate((ww + muL,np.array([wS]), ww + muR + 1e-14))
    vk = np.concatenate((vk,np.array([0]), vk))
    dV = np.concatenate((-dV * .5 + np.zeros(L) , np.array([0]), +dV * .5 + np.zeros(L)))
    temp = np.concatenate((tempL + np.zeros(L) , np.array([0]), tempR + np.zeros(L)))
    
    II = np.kron(I,I)

    if ordered:        
        # sort by energy
        id = np.argsort(wk)
        LSR = LSR[id]
        wk = wk[id]
        vk = vk[id]
        dV = dV[id]
        temp = temp[id]     
        n1 = np.argwhere(LSR == 2)[0,0]
        # dim is virt, phys_out, phys_in, virt
        H = yamps.Mps(N, nr_phys=2, nr_aux=0)
        if vert==0:
            II = np.vstack([np.array([1,0,0,0]),
                            np.array([0,1,0,0]), 
                            np.array([0,0,1,0]),
                            np.array([0,0,0,1]) 
                            ])
            q_z = np.vstack([np.array([0,1,0,0]),
                            np.array([1,0,0,0]), 
                            np.array([0,0,0,1j]),
                            np.array([0,0,-1j,0]) 
                            ])
            z_q = np.vstack([np.array([0,1,0,0]),
                            np.array([1,0,0,0]), 
                            np.array([0,0,0,-1j]),
                            np.array([0,0,1j,0]) 
                            ])
            q_n = np.vstack([.5*np.array([1,-1,0,0]),
                            .5*np.array([-1,1,0,0]), 
                            .5*np.array([0,0,1,1j]),
                            .5*np.array([0,0,-1j,1]) 
                            ])
            n_q = np.vstack([.5*np.array([1,-1,0,0]),
                            .5*np.array([-1,1,0,0]), 
                            .5*np.array([0,0,1,-1j]),
                            .5*np.array([0,0,1j,1]) 
                            ])
            c_q_cp = np.vstack([.5*np.array([1,+1,0,0]),
                            .5*np.array([0,0,0,0]), 
                            .5*np.array([0,0,0,0]),
                            .5*np.array([0,0,0,0]) 
                            ])
            cp_q_c = np.vstack([.5*np.array([1,-1,0,0]),
                            .5*np.array([0,0,0,0]), 
                            .5*np.array([0,0,0,0]),
                            .5*np.array([0,0,0,0]) 
                            ])
            c_q = np.vstack([.5*np.array([0,0,1,+1j]),
                            .5*np.array([0,0,+1,1j]), 
                            .5*np.array([1,-1,0,0]),
                            .5*np.array([+1j,-1j,0,0]) 
                            ])
            cp_q = np.vstack([.5*np.array([0,0,1,-1j]),
                            .5*np.array([0,0,-1,1j]), 
                            .5*np.array([1,+1,0,0]),
                            .5*np.array([-1j,-1j,0,0]) 
                            ])
            q_c = np.vstack([.5*np.array([0,0,1,+1j]),
                            .5*np.array([0,0,-1,-1j]), 
                            .5*np.array([1,+1,0,0]),
                            .5*np.array([+1j,1j,0,0]) 
                            ])
            q_cp = np.vstack([.5*np.array([0,0,1,-1j]),
                            .5*np.array([0,0,+1,-1j]), 
                            .5*np.array([1,-1,0,0]),
                            .5*np.array([-1j,1j,0,0]) 
                            ])
        elif vert==1:
            II = np.vstack([np.array([1,0,0,0]),
                            np.array([0,1,0,0]), 
                            np.array([0,0,1,0]),
                            np.array([0,0,0,1]) 
                            ])
            q_z = np.vstack([np.array([-1,0,0,0]),
                            np.array([0,1,0,0]), 
                            np.array([0,0,-1,0]),
                            np.array([0,0,0,1]) 
                            ])
            z_q = np.vstack([np.array([-1,0,0,0]),
                            np.array([0,1,0,0]), 
                            np.array([0,0,1,0]),
                            np.array([0,0,0,-1]) 
                            ])
            q_n = np.zeros((4,4))
            q_n[0,0]=1
            q_n[2,2]=1
            
            n_q = np.zeros((4,4))
            n_q[0,0]=1
            n_q[-1,-1]=1
            
            c_q_cp = np.zeros((4,4))
            c_q_cp[1,0]=1
            
            cp_q_c = np.zeros((4,4))
            cp_q_c[0,1]=1
            
            c_q = np.zeros((4,4))
            c_q[2,0]=1
            c_q[1,-1]=1
            
            cp_q = np.zeros((4,4))
            cp_q[-1,1]=1
            cp_q[0,2]=1
            
            q_c = np.zeros((4,4))
            q_c[2,1]=1
            q_c[1,-1]=1
            
            q_cp = np.zeros((4,4))
            q_cp[-1,0]=1

        for n in range(N):  # empty tensors
            H.A[n] = Tensor.Tensor(settings=settings)
            v = vk[n] * (-1j * .5)
            
            if LSR[n] == -1: #L
                vL, vR = v,0
            elif LSR[n] == +1: #R
                vL, vR = 0,v
            
            # local operator - including dissipation
            if abs(LSR[n]) == 1:
                en = wk[n] + dV[n]
                p = 1. / (1. + np.exp(en / temp[n])) if temp[n] > 1e-6 else (1. - np.sign(en))*.5
                pp = np.sqrt(p)
                pm = np.sqrt(1 - p)
                #
                On_Site = -1j*wk[n]*(n_q - q_n) + pp*cp_q_c + pm*c_q_cp + (pp-pm)*.5*(n_q + q_n)
            else: # impurity
                On_Site = -1j*wk[n]*(n_q - q_n)

            if abs(LSR[n]) == 1 and n < n1:
                if n == 0:
                    tmp = np.hstack([On_Site, +vL * c_q, -vL * q_c, vL * cp_q, -vL * q_cp, +vR * c_q, -vR * q_c, vR * cp_q, -vR * q_cp, II])
                    tmp = tmp.reshape((1, 4, 10, 4))
                else:
                    tmp = np.vstack([np.hstack([II    , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([II * 0,     z_q, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([II * 0, II * 0,     q_z, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([II * 0, II * 0, II * 0,     z_q, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([II * 0, II * 0, II * 0, II * 0,     q_z, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([II * 0, II * 0, II * 0, II * 0, II * 0,     z_q, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([II * 0, II * 0, II * 0, II * 0, II * 0, II * 0,     q_z, II * 0, II * 0, II * 0]),
                                     np.hstack([II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0,     z_q, II * 0, II * 0]),
                                     np.hstack([II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0,     q_z, II * 0]),
                                     np.hstack([On_Site, +vL * c_q, -vL * q_c, +vL * cp_q, -vL * q_cp, +vR * c_q, -vR * q_c, +vR * cp_q, -vR * q_cp,  II])])
                    tmp = tmp.reshape((10, 4, 10, 4))
            elif abs(LSR[n]) == 1 and n > n1:
                if n == N - 1:
                    tmp = np.vstack([II, c_q, q_c, cp_q, q_cp, c_q, q_c, cp_q, q_cp, On_Site])
                    tmp = tmp.reshape((10, 4, 1, 4))
                else:
                    tmp = np.vstack([np.hstack([II          , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([+vL * c_q    ,     z_q, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([-vL * q_c    , II * 0,     q_z, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([+vL * cp_q    , II * 0, II * 0,     z_q, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([-vL * q_cp    , II * 0, II * 0, II * 0,     q_z, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([+vR * c_q    , II * 0, II * 0, II * 0, II * 0,     z_q, II * 0, II * 0, II * 0, II * 0]),
                                     np.hstack([-vR * q_c    , II * 0, II * 0, II * 0, II * 0, II * 0,     q_z, II * 0, II * 0, II * 0]),
                                     np.hstack([+vR * cp_q    , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0,     z_q, II * 0, II * 0]),
                                     np.hstack([-vR * q_cp    , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0,     q_z, II * 0]),
                                     np.hstack([On_Site, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0,  II])])
                    tmp = tmp.reshape((10, 4, 10, 4))
            elif n == n1: # site 1 of S in LSR
                tmp = np.vstack([np.hstack([II        , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([c_q    , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([q_c    , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([cp_q   , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([q_cp   , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([c_q    , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([q_c    , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([cp_q   , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([q_cp   , II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([On_Site,    c_q,    q_c,   cp_q,   q_cp,    c_q,    q_c,   cp_q,   q_cp,    II])])
                tmp = tmp.reshape((10, 4, 10, 4))
            # permute legs and save as MPO
            tmp = tmp.transpose((0, 1, 3, 2))
            H.A[n].set_block(val=tmp)
        if AdagA:
            HdagH = stack_MPOs(H, H, [1,1], [1,0])
        else:
            HdagH = None
        return H, LSR, wk, vk, temp, HdagH

def stack_MPOs(UP, DOWN, connection, conj):
    # UP set of mpo - will leave the physical element up
    # DOWN set of mpo - will leave the physical element down
    # the legs - connection[0] of UP and connection[1] of DOWN are contracted
    # conj - list for conjugation
    NEW = UP.copy()
    for it in range(UP.N):
        up = UP.A[it]
        down = DOWN.A[it]
        leg_up = np.zeros(4)
        leg_up[0] = -1
        leg_up[connection[0]] = 1
        leg_up[1 + (connection[0]) % 2] = -4
        leg_up[-1] = -5
        
        leg_dn = np.zeros(4)
        leg_dn[0] = -2
        leg_dn[connection[1]] = 1
        leg_dn[1 + (connection[1]) % 2] = -3
        leg_dn[-1] = -6
        
        # rest
        tmp = ncon([up, down], [leg_up, leg_dn], conj)
        x = tmp.get_shape()
        NEW.A[it].set_block(val = np.reshape(tmp.to_numpy(), (x[0] * x[1], x[2], x[3], x[4] * x[5])))
    return NEW

# MEASURE
def current(LSR, vk, cut = 'LS'):
    """
    Generate a vectorization of an operator to measure the current from reservoir L/R to impurity.

    Parameters
    ----------
    vk: list
        List of hopping aplitudes.

    LSR: list
        Information about the meaning of a site. Is -1 for L, 2 for impurity and +1 for R.

    cut: 'LS' or 'SR'
        Which current should be measured. Only two options 'LS' or 'SR', where S stands for impurity.
    """
    
    N = len(LSR) # total number of sites
    n1 = np.argwhere(LSR == 2)[0,0]
    # dim is virt, phys_out, phys_in, virt
    H = yamps.Mps(N, nr_phys=1, nr_aux=0)
    # vectorized operators
    II = np.array([1,0,0,1])
    z_q = np.array([1,0,0,-1])
    A1 = np.array([0,1,1,0])
    B1 = np.array([0,1,-1,0])
    for n in range(N):  # empty tensors
        H.A[n] = Tensor.Tensor(settings=settings)
        v = vk[n] * .25 
        if cut=='LS':
            vv = v if LSR[n] == -1 else 0
        if cut=='SR':
            vv = v if LSR[n] == +1 else 0
        # local operator - including dissipation
        if abs(LSR[n]) == 1 and n < n1:
            if n == 0:
                tmp = np.hstack([II * 0, vv * A1, -vv * B1, -vv * B1, vv * A1, II])
                tmp = tmp.reshape((1, 4, 6))
            else:
                tmp = np.vstack([np.hstack([II    , II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([II * 0,     z_q, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([II * 0, II * 0,     z_q, II * 0, II * 0, II * 0]),
                                    np.hstack([II * 0, II * 0, II * 0,     z_q, II * 0, II * 0]),
                                    np.hstack([II * 0, II * 0, II * 0, II * 0,     z_q, II * 0]),
                                    np.hstack([II * 0, +vv * A1, -vv * B1, -vv * B1, +vv * A1, II])])
                tmp = tmp.reshape((6, 4, 6))
        elif abs(LSR[n]) == 1 and n > n1:
            if n == N - 1:
                tmp = np.vstack([II, A1, B1, A1, B1, II * 0])
                tmp = tmp.reshape((6, 4, 1))
            else:
                tmp = np.vstack([np.hstack([II    , II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([+vv * A1    , II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([-vv * B1    , II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([+vv * A1    , II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([-vv * B1    , II * 0, II * 0, II * 0, II * 0, II * 0]),
                                    np.hstack([II * 0, II * 0, II * 0, II * 0, II * 0, II])])
                tmp = tmp.reshape((6, 4, 6))
        elif n == n1: # site 1 of S in LSR
            tmp = np.vstack([np.hstack([II    , II * 0, II * 0, II * 0, II * 0, II * 0]),
                                np.hstack([A1    ,     z_q, II * 0, II * 0, II * 0, II * 0]),
                                np.hstack([B1    , II * 0,     z_q, II * 0, II * 0, II * 0]),
                                np.hstack([A1    , II * 0, II * 0,     z_q, II * 0, II * 0]),
                                np.hstack([B1    , II * 0, II * 0, II * 0,     z_q, II * 0]),
                                np.hstack([II * 0,    A1,     B1,     B1,     A1, II])])
            tmp = tmp.reshape((6, 4, 6))
        H.A[n].set_block(val=tmp)
    return H

def occupancy(N, id):
    """
    Generate a vectorization of an operator to measure the occupancy of nthsite in a lattice.

    Parameters
    ----------
    N: int
        Number of sites in a lattice.

    id: int
        Site number to measure occupancy. 
    """
    H = yamps.Mps(N, nr_phys=1, nr_aux=0)
    # vectorized operators
    II = np.array([1,0,0,1])
    nn = np.array([0,0,0,1])
    for n in range(N):  # empty tensors
        H.A[n] = Tensor.Tensor(settings=settings)
        if n==id:
            tmp = np.hstack([nn])
        else:
            tmp = np.hstack([II])
        tmp = tmp.reshape((1, 4, 1))
        H.A[n].set_block(val=tmp)
    return H
