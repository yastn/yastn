import numpy as np
import yast
from yast import ncon
from yast.tensor.linalg import svd_with_truncation


def fuse_ancilla_1s(G_loc, fid):
    """ kron and fusion of local gate with identity for ancilla. """
   # Gas = ncon((fid, G_loc), ((-2, -0), (-1, -3)))
    Gas = ncon((G_loc, fid), ((-0, -1), (-2, -3)))
    return Gas.fuse_legs(axes=((0, 3), (1, 2)))


def match_ancilla_1s(G_loc, A):
    """ kron and fusion of local gate with identity for ancilla. Identity is read from ancila of A. """
    leg = A.get_legs(axis=-1)
    if len(leg.legs) == 1 and leg.legs[0].tree[0] == 1:
        return G_loc
    leg, _ = yast.leg_undo_product(leg) # last leg of A should be fused
    fid = yast.eye(config=A.config, legs=[leg.conj(), leg]).diag()
    Gas = ncon((G_loc, fid), ((-0, -1), (-2, -3)))
    Gas = Gas.fuse_legs(axes=((0, 3), (1, 2)))
    return Gas


def fuse_ancilla_2s(GA, GB, fid):
    """ kron and fusion of local gate with identity for ancilla. """
    if GA.ndim == 2:
        GA = GA.add_leg(s=1).swap_gate(axes=(0, 2))
    if GB.ndim == 2:
        GB = GB.add_leg(s=-1)
  #  GAas = ncon((fid, GA), ((-2, -0), (-1, -3, -4)))

    GAas = ncon((GA, fid), ((-0, -1, -4), (-2, -3)))
    
    GAas = GAas.swap_gate(axes=(2, 4))    # swap of connecting axis with ancilla is always in GA gate
    
    #GAas = GAas.fuse_legs(axes=((0, 1), (2, 3), 4))
    GAas = GAas.fuse_legs(axes=((0, 3), (1, 2), 4))

  #  GBas = ncon((fid, GB), ((-2, -0), (-1, -3, -4)))
    GBas = ncon((GB, fid), ((-0, -1, -4), (-2, -3)))
   # GBas = GBas.fuse_legs(axes=((0, 1), (2, 3), 4))
    GBas = GBas.fuse_legs(axes=((0, 3), (1, 2), 4))

    return GAas, GBas


def gates_hopping(t, beta, fid, fc, fcdag, ancilla, purification):
    """ gates for exp[beta * t * (cdag1 c2 + c2dag c1) / 4] """
    # below note that local operators follow convention where
    # they are transposed comparing to typical matrix notation

    if purification == 'False':
        coeff = 0.5
    elif purification == 'True':
        coeff = 0.25

    fn = fcdag@fc
    one = ncon([fid, fid], ((-0, -1), (-2, -3)))
    n1 = ncon([fn, fid], ((-0, -1), (-2, -3)))
    n2 = ncon([fid, fn], ((-0, -1), (-2, -3)))
    nn = ncon([fn, fn], ((-0, -1), (-2, -3)))

    c1dag = fcdag.add_leg(s=1).swap_gate(axes=(0, 2))
    c2 = fc.add_leg(s=-1)
    c1 = fc.add_leg(s=1).swap_gate(axes=(1, 2))
    c2dag = fcdag.add_leg(s=-1)
    cc = ncon([c1dag, c2], ((-0, -1, 1) , (-2, -3, 1))) + \
         ncon([c1, c2dag], ((-0, -1, 1) , (-2, -3, 1)))

    step = coeff * t * beta
    U =  one + (np.cosh(step) - 1) * (n1 + n2 - 2 * nn) + np.sinh(step) * cc
    U, S, V = svd_with_truncation(U, axes = ((0, 1), (2, 3)), sU = -1, tol = 1e-15, Vaxis=2)
    S = S.sqrt()
    GA = S.broadcast(U, axis=2)
    GB = S.broadcast(V, axis=2)
    if ancilla == 'True':
        GA, GB =  fuse_ancilla_2s(GA, GB, fid)
    return GA, GB


def gate_local_Hubbard(mu_up, mu_dn, U, beta, fid, fc_up, fc_dn, fcdag_up, fcdag_dn, ancilla=True, purification = False):
    # local Hubbard gate with chemical potential and Coulomb interaction
    fn_up = fcdag_up @ fc_up
    fn_dn = fcdag_dn @ fc_dn
    fnn = fn_up @ fn_dn

    if purification == 'False':
        coeff = 0.5
    elif purification == 'True':
        coeff = 0.25

    G_loc = fid
    G_loc = G_loc + (fn_dn - fnn) * (np.exp((coeff * beta * (mu_dn + 0.5 * U))) - 1)
    G_loc = G_loc + (fn_up - fnn) * (np.exp((coeff * beta * (mu_up + 0.5 * U))) - 1)
    G_loc = G_loc + fnn * (np.exp((coeff * beta * (mu_up + mu_dn))) - 1)
    if ancilla == 'True':
        G_loc = fuse_ancilla_1s(G_loc, fid)
    return G_loc

def gate_local_fermi_sea(mu, beta, fid, fc, fcdag, ancilla=True, purification=False):
    """ gates for exp[beta * mu * fn / 4] """
    # below note that local operators follow convention where
    # they are transposed comparing to typical matrix notation

    if purification == 'False':
        coeff = 0.5
    elif purification == 'True':
        coeff = 0.25
    fn = fcdag @ fc
    step = coeff * beta * mu
    G_loc = fid + fn * (np.exp(step) - 1)
    if ancilla=='True':
        G_loc = fuse_ancilla_1s(G_loc, fid)
    return G_loc


def Gate_Ising(id, z, J, beta, evolution='imaginary', ancilla=True):

    if evolution == 'imaginary':
        coeff = 0.25
        step = -coeff * beta # coeff comes from either purification and ST 
    elif evolution == 'real':
        coeff = 0.5
        step = -1j *coeff * beta # coeff comes from either purification and ST or ST only 
    """ define 2-site gate for Ising model with longitudinal (hz) and transverse (hx) fields. """
    F = - J * ncon([z, z], ((-0, -1), (-2, -3)))

    F = F.fuse_legs(axes = ((0, 2), (1, 3)))
    D, S = yast.eigh(F, axes = (0, 1))
    D = yast.exp(D, step=step) 
    U = yast.ncon((S, D, S), ([-1, 1], [1, 2], [-3, 2]), conjs=(0, 0, 1))
    U = U.unfuse_legs(axes=(0, 1))
    U = U.transpose(axes=(0, 2, 1, 3))

    U, S, V = svd_with_truncation(U, axes = ((0, 1), (2, 3)), sU = -1, tol = 1e-15, Vaxis=2)
    S = S.sqrt()
    GA = S.broadcast(U, axis=2)
    GB = S.broadcast(V, axis=2)

    if ancilla == True:
        GA = ncon((id, GA), ((-2, -0), (-1, -3, -4)))
        GB = ncon((id, GB), ((-2, -0), (-1, -3, -4)))
        GA = GA.fuse_legs(axes=((0, 1), (2, 3), 4))
        GB = GB.fuse_legs(axes=((0, 1), (2, 3), 4))

    return GA, GB


def Gate_local_dense(x, z, hx, hz, beta, evolution, disorder_ancilla=True, gen_fsigz=None):

    if evolution == 'imaginary':
        coeff = 0.25
        step = -coeff * beta # coeff comes from either purification and ST 
    elif evolution == 'real':
        coeff = 0.5
        step = -1j * coeff * beta # coeff comes from either purification and ST or ST only 

    F = - hx * x
    F = F -  hz * z
    D, S = yast.eigh(F, axes = (0, 1))
    D = yast.exp(D, step=step) 
    G_loc = yast.ncon((S, D, S), ([-1, 1], [1, 2], [-3, 2]), conjs=(0, 0, 1))
    
    if disorder_ancilla == True:
        Gas = ncon((gen_fsigz, G_loc), ((-2, -0), (-1, -3)))
        G_loc = Gas.fuse_legs(axes=((0, 1), (2, 3)))
    
    return G_loc


def Gate_local_dense_floquet(x, h_0, w, beta, dbeta):

    coeff = 0.5   # from suzuki trotter
    step = -1j *coeff * dbeta # coeff comes from either purification and ST 
    hx = h_0*np.cos(w*beta)
    print('field hx', hx)
    F = - hx * x
    D, S = yast.eigh(F, axes = (0, 1))
    D = yast.exp(D, step=step) 
    G_loc = yast.ncon((S, D, S), ([-1, 1], [1, 2], [-3, 2]), conjs=(0, 0, 1))
    
    return G_loc



def Gate_Heisenberg(sp, sm, sz, Jpm, Jmp, Jz, beta, fid_ancilla, ancilla=True):

    coeff = 0.5 # coeff comes from Suzuki Trotter
    step = -1j *coeff * beta 
    """ define 2-site gate for Heisenberg model """
    F =  0.25 * Jz * ncon([sz, sz], ((-0, -1), (-2, -3)))
    F = F + 0.25 * 0.5 * Jpm * ncon([sp, sm], ((-0, -1), (-2, -3)))
    F = F + 0.25 * 0.5 * Jmp * ncon([sm, sp], ((-0, -1), (-2, -3)))

    F = F.fuse_legs(axes = ((0, 2), (1, 3)))
    D, S = yast.eigh(F, axes = (0, 1))
    D = yast.exp(D, step=step) 
    U = yast.ncon((S, D, S), ([-1, 1], [1, 2], [-3, 2]), conjs=(0, 0, 1))
    U = U.unfuse_legs(axes=(0, 1))
    U = U.transpose(axes=(0, 2, 1, 3))

    U, S, V = svd_with_truncation(U, axes = ((0, 1), (2, 3)), sU = -1, tol = 1e-15, Vaxis=2)
    S = S.sqrt()
    GA = S.broadcast(U, axis=2)
    GB = S.broadcast(V, axis=2)

    if ancilla == True:
        GA = ncon((fid_ancilla, GA), ((-2, -0), (-1, -3, -4)))
        GB = ncon((fid_ancilla, GB), ((-2, -0), (-1, -3, -4)))
        GA = GA.fuse_legs(axes=((0, 1), (2, 3), 4))
        GB = GB.fuse_legs(axes=((0, 1), (2, 3), 4))

    return GA, GB


def trivial_tensor(fid):
    """
    fid is identity operator in local space with desired symmetry
    """
    A = (1/np.sqrt(fid.get_shape()[0])) * fid
    A = A.fuse_legs(axes=[(0, 1)])
    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)
    return A.fuse_legs(axes=((0, 1), (2, 3), 4))


