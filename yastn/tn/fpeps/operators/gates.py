import numpy as np
import yastn
from yastn import ncon
from yastn.tensor.linalg import svd_with_truncation


def match_ancilla_1s(G, A):
    """ kron and fusion of local gate with identity for ancilla. Identity is read from ancila of A. """
    leg = A.get_legs(axes=-1)

    if not leg.is_fused():
        if any(n != 0 for n in G.n):
            G = G.add_leg(axis=1, s=-1)
            G = G.fuse_legs(axes=((0, 1), 2))  # new ancilla on outgoing leg
        return G

    _, leg = yastn.leg_undo_product(leg)  # unfuse to get ancilla leg
    one = yastn.eye(config=A.config, legs=[leg, leg.conj()]).diag()
    if all(n == 0 for n in G.n):
        Gsa = ncon((G, one), ((-0, -2), (-1, -3)))
    else:
        G = G.add_leg(axis=1, s=-1)
        Gsa = ncon((G, one), ((-0, -1, -3), (-2, -4)))
        Gsa = Gsa.fuse_legs(axes=(0, (1, 2), 3, 4))
        Gsa = Gsa.drop_leg_history(axes=1)
    Gsa = Gsa.fuse_legs(axes=((0, 1), (2, 3)))
    return Gsa


def match_ancilla_2s(G, A, dir=None):
    """ kron and fusion of local gate with identity for ancilla. """
    leg = A.get_legs(axes=-1)

    if not leg.is_fused():
        return G

    _, leg = yastn.leg_undo_product(leg)  # unfuse to get ancilla leg
    one = yastn.eye(config=A.config, legs=[leg, leg.conj()]).diag()

    if G.ndim == 2:
        if dir=='l':
            G = G.add_leg(s=1, axis=-1).swap_gate(axes=(0, 2))
        elif dir=='r':
            G = G.add_leg(s=-1, axis=-1)
    Gsa = ncon((G, one), ((-0, -2, -4), (-1, -3)))
    if dir == 'l':
        Gsa = Gsa.swap_gate(axes=(3, 4))  # swap of connecting axis with ancilla is always in G gate
    Gsa = Gsa.fuse_legs(axes=((0, 1), (2, 3), 4))
    return Gsa


def gates_hopping(t, beta, fid, fc, fcdag, purification):
    """ gates for exp[beta * t * (cdag1 c2 + c2dag c1) / 4] """
    # below note that local operators follow convention where
    # they are transposed comparing to typical matrix notation

    if purification == 'False':
        coeff = 0.5
    elif purification == 'True':
        coeff = 0.25
    elif purification == 'Time':
        coeff = 1j*0.5

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
    GA = S.broadcast(U, axes=2)
    GB = S.broadcast(V, axes=2)
    return GA, GB

def gate_local_Hubbard(mu_up, mu_dn, U, beta, fid, fc_up, fc_dn, fcdag_up, fcdag_dn, purification = False, checkerboard=False):
    # local Hubbard gate with chemical potential and Coulomb interaction
    fn_up = fcdag_up @ fc_up
    fn_dn = fcdag_dn @ fc_dn
    fnn = fn_up @ fn_dn

    if purification == 'False':
        coeff = 0.5
    elif purification == 'True':
        coeff = 0.25
    elif purification == 'Time':
        coeff = 1j*0.5
    
    if checkerboard is True:
        coeff = 0.5 * coeff

    G_loc = fid
    G_loc = G_loc + (fn_dn - fnn) * (np.exp((coeff * beta * (mu_dn + 0.5 * U))) - 1)
    G_loc = G_loc + (fn_up - fnn) * (np.exp((coeff * beta * (mu_up + 0.5 * U))) - 1)
    G_loc = G_loc + fnn * (np.exp((coeff * beta * (mu_up + mu_dn))) - 1)
    return G_loc

def gate_local_fermi_sea(mu, beta, fid, fc, fcdag, purification=False):
    """ gates for exp[beta * mu * fn / 4] """
    # below note that local operators follow convention where
    # they are transposed comparing to typical matrix notation

    if purification == 'False':
        coeff = -0.5
    elif purification == 'True':
        coeff = -0.25
    elif purification == 'Time':
        coeff = 1j*0.5
    fn = fcdag @ fc
    step = coeff * beta * mu
    G_loc = fid + fn * (np.exp(step) - 1)
    return G_loc


def gate_Ising(sid, sz, J, beta, purification='imaginary'):
    """ define 2-site gate for Ising model with ZZ interactions. """
   
    if purification == 'False': 
        coeff = -0.5*1j
    elif purification == 'True':
        coeff = -0.25*1j
    elif purification == 'Time':
        coeff = 0.5

    one = ncon([sid, sid], ((-0, -1), (-2, -3)))
    ZZ = ncon([sz, sz], ((-0, -1), (-2, -3)))

    step = coeff * beta * J
    D =  np.cos(step) * one + 1j * np.sin(step) * ZZ

    U, S, V = svd_with_truncation(D, axes=((0, 1), (2, 3)), sU=-1, tol=1e-15, Vaxis=2)
    S = S.sqrt()
    GA = S.broadcast(U, axes=2)
    GB = S.broadcast(V, axes=2)
    return GA, GB


def gate_local_dense(sid, sx, sz, hx, hz, beta, purification='imaginary'):

    if purification == 'False':
        coeff = -0.5 * 1j
    elif purification == 'True':
        coeff = -0.25 * 1j
    elif purification == 'Time':
        coeff = 0.5 
    
    step_x = coeff * beta * hx
    step_z = coeff * beta * hz

    G_loc_x = np.cos(step_x) * sid + 1j * np.sin(step_x) * sx
    G_loc_z = np.cos(step_z) * sid + 1j * np.sin(step_z) * sz

    G_loc = G_loc_x @ G_loc_z
    return G_loc




def trivial_tensor(fid):
    """
    fid is identity operator in local space with desired symmetry
    """
    A = (1/np.sqrt(fid.get_shape()[0])) * fid
    A = A.fuse_legs(axes=[(0, 1)])
    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)
    return A.fuse_legs(axes=((0, 1), (2, 3), 4))


