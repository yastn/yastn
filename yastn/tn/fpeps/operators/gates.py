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


def gates_hopping(t, step, fid, fc, fcdag):
    """ gates for exp[step * t * (cdag1 c2 + c2dag c1)] """
    # below note that local operators follow convention where
    # they are transposed comparing to typical matrix notation

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

    tr_step = t * step
    U =  one + (np.cosh(tr_step) - 1) * (n1 + n2 - 2 * nn) + np.sinh(tr_step) * cc
    U, S, V = svd_with_truncation(U, axes = ((0, 1), (2, 3)), sU = -1, tol = 1e-15, Vaxis=2)
    S = S.sqrt()
    GA = S.broadcast(U, axes=2)
    GB = S.broadcast(V, axes=2)
    return GA, GB

def gate_local_Hubbard(mu_up, mu_dn, U, step, fid, fc_up, fc_dn, fcdag_up, fcdag_dn):
    """ gates for exp[- beta * (U * (fn_up-0.5*ident) * (fn_dn-0.5*iden) - mu_up * fn_up - mu_dn * fn_dn];
    we ignore a total of contant U/4 in the gate constructed below """
    # local Hubbard gate with chemical potential and Coulomb interaction
    # below note that local operators follow convention where
    # they are transposed comparing to typical matrix notation
    fn_up = fcdag_up @ fc_up
    fn_dn = fcdag_dn @ fc_dn
    fnn = fn_up @ fn_dn

    G_loc = fid
    G_loc = G_loc + (fn_dn - fnn) * (np.exp((step * (mu_dn + 0.5 * U))) - 1)
    G_loc = G_loc + (fn_up - fnn) * (np.exp((step * (mu_up + 0.5 * U))) - 1)
    G_loc = G_loc + fnn * (np.exp((step * (mu_up + mu_dn))) - 1)
    return G_loc

def gate_local_fermi_sea(mu, step, fid, fc, fcdag):
    """ gates for exp[beta * mu * fn] """
    # below note that local operators follow convention where
    # they are transposed comparing to typical matrix notation

    fn = fcdag @ fc
    tr_step = step * mu
    G_loc = fid + fn * (np.exp(tr_step) - 1)
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


