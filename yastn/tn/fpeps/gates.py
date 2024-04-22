import numpy as np
from typing import NamedTuple
from ... import ncon


class Gate_nn(NamedTuple):
    """ A should be before B in the fermionic order. """
    G0 : tuple = None
    G1 : tuple = None
    bond : tuple = None


class Gate_local(NamedTuple):
    G : tuple = None
    site : tuple = None


class Gates(NamedTuple):
    nn : list = None   # list of NN gates
    local : list = None   # list of local gates


def decompose_nn_gate(Gnn):
    U, S, V = Gnn.svd_with_truncation(axes=((0, 2), (1, 3)), sU=-1, tol=1e-14, Vaxis=2)
    S = S.sqrt()
    return Gate_nn(S.broadcast(U, axes=2), S.broadcast(V, axes=2))


def gate_nn_hopping(t, step, I, c, cdag):
    """
    Nearest-neighbor gate G = exp(-step * H)
    for H = -t * (cdag1 c2 + c2dag c1)

    G = I + (cosh(x) - 1) * (n1 + n2 - 2 n1 n2) + sinh(x) * (c1dag c2 + c2dag c1)
    """
    n = cdag @ c
    II = ncon([I, I], [(-0, -2), (-1, -3)])
    n1 = ncon([n, I], [(-0, -2), (-1, -3)])
    n2 = ncon([I, n], [(-0, -2), (-1, -3)])
    nn = ncon([n, n], [(-0, -2), (-1, -3)])

    # site-1 is before site-2 in fermionic order
    # c1dag c2;
    c1dag = cdag.add_leg(s=1).swap_gate(axes=(0, 2))
    c2 = c.add_leg(s=-1)
    cc = ncon([c1dag, c2], [(-0, -2, 1) , (-1, -3, 1)])

    # c2dag c1
    c1 = c.add_leg(s=1).swap_gate(axes=(1, 2))
    c2dag = cdag.add_leg(s=-1)
    cc = cc + ncon([c1, c2dag], [(-0, -2, 1) , (-1, -3, 1)])

    G =  II + (np.cosh(t * step) - 1) * (n1 + n2 - 2 * nn) + np.sinh(t * step) * cc
    return decompose_nn_gate(G)


# def Hamiltonian_nn_hopping():
#     #  c1 dag c2 + c2dag c1
#     pass
#     c1 = c.add_leg(s=1).swap_gate(axes=(1, 2))
#     c2dag = cdag.add_leg(s=-1)
#     cc = cc + ncon([c1, c2dag], [(-0, -2, 1) , (-1, -3, 1)])
#     return decompose_nn_gate(cc)



def gate_local_Coulomb(mu_up, mu_dn, U, step, I, n_up, n_dn):
    """
    Local gate exp(-step * H)
    for H = U * (n_up - I / 2) * (n_dn - I / 2) - mu_up * n_up - mu_dn * n_dn

    We ignore a constant U / 4 in the above Hamiltonian.
    """
    nn = n_up @ n_dn
    G_loc = I
    G_loc = G_loc + (n_dn - nn) * (np.exp(step * (mu_dn + U / 2)) - 1)
    G_loc = G_loc + (n_up - nn) * (np.exp(step * (mu_up + U / 2)) - 1)
    G_loc = G_loc + nn * (np.exp(step * (mu_up + mu_dn)) - 1)
    return Gate_local(G_loc)


# def Hamiltonian_local_Coulomb(n_up, n_dn):
#     return Gate_local(n_up @ n_dn)

def gate_local_occupation(mu, step, I, n):
    """
    Local gate exp(-step * H)
    for H = -mu * n
    """
    return Gate_local(I + n * (np.exp(mu * step) - 1))


def distribute(geometry, gates_nn=None, gates_local=None) -> Gates:
    """
    Distributes gates homogeneous over the lattice.

    Parameters
    ----------
    geomtry : yastn.tn.fpeps.SquareLattice | yastn.tn.fpeps.CheckerboardLattice | yast.tn.fpeps.Peps
        Geometry of PEPS lattice.
        Can be any structure that includes geometric information about the lattice, like the Peps class.

    nn : Gate_nn | Sequence[Gate_nn]
        Nearest-neighbor gate, or a list of gates, to be distributed over lattice bonds.

    local : Gate_local | Sequence[Gate_local]
        Local gate, or a list of local gates, to be distributed over lattice sites.
    """

    if isinstance(gates_nn, Gate_nn):
        gates_nn = [gates_nn]

    nn = []
    if gates_nn is not None:
        for bond in geometry.bonds():
            for Gnn in gates_nn:
                nn.append(Gnn._replace(bond=bond))

    if isinstance(gates_local, Gate_local):
        gates_local = [gates_local]

    local = []
    if gates_local is not None:
        for site in geometry.sites():
            for Gloc in gates_local:
                local.append(Gloc._replace(site=site))

    return Gates(nn=nn, local=local)
