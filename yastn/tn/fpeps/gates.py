# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
from typing import NamedTuple
from ... import exp, ncon, eigh
from ._gates_auxiliary import fkron

class Gate_nn(NamedTuple):
    """
    ``G0`` should be before ``G1`` in the fermionic and lattice orders
    (``G0`` acts on the left/top site; ``G1`` acts on the right/bottom site from a pair of nearest-neighbor sites).
    The third legs of ``G0`` and ``G1`` are auxiliary legs connecting them into a two-site operator.

    If a ``bond`` is ``None``, this is a general operator.
    Otherwise, ``bond`` carries information where it should be applied
    (potentially, after fixing order mismatches).
    """
    G0 : tuple = None
    G1 : tuple = None
    bond : tuple = None

class Gate_nnn(NamedTuple):
    """
    ``G0``, ``G1`` and ``G2`` should be ordered in the fermionic and lattice orders.
    ``s0``, ``s1`` and ``s2`` correspond to the sites of ``G0``, ``G1`` and ``G2``, respectively.
    """
    G0 : tuple = None
    G1 : tuple = None
    G2 : tuple = None
    site0 : tuple = None
    site1 : tuple = None
    site2 : tuple = None

class Gate_local(NamedTuple):
    r"""
    ``G`` is a local operator with ``ndim=2``.

    If ``site`` is ``None``, this is a general operator.
    Otherwise, ``site`` carries information where it should be applied.
    """
    G : tuple = None
    site : tuple = None


class Gates(NamedTuple):
    r"""
    List of nearest-neighbor and local operators to be applied to PEPS by :meth:`yastn.tn.fpeps.evolution_step_`.
    """
    nn : list = ()   # list of NN gates
    local : list = ()   # list of local gates
    nnn: list = ()   # list of NNN gates


def decompose_nn_gate(Gnn, bond=None) -> Gate_nn:
    r"""
    Auxiliary function cutting a two-site gate with SVD
    into two local operators with the connecting legs.
    """
    U, S, V = Gnn.svd_with_truncation(axes=((0, 2), (1, 3)), sU=-1, tol=1e-14, Vaxis=2)
    S = S.sqrt()
    return Gate_nn(S.broadcast(U, axes=2), S.broadcast(V, axes=2), bond=bond)

def decompose_nnn_gate(Gnnn, s0, s1, s2) -> Gate_nnn:
    r"""
    Auxiliary function cutting a three-site gate with SVD
    into two local operators with the connecting legs.
    """
    U1, S1, V1 = Gnnn.svd_with_truncation(axes=((0, 3), (1, 4, 2, 5)), sU=-1, tol=1e-14, Vaxis=0)
    S1 = S1.sqrt()
    G0 = S1.broadcast(U1, axes=2)
    V1 = S1.broadcast(V1, axes=0)
    U2, S2, V2 = V1.svd_with_truncation(axes=((0, 1, 2), (3, 4)), sU=-1, tol=1e-14, Vaxis=2)
    S2 = S2.sqrt()
    G1 = S2.broadcast(U2, axes=3)
    G2 = S2.broadcast(V2, axes=2)
    return Gate_nnn(G0, G1, G2, site0=s0, site1=s1, site2=s2)

def gate_nn_hopping(t, step, I, c, cdag, bond=None) -> Gate_nn:
    r"""
    Nearest-neighbor gate :math:`G = \exp(-step \cdot H)` for
    :math:`H = -t \cdot (cdag_1 c_2 + cdag_2 c_1)`

    :math:`G = I + (\cosh(x) - 1) (n_1 h_2 + h_1 n_2) + \sinh(x) (cdag_1 c_2 + cdag_2 c_1)`,
    where :math:`x = t \cdot step`
    """
    n = cdag @ c
    h = c @ cdag

    II = fkron(I, I, sites=(0, 1))
    nh = fkron(n, h, sites=(0, 1))
    hn = fkron(h, n, sites=(0, 1))

    cc = fkron(cdag, c, sites=(0, 1)) + fkron(cdag, c, sites=(1, 0))

    G =  II + (np.cosh(t * step) - 1) * (nh + hn) + np.sinh(t * step) * cc
    return decompose_nn_gate(G, bond)


def gate_nn_Ising(J, step, I, X, bond=None) -> Gate_nn:
    r"""
    Nearest-neighbor gate :math:`G = \exp(-step \cdot H)` for
    :math:`H = J X_1 X_2`,
    where :math:`X` is a Pauli matrix.

    :math:`G = \cosh(x) I - \sinh(x) X_1 X_2`,
    where :math:`x = step \cdot J`.
    """
    II = fkron(I, I, sites=(0, 1))
    XX = fkron(X, X, sites=(0, 1))

    G = np.cosh(J * step) * II - np.sinh(J * step) * XX
    return decompose_nn_gate(G, bond)


def gate_nn_tJ(J, tu, td, muu0, muu1, mud0, mud1, step, I, cu, cpu, cd, cpd, bond=None) -> Gate_nn:
    r"""
    Nearest-neighbor gate :math:`G = \exp(-step \cdot H_{tj})`
    """
    nu = cpu @ cu
    nd = cpd @ cd
    Sp = cpu @ cd
    Sm = cpd @ cu

    H = 0 * fkron(I, I, sites=(0, 1))
    H = H + 0.5 * J * fkron(Sp, Sm, sites=(0, 1))
    H = H + 0.5 * J * fkron(Sm, Sp, sites=(0, 1))
    H = H - 0.5 * J * fkron(nu, nd, sites=(0, 1))
    H = H - 0.5 * J * fkron(nd, nu, sites=(0, 1))
    H = H - tu * fkron(cpu, cu, sites=(0, 1))
    H = H - tu * fkron(cpu, cu, sites=(1, 0))
    H = H - td * fkron(cpd, cd, sites=(0, 1))
    H = H - td * fkron(cpd, cd, sites=(1, 0))
    H = H - muu0 * fkron(cpu @ cu, I, sites=(0, 1))
    H = H - muu1 * fkron(I, cpu @ cu, sites=(0, 1))
    H = H - mud0 * fkron(cpd @ cd, I, sites=(0, 1))
    H = H - mud1 * fkron(I, cpd @ cd, sites=(0, 1))

    H = H.fuse_legs(axes = ((0, 1), (2, 3)))
    D, S = eigh(H, axes = (0, 1))
    D = exp(D, step=-step)
    G = ncon((S, D, S), ([-1, 1], [1, 2], [-3, 2]), conjs=(0, 0, 1))
    G = G.unfuse_legs(axes=(0, 1))
    return decompose_nn_gate(G, bond)


def gate_local_Coulomb(mu_up, mu_dn, U, step, I, n_up, n_dn, site=None) -> Gate_local:
    r"""
    Local gate :math:`\exp(-step \cdot H)` for
    :math:`H = U \cdot (n_{up} - I / 2) \cdot (n_{dn} - I / 2) - mu_{up} \cdot n_{up} - mu_{dn} \cdot n_{dn}`

    We ignore a constant :math:`U / 4` in the above Hamiltonian.
    """
    nn = n_up @ n_dn
    G_loc = I
    G_loc = G_loc + (n_dn - nn) * (np.exp(step * (mu_dn + U / 2)) - 1)
    G_loc = G_loc + (n_up - nn) * (np.exp(step * (mu_up + U / 2)) - 1)
    G_loc = G_loc + nn * (np.exp(step * (mu_up + mu_dn)) - 1)
    return Gate_local(G_loc, site)


def gate_local_occupation(mu, step, I, n, site=None) -> Gate_local:
    r"""
    Local gate :math:`G = \exp(-step \cdot H)` for
    :math:`H = -mu \cdot n`

    :math:`G = I + n \cdot (\exp(mu \cdot step) - 1)`
    """
    G_loc = I + n * (np.exp(mu * step) - 1)
    return Gate_local(G_loc, site)


def gate_local_field(h, step, I, X, site=None) -> Gate_local:
    r"""
    Local gate :math:`G = \exp(-step \cdot H)` for
    :math:`H = -h \cdot X`
    where :math:`X` is a Pauli matrix.

    :math:`G = \cosh(h \cdot step) \cdot I + \sinh(h \cdot step) \cdot X`
    """
    G_loc = np.cosh(h * step) * I + np.sinh(h * step) * X
    return Gate_local(G_loc, site)


def distribute(geometry, gates_nn=None, gates_local=None) -> Gates:
    r"""
    Distributes gates homogeneous over the lattice.

    Parameters
    ----------
    geometry : yastn.tn.fpeps.SquareLattice | yastn.tn.fpeps.CheckerboardLattice | yast.tn.fpeps.Peps
        Geometry of PEPS lattice.
        Can be any structure that includes geometric information about the lattice, like the Peps class.

    nn : Gate_nn | Sequence[Gate_nn]
        Nearest-neighbor gate, or a list of gates, to be distributed over all unique lattice bonds.

    local : Gate_local | Sequence[Gate_local]
        Local gate, or a list of local gates, to be distributed over all unique lattice sites.
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
