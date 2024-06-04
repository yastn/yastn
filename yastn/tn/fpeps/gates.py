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
from ._gates_auxiliary import fkron


class Gate_nn(NamedTuple):
    """
    G0 should be before G1 in the fermionic and lattice orders.
    The third legs of G0 and G1 are auxiliary legs connecting them into a two-site operator.

    If a bond is None, this is a general operator.
    Otherwise, the bond can carry information where it should be applied
    (potentially, after fixing the order mismatches).
    """
    G0 : tuple = None
    G1 : tuple = None
    bond : tuple = None


class Gate_local(NamedTuple):
    """
    G is a local operator with ndim==2.

    If site is None, this is a general operator.
    Otherwise, the site can carry information where it should be applied.
    """
    G : tuple = None
    site : tuple = None


class Gates(NamedTuple):
    """
    List of nearest-neighbor and local operators to be applied to PEPS during evolution_step.
    """
    nn : list = None   # list of NN gates
    local : list = None   # list of local gates


def decompose_nn_gate(Gnn) -> Gate_nn:
    """
    Auxiliary function cutting a two-site gate with SVD
    into two local operators with the connecting legs.
    """
    U, S, V = Gnn.svd_with_truncation(axes=((0, 2), (1, 3)), sU=-1, tol=1e-14, Vaxis=2)
    S = S.sqrt()
    return Gate_nn(S.broadcast(U, axes=2), S.broadcast(V, axes=2))


def gate_nn_hopping(t, step, I, c, cdag) -> Gate_nn:
    """
    Nearest-neighbor gate G = exp(-step * H)
    for H = -t * (cdag_1 c_2 + cdag_2 c_1)

    G = I + (cosh(x) - 1) * (n_1 h_2 + h_1 n_2) + sinh(x) * (cdag_1 c_2 + cdag_2 c_1)
    """
    n = cdag @ c
    h = c @ cdag

    II = fkron(I, I, sites=(0, 1))
    nh = fkron(n, h, sites=(0, 1))
    hn = fkron(h, n, sites=(0, 1))

    cc = fkron(cdag, c, sites=(0, 1)) \
       + fkron(cdag, c, sites=(1, 0))

    G =  II + (np.cosh(t * step) - 1) * (nh + hn) + np.sinh(t * step) * cc
    return decompose_nn_gate(G)


def gate_local_Coulomb(mu_up, mu_dn, U, step, I, n_up, n_dn) -> Gate_local:
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


def gate_local_occupation(mu, step, I, n) -> Gate_local:
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
