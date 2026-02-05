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
from typing import Sequence

import numpy as np

from ._gates_auxiliary import fkron, Gate
from ...tensor import exp, ncon, eigh


def Gate_local(G, site):
    """
    Legacy function, generating :class:`Gate` instance.
    """
    return Gate(G=(G,), sites=(site,))


def Gate_nn(G0, G1, bond):
    """
    Legacy function, generating :class:`Gate` instance.

    ``G0`` should be before ``G1`` in the fermionic and lattice orders
    (``G0`` acts on the left/top site; ``G1`` acts on the right/bottom site from a pair of nearest-neighbor sites).
    The third legs of ``G0`` and ``G1`` are auxiliary legs connecting them into a two-site operator.
    """
    return Gate(G=(G0, G1), sites=bond)


def decompose_nn_gate(Gnn, bond=None) -> Gate:
    r"""
    Auxiliary function to generate Gate by cutting a two-site operator,
    using SVD, into two local operators with the connecting legs.
    """
    U, S, V = Gnn.svd_with_truncation(axes=((0, 1), (2, 3)), sU=-1, tol=1e-14, Vaxis=2)
    S = S.sqrt()
    return Gate_nn(S.broadcast(U, axes=2), S.broadcast(V, axes=2), bond)


def gate_nn_exp(step, I, H, bond=None) -> Gate:
    r"""
    Gate exp(-step * H) for a Hermitian Hamiltonian H,
    consistent with leg order of :meth:`yastn.tn.fpeps.gates.fkron`.
    Add 0 * I to the Hamiltonian to avoid situation,
    where some blocks are missing in the Hamiltonian.
    """
    H = H + 0 * fkron(I, I)
    H = H.fuse_legs(axes = ((0, 2), (1, 3)))
    D, U = eigh(H, axes = (0, 1))
    D = exp(D, step=-step)
    G = ncon((U, D, U.conj()), ([-1, 1], [1, 2], [-3, 2]))
    G = G.unfuse_legs(axes=(0, 1)).transpose(axes=(0, 2, 1, 3))
    return decompose_nn_gate(G, bond)


def gate_local_exp(step, I, H, site=None) -> Gate:
    r"""
    Gate exp(-step * H) for local Hamiltonian H.
    Add 0 * I to the Hamiltonian to avoid situation,
    where some blocks are missing in the Hamiltonian.
    """
    H = H + 0 * I
    D, S = eigh(H, axes = (0, 1))
    D = exp(D, step=-step)
    G = ncon((S, D, S), ([-1, 1], [1, 2], [-3, 2]), conjs=(0, 0, 1))
    return Gate_local(G, site)


def gate_nn_hopping(t, step, I, c, cdag, bond=None) -> Gate:
    r"""
    Nearest-neighbor gate :math:`G = \exp(-step \cdot H)` for
    :math:`H = -t \cdot (c^\dagger_1 c_2 + c^\dagger_2 c_1)`

    :math:`G = I + (\cosh(x) - 1) (n_1 h_2 + h_1 n_2) + \sinh(x) (c^\dagger_1 c_2 + c^\dagger_2 c_1)`,
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


def gate_nn_Ising(J, step, I, X, bond=None) -> Gate:
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


def gate_nn_Heisenberg(J, step, I, Sz, Sp, Sm, bond=None) -> Gate:
    r"""
    Nearest-neighbor gate :math:`G = \exp(-step \cdot H)` for
    :math:`H = J (S_z S_z + 2 * S_+ S_- + 2 * S_- S_+)`,
    for spin operators :math:`S_z, S_+, S_-`.
    """
    H = 0.5 * J * fkron(Sp, Sm) \
      + 0.5 * J * fkron(Sm, Sp) \
      + J * fkron(Sz, Sz)

    return gate_nn_exp(step, I, H, bond)


def gate_nn_tJ(J, tu, td, muu0, muu1, mud0, mud1, step, I, cu, cpu, cd, cpd, bond=None) -> Gate:
    r"""
    Nearest-neighbor gate :math:`G = \exp(-step \cdot H)` for
    :math:`H = -t \sum_{\sigma} (c_{0,\sigma}^\dagger c_{1,\sigma} + c_{1,\sigma}^\dagger c_{0,\sigma}) + J (S_i \cdot S_j - \frac{n_i n_j}{4}) - \sum_{i, \sigma} \mu_{i,\sigma} n_{i,\sigma}`
    """
    nu = cpu @ cu
    nd = cpd @ cd
    Sp = cpu @ cd
    Sm = cpd @ cu

    H = 0.5 * J * fkron(Sp, Sm, sites=(0, 1)) \
      + 0.5 * J * fkron(Sm, Sp, sites=(0, 1)) \
      - 0.5 * J * fkron(nu, nd, sites=(0, 1)) \
      - 0.5 * J * fkron(nd, nu, sites=(0, 1)) \
      - tu * fkron(cpu, cu, sites=(0, 1)) \
      - tu * fkron(cpu, cu, sites=(1, 0)) \
      - td * fkron(cpd, cd, sites=(0, 1)) \
      - td * fkron(cpd, cd, sites=(1, 0)) \
      - muu0 * fkron(cpu @ cu, I, sites=(0, 1)) \
      - muu1 * fkron(I, cpu @ cu, sites=(0, 1)) \
      - mud0 * fkron(cpd @ cd, I, sites=(0, 1)) \
      - mud1 * fkron(I, cpd @ cd, sites=(0, 1)) \

    return gate_nn_exp(step, I, H, bond)


def gate_local_Coulomb(mu_up, mu_dn, U, step, I, n_up, n_dn, site=None) -> Gate:
    r"""
    Local gate :math:`\exp(-step \cdot H)` for
    :math:`H = U \cdot (n_{up} - I / 2) \cdot (n_{dn} - I / 2) - \mu_{up} \cdot n_{up} - \mu_{dn} \cdot n_{dn}`

    We ignore a constant :math:`U / 4` in the above Hamiltonian.
    """
    nn = n_up @ n_dn
    G_loc = I
    G_loc = G_loc + (n_dn - nn) * (np.exp(step * (mu_dn + U / 2)) - 1)
    G_loc = G_loc + (n_up - nn) * (np.exp(step * (mu_up + U / 2)) - 1)
    G_loc = G_loc + nn * (np.exp(step * (mu_up + mu_dn)) - 1)
    return Gate_local(G_loc, site)


def gate_local_occupation(mu, step, I, n, site=None) -> Gate:
    r"""
    Local gate :math:`G = \exp(-step \cdot H)` for
    :math:`H = -\mu \cdot n`

    :math:`G = I + n \cdot (\exp(\mu \cdot step) - 1)`
    """
    G_loc = I + n * (np.exp(mu * step) - 1)
    return Gate_local(G_loc, site)


def gate_local_field(h, step, I, X, site=None) -> Gate:
    r"""
    Local gate :math:`G = \exp(-step \cdot H)` for
    :math:`H = -h \cdot X`
    where :math:`X` is a Pauli matrix.

    :math:`G = \cosh(h \cdot step) \cdot I + \sinh(h \cdot step) \cdot X`
    """
    G_loc = np.cosh(h * step) * I + np.sinh(h * step) * X
    return Gate_local(G_loc, site)


def distribute(geometry, gates_nn=None, gates_local=None, symmetrize=True, reverse_sites=False) -> Sequence[Gate]:
    r"""
    Distributes gates homogeneously over the lattice.

    Parameters
    ----------
    geometry : yastn.tn.fpeps.SquareLattice | yastn.tn.fpeps.CheckerboardLattice | yast.tn.fpeps.Peps
        Geometry of PEPS lattice.
        Can be any structure that includes geometric information about the lattice, like the Peps class.

    gates_nn : Gate | Sequence[Gate]
        Nearest-neighbor gate, or a list of gates, to be distributed over all unique lattice bonds.

    gates_local : Gate | Sequence[Gate]
        Local gate, or a list of local gates, to be distributed over all unique lattice sites.

    symmetrize: bool
        Whether to iterate through provided gates forward and then backward, resulting in a 2nd order method.
        In that case, each gate should correspond to half of the desired timestep. The default is ``True``.

    reverse_sites: bool
        If symmetrize, whether to also reverse the sites associated with the gate.
        This affects truncation order in 3-site gates, but requires a symmetric gate operator.
        The default is ``False``.
    """
    nn = []
    if gates_nn is not None:
        if isinstance(gates_nn, Gate):
            gates_nn = [gates_nn]
        for bond in geometry.bonds():
            for Gnn in gates_nn:
                nn.append(Gnn._replace(sites=bond))

    local = []
    if gates_local is not None:
        if isinstance(gates_local, Gate):
            gates_local = [gates_local]
        for site in geometry.sites():
            for Gloc in gates_local:
                local.append(Gloc._replace(sites=(site,)))

    gates = nn + local

    if symmetrize:
        gates_back = gates[::-1]
        if reverse_sites:
            gates_back = [gate._replace(sites=gate.sites[::-1]) for gate in gates_back]
        gates = gates + gates_back
    return gates
