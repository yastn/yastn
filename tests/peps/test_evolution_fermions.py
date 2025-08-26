# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
""" Test evolution of PEPS using an example of quantum Fourier transform. """
import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps

tol = 1e-12  #pylint: disable=invalid-name


def mpo_hopping(ops, N, p0, p1, angle, f_map=None):
    r"""
    :math:`G = I + (\cosh(x) - 1) (n_0 h_1 + h_0 n_1) + \sinh(x) (cdag_0 c_1 + cdag_1 c_0)`,
    """
    I, cp, c = ops.I(), ops.cp(), ops.c()
    n, h = cp @ c, c @ cp
    terms = [mps.Hterm(1, [p0, p1], [I, I]),
             mps.Hterm(np.cosh(angle) - 1, [p0, p1], [n, h]),
             # mps.Hterm(np.cosh(angle) - 1, [p0, p1], [h, n]),
             # mps.Hterm(-np.sinh(angle), [p0, p1], [cp, c]),
             mps.Hterm(np.sinh(angle), [p1, p0], [cp, c])]
    return mps.generate_mpo(I, terms, N=N, f_map=f_map)


def gate_hopping(ops, sites, angle):
    G = mpo_hopping(ops, len(sites), 0, len(sites) - 1, angle)

    # G[0] = G[0].swap_gate(axes=(2, 2))
    #G1 = mpo_hopping(ops, len(sites), 0, len(sites) - 1, angle, f_map= [1, 0])

    # G0 = G.shallow_copy()
    # G0[0] = G0[0].swap_gate(axes=(2, 3))
    # G0[len(sites) - 1] = G0[len(sites) - 1].swap_gate(axes=(0, 1))

    # print((G - G0).norm())
    #print((G - G1).norm())

    return fpeps.Gate(G=G, sites=sites)


def test_peps_evolution_hopping(config_kwargs):
    """ Simulate spinless fermions hopping on a small finite system. """
    #
    Nx, Ny = 3, 2
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary='obc')
    #geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary='cylinder')

    sites = geometry.sites()
    bonds = list(geometry.bonds())
    #bonds = []

    #bonds = bonds + [((2, 0), (0, 0))]

    # bonds = bonds + [((0, 0), (1, 0), (1, 1), (2, 1))]
    # bonds = bonds + [((2, 0), (1, 0), (1, 1), (0, 1))]
    # # # bonds = bonds + [((0, 0), (1, 0), (2, 0))]
    # bonds = bonds + [((2, 0), (1, 0), (0, 0))]

    # bonds = bonds + [((1, 0), (0, 0), (0, 1), (1, 1))]

    # bonds = bonds + [((0, 0), (1, 0), (1, 1))]
    # bonds = bonds + [((0, 0), (0, 1), (1, 1))]
    # bonds = bonds + [((0, 1), (0, 0), (1, 0))]
    # bonds = bonds + [((1, 0), (1, 1), (0, 1))]
    #bonds = bonds + [((1, 0), (0, 0), (0, 1))]
    #bonds = bonds + [((2, 0), (1, 0), (1, 1), (0, 1))]

    #bonds = bonds + [((2, 0), (1, 0), (1, 1), (0, 1))[::-1]]


    np.random.seed(seed=0)
    angles = {bond: np.random.rand() + 1j * np.random.rand() - 0.5 - 0.5j for bond in bonds}
    ops = yastn.operators.SpinlessFermions(sym='U1', default_dtype='complex128', **config_kwargs)
    #
    psi = fpeps.product_peps(geometry, ops.I())
    gates = [gate_hopping(ops, sites, angle) for sites, angle in angles.items()]

    for gate in gates:
        psi.apply_gate_(gate)
    psi = psi.to_tensor()
    #
    i2s = dict(enumerate(sites))
    s2i = {s: i for i, s in i2s.items()}  # 1d order of sites for mps
    #
    phi = mps.product_mpo(ops.I(), N=Nx * Ny)
    gates = [mpo_hopping(ops, Nx * Ny, s2i[bond[0]], s2i[bond[-1]], angle) for bond, angle in angles.items()]
    for gate in gates:
        phi = gate @ phi
    phi = phi.to_tensor()

    assert (psi - phi).norm() < tol


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
