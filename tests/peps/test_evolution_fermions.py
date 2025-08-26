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
             mps.Hterm(np.cosh(angle) - 1, [p0, p1], [h, n]),
             mps.Hterm(np.sinh(angle), [p0, p1], [cp, c]),
             mps.Hterm(np.sinh(angle), [p1, p0], [cp, c])]
    return mps.generate_mpo(I, terms, N=N, f_map=f_map)

@pytest.mark.parametrize('boundary', ['obc', 'cylinder'])
def test_peps_evolution_hopping(config_kwargs, boundary):
    """ Simulate spinless fermions hopping on a small finite system. """
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', default_dtype='complex128', **config_kwargs)
    #
    Nx, Ny = 3, 2
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary=boundary)
    #
    paths_list = [geometry.bonds(),
                  [((0, 0), (1, 0), (1, 1), (2, 1)), ((2, 0), (1, 0), (1, 1), (0, 1))],
                  [((2, 0), (1, 0), (1, 1), (2, 1)), ((2, 1), (2, 0), (1, 0), (1, 1), (0, 1), (0, 0))]
                  ]
    if boundary == 'cylinder':
        paths_list.append([((0, 0), (2, 0), (1, 0), (1, 1)), ((2, 1), (0, 1), (0, 0))])

    for paths in paths_list:
        angles = {path: np.random.rand() + 1j * np.random.rand() - 0.5 - 0.5j for path in paths}
        #
        psi = fpeps.product_peps(geometry, ops.I())
        gates = [fpeps.gates.gate_nn_hopping(angle, 1, ops.I(), ops.c(), ops.cp(), path)
                  for path, angle in angles.items()]
        # gates = [fpeps.Gate(mpo_hopping(ops, len(path), 0, len(path) - 1, angle), path)
        #          for path, angle in angles.items()]
        for gate in gates:
            psi.apply_gate_(gate)
        psi = psi.to_tensor()
        #
        s2i = {s: i for i, s in enumerate(geometry.sites())}  # 1d order of fermionically-ordered sites for mps
        #
        phi = mps.product_mpo(ops.I(), N=Nx * Ny)
        gates = [mpo_hopping(ops, Nx * Ny, s2i[path[0]], s2i[path[-1]], angle) for path, angle in angles.items()]
        for gate in gates:
            phi = gate @ phi
        phi = phi.to_tensor()

        assert (psi - phi).norm() < tol


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
