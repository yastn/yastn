# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
""" extending krylov to mps """
import pytest
import yastn.tn.mps as mps
import yastn


def test_krylov(config_kwargs):
    """ krylov in mps """
    #
    N = 10  # Consider a system of 10 sites
    #
    # Load spin-1/2 operators
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    ops.random_seed(seed=0)
    #
    # Hterm-s to generate H = -sum_i X_i X_{i+1} - g * Z_i
    #
    I = mps.product_mpo(ops.I(), N)  # identity MPO
    termsXX = [mps.Hterm(-1, [i, (i + 1) % N], [ops.x(), ops.x()]) for i in range(N)]
    HXX = mps.generate_mpo(I, termsXX)
    termsZ = [mps.Hterm(-1, i, ops.z()) for i in range(N)]
    HZ = mps.generate_mpo(I, termsZ)
    #
    H = HXX + HZ
    psi = mps.random_mps(I, D_total=4)

    # yastn.expmv(lambda x: H @ x, psi, 0.1)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
