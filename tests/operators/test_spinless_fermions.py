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
""" Predefined spinless fermion operators. """
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_spinless_fermions(config_kwargs):
    """ Generate standard operators in two-dimensional Hilbert space for various symmetries. """
    ops_Z2 = yastn.operators.SpinlessFermions(sym='Z2', **config_kwargs)
    # other way to initialize
    config_U1 = yastn.make_config(fermionic=True, sym="U1", **config_kwargs)
    ops_U1 = yastn.operators.SpinlessFermions(**config_U1._asdict())

    assert all(ops.config.fermionic == True for ops in (ops_Z2, ops_U1))

    opss = [ops_Z2, ops_U1]
    Is = [ops.I() for ops in opss]
    legs = [ops.space() for ops in opss]

    assert all(ops.key() is None for ops in opss) # SpinlessFermions charge are naturally ordered for all symmetries, so key is None
    assert all(leg == I.get_legs(axes=0) for (leg, I) in zip(legs, Is))
    assert all(np.allclose(I.to_numpy(), np.eye(2)) for I in Is)

    lss = [{0: I.get_legs(0), 1: I.get_legs(1)} for I in Is]

    ns = [ops_Z2.n(), ops_U1.n()]
    assert all(np.allclose(n.to_numpy(legs=ls), np.array([[0, 0], [0, 1]])) for n, ls in zip(ns, lss))

    cps = [ops_Z2.cp(), ops_U1.cp()]
    assert all(np.allclose(cp.to_numpy(legs=ls), np.array([[0, 0], [1, 0]])) for cp, ls in zip(cps, lss))

    cs = [ops_Z2.c(), ops_U1.c()]
    assert all(np.allclose(c.to_numpy(legs=ls), np.array([[0, 1], [0, 0]])) for c, ls in zip(cs, lss))

    assert all(yastn.norm(cp @ c - n) < tol for cp, c, n in zip(cps, cs, ns))

    n0s = [ops_Z2.vec_n(val=0), ops_U1.vec_n(val=0)]
    n1s = [ops_Z2.vec_n(val=1), ops_U1.vec_n(val=1)]

    assert all(yastn.norm(n @ v) < tol for n, v in zip(ns, n0s))
    assert all(yastn.norm(n @ v - v) < tol for n, v in zip(ns, n1s))
    assert all(yastn.norm(cp @ v0 - v1) < tol for cp, v0, v1 in zip(cps, n0s, n1s))
    assert all(yastn.norm(c @ v1 - v0) < tol for c, v0, v1 in zip(cs, n0s, n1s))
    assert all(yastn.norm(c @ v0) < tol for c, v0 in zip(cs, n0s))
    assert all(yastn.norm(cp @ v1) < tol for cp, v1 in zip(cps, n1s))

    with pytest.raises(yastn.YastnError):
        yastn.operators.SpinlessFermions(sym='dense')
        # For SpinlessFermions sym should be in ('Z2', 'U1').
    with pytest.raises(yastn.YastnError):
        yastn.operators.SpinlessFermions(sym='U1', fermionic=False)
        # For SpinlessFermions config.fermionic should be True.
    with pytest.raises(yastn.YastnError):
        ops_U1.vec_n(val=-1)
        # Occupation val should be in (0, 1).

    d = ops_Z2.to_dict()  # dict used in mps.Generator
    (d["I"](3) - ops_Z2.I()).norm() < tol  # here 3 is a posible position in the mps
    assert all(k in d for k in ('I', 'n', 'c', 'cp'))


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
