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
""" Predefined spin-1/2 operators. """
from itertools import chain
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_spin12(config_kwargs):
    """ Standard operators and some vectors in two-dimensional Hilbert space for various symmetries. """
    ops_dense = yastn.operators.Spin12(sym='dense', **config_kwargs)
    ops_Z2 = yastn.operators.Spin12(sym='Z2', **config_kwargs)
    # other way to initialize
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    ops_U1 = yastn.operators.Spin12(**config_U1._asdict())

    rs = (False, False, True) # reverse option for to_numpy/to_dense/to_nonsymmetric

    assert all(ops.config.fermionic == False for ops in (ops_dense, ops_Z2, ops_U1))

    Is = [ops_dense.I(), ops_Z2.I(), ops_U1.I()]
    legs = [ops_dense.space(), ops_Z2.space(), ops_U1.space()]

    assert all(leg == I.get_legs(axes=0) for (leg, I) in zip(legs, Is))
    assert all(np.allclose(I.to_numpy(reverse=r), np.eye(2)) for (I, r) in zip(Is, rs))

    zs = [ops_dense.z(), ops_Z2.z(), ops_U1.z()]
    szs = [ops_dense.sz(), ops_Z2.sz(), ops_U1.sz()]
    assert all(np.allclose(z.to_numpy(reverse=r), np.array([[1, 0], [0, -1]])) for (z, r) in zip(zs, rs))

    xs = [ops_dense.x(), ops_Z2.x()]
    sxs = [ops_dense.sx(), ops_Z2.sx()]
    assert all(np.allclose(x.to_numpy(reverse=r), np.array([[0, 1], [1, 0]])) for (x, r) in zip(xs, rs))

    ys = [ops_dense.y(), ops_Z2.y()]
    sys = [ops_dense.sy(), ops_Z2.sy()]
    assert all(np.allclose(y.to_numpy(reverse=r), np.array([[0, -1j], [1j, 0]])) for (y, r) in zip(ys, rs))

    iys = [ops_dense.iy(), ops_Z2.iy()]
    isys = [ops_dense.isy(), ops_Z2.isy()]
    assert all((1j * y - iy).norm() < tol for (y, iy) in zip(ys, iys))
    assert all((1j * sy - isy).norm() < tol for (sy, isy) in zip(sys, isys))

    lss = [{0: I.get_legs(0), 1: I.get_legs(1)} for I in Is]

    sps = [ops_dense.sp(), ops_Z2.sp(), ops_U1.sp()]
    assert all(np.allclose(sp.to_numpy(legs=ls, reverse=r), np.array([[0, 1], [0, 0]])) for sp, ls, r in zip(sps, lss, rs))

    sms = [ops_dense.sm(), ops_Z2.sm(), ops_U1.sm()]
    assert all(np.allclose(Sm.to_numpy(legs=ls, reverse=r), np.array([[0, 0], [1, 0]])) for Sm, ls, r in zip(sms, lss, rs))

    assert all(yastn.norm(sx + 1j * sy - sp) < tol for sx, sy, sp in zip(sxs, sys, sps))
    assert all(yastn.norm(sx - 1j * sy - sm) < tol for sx, sy, sm in zip(sxs, sys, sms))

    assert all(yastn.norm(sp @ sm - sm @ sp - 2 * sz) < tol for sp, sm, sz in zip(sps, sms, szs))
    assert all(yastn.norm(sx @ sy - sy @ sx - 1j * sz) < tol for sx, sy, sz in zip(sxs, sys, szs))
    assert all(yastn.norm(sz @ sp - sp @ sz - sp) < tol for sz, sp in zip(szs, sps))
    assert all(yastn.norm(sz @ sm - sm @ sz + sm) < tol for sz, sm in zip(szs, sms))

    zp1s = [ops_dense.vec_z(val=+1), ops_Z2.vec_z(val=+1), ops_U1.vec_z(val=+1)]
    zm1s = [ops_dense.vec_z(val=-1), ops_Z2.vec_z(val=-1), ops_U1.vec_z(val=-1)]
    xp1s = [ops_dense.vec_x(val=+1)]
    xm1s = [ops_dense.vec_x(val=-1)]
    yp1s = [ops_dense.vec_y(val=+1)]
    ym1s = [ops_dense.vec_y(val=-1)]

    assert all(yastn.norm(z @ v - v) < tol for z, v in zip(zs, zp1s))
    assert all(yastn.norm(z @ v + v) < tol for z, v in zip(zs, zm1s))
    assert all(yastn.norm(x @ v - v) < tol for x, v in zip(xs, xp1s))
    assert all(yastn.norm(x @ v + v) < tol for x, v in zip(xs, xm1s))
    assert all(yastn.norm(y @ v - v) < tol for y, v in zip(ys, yp1s))
    assert all(yastn.norm(y @ v + v) < tol for y, v in zip(ys, ym1s))

    assert all(abs(v.norm() - 1) < tol for v in chain(zp1s, zm1s, xp1s, xm1s, yp1s, ym1s))

    with pytest.raises(yastn.YastnError):
        _ = ops_U1.x()
        # Cannot define sigma_x operator for U1 symmetry
    with pytest.raises(yastn.YastnError):
        _ = ops_U1.y()
        # Cannot define sigma_y operator for U1 symmetry
    with pytest.raises(yastn.YastnError):
        _ = ops_U1.iy()
        # Cannot define sigma_y operator for U1 symmetry.
    with pytest.raises(yastn.YastnError):
        yastn.operators.Spin12(sym='wrong symmetry')
        # For Spin12 sym should be in ('dense', 'Z2', 'U1').
    with pytest.raises(yastn.YastnError):
        yastn.operators.Spin12(sym='U1', fermionic=True)
        # For Spin12 config.fermionic should be False.
    with pytest.raises(yastn.YastnError):
        yastn.operators.Spin12(sym='U1xU1')
        # For Spin12 sym should be in ('dense', 'Z2', 'U1').
    with pytest.raises(yastn.YastnError):
        ops_U1.vec_z(val=10)
        # Eigenvalues val should be in (-1, 1).
    with pytest.raises(yastn.YastnError):
        ops_dense.vec_x(val=10)
        # Eigenvalues val should be in (-1, 1) and eigenvectors of Sx are well defined only for dense tensors.
    with pytest.raises(yastn.YastnError):
        ops_Z2.vec_y(val=1)
        # Eigenvalues val should be in (-1, 1) and eigenvectors of Sy are well defined only for dense tensors.

    # used in mps Generator
    d = ops_dense.to_dict()
    (d["I"](3) - ops_dense.I()).norm() < tol  # here 3 is a posible position in the mps
    assert all(k in d for k in ('I', 'x', 'y', 'z', 'sx', 'sy', 'sz', 'sp', 'sm'))


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
