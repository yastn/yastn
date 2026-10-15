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
import pytest
import yastn

numtol = 1e-16
errtol = 1e-12
pinv_tol = 1e-13

def test_dense(config_kwargs):
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=0)  # fix seed for testing

    D = 10
    a = yastn.rand(config=config, s=(-1, 1), D=[D, D])
    b = yastn.rand(config=config, s=(-1,), D=[D])
    f = lambda x: a @ x

    v0 = b.copy() * 0
    _, res = yastn.lin_solver(f, b, v0, ncv=D, tol=numtol, pinv_tol=pinv_tol, hermitian=False)
    assert res < errtol

    d1, d2 = 10, 2
    a = yastn.rand(config=config, s=(-1, 1, 1, -1), D=[d1, d2, d1, d2])
    b = yastn.rand(config=config, s=(-1, 1), D=[d1, d2])
    f = lambda x: a.tensordot(x, axes=((2,3), (0,1)))

    v0 = b.copy() * 0
    _, res = yastn.lin_solver(f, b, v0, ncv=d1*d2, tol=numtol, pinv_tol=pinv_tol, hermitian=False)
    assert res < errtol


def test_U1(config_kwargs):
    config = yastn.make_config(sym='U1', **config_kwargs)
    config.backend.random_seed(seed=0)  # fix seed for testing

    leg = yastn.Leg(config, s=-1, t=(-1, 0, 1), D=(2, 3, 4))
    a = yastn.rand(config=config, legs=[leg, leg.conj()])
    b = yastn.rand(config=config, legs=[leg])
    f = lambda x: a @ x
    v0 = b.copy() * 0
    _, res = yastn.lin_solver(f, b, v0, ncv=sum(leg.D), tol=numtol, pinv_tol=pinv_tol, hermitian=False)
    assert res < errtol

    leg1 = yastn.Leg(config, s=-1, t=(-1, 0, 1), D=(2, 3, 4))
    leg2 = yastn.Leg(config, s=1, t=(-1, 0, 1), D=(3, 2, 2))
    a = yastn.rand(config=config, legs=[leg, leg.conj(), leg.conj(), leg])
    b = yastn.rand(config=config, legs=[leg, leg.conj()])
    f = lambda x: a.tensordot(x, axes=((2,3), (0,1)))

    v0 = b.copy() * 0
    _, res = yastn.lin_solver(f, b, v0, ncv=sum(leg1.D)*sum(leg2.D), tol=numtol, pinv_tol=pinv_tol, hermitian=False)
    assert res < errtol

if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--device", "cuda"])
