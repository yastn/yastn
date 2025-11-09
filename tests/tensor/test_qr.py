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
""" yastn.linalg.qr() """
from itertools import product
import pytest
import yastn

tol = 1e-10  #pylint: disable=invalid-name


def run_qr_combine(a):
    """ decompose and contracts tensor ``a`` using qr decomposition """
    assert a.ndim == 4

    def check_diag_R_nonnegative(R):
        """ checks that diagonal of R is selected to be non-negative """
        for t in R.struct.t:
            assert all(R.config.backend.diag_get(R.real()[t]) >= 0)
            assert all(R.config.backend.diag_get(R.imag()[t]) == 0)

    Q, R = yastn.linalg.qr(a, axes=((3, 1), (2, 0)))
    QR = yastn.tensordot(Q, R, axes=(2, 0))
    QR = QR.transpose(axes=(3, 1, 2, 0))
    assert yastn.norm(a - QR) < 1e-12  # == 0.0
    assert Q.is_consistent()
    assert R.is_consistent()
    check_diag_R_nonnegative(R.fuse_legs(axes=(0, (1, 2)), mode='hard'))

    # change signature of new leg; and position of new leg
    Q2, R2 = yastn.qr(a, axes=((3, 1), (2, 0)), sQ=-1, Qaxis=0, Raxis=-1)
    QR2 = yastn.tensordot(R2, Q2, axes=(2, 0)).transpose(axes=(1, 3, 0, 2))
    assert yastn.norm(a - QR2) < 1e-12  # == 0.0
    assert Q2.is_consistent()
    assert R2.is_consistent()
    check_diag_R_nonnegative(R2.fuse_legs(axes=((0, 1), 2), mode='hard'))




def test_qr_basic(config_kwargs):
    """ test qr decomposition for various symmetries """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    a = yastn.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    run_qr_combine(a)

    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(5, 6, 7)),
            yastn.Leg(config_U1, s=1, t=(-2, -1, 0, 1, 2), D=(6, 5, 4, 3, 2)),
            yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))]
    a = yastn.rand(config=config_U1, legs=legs, n=1)
    run_qr_combine(a)

    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    legs = [yastn.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=(2, 3, 4, 5)),
            yastn.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=(5, 4, 3, 2)),
            yastn.Leg(config_Z2xU1, s=-1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=(3, 4, 5, 6)),
            yastn.Leg(config_Z2xU1, s=-1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=(1, 2, 3, 4))]
    a = yastn.ones(config=config_Z2xU1, legs=legs)
    run_qr_combine(a)

    # test qr of empty Tensor
    for config in [config_dense, config_U1, config_Z2xU1]:
        a = yastn.Tensor(config, s=(1, -1, 1))
        Q, R = yastn.linalg.qr(a, axes=(0, (1, 2)))
        assert Q.size == R.size == 0


def test_qr_Z3(config_kwargs):
    # Z3
    config_Z3 = yastn.make_config(sym='Z3', **config_kwargs)
    sset = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    nset = (0, 1, 2)
    sQset = (-1, 1)
    for s, n, sQ in product(sset, nset, sQset):
        a = yastn.rand(config=config_Z3, s=s, n=n, t=[(0, 1, 2), (0, 1, 2)], D=[(2, 5, 3), (5, 2, 3)], dtype='complex128')
        Q, R = yastn.linalg.qr(a, axes=(0, 1), sQ=sQ)
        assert yastn.norm(a - Q @ R) < tol  # == 0.0
        assert Q.is_consistent()
        assert R.is_consistent()
        for t in R.struct.t:
            assert all(R.config.backend.diag_get(R.real()[t]) >= 0)
            assert all(R.config.backend.diag_get(R.imag()[t]) == 0)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
