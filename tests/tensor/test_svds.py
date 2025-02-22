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
import pytest
import yastn

tol = 1e-10  #pylint: disable=invalid-name

torch_test = pytest.mark.skipif("'torch' not in config.getoption('--backend')",
                                reason="Uses torch.autograd.gradcheck().")

def test_svds_truncate_U1(config_kwargs):
    #
    # Start with random tensor with 4 legs.
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=1, t=(0, 1), D=(5, 6)),
            yastn.Leg(config_U1, s=1, t=(-1, 0), D=(5, 6)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4))]
    a = yastn.rand(config=config_U1, n=1, legs=legs)

    #
    # Fixing singular values for testing.
    # Create new tensor *a* that will be used for testing.
    Ur, Sr, Vr = yastn.linalg.svd(a, axes=((0, 1), (2, 3)), sU=-1)
    Sr.set_block(ts=(-2, -2), Ds=4, val=[2**(-ii - 6) for ii in range(4)])
    Sr.set_block(ts=(-1, -1), Ds=12, val=[2**(-ii - 2) for ii in range(12)])
    Sr.set_block(ts=(0, 0), Ds=25, val=[2**(-ii - 1) for ii in range(25)])
    a = yastn.ncon([Ur, Sr, Vr], [(-1, -2, 1), (1, 2), (2, -3, -4)])

    #
    # Opts for truncation; truncates below (global) relative tolerance reltol with respect to largest singular value
    # up to a total of D_total singular values, with at most D_block singular values for each charge sector.
    opts = {'reltol': 0.01, 'D_block': 100, 'D_total': 12}
    U1, S1, V1 = yastn.svds(a, axes=((0, 1), (2, 3)), sU=-1, nU=True, **opts)
    assert S1.get_blocks_charge() == ((-2, -2), (-1, -1), (0, 0))
    assert S1.s[0] == 1 and U1.n == a.n and V1.n == (0,) and S1.get_shape() == (12, 12)

    #
    # svd_with_truncation is a shorthand for
    U, S, V = yastn.svds(a, axes=((0, 1), (2, 3)), k=opts['D_total'], sU=-1,) # nU=True
    mask = yastn.linalg.truncation_mask(S, tol=opts['reltol'], **opts)
    U1p, S1p, V1p = mask.apply_mask(U, S, V, axes=(2, 0, 0))
    assert all((x - y).norm() < 1e-12 for x, y in ((U1, U1p), (S1, S1p), (V1, V1p)))

    #
    # Specific charges of S1 depend on signature of a new leg comming from U,
    # and where charge of tensor 'a' is attached
    U1, S1, V1 = yastn.svds(a, axes=((0, 1), (2, 3)), **opts)  # sU=1, nU=True
    assert S1.get_blocks_charge() == ((0, 0), (1, 1), (2, 2))
    assert S1.s[0] == -1 and U1.n == a.n and V1.n == (0,) and S1.get_shape() == (12, 12)

    U1, S1, V1 = yastn.svds(a, axes=((0, 1), (2, 3)), sU=-1, nU=False, **opts)
    assert S1.get_blocks_charge() == ((-1, -1), (0, 0), (1, 1))
    assert S1.s[0] == 1 and U1.n == (0,) and V1.n == a.n and S1.get_shape() == (12, 12)

    U1, S1, V1 = yastn.svds(a, axes=((0, 1), (2, 3)), sU=1, nU=False, **opts)
    assert S1.get_blocks_charge() == ((-1, -1), (0, 0), (1, 1))
    assert S1.s[0] == -1 and U1.n == (0,) and V1.n == a.n and S1.get_shape() == (12, 12)

    #
    # Different truncation options.
    opts = {'tol': 0.02, 'D_block': 5, 'D_total': 100}
    _, S1, _ = yastn.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), sU=1, **opts)
    assert S1.get_shape() == (11, 11)

    #
    # tol_block decides on block truncation tolerance with respect to largest singular value of each block
    opts = {'tol': 0, 'tol_block': 0.2, 'D_total': 100}
    _, S1, _ = yastn.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), sU=1, **opts)
    assert S1.get_shape() == (9, 9)

    #
    # We can specify D_block and tol_block for each charge sector independently
    opts = {'D_block': {(0,): 2, (-1,): 3}, 'tol_block': {(0,): 0.6, (-1,): 0.1}}
    _, S1, _ = yastn.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), sU=-1, **opts)
    assert S1.get_shape() == (4, 4)

    #
    # Here use truncate_multiplets option to shift the cut to largest D to retain multiplet
    opts = {'D_total': 10, "truncate_multiplets": True}
    _, S1, _ = yastn.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), sU=-1, **opts)
    assert S1.get_shape() == (12, 12)

    #
    # Empty tensor
    opts = {'D_total': 0}
    _, S2, _ = yastn.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), nU=False, sU=-1, **opts)
    assert S2.norm() < 1e-12 and S2.size == 0


@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
def test_svds_U1_matrix(config_kwargs, dtype):
    """ check lowrank svd vs full-rank svd with truncation. """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg = yastn.Leg(config_U1, s=1, t=(0, 1, 2, 3, 4, 5), D=(1, 2, 4, 8, 64, 128))
    config_U1.backend.random_seed(seed=0)  # fix seed for testing
    
    a = yastn.rand(config=config_U1, legs=[leg.conj(), leg], dtype=dtype)
    #
    # fixing singular values for testing;
    U, S, V = yastn.linalg.svd(a)
    S = yastn.exp(-S)
    a = U @ (S / S.norm()) @ V
    #
    # actual test
    U1, S1, V1 = yastn.linalg.svd_with_truncation(a, D_total=15, fix_signs=True)
    U2, S2, V2 = yastn.svds(a, D_total=15, fix_signs=True)
    assert (S1 - S2).norm() < tol # this tests the compatibility of block structure between svd and svds
    assert (U1 @ S1 @ V1 - U2 @ S2 @ V2).norm() < tol
    l1 = S1.get_legs(axes=0)
    l2 = S2.get_legs(axes=0)
    assert l1 == l2
    assert U1.yast_dtype == dtype
    assert S1.yast_dtype == 'float64'
    assert V1.yast_dtype == dtype
    assert U2.yast_dtype == dtype
    assert S2.yast_dtype == 'float64'
    assert V2.yast_dtype == dtype


@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
def test_svds_U1_rank4(config_kwargs, dtype):
    """ check lowrank svd vs full-rank svd with truncation. """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg0 = yastn.Leg(config_U1, s=1, t=(0, 1, 2,), D=(1, 2, 8))
    leg1 = yastn.Leg(config_U1, s=1, t=(0, 1, 2,), D=(1, 2, 8))
    config_U1.backend.random_seed(seed=0)  # fix seed for testing
    
    a = yastn.rand(config=config_U1, legs=[leg0.conj(), leg1.conj(), leg0, leg1], dtype=dtype)
    #
    # fixing singular values for testing;
    U, S, V = yastn.linalg.svd(a, axes=((0, 1), (2, 3)))
    S = yastn.exp(-S)
    a = U @ (S / S.norm()) @ V
    #
    # actual test
    U1, S1, V1 = yastn.linalg.svd_with_truncation(a, axes=((0,1), (2,3)), D_total=15, fix_signs=True)
    U2, S2, V2 = yastn.svds(a, axes=((0,1), (2,3)), D_total=15, fix_signs=True)
    assert (S1 - S2).norm() < tol # this tests the compatibility of block structure between svd and svds
    assert (U1 @ S1 @ V1 - U2 @ S2 @ V2).norm() < tol
    l1 = S1.get_legs(axes=0)
    l2 = S2.get_legs(axes=0)
    assert l1 == l2


@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
def test_svds_Z2xU1_matrix(config_kwargs, dtype):
    """ check lowrank svd vs full-rank svd with truncation. """
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    l0 = yastn.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=(7, 8, 9, 10))
    config_Z2xU1.backend.random_seed(seed=0)  # fix seed for testing
    
    a = yastn.rand(config=config_Z2xU1, legs=[l0, l0.conj()], dtype=dtype)
    #
    # fixing singular values for testing;
    U, S, V = yastn.linalg.svd(a)
    S = yastn.exp(-S)
    a = U @ (S / S.norm()) @ V
    #
    # actual test
    U1, S1, V1 = yastn.linalg.svd_with_truncation(a, D_total=15, fix_signs=True)
    U2, S2, V2 = yastn.svds(a, D_total=15, fix_signs=True)
    assert (S1 - S2).norm() < tol # this tests the compatibility of block structure between svd and svds
    assert (U1 @ S1 @ V1 - U2 @ S2 @ V2).norm() < tol
    l1 = S1.get_legs(axes=0)
    l2 = S2.get_legs(axes=0)
    assert l1 == l2


@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
def test_svds_Z2xU1_rank4(config_kwargs, dtype):
    """ check lowrank svd vs full-rank svd with truncation. """
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    l0 = yastn.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=(7, 8, 9, 10))
    l1 = yastn.Leg(config_Z2xU1, s=1, t=[(0, -1), (0, 1), (1, -1), (1, 1)], D= (9, 4, 3, 2))
    config_Z2xU1.backend.random_seed(seed=0)  # fix seed for testing

    a = yastn.rand(config=config_Z2xU1, legs=[l0, l1, l0.conj(), l1.conj()], dtype=dtype)
    #
    # fixing singular values for testing;
    U, S, V = yastn.linalg.svd(a, axes=((0, 1), (2, 3)))
    S = yastn.exp(-S)
    a = U @ (S / S.norm()) @ V
    #
    # actual test
    U1, S1, V1 = yastn.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), D_total=15, fix_signs=True)
    U2, S2, V2 = yastn.svds(a, axes=((0, 1), (2, 3)), D_total=15, fix_signs=True)
    assert (S1 - S2).norm() < tol # this tests the compatibility of block structure between svd and svds
    assert (U1 @ S1 @ V1 - U2 @ S2 @ V2).norm() < tol
    l1 = S1.get_legs(axes=0)
    l2 = S2.get_legs(axes=0)
    assert l1 == l2