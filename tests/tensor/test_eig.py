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
""" yastn.linalg.svd() and truncation of its singular values """
from itertools import product
import numpy as np
import pytest
import yastn

tol = 1e-10  #pylint: disable=invalid-name
seed = 1

torch_test = pytest.mark.skipif("'torch' not in config.getoption('--backend')",
                                reason="Uses torch.autograd.gradcheck().")

def eig_combine(a):
    """ decompose and contracts tensor using svd decomposition """
    U, S, V = yastn.linalg.eig(a, axes=((3, 1), (2, 0)), sU=-1)
    US = yastn.tensordot(U, S, axes=(2, 0))
    USV = yastn.tensordot(US, V, axes=(2, 0))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert yastn.norm(a - USV) < tol  # == 0.0
    assert all(x.is_consistent() for x in (a, U, S, V))

    onlyS = yastn.linalg.eig(a, axes=((3, 1), (2, 0)), sU=-1, compute_uv=False)
    assert yastn.norm(S - onlyS) < tol

    # changes signature of new leg; and position of new leg
    U, S, V = yastn.linalg.eig(a, axes=((3, 1), (2, 0)), sU=1, nU=False, Uaxis=0, Vaxis=-1, fix_signs=True)
    US = yastn.tensordot(S, U, axes=(0, 0))
    USV = yastn.tensordot(US, V, axes=(0, 2))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert yastn.norm(a - USV) < tol  # == 0.0
    assert all(x.is_consistent() for x in (U, S, V))

    onlyS = yastn.linalg.eig(a, axes=((3, 1), (2, 0)), sU=1, nU=False, compute_uv=False)
    assert yastn.norm(S - onlyS) < tol


def test_eig_basic(config_kwargs):
    """ test svd decomposition for various symmetries """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    config_dense.backend.random_seed(seed=seed)

    a = yastn.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 11, 13, 13])
    eig_combine(a)

    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=1, t=(-2, 0, 2), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(2, 3, 4))]
    a = yastn.rand(config=config_U1, n=0, legs=legs)
    eig_combine(a)

    # test eig of empty Tensor
    for config in [config_dense, config_U1]:
        a = yastn.Tensor(config, s=(1, -1, 1))
        U, S, V = yastn.linalg.eig(a, axes=((0, 1), 2))
        assert U.size == S.size == V.size == 0


def test_eig_degeneracy_fail(config_kwargs):
    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 1)),
            yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 1)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 1)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 1))]
    a = yastn.ones(config=config_Z2xU1, legs=legs)
    with pytest.raises(ValueError):
        eig_combine(a)
        # fails to biorthogonalize due to degeneracies in eigenspaces


def test_eig_complex(config_kwargs):
    """ test eig decomposition and dtype propagation """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    config_dense.backend.random_seed(seed=seed)

    a = yastn.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 11, 11, 11], dtype='complex128')
    U, S, V = yastn.linalg.eig(a, axes=((0, 1), (2, 3)), sU=-1)
    assert U.yastn_dtype == 'complex128'
    assert S.yastn_dtype == 'complex128'

    US = yastn.tensordot(U, S, axes=(2, 0))  # here tensordot goes though broadcasting
    USV = yastn.tensordot(US, V, axes=(2, 0))
    assert yastn.norm(a - USV) < tol  # == 0.0

    SS = yastn.diag(S)
    assert SS.yastn_dtype == 'complex128'
    US = yastn.tensordot(U, SS, axes=(2, 0))
    USV = yastn.tensordot(US, V, axes=(2, 0))
    assert yastn.norm(a - USV) < tol  # == 0.0

    eig_combine(a)


# def test_eig_multiplets(config_kwargs):
#     config_U1 = yastn.make_config(sym='U1', **config_kwargs)
#     config_U1.backend.random_seed(seed=0)  # to fix consistency of tests
#     legs = [yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 2)),
#             yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(3, 4, 3)),
#             yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(4, 5, 4)),
#             yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(5, 6, 5))]
#     a = yastn.rand(config=config_U1, n=0, legs=legs)

#     U, S, V = yastn.linalg.eig(a, axes=((0, 1), (2, 3)))

#     # fixing singular values for testing
#     v00 = [1, 1, 0.1001, 0.1000, 0.1000, 0.0999, 0.001001, 0.001000] + [0] * 16
#     S.set_block(ts=(0, 0), Ds=24, val=v00)

#     v11 = [1, 1, 0.1001, 0.1000, 0.0999, 0.001000, 0.000999] + [0] * 10
#     S.set_block(ts=(1, 1), Ds=17, val=v11)
#     S.set_block(ts=(-1, -1), Ds=17, val=v11)

#     v22 = [1, 1, 0.1001, 0.1000, 0.001000, 0]
#     S.set_block(ts=(2, 2), Ds=6, val=v22)
#     S.set_block(ts=(-2, -2), Ds=6, val=v22)

#     a = yastn.ncon([U, S, V], [(-1, -2, 1), (1, 2), (2, -3, -4)])

#     opts = {'tol': 0.0001, 'D_block': 7, 'D_total': 30}
#     _, S1, _ = yastn.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), **opts)
#     assert S1.get_shape() == (30, 30)

#     mask_f = lambda x: yastn.truncation_mask_multiplets(x, tol=0.0001, D_total=30, eps_multiplet=0.001)
#     _, S1, _ = yastn.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), mask_f=mask_f)
#     # assert S1.get_shape() == (24, 24)
#     # TODO: CI gives an error (30, 30) != (24, 24)
#     # in test-full (torch, 3.9, 1.26.4, 1.13.1, 2.4); cannot reproduce it locally ...
#     # config_kwargs = {'backend': 'torch', 'default_device': 'cpu'}

#     # below extend the cut to largest gap in singular values;
#     # enforcing that multiplets are kept
#     opts = {'tol': 0.001, 'truncate_multiplets': True}
#     _, S1, _ = yastn.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), **opts)
#     assert S1.get_shape() == (32, 32)

#     opts = {'D_total': 17, 'truncate_multiplets': True}
#     _, S1, _ = yastn.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), **opts)
#     assert S1.get_shape() == (24, 24)

# this is covered by svd as we use _meta_svd internally
#def test_eig_tensor_charge_division(config_kwargs):


def test_eig_exceptions(config_kwargs):
    """ raising exceptions by eig(), and some corner cases. """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=1, t=(0, 1), D=(5, 6)),
            yastn.Leg(config_U1, s=-1, t=(0, 1), D=(5, 6)),]
    a = yastn.rand(config=config_U1, legs=legs)

    with pytest.raises(yastn.YastnError):
        _ = yastn.eig(a, axes=((0, 1), 2), policy='wrong_policy')
        # svd policy should be one of (`lowrank`, `fullrank`)
    with pytest.raises(yastn.YastnError):
        _ = yastn.eig(a, axes=((0, 1), 2), policy='lowrank')
        # lowrank policy in svd requires passing argument D_block

if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
