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


def test_eig_transpose_meta(config_kwargs):
    """ test eig decomposition with meta-fuse and transpose """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (1, 2), (3, 4)))
    #
    af = a.fuse_legs(axes=((1, 2), (3, 0)), mode='meta')
    assert af.trans == (1, 2, 3, 0)
    #
    aft = af.transpose(axes=(1, 0))
    Uf, Sf, Vf = yastn.linalg.eig(aft, axes=(1, 0))
    USVf = Uf @ Sf @ Vf
    assert yastn.norm(USVf - af) < tol  # == 0.0
    #
    U, S, V = yastn.linalg.eig(a, axes=((1, 2), (3, 0)))
    Um = U.fuse_legs(axes=((0, 1), 2), mode='meta')
    Vm = V.fuse_legs(axes=(0, (1, 2)), mode='meta')
    assert yastn.norm(Uf - Um) < tol  # == 0.0
    assert yastn.norm(Sf - S) < tol  # == 0.0
    assert yastn.norm(Vf - Vm) < tol  # == 0.0


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
