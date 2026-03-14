# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
""" yastn.linalg.eigh() """
from itertools import product
import pytest
import yastn

tol = 1e-9  #pylint: disable=invalid-name


def eighs_combine(a,D_block,which='SR'):
    """ decompose and contracts Hermitian tensor using eigh decomposition """
    _tol, _tol_block = 0, 0
    if which in ['SR', 'SM']:
        _tol, _tol_block = -float('inf'), -float('inf')

    a2 = yastn.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))  # makes Hermitian matrix from a
    S, U = yastn.linalg.eigh(a2, axes=((0, 1), (2, 3)), which=which, policy='block_lanczos', D_block=D_block)
    S_full, U_full = yastn.linalg.eigh_with_truncation(a2, axes=((0, 1), (2, 3)), which=which, D_block=D_block,
                        sU=1, Uaxis=-1, policy='fullrank',
                        tol=_tol, tol_block=_tol_block, D_total=float('inf'),
                        truncate_multiplets=False, mask_f=None)
    
    assert yastn.norm(S - S_full) < tol
    US = yastn.tensordot(U, S, axes=(2, 0))
    USU = yastn.tensordot(US, U, axes=(2, 2), conj=(0, 1))
    US_full = yastn.tensordot(U_full, S_full, axes=(2, 0))
    USU_full = yastn.tensordot(US_full, U_full, axes=(2, 2), conj=(0, 1))
    assert yastn.norm(USU - USU_full) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()

    # changes signature of new leg; and position of new leg
    S, U = yastn.linalg.eigh(a2, axes=((0, 1), (2, 3)), Uaxis=0, sU=-1, which=which, 
                             policy='block_lanczos', D_block=D_block)
    S_full, U_full = yastn.linalg.eigh_with_truncation(a2, axes=((0, 1), (2, 3)), which=which, D_block=D_block,
                        sU=-1, Uaxis=0, policy='fullrank',
                        tol=_tol, tol_block=_tol_block, D_total=float('inf'),
                        truncate_multiplets=False, mask_f=None)

    assert yastn.norm(S - S_full) < tol
    US = yastn.tensordot(S, U, axes=(0, 0))
    USU = yastn.tensordot(US, U, axes=(0, 0), conj=(0, 1))
    US_full = yastn.tensordot(S_full, U_full, axes=(0, 0))
    USU_full = yastn.tensordot(US_full, U_full, axes=(0, 0), conj=(0, 1))
    assert yastn.norm(USU - USU_full) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()


@pytest.mark.parametrize("which", ['LM',])
def test_eigh_basic(config_kwargs,which):
    """ test eigh decomposition for various symmetries """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    a = yastn.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    eighs_combine(a,D_block=5,which=which)

    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(5, 6, 7)),
            yastn.Leg(config_U1, s=1, t=(-2, -1, 0, 1, 2), D=(6, 5, 4, 3, 2)),
            yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))]
    a = yastn.rand(config=config_U1, n=1, legs=legs)
    eighs_combine(a,D_block=5,which=which)

    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(2, 3, 4, 5)),
            yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(5, 4, 3, 2)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(3, 4, 5, 6)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 4))]
    a = yastn.ones(config=config_Z2xU1, legs=legs)
    eighs_combine(a,D_block=5,which=which)

    # test eigh of empty Tensor
    for config in [config_dense, config_U1, config_Z2xU1]:
        a = yastn.Tensor(config, s=(1, -1, 1, -1))
        S, U = yastn.linalg.eigh(a, axes=((0, 1), (2, 3)), policy='block_lanczos', D_block=1)
        assert S.size == U.size == 0


@pytest.mark.parametrize("which", ['LM','SR'])
def test_eigh_basic_2(config_kwargs,which):
    """ test eigh decomposition for various symmetries """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    a = yastn.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 11, 12])
    eighs_combine(a,D_block=5,which=which)

    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=1,t=(-1, 0, 1), D=(2, 3, 4))]
    a = yastn.rand(config=config_U1, n=1, legs=legs)
    eighs_combine(a,D_block=5,which=which)


@pytest.mark.parametrize("which", ['LM',])
def test_eigh_Z3(config_kwargs,which):
    # Z3
    config_Z3 = yastn.make_config(sym='Z3', **config_kwargs)
    s0set = (-1, 1)
    sUset = (-1, 1)
    for s, sU in product(s0set, sUset):
        leg = yastn.Leg(config_Z3, s=s, t=(0, 1, 2), D=(2, 5, 3))
        a = yastn.rand(config=config_Z3,legs=[leg, leg.conj()], dtype='complex128')
        a = a + a.transpose(axes=(1, 0)).conj()
        S, U = yastn.linalg.eigh(a, axes=(0, 1), sU=sU, which=which)
        assert yastn.norm(a - U @ S @ U.transpose(axes=(1, 0)).conj()) < tol  # == 0.0
        assert U.is_consistent()
        assert S.is_consistent()