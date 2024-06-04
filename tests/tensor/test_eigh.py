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
""" yastn.linalg.eigh() """
import pytest
from itertools import product
import yastn
try:
    from .configs import config_dense, config_U1, config_Z2xU1, config_Z3
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1, config_Z3

tol = 1e-10  #pylint: disable=invalid-name


def eigh_combine(a):
    """ decompose and contracts Hermitian tensor using eigh decomposition """
    a2 = yastn.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))  # makes Hermitian matrix from a
    S, U = yastn.linalg.eigh(a2, axes=((0, 1), (2, 3)))
    US = yastn.tensordot(U, S, axes=(2, 0))
    USU = yastn.tensordot(US, U, axes=(2, 2), conj=(0, 1))
    assert yastn.norm(a2 - USU) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()

    # changes signature of new leg; and position of new leg
    S, U = yastn.eigh(a2, axes=((0, 1), (2, 3)), Uaxis=0, sU=-1)
    US = yastn.tensordot(S, U, axes=(0, 0))
    USU = yastn.tensordot(US, U, axes=(0, 0), conj=(0, 1))
    assert yastn.norm(a2 - USU) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()


def test_eigh_basic():
    """ test eigh decomposition for various symmetries """
    # dense
    a = yastn.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    eigh_combine(a)

    # U1
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(5, 6, 7)),
            yastn.Leg(config_U1, s=1, t=(-2, -1, 0, 1, 2), D=(6, 5, 4, 3, 2)),
            yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))]
    a = yastn.rand(config=config_U1, n=1, legs=legs)
    eigh_combine(a)

    # Z2xU1
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(2, 3, 4, 5)),
            yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(5, 4, 3, 2)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(3, 4, 5, 6)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 4))]
    a = yastn.ones(config=config_Z2xU1, legs=legs)
    eigh_combine(a)


def test_eigh_Z3():
    # Z3
    s0set = (-1, 1)
    sUset = (-1, 1)
    for s, sU in product(s0set, sUset):
        leg = yastn.Leg(config_Z3, s=s, t=(0, 1, 2), D=(2, 5, 3))
        a = yastn.rand(config=config_Z3,legs=[leg, leg.conj()], dtype='complex128')
        a = a + a.transpose(axes=(1, 0)).conj()
        S, U = yastn.linalg.eigh(a, axes=(0, 1), sU=sU)
        assert yastn.norm(a - U @ S @ U.transpose(axes=(1, 0)).conj()) < tol  # == 0.0
        assert U.is_consistent()
        assert S.is_consistent()


def test_eigh_exceptions():
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0), D=(2, 3)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0), D=(5, 6))]
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config_U1, n=2, legs=legs)
        _ = yastn.eigh(a, axes=(0, 1))
        # eigh requires tensor charge to be zero.
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config_U1, n=0, legs=legs)
        _ = yastn.eigh(a, axes=(0, 1))
        # Tensor likely not Hermitian. Legs of effective square blocks not match.


if __name__ == '__main__':
    test_eigh_basic()
    test_eigh_Z3()
    test_eigh_exceptions()
