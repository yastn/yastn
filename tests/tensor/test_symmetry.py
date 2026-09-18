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
""" Test yastn.sym classes"""
import numpy as np
import pytest

import yastn.sym.sym_U1 as sym_U1
import yastn.sym.sym_Z2 as sym_Z2
import yastn.sym.sym_Z3 as sym_Z3
import yastn.sym.sym_none as sym_none
import yastn.sym.sym_U1xU1 as sym_U1xU1
import yastn.sym.sym_Z2xU1 as sym_Z2xU1
import yastn.sym.sym_U1xU1xZ2 as sym_U1xU1xZ2
import yastn.sym.sym_SU2 as sym_SU2
import yastn.sym.sym_SU2xU1 as sym_SU2xU1
from yastn.sym import sym_nonabelian


def test_symmetry():
    """ tests of predifined symmery classes. """
    #
    # no sym
    assert sym_none.NSYM == 0
    assert str(sym_none) == 'dense'
    assert sym_none.zero() == ()
    charges = [[[], [], []],
               [[], [], []],
               [[], [], []],
               [[], [], []]]
    charges = np.array(charges, dtype=np.int64)
    fused_charges = sym_none.fuse(charges, signatures=(1, 1, -1), new_signature=1)
    ref_charges = np.array([[], [], [], []], dtype=np.int64)
    assert 'int' in fused_charges.dtype.name
    assert np.allclose(fused_charges, ref_charges)
    #
    # U1
    assert sym_U1.NSYM == 1
    assert str(sym_U1) == 'U1'
    assert sym_U1.zero() == (0,)
    charges = [[[1], [-1], [1]],
               [[1], [0], [-1]],
               [[-2], [2], [1]],
               [[-1], [0], [3]]]
    charges = np.array(charges, dtype=np.int64)
    fused_charges = sym_U1.fuse(charges, signatures=(1, 1, -1), new_signature=-1)
    ref_charges = np.array([[1], [-2], [1], [4]], dtype=np.int64)
    assert 'int' in fused_charges.dtype.name
    assert np.allclose(fused_charges, ref_charges)
    #
    # Z2
    assert sym_Z2.NSYM == 1
    assert str(sym_Z2) == 'Z2'
    assert sym_Z2.zero() == (0,)
    charges = [[[1], [1], [0]],
               [[1], [0], [0]],
               [[0], [1], [1]],
               [[1], [0], [0]]]
    charges = np.array(charges, dtype=np.int64)
    fused_charges = sym_Z2.fuse(charges, signatures=(1, 1, -1), new_signature=1)
    ref_charges = np.array([[0], [1], [0], [1]], dtype=np.int64)
    assert 'int' in fused_charges.dtype.name
    assert np.allclose(fused_charges, ref_charges)
    #
    # Z3
    assert sym_Z3.NSYM == 1
    assert str(sym_Z3) == 'Z3'
    assert sym_Z3.zero() == (0,)
    charges = [[[1], [2], [0]],
               [[1], [0], [1]],
               [[2], [2], [2]],
               [[1], [2], [1]]]
    charges = np.array(charges, dtype=np.int64)
    fused_charges = sym_Z3.fuse(charges, signatures=(1, 1, -1), new_signature=1)
    ref_charges = np.array([[0], [0], [2], [2]], dtype=np.int64)
    assert 'int' in fused_charges.dtype.name
    assert np.allclose(fused_charges, ref_charges)
    #
    # U1xU1
    assert sym_U1xU1.NSYM == 2
    assert str(sym_U1xU1) == 'U1xU1'
    assert sym_U1xU1.zero() == (0, 0)
    charges = [[[1, 0], [-1, 1], [1, -2]],
               [[1, 2], [0, -2], [-1, 0]],
               [[-2, 2], [2, 1], [1, -2]],
               [[-1, 2], [0, 3], [3, 2]]]
    charges = np.array(charges, dtype=np.int64)
    fused_charges = sym_U1xU1.fuse(charges, signatures=(1, 1, -1), new_signature=-1)
    ref_charges = np.array([[1, -3], [-2, 0], [1, -5], [4, -3]], dtype=np.int64)
    assert 'int' in fused_charges.dtype.name
    assert np.allclose(fused_charges, ref_charges)
    #
    # U1xU1xZ2
    assert sym_U1xU1xZ2.NSYM == 3
    assert str(sym_U1xU1xZ2) == 'U1xU1xZ2'
    assert sym_U1xU1xZ2.zero() == (0, 0, 0)
    charges = [[[1, 0, 1], [-1, 1, 0], [1, -2, 1]],
               [[1, 2, 1], [0, -2, 0], [-1, 0, 1]],
               [[-2, 2, 0], [2, 1, 1], [1, -2, 1]],
               [[-1, 2, 1], [0, 3, 1], [3, 2, 1]]]
    charges = np.array(charges, dtype=np.int64)
    fused_charges = sym_U1xU1xZ2.fuse(charges, signatures=(1, 1, -1), new_signature=-1)
    ref_charges = np.array([[1, -3, 0], [-2, 0, 0], [1, -5, 0], [4, -3, 1]], dtype=np.int64)
    assert 'int' in fused_charges.dtype.name
    assert np.allclose(fused_charges, ref_charges)
    #
    #
    # Z2xU1
    assert sym_Z2xU1.NSYM == 2
    assert str(sym_Z2xU1) == 'Z2xU1'
    assert sym_Z2xU1.zero() == (0, 0)
    charges = [[[1, 0], [1, 1], [1, -2]],
               [[1, 2], [0, -2], [1, 0]],
               [[0, 2], [0, 1], [1, -2]],
               [[1, 2], [0, 3], [1, 2]]]
    charges = np.array(charges, dtype=np.int64)
    fused_charges = sym_Z2xU1.fuse(charges, signatures=(1, 1, -1), new_signature=-1)
    ref_charges = np.array([[1, -3], [0, 0], [1, -5], [0, -3]], dtype=np.int64)
    assert 'int' in fused_charges.dtype.name
    assert np.allclose(fused_charges, ref_charges)


def test_add_charges():
    """ tests auxiliary symmetry function: add_charges, conj_charge. """
    assert sym_none.add_charges((), (), ()) == ()
    #
    assert sym_Z2.add_charges((1,), (1,), (0,), (1,)) == (1,)
    assert sym_Z2.add_charges((1,), (1,), (1,), signatures=(1, -1, -1), new_signature=-1) == (1,)
    assert sym_Z2.add_charges(1, 0, 0, 1,) == (0,)  # flattened input also works, but only for NSYM=1
    #
    assert sym_Z3.add_charges((2,), (2,), (1,), signatures=[1, 1, -1]) == (0,)
    assert sym_Z3.add_charges((2,), (2,), (1,)) == (2,)
    assert sym_Z3.add_charges((2,), (2,), (1,), signatures=[1, 1, 1]) == (2,)
    assert sym_Z3.add_charges((2,), (2,), (1,), new_signature=-1) == (1,)
    assert sym_Z3.add_charges((1,), new_signature=-1) == (2,)
    #
    assert sym_U1.add_charges((1,), (2,), (-1,)) == (2,)
    assert sym_U1.add_charges((1,), (2,), signatures=[1, -1]) == (-1,)
    assert sym_U1.add_charges((2,), new_signature=-1) == (-2,)
    #
    assert sym_U1xU1.add_charges((1, 0), (-1, 1), (1, -2)) == (1, -1)
    #
    assert sym_U1xU1xZ2.add_charges((2, 1, 1), new_signature=-1) == (-2, -1, 1)
    charge = sym_U1xU1xZ2.add_charges((1, 0, 1), (-1, 1, 0), (1, -2, 1))
    assert charge == (1, -1, 0)
    assert all(isinstance(x, int) for x in charge)
    #
    assert sym_U1.conj_charge((1,)) == (-1,)
    assert sym_Z2.conj_charge((1,)) == (1,)
    assert sym_Z3.conj_charge((1,)) == (2,)
    assert sym_U1xU1xZ2.conj_charge((-1, 2, 0)) == (1, -2, 0)


def test_symmetry_SU2_category_rules():
    assert issubclass(sym_SU2, sym_nonabelian)
    assert str(sym_SU2) == 'SU2'
    assert sym_SU2.IS_ABELIAN is False
    assert sym_SU2.zero() == (0,)
    assert sym_SU2.conj_charge((3,)) == (3,)
    assert sym_SU2.fusion_outcomes((1,), (1,)) == ((0,), (2,))
    outcomes = sym_SU2.fusion_outcomes((1,), (1,), (1,), (1,))
    assert outcomes.count((0,)) == 2
    assert sym_SU2.fusion_paths(((1,),) * 4, (0,)) == ((0, 1), (2, 1))
    assert sym_SU2.fusion_multiplicity(((1,),) * 4, (0,)) == 2
    assert sym_SU2.irrep_dimension((3,)) == 4
    with pytest.raises(TypeError, match='branches'):
        sym_SU2.fuse(np.ones((1, 2, 1), dtype=np.int64), (1, 1), 1)
    with pytest.raises(ValueError, match='non-negative'):
        sym_SU2.fusion_outcomes((-1,), (1,))


def test_symmetry_SU2xU1_category_rules():
    assert issubclass(sym_SU2xU1, sym_nonabelian)
    assert str(sym_SU2xU1) == 'SU2xU1'
    assert sym_SU2xU1.zero() == (0, 0)
    assert sym_SU2xU1.conj_charge((3, -2)) == (3, 2)
    assert sym_SU2xU1.fusion_outcomes((1, 2), (1, -1)) == ((0, 1), (2, 1))
    assert sym_SU2xU1.fusion_paths(((1, 2), (1, -1), (1, 0), (1, 0)), (0, 1)) == (
        ((0, 1), (1, 1)), ((2, 1), (1, 1)))
    assert sym_SU2xU1.signed_fusion_outcomes(((1, 2), (1, -1)), (1, -1)) == ((0, 3), (2, 3))
    assert sym_SU2xU1.irrep_dimension((3, 99)) == 4
    assert sym_SU2xU1.add_charges((0, 2), (0, -1), signatures=(1, -1)) == (0, 3)
    with pytest.raises(TypeError, match='branches'):
        sym_SU2xU1.fuse(np.array([[[1, 2], [1, -1]]]), (1, 1), 1)
    assert sym_SU2xU1.f_symbol((1, 2), (1, 2), (1, -2), (1, 2),
                               (0, 4), (0, 0)) != 0
    assert sym_SU2xU1.f_symbol((1, 2), (1, 2), (1, -2), (1, 2),
                               (0, 0), (0, 0)) == 0


def test_symmetry_SU2_cg_and_f_move_are_unitary():
    rt2 = np.sqrt(2.)
    assert np.isclose(sym_SU2.clebsch_gordan(1, 1, 1, -1, 0, 0), 1 / rt2)
    assert np.isclose(sym_SU2.clebsch_gordan(1, -1, 1, 1, 0, 0), -1 / rt2)

    # Recoupling three spin halves in the total spin-half sector.
    channels = (0, 2)
    F = np.array([[sym_SU2.f_symbol(1, 1, 1, 1, left, right)
                   for right in channels] for left in channels])
    assert np.allclose(F @ F.T, np.eye(2), atol=1e-14)

    cg = sym_SU2.fusion_isometry(1, 1)
    coupled = np.concatenate([cg[(0,)], cg[(2,)]], axis=0)
    assert np.allclose(coupled @ coupled.T, np.eye(4), atol=1e-14)
    assert np.allclose(coupled.T @ coupled, np.eye(4), atol=1e-14)

    cg_u1 = sym_SU2xU1.fusion_isometry((1, 2), (1, -1), (0, 1))
    assert np.allclose(cg_u1 @ cg_u1.T, np.eye(1), atol=1e-14)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
