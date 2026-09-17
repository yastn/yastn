import numpy as np
import pytest

import yastn
from yastn.su2 import Leg, SU2Tensor, tensordot
from yastn.su2 import truncation_mask
from yastn.su2 import SU2U1Leg, SU2U1Tensor
from yastn.tn import fpeps


def test_spin_half_singlet_roundtrip_and_serialization():
    leg = Leg(t=(1,), D=(1,))
    singlet = SU2Tensor((leg, leg), {((1, 1), ()): np.array(2.0)})
    assert np.allclose(singlet.to_dense(), [[0, -np.sqrt(2)], [np.sqrt(2), 0]])
    restored = SU2Tensor.from_dense(singlet.to_dense(), (leg, leg))
    assert np.allclose(restored.blocks[((1, 1), ())], 2.0)
    assert np.allclose(SU2Tensor.from_dict(singlet.to_dict()).to_dense(), singlet.to_dense())
    assert np.allclose(yastn.from_dict(singlet.to_dict()).to_dense(), singlet.to_dense())


def test_branching_path_is_explicit():
    leg = Leg(t=(1,), D=(1,))
    # Four spin halves contain two independent singlets, labelled by paths 0, 2.
    tensor = SU2Tensor((leg, leg, leg, leg), {
        ((1, 1, 1, 1), (0, 1)): np.array(1.0),
        ((1, 1, 1, 1), (2, 1)): np.array(3.0),
    })
    assert len(tensor.blocks) == 2
    assert np.allclose(SU2Tensor.from_dense(tensor.to_dense(), tensor.legs).to_dense(), tensor.to_dense())


def test_contraction_requires_conjugate_legs_and_stays_symmetric():
    out, incoming = Leg((1,), (1,), 1), Leg((1,), (1,), -1)
    a = SU2Tensor((out, incoming), {((1, 1), ()): np.array(1.0)})
    b = SU2Tensor((out, incoming), {((1, 1), ()): np.array(1.0)})
    result = tensordot(a, b, axes=((1,), (0,)))
    assert result.shape == (2, 2)
    with pytest.raises(ValueError, match='conjugate'):
        tensordot(a, a, axes=((0,), (0,)))


def test_legacy_tensor_and_config_remain_abelian_only():
    config = yastn.make_config(sym='U1')
    assert yastn.ones(config=config, s=(1, -1), t=((0,), (0,)), D=((1,), (1,))).to_dense().shape == (1, 1)
    with pytest.raises(TypeError, match='non-abelian'):
        yastn.Tensor(config=yastn.make_config(sym=yastn.sym.sym_SU2), s=(1, -1))


def test_multiplet_truncation_never_cuts_magnetic_states():
    # The largest spin-half singular value costs two dense states.  A budget
    # of three cannot retain it plus a spin-one multiplet, so it keeps the
    # former and one scalar instead of a partial spin-one or spin-half.
    mask = truncation_mask({0: np.array([0.8, 0.1]), 1: np.array([1.0]), 2: np.array([0.9])}, D_total=3)
    assert mask.kept_dimension == 3
    assert mask.masks[1].tolist() == [True]
    assert mask.masks[2].tolist() == [False]
    assert mask.masks[0].tolist() == [True, False]


def test_su2xu1_fusion_preserves_additive_charge():
    sym = yastn.sym.sym_SU2xU1
    assert sym.fusion_outcomes((1, 2), (1, -1)) == ((0, 1), (2, 1))
    assert sym.conj_charge((3, -4)) == (3, 4)
    with pytest.raises(TypeError, match='non-abelian'):
        yastn.Tensor(config=yastn.make_config(sym='SU2xU1'), s=(1, -1))


def test_su2xu1_tensor_basic_operations_and_serialization():
    out = SU2U1Leg(((1, 1),), (1,), 1)
    incoming = out.conj()
    a = SU2U1Tensor((out, incoming), {(((1, 1), (1, 1)), ()): 2.0})
    b = a.copy().transpose((1, 0)).transpose((1, 0))
    np.testing.assert_allclose((a + b).to_dense(), 2 * a.to_dense())
    np.testing.assert_allclose(a.conj().conj().to_dense(), a.to_dense())
    np.testing.assert_allclose(yastn.from_dict(a.to_dict()).to_dense(), a.to_dense())
    assert tensordot(a, a, ((1,), (0,))).ndim == 2


def test_peps_local_su2_gate_dispatch():
    scalar = Leg((0,), (1,))
    spin_out, spin_in = Leg((1,), (1,), 1), Leg((1,), (1,), -1)
    # A rank-five invariant site tensor; its first virtual leg and physical
    # leg are spin halves, all other virtual legs are scalars.
    site = SU2Tensor((spin_in, scalar, scalar, scalar, spin_out), {
        ((1, 0, 0, 0, 1), (1, 1, 1)): 1.0})
    gate = SU2Tensor((spin_out, spin_in), {((1, 1), ()): 1.0})
    psi = fpeps.Peps(fpeps.SquareLattice(dims=(1, 1)), tensors=site)
    psi.apply_gate_(fpeps.Gate_local(gate, (0, 0)))
    assert isinstance(psi[0, 0], SU2Tensor)
    assert psi[0, 0].shape == site.shape
