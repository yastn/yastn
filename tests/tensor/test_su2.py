import numpy as np

import yastn
from yastn.tensor._auxiliary import find_matching_block_keys, get_blocks


def test_su2_reduced_matrix_algebra_and_roundtrip(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    leg = yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 2, 1))
    tensor = yastn.rand(config, legs=(leg, leg.conj()))

    assert tensor.get_blocks_charge() == ((0, 0), (1, 1), (2, 2))
    U, S, V = tensor.svd(axes=(0, 1))
    assert (U @ S @ V - tensor).norm() < 1e-12
    assert (tensor.transpose().transpose() - tensor).norm() < 1e-12

    restored = yastn.from_dict(tensor.to_dict())
    assert restored.config.sym.SYM_ID == 'SU2'
    assert (restored - tensor).norm() < 1e-12


def test_su2_truncation_counts_complete_irreps(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    leg = yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 1, 1))
    spectrum = yastn.zeros(config, legs=(leg, leg.conj()), isdiag=True)
    spectrum[(0, 0)] = np.array([0.8])
    spectrum[(1, 1)] = np.array([1.0])
    spectrum[(2, 2)] = np.array([0.9])

    mask = spectrum.truncation_mask(D_total=3)
    assert mask[(1, 1)].item()
    assert mask[(0, 0)].item()
    assert not mask[(2, 2)].item()
    # spin 1/2 costs two magnetic states and the singlet costs one.
    assert sum(config.sym.irrep_dimension((j,)) * mask[(j, j)].item()
               for j in (0, 1, 2)) == 3


def test_su2_contraction_automatically_uses_branch_safe_kernel(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    half = yastn.Leg(config, s=1, t=(1,), D=(2,))
    scalar = yastn.Leg(config, s=1, t=(0,), D=(1,))
    tensor = yastn.rand(config, legs=(half, half, scalar))
    result = yastn.tensordot(tensor, tensor.conj(), axes=((0, 1), (0, 1)))
    assert result.get_blocks_charge() == ((0, 0),)
    assert result.norm() >= 0


def test_su2xu1_tensor_selection_svd_and_roundtrip(config_kwargs):
    config = yastn.make_config(sym='SU2xU1', **config_kwargs)
    leg = yastn.Leg(config, s=1, t=((0, -1), (1, 2), (2, 0)), D=(1, 2, 1))
    tensor = yastn.rand(config, legs=(leg, leg.conj()))

    assert tensor.get_blocks_charge() == (
        (0, -1, 0, -1),
        (1, 2, 1, 2),
        (2, 0, 2, 0),
    )
    U, S, V = tensor.svd(axes=(0, 1))
    assert (U @ S @ V - tensor).norm() < 1e-12

    restored = yastn.from_dict(tensor.to_dict())
    assert restored.config.sym.SYM_ID == 'SU2xU1'
    assert (restored - tensor).norm() < 1e-12


def test_su2_hard_fusion_unique_channel_roundtrip(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    half = yastn.Leg(config, s=1, t=(1,), D=(2,))
    scalar = yastn.Leg(config, s=1, t=(0,), D=(1,))
    tensor = yastn.rand(config, legs=(half, half, scalar))

    fused = tensor.fuse_legs(axes=((0, 1), 2), mode='hard')
    assert fused.get_legs(0).t == ((0,),)
    assert fused.get_legs(0).D == (4,)
    assert (fused.unfuse_legs(0) - tensor).norm() < 1e-12


def test_su2_multichannel_hard_fusion_roundtrip(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    half = yastn.Leg(config, s=1, t=(1,), D=(1,))
    # Four spin halves have two singlet channels, both stored as independent
    # blocks with identical external charges.
    tensor = yastn.rand(config, legs=(half,) * 4)
    blocks = get_blocks(config.sym, tensor.struct)
    assert blocks.channels == ((0, 1), (2, 1))
    assert tensor.size == 2

    fused = tensor.fuse_legs(axes=((0, 1), 2, 3), mode='hard')
    restored = fused.unfuse_legs(0)
    assert (restored - tensor).norm() < 1e-12
    assert np.isclose(yastn.vdot(tensor, tensor).real, tensor.norm() ** 2)

    paired = tensor.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
    assert paired.size == tensor.size
    assert (paired.unfuse_legs((0, 1)) - tensor).norm() < 1e-12


def test_fusion_channel_metadata_roundtrip_and_matching(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    half = yastn.Leg(config, s=1, t=(1,), D=(1,))
    scalar = yastn.Leg(config, s=1, t=(0,), D=(1,))
    tensor = yastn.rand(config, legs=(half, half, scalar))
    tensor = tensor._replace(struct=tensor.struct.replace(channels=((0,),)))

    restored = yastn.from_dict(tensor.to_dict())
    assert restored.struct.channels == ((0,),)
    assert get_blocks(config.sym, restored.struct).channels == ((0,),)
    assert (restored - tensor).norm() < 1e-12

    charges = np.array([[[1], [1]], [[1], [1]]], dtype=np.int64)
    ind1, ind2 = find_matching_block_keys(
        charges, ((0,), (2,)), charges[::-1], ((2,), (0,)))
    assert ind1.tolist() == [1, 0]
    assert ind2.tolist() == [0, 1]


def test_su2_multichannel_svd_reconstructs_tensor(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    half = yastn.Leg(config, s=1, t=(1,), D=(2,))
    tensor = yastn.rand(config, legs=(half,) * 4)

    U, S, V = tensor.svd(axes=((0, 1), (2, 3)))
    restored = (U @ S) @ V
    assert (restored - tensor).norm() < 1e-12
    assert S.get_legs(0).t == ((0,), (2,))
