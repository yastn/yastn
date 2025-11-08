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
""" yastn.save_to_dict() yastn.load_from_dict() yastn.save_to_hdf5() yastn.load_from_hdf5(). """
import os
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def identical_tensors(a, b):
    da, db = a.__dict__, b.__dict__
    assert da.keys() == db.keys()
    for k in da:
        if k != '_data':
            assert da[k] == db[k]
    assert type(a.data) == type(b.data)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_to_from_dict(config_kwargs):
    config = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config, s=1, t=(0, 1, 2), D= (3, 5, 2)),
            yastn.Leg(config, s=-1, t=(0, 1, 3), D= (1, 2, 3)),
            yastn.Leg(config, s=1, t=(-1, 0, 1), D= (2, 3, 4)),
            yastn.Leg(config, s=1, t=(-1, 0, 1), D= (4, 3, 2))]

    a = yastn.rand(config, legs=legs)
    a = a.fuse_legs(axes=(0, (1, 2), 3), mode='hard')
    a = a.fuse_legs(axes=(0, (1, 2)), mode='meta')

    for level, ind in zip([0, 1, 2], [False, False, True]):
        d = a.to_dict(level=level)
        b = yastn.Tensor.from_dict(d)

        data, meta = yastn.split_data_and_meta(d)
        dd = yastn.combine_data_and_meta(data, meta)
        c = yastn.Tensor.from_dict(dd)
        d = yastn.from_dict(dd)
        #
        identical_tensors(a, b)
        identical_tensors(a, c)
        identical_tensors(a, d)
        assert yastn.are_independent(a, b) == ind
        #
        # the result below depends on the backend: TODO
        # assert yastn.are_independent(b, c) == ind  # for numpy
        # # assert yastn.are_independent(b, c) == False  # for torch
        # torch.as_tensor and numpy.array have different behavior in creating a copy


def check_to_numpy(a1, config):
    """ save/load to numpy and tests consistency."""
    d1 = a1.save_to_dict()
    a2 = 2 * a1  # second tensor to be saved
    d2 = yastn.save_to_dict(a2)
    data={'tensor1': d1, 'tensor2': d2}  # two tensors to be saved
    np.save('tmp.npy', data)
    ldata = np.load('tmp.npy', allow_pickle=True).item()
    os.remove('tmp.npy')

    b1 = yastn.load_from_dict(config=config, d=ldata['tensor1'])
    b2 = yastn.load_from_dict(config=config, d=ldata['tensor2'])

    assert all(yastn.norm(a - b) < tol for a, b in [(a1, b1), (a2, b2)])
    assert all(b.is_consistent for b in (b1, b2))
    assert all(yastn.are_independent(a, b) for a, b in [(a1, b1), (a2, b2)])


def check_to_hdf5(a, *args):
    """ Test if two Tensor-s have the same values. """
    # os.remove("tmp.h5") remove if exists .. perhaps 'w' in the line below
    h5py = pytest.importorskip("h5py")
    try:
        os.remove("tmp.h5")
    except OSError:
        pass
    with h5py.File('tmp.h5', 'a') as f:
        a.save_to_hdf5(f, './')
    with h5py.File('tmp.h5', 'r') as f:
        b = yastn.load_from_hdf5(a.config, f, './')
    os.remove("tmp.h5")
    b.is_consistent()
    assert yastn.are_independent(a, b)
    assert yastn.norm(a - b) < tol


@pytest.mark.parametrize("test_f", [check_to_numpy, check_to_hdf5])
def test_dict(config_kwargs, test_f):
    """ test exporting tensor to native python data-structure,
        that allows robust saving/loading with np.save/load."""
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    a = yastn.rand(config=config_dense)  # s=() i.e. a scalar
    assert a.size == 1
    test_f(a, config_dense)

    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=1, t=(0, 1, 2), D= (3, 5, 2)),
            yastn.Leg(config_U1, s=-1, t=(0, 1, 3), D= (1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D= (2, 3, 4))]

    a = yastn.rand(config=config_U1, legs=legs)
    test_f(a, config_U1)

    a = yastn.randC(config=config_U1, legs=legs, n=1)
    test_f(a, config_U1) # here a is complex

    a = yastn.rand(config=config_U1, isdiag=True, legs=legs[0])
    test_f(a, config_U1)

    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 4)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, -2), (0, 2)), D=(2, 1)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)), D=(2, 3, 5, 4, 1, 6))]
    a = yastn.ones(config=config_Z2xU1, legs=legs, n=(0, -2))
    test_f(a, config_Z2xU1)

    a = yastn.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    a = a.fuse_legs(axes=((0, 2), 1, (4, 3), 5), mode='hard')
    a = a.fuse_legs(axes=((0, 2), 1, 3), mode='hard')
    a = a.fuse_legs(axes=((0, 2), 1), mode='meta')
    test_f(a, config_U1)



def test_load_exceptions(config_kwargs):
    """ handling exceptions """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    with pytest.raises(yastn.YastnError):
        _ = yastn.load_from_dict(config=config_U1)  # Dictionary d is required

    config_Z2 = yastn.make_config(sym='Z2', **config_kwargs)
    leg = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(2, 3))
    a = yastn.randC(config=config_Z2, n=1, legs=[leg, leg, leg.conj()])
    check_to_numpy(a, config_Z2)  # OK

    with pytest.raises(yastn.YastnError):
        check_to_numpy(a, config_U1)  # Symmetry rule in config do not match loaded one.

    config_Z2_fermionic = yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    with pytest.raises(yastn.YastnError):
        check_to_numpy(a, config_Z2_fermionic)  # Fermionic statistics in config do not match loaded one.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
