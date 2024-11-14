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
""" yastn.vdot() yastn.moveaxis() """
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name

def run_moveaxis(a, ad, source, destination, result):
    newa = a.moveaxis(source=source, destination=destination)
    assert newa.to_numpy().shape == result
    assert newa.get_shape() == result
    assert np.moveaxis(ad, source=source, destination=destination).shape == result
    assert newa.is_consistent()
    assert a.are_independent(newa)


def run_transpose(a, ad, axes, result):
    newa = a.transpose(axes=axes)
    assert newa.to_numpy().shape == result
    assert newa.get_shape() == result
    assert np.transpose(ad, axes=axes).shape == result
    assert newa.is_consistent()
    assert a.are_independent(newa)


def test_transpose_syntax(config_kwargs):
    #
    # Define rank-6 U1-symmetric tensor.
    #
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                D=[(2, 3), (4, 5), (6, 7), (6, 5), (4, 3), (2, 1)])

    #
    # For each leg its dense dimension is given by the sum of dimensions
    # of individual sectors. Hence, the (dense) shape of this tensor
    # is (2+3, 4+5, 6+7, 6+5, 4+3, 2+1).
    assert a.get_shape() == (5, 9, 13, 11, 7, 3)

    #
    # Permute the legs of the tensor and check the shape
    # is changed accordingly.
    b = a.transpose(axes=(0, 2, 4, 1, 3, 5))
    assert b.get_shape() == (5, 13, 7, 9, 11, 3)

    #
    #  If axes is not provided, reverse the order
    #  This can be also done using a shorthand self.T
    a.transpose().get_shape() == (3, 7, 11, 13, 9, 5)
    a.T.get_shape() == (3, 7, 11, 13, 9, 5)

    #
    # Sometimes, instead of writing explicit permutation of all legs
    # it is more convenient to only specify pairs of legs to switched.
    # In this example, we reverse the permutation done previously thus
    # ending up with tensor numerically identical to a.
    #
    c = b.moveaxis(source=(1, 2), destination=(2, 4))
    assert c.get_shape() == a.get_shape()
    assert yastn.norm(a - c) < 1e-12


def test_transpose_basic(config_kwargs):
    """ test transpose for different symmetries. """
    # dense
    config_dense = yastn.make_config(sym='dense', **config_kwargs)
    a = yastn.ones(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    assert a.get_shape() == (2, 3, 4, 5)
    ad = a.to_numpy()
    run_transpose(a, ad, axes=(1, 3, 2, 0), result=(3, 5, 4, 2))
    run_moveaxis(a, ad, source=1, destination=-1, result=(2, 4, 5, 3))
    run_moveaxis(a, ad, source=(1, 3), destination=(1, 0), result=(5, 3, 2, 4))
    run_moveaxis(a, ad, source=(3, 1), destination=(0, 1), result=(5, 3, 2, 4))
    run_moveaxis(a, ad, source=(3, 1), destination=(1, 0), result=(3, 5, 2, 4))
    run_moveaxis(a, ad, source=(1, 3), destination=(0, 1), result=(3, 5, 2, 4))

    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(2, 3), (4, 5), (6, 7), (6, 5), (4, 3), (2, 1)])
    ad = a.to_numpy()
    assert a.get_shape() == (5, 9, 13, 11, 7, 3)
    run_transpose(a, ad, axes=(1, 2, 3, 0, 5, 4), result=(9, 13, 11, 5, 3, 7))
    run_moveaxis(a, ad, source=1, destination=4, result=(5, 13, 11, 7, 9, 3))
    run_moveaxis(a, ad, source=(2, 0), destination=(0, 2), result=(13, 9, 5, 11, 7, 3))
    run_moveaxis(a, ad, source=(2, -1, 0), destination=(-1, 2, -2), result=(9, 11, 3, 7, 5, 13))

    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    legs = [yastn.Leg(config_Z2xU1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(7, 3, 4, 5), s=-1),
            yastn.Leg(config_Z2xU1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(5, 4, 3, 2), s=-1),
            yastn.Leg(config_Z2xU1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(3, 4, 5, 6), s=1),
            yastn.Leg(config_Z2xU1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 4), s=1)]
    a = yastn.ones(config=config_Z2xU1, legs=legs)
    assert a.get_shape() == (19, 14, 18, 10)
    ad = a.to_numpy()
    run_transpose(a, ad, axes=(1, 2, 3, 0), result=(14, 18, 10, 19))
    run_moveaxis(a, ad, source=-1, destination=-3, result=(19, 10, 14, 18))


def test_transpose_diag(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.eye(config=config_U1, t=(-1, 0, 2), D=(2, 2 ,4))
    at = a.transpose(axes=(1, 0))
    assert yastn.tensordot(a, at, axes=((0, 1), (0, 1))).item() == 8.
    assert yastn.vdot(a, at, conj=(0, 0)).item() == 8.
    a = a.transpose(axes=(1, 0))
    assert yastn.vdot(a, at).item() == 8.


def test_transpose_exceptions(config_kwargs):
    """ test handling expections """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    with pytest.raises(yastn.YastnError):
        _ = a.transpose(axes=(0, 1, 3, 5))  # Provided axes do not match tensor ndim.
    with pytest.raises(yastn.YastnError):
        _ = a.transpose(axes=(0, 1, 1, 2, 2, 3))  # Provided axes do not match tensor ndim.


@pytest.mark.skipif("'torch' not in config.getoption('--backend')",
                    reason="Uses torch.autograd.gradcheck().")
def test_transpose_backward(config_kwargs):
    import torch

    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(2, 3), (4, 5), (6, 7), (6, 5), (4, 3), (2, 1)])
    b = a.transpose(axes=(1, 2, 3, 0, 5, 4))
    target_block = (0, 1, 0, 0, 1, 0)
    target_block_size = a[target_block].size()

    def test_f(block):
        a.set_block(ts=target_block, val=block)
        tmp_a = a.transpose(axes=(1, 2, 3, 0, 5, 4))
        ab = b.vdot(tmp_a)
        return ab

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
    test = torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)
    assert test


if __name__ == '__main__':
    # pytest.main([__file__, "-vs", "--durations=0"])
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
