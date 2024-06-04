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
""" Test adding a single block to the tensor with yastn.set_block """
import numpy as np
import pytest
import yastn
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def test_U1():
    """ initialization of tensor with U1 symmetry """
    # 3-dim tensor
    a = yastn.Tensor(config=config_U1, s=(-1, 1, 1))  # initialize empty tensor
    assert a.get_shape() == (0, 0, 0)
    a.set_block(ts=(1, -1, 2), Ds=(2, 5, 3), val='rand')  # add a block filled with random numbers
    a.set_block(ts=(2, 0, 2), Ds=(3, 6, 3), val='rand')  # add a block filled with random numbers
    assert (1, -1, 2) in a
    assert (2, 0, 2) in a
    assert a.to_numpy().shape == a.get_shape() == (5, 11, 3)

    b = yastn.Tensor(config=config_U1, s=(-1, 1, 1))  # initialize empty tensor
    b.set_block(ts=(1, 0, 1), Ds=(2, 6, 2), val='rand')  # add a block filled with random numbers
    b.set_block(ts=(2, 0, 2), Ds=(3, 6, 3), val='rand')  # add a block filled with random numbers
    assert b.to_numpy().shape == b.get_shape() == (5, 6, 5)

    legs = (yastn.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(0, 2), D=(1, 2,)),
            yastn.Leg(config_U1, s=1, t=(-2, 0, 2), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(0,), D=(1,)))
    c = yastn.ones(config=config_U1, legs=legs)
    assert c.get_shape() == (6, 3, 6, 1)
    assert pytest.approx(c.norm().item() ** 2, rel=tol) == 30
    c.set_block(ts=(-2, 0, -2, 0), val='zeros')  # replaces existing block
    assert (-2, 0, -2, 0) in c
    assert c.get_shape() == (6, 3, 6, 1)
    assert pytest.approx(c.norm().item() ** 2, rel=tol) == 29
    c.set_block(ts=(2, 0, 0, 2), Ds=(3, 1, 2, 3), val='ones')  # adds a new block changing tensor shape
    assert c.get_shape() == (6, 3, 6, 4)
    assert pytest.approx(c.norm().item() ** 2, rel=tol) == 47

    # 0-dim tensor
    a = yastn.ones(config=config_U1)  # s=() # t=(), D=()
    assert pytest.approx(a.item(), rel=tol) == 1
    a.set_block(val=3)
    assert pytest.approx(a.item(), rel=tol) == 3
    assert a.get_shape() == ()

    # diagonal tensor
    a = yastn.rand(config=config_U1, isdiag=True, t=0, D=5)
    # for U1 simplified notation of charges avoiding some brackets is usually supported, if unambiguous.
    a.set_block(ts=0, val='rand')
    a.set_block(ts=1, val='rand', Ds=4)
    assert 0 in a
    npa = a.to_numpy()
    assert npa.shape == a.get_shape() == (9, 9)
    assert a.is_consistent()
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0


def test_Z2xU1():
    """ initialization of tensor with more complicated symmetry indexed by 2 numbers"""
    # 3-dim tensor
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=[(0, 0), (1, 0), (0, 2), (1, 2)], D=[1, 2, 2, 4]),
            yastn.Leg(config_Z2xU1, s=1, t=[(0, -2), (0, 2)], D=[1, 2]),
            yastn.Leg(config_Z2xU1, s=1, t=[(0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)], D=[2, 4, 6, 3, 6, 9])]
    a = yastn.ones(config=config_Z2xU1, legs=legs)
    assert a.get_shape() == (9, 3, 30)
    assert pytest.approx(a.norm().item() ** 2, rel=tol) == a.size == 104

    a.set_block(ts=((0, 0), (0, 0), (0, 0)), Ds=(1, 5, 4), val=np.sqrt(np.arange(20)))
    assert ((0, 0), (0, 0), (0, 0)) in a
    assert pytest.approx(a.norm().item() ** 2, rel=tol) == 294  # sum(range(20)) == 190
    assert a.get_shape() == (9, 8, 30)
    assert a.is_consistent()

    # setting values in the exhisting block are also possible using __setitem__
    a[(0, 0, 0, 0, 0, 0)] = a[(0, 0, 0, 0, 0, 0)] * 2
    assert pytest.approx(a.norm().item() ** 2, rel=tol) == 864  # sum(4 * range(20)) == 760

    b = a.fuse_legs(axes=((0, 1), 2), mode='hard')  # if tensor is hard-fused, have to refer to fused blocks
    assert ((0, 0), (0, 0)) in b

    a = a.fuse_legs(axes=((0, 1), 2), mode='meta')  # if tensor is meta-fused, have to refer to unfused blocks
    assert ((0, 0), (0, 0), (0, 0)) in a
    #TODO assert (0, 0, 0, 0) in a
    a.set_block(ts=((0, 0), (0, 0), (0, 0)), Ds=(1, 5, 4), val=np.sqrt(np.arange(20)))
    assert pytest.approx(a.norm().item() ** 2, rel=tol) == 294  # sum(range(20)) == 190
    assert a.get_shape() == (26, 30)
    assert a.get_shape(native=True) == (9, 8, 30)

    # 3-dim tensor
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=[(0, 1), (1, 0)], D=[1, 2]),
            yastn.Leg(config_Z2xU1, s=1, t=[(0, 0)], D=[3]),
            yastn.Leg(config_Z2xU1, s=1, t=[(0, 1), (1, 0)], D=[1, 2])]
    a = yastn.ones(config=config_Z2xU1, legs=legs)
    assert a.get_shape() == (3, 3, 3)

    a.set_block(ts=((0, 1), (0, -2), (0, 3)), Ds=(1, 5, 6), val='ones')
    a.set_block(ts=(0, 1, 0, -2, 0, 3), Ds=(1, 5, 6), val='ones') # those two have the same effect
    assert ((0, 1), (0, -2), (0, 3)) in a
    assert (0, 1, 0, -2, 0, 3) in a
    assert a.get_shape() == (3, 8, 9)
    assert a.is_consistent()

    # diagonal tensor
    leg = yastn.Leg(config_Z2xU1, s=1, t=[(0, 0), (1, 1), (0, 2)], D=[2, 3, 5])
    a = yastn.rand(config=config_Z2xU1, isdiag=True, legs=[leg, leg.conj()])
    assert a.get_shape() == (10, 10)

    a.set_block(ts=(0, 0), val='ones')  # in a diagonal tensor, can fill-in matching charge on a second leg
    a.set_block(ts=((1, 1), (1, 1)), val='ones')
    a.set_block(ts=((0, 2), (0, 2)), val='ones')
    a.set_block(ts=(1, 3), val='ones', Ds=1)
    assert (1, 3) in a
    npa = a.to_numpy()
    assert npa.shape == a.get_shape() == (11, 11)
    assert np.allclose(npa, np.eye(11), rtol=tol, atol=tol)
    assert a.is_consistent()


def test_dense():
    """ initialization of dense tensor with no symmetry """
    # 3-dim tensor
    a = yastn.Tensor(config=config_dense, s=(-1, 1, 1))  # initialize empty tensor
    a.set_block(Ds=(4, 5, 6), val='rand')  # add the only possible block for dense tensor
    npa = a.to_numpy()
    assert np.isrealobj(npa) == (config_dense.default_dtype == 'float64')
    assert npa.shape == a.get_shape() == (4, 5, 6)
    assert a.is_consistent()

    # 0-dim tensor
    a = yastn.Tensor(config=config_dense)  # s=()
    a.set_block(val=3)  # 0-dim tensor is a number
    npa = a.to_numpy()
    assert np.isrealobj(npa) == (config_dense.default_dtype == 'float64')
    assert npa.shape == a.get_shape() == ()
    assert pytest.approx(a.item(), rel=tol) == 3
    assert a.is_consistent()

    # 1-dim tensor
    a = yastn.Tensor(config=config_dense, s=1)  # s=(1,)
    a.set_block(Ds=5, val='ones')
    npa = a.to_numpy()
    assert np.isrealobj(npa) == (config_dense.default_dtype == 'float64')
    assert npa.shape == a.get_shape() == (5,)
    assert a.is_consistent()

    # diagonal tensor
    a = yastn.Tensor(config=config_dense, isdiag=True)
    a.set_block(Ds=5, val='ones')
    npa = a.to_numpy()
    assert np.isrealobj(npa) == (config_dense.default_dtype == 'float64')
    assert npa.shape == a.get_shape() == (5, 5)
    assert a.is_consistent()
    assert np.linalg.norm(np.diag(np.diag(npa)) - npa) < tol  # == 0.0


def test_set_block_exceptions():
    """ test raise YaseError by set_block()"""
    # 3-dim tensor
    leg = yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))
    a = yastn.ones(config=config_U1, legs=[leg, leg, leg.conj()])
    b = a.copy()
    with pytest.raises(yastn.YastnError):
        b = a.copy()
        b.set_block(ts=(0, 0, 0), Ds=(3, 2, 2), val='ones')  # here (3, ...) is inconsistent bond dimension
        # Inconsistend bond dimension of charge.
    with pytest.raises(yastn.YastnError):
        b = a.copy()
        b.set_block(ts=(0, 0), Ds=(2, 2), val='ones')
        # Size of ts is not consistent with tensor rank and the number of symmetry sectors.
    with pytest.raises(yastn.YastnError):
        b = a.copy()
        b.set_block(ts=(0, 0, 0), Ds=(2, 2), val='ones')
        # 'Size of Ds is not consistent with tensor rank.'
    with pytest.raises(yastn.YastnError):
        b = a.copy()
        b.set_block(ts=(3, 0, 0), Ds=(2, 2, 2), val='ones')
        # Charges ts are not consistent with the symmetry rules: f(t @ s) == n
    with pytest.raises(yastn.YastnError):
        b = a.copy()
        b.set_block(ts=(1, 1, 2), val='ones')
        # Provided Ds. Cannot infer all bond dimensions from existing blocks.
    with pytest.raises(yastn.YastnError):
        b = yastn.Tensor(config=config_U1, isdiag=True)
        b.set_block(ts=(0, 0), Ds=(1, 2), val='ones')
        # Diagonal tensor requires the same bond dimensions on both legs.
    with pytest.raises(yastn.YastnError):
        b= a.copy()
        b.set_block(ts=(0, 0, 0), val='four')
        # val should be in ("zeros", "ones", "rand")
    with pytest.raises(yastn.YastnError):
        a[(1, 1, 1)] = np.ones((3, 3, 3))
        # tensor does not have block specified by key
    with pytest.raises(yastn.YastnError):
        a[(1, 1, 1)]
        # tensor does not have block specified by key
    with pytest.raises(yastn.YastnError):
        b= a.fuse_legs(axes=((0,1),2))
        b.set_block(ts=(0,0), Ds=a[(0,0)].shape, val='zeros') # cannot set blocks on fused tensors


if __name__ == '__main__':
    test_U1()
    test_Z2xU1()
    test_dense()
    test_set_block_exceptions()
