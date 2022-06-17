""" yast.linalg.svd() and truncation of its singular values """
import pytest
import numpy as np
from itertools import product
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1, config_Z3
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1, config_Z3

tol = 1e-10  #pylint: disable=invalid-name


def svd_combine(a):
    """ decompose and contracts tensor using svd decomposition """
    U, S, V = yast.linalg.svd(a, axes=((3, 1), (2, 0)), sU=-1)
    US = yast.tensordot(U, S, axes=(2, 0))
    USV = yast.tensordot(US, V, axes=(2, 0))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert yast.norm(a - USV) < tol  # == 0.0
    assert all(x.is_consistent() for x in (a, U, S, V))

    # changes signature of new leg; and position of new leg
    U, S, V = yast.linalg.svd(a, axes=((3, 1), (2, 0)), sU=1, Uaxis=0, Vaxis=-1)
    US = yast.tensordot(S, U, axes=(0, 0))
    USV = yast.tensordot(US, V, axes=(0, 2))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert yast.norm(a - USV) < tol  # == 0.0
    assert all(x.is_consistent() for x in (U, S, V))


def test_svd_basic():
    """ test svd decomposition for various symmetries """
    # dense
    a = yast.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    svd_combine(a)

    # U1
    legs = [yast.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yast.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(5, 6, 7)),
            yast.Leg(config_U1, s=1, t=(-2, -1, 0, 1, 2), D=(6, 5, 4, 3, 2)),
            yast.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))]
    a = yast.rand(config=config_U1, n=1, legs=legs)
    svd_combine(a)

    # Z2xU1
    legs = [yast.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(2, 3, 4, 5)),
            yast.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(5, 4, 3, 2)),
            yast.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(2, 1, 3, 3)),
            yast.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 4))]
    a = yast.ones(config=config_Z2xU1, legs=legs)
    svd_combine(a)


def test_svd_Z3():
    # Z3
    sset = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    nset = (0, 1, 2)
    sUset = (-1, 1)
    nUset = (True, False)
    for s, n, sU, nU in product(sset, nset, sUset, nUset):
        a = yast.rand(config=config_Z3, s=s, n=n, t=[(0, 1, 2), (0, 1, 2)], D=[(2, 5, 3), (5, 2, 3)], dtype='complex128')
        U, S, V = yast.linalg.svd(a, axes=(0, 1), sU=sU, nU=nU)
        assert yast.norm(a - U @ S @ V) < tol  # == 0.0
        assert all(x.is_consistent() for x in (U, S, V))


def test_svd_complex():
    """ test svd decomposition and dtype propagation """
    # dense
    a = yast.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21], dtype='complex128')
    U, S, V = yast.linalg.svd(a, axes=((0, 1), (2, 3)), sU=-1)
    assert U.yast_dtype == 'complex128'
    assert S.yast_dtype == 'float64'

    US = yast.tensordot(U, S, axes=(2, 0))  # here tensordot goes though broadcasting
    USV = yast.tensordot(US, V, axes=(2, 0))
    assert yast.norm(a - USV) < tol  # == 0.0

    SS = yast.diag(S)
    assert SS.yast_dtype == 'float64'
    US = yast.tensordot(U, SS, axes=(2, 0))
    USV = yast.tensordot(US, V, axes=(2, 0))
    assert yast.norm(a - USV) < tol  # == 0.0


def test_svd_sparse():
    a = yast.Tensor(config=config_U1, s=(-1, -1, -1, 1, 1, 1), n=0)
    a.set_block(ts=(1, 0, 0, 1, 0, 0), Ds=(2, 2, 2, 2, 2, 2), val='rand')
    a.set_block(ts=(0, 1, 0, 0, 1, 0), Ds=(2, 2, 2, 2, 2, 2), val='rand')
    a.set_block(ts=(0, 0, 1, 0, 0, 1), Ds=(2, 2, 2, 2, 2, 2), val='rand')
    axes = ((1, 2, 0), (5, 3, 4))
    U, S, V = yast.linalg.svd(a, axes=axes)
    assert all(x.is_consistent() for x in (U, S, V))

    USV = U @ S @ V
    b = a.transpose(axes=axes[0] + axes[1])
    assert yast.norm(b - USV) < tol  # == 0.0


def test_svd_truncate():
    legs = [yast.Leg(config_U1, s=1, t=(0, 1), D=(5, 6)),
            yast.Leg(config_U1, s=1, t=(-1, 0), D=(5, 6)),
            yast.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yast.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4))]
    a = yast.rand(config=config_U1, n=1, legs=legs)

    U, S, V = yast.linalg.svd(a, axes=((0, 1), (2, 3)), sU=-1)

    # fixing singular values for testing
    S.set_block(ts=(-2, -2), Ds=4, val=[2**(-ii - 6) for ii in range(4)])
    S.set_block(ts=(-1, -1), Ds=12, val=[2**(-ii - 2) for ii in range(12)])
    S.set_block(ts=(0, 0), Ds=25, val=[2**(-ii - 1) for ii in range(25)])

    a = yast.ncon([U, S, V], [(-1, -2, 1), (1, 2), (2, -3, -4)])

    opts = {'tol': 0.01, 'D_block': 100, 'D_total': 12}
    U1, S1, V1 = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), sU=-1, **opts)  # nU=True
    assert S1.get_blocks_charge() == ((-2, -2), (-1, -1), (0, 0))
    assert S1.s[0] == 1 and U1.n == a.n and V1.n == (0,) and S1.get_shape() == (12, 12)

    # specific charges of S1 depend on signature of a new leg comming from U, 
    # and where charge of tensor 'a' is attached
    U1, S1, V1 = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), **opts)  # sU=1, nU=True
    assert S1.get_blocks_charge() == ((0, 0), (1, 1), (2, 2))
    assert S1.s[0] == -1 and U1.n == a.n and V1.n == (0,) and S1.get_shape() == (12, 12)

    U1, S1, V1 = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), sU=-1, nU=False, **opts)
    assert S1.get_blocks_charge() == ((-1, -1), (0, 0), (1, 1))
    assert S1.s[0] == 1 and U1.n == (0,) and V1.n == a.n and S1.get_shape() == (12, 12)

    U1, S1, V1 = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), sU=1, nU=False, **opts)
    assert S1.get_blocks_charge() == ((-1, -1), (0, 0), (1, 1))
    assert S1.s[0] == -1 and U1.n == (0,) and V1.n == a.n and S1.get_shape() == (12, 12)

    try:
        _, S2, _ = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), sU=-1, **opts, policy='lowrank')
        assert S2.get_shape() == (12, 12)
    except NameError:
        pass

    opts = {'tol': 0.02, 'D_block': 5, 'D_total': 100}
    _, S1, _ = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), sU=1, **opts)
    assert S1.get_shape() == (11, 11)

    opts = {'tol': 0, 'tol_block': 0.2, 'D_total': 100}
    _, S1, _ = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), sU=1, **opts)
    assert S1.get_shape() == (9, 9)

    opts = {'D_block': {(0,): 2, (-1,): 3}, 'tol_block': {(0,): 0.6, (-1,): 0.1}}
    _, S1, _ = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), sU=-1, **opts)
    assert S1.get_shape() == (4, 4)

    opts = {'D_block': {(0,): 2, (-1,): 0}, 'policy': 'lowrank'}
    U1, S1, V1 = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), nU=True, sU=-1, **opts)
    assert S1.get_shape() == (2, 2)
    a1 = U1 @ S1 @ V1

    # truncation by charge requires some care to properly assign charges, given nU and sU.
    opts = {'D_block': {(1,): 2}, 'policy': 'lowrank'}
    U2, S2, V2 = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), nU=False, sU=-1, **opts)
    assert S1.get_shape() == (2, 2)
    a2 = U2 @ S2 @ V2
    assert yast.norm(a1 - a2) < tol

    opts = {'D_total': 0}
    _, S2, _ = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), nU=False, sU=-1, **opts)
    assert S2.norm() < tol


def test_svd_multiplets():
    config_U1.backend.random_seed(seed=0)  # to fix consistency of tests
    legs = [yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 2)),
            yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(3, 4, 3)),
            yast.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(4, 5, 4)),
            yast.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(5, 6, 5))]
    a = yast.rand(config=config_U1, n=0, legs=legs)

    U, S, V = yast.linalg.svd(a, axes=((0, 1), (2, 3)))

    # fixing singular values for testing
    v00 = [1, 1, 0.1001, 0.1000, 0.1000, 0.0999, 0.001001, 0.001000] + [0] * 16
    S.set_block(ts=(0, 0), Ds=24, val=v00)

    v11 = [1, 1, 0.1001, 0.1000, 0.0999, 0.001000, 0.000999] + [0] * 10
    S.set_block(ts=(1, 1), Ds=17, val=v11)
    S.set_block(ts=(-1, -1), Ds=17, val=v11)

    v22 = [1, 1, 0.1001, 0.1000, 0.001000, 0]
    S.set_block(ts=(2, 2), Ds=6, val=v22)
    S.set_block(ts=(-2, -2), Ds=6, val=v22)

    a = yast.ncon([U, S, V], [(-1, -2, 1), (1, 2), (2, -3, -4)])

    opts = {'tol': 0.0001, 'D_block': 7, 'D_total': 30}
    _, S1, _ = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), **opts)
    print(sorted(np.diag(S1.to_numpy())))
    assert S1.get_shape() == (30, 30)

    mask_f = lambda x: yast.truncation_mask_multiplets(x, tol=0.00001, D_total=30, eps_multiplet=0.001)
    _, S1, _ = yast.linalg.svd_with_truncation(a, axes=((0, 1), (2, 3)), mask_f=mask_f)
    assert S1.get_shape() == (24, 24)


def test_svd_tensor_charge_division():
    legs = [yast.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yast.Leg(config_U1, s=-1, t=(-2, 0, 2), D=(5, 6, 7)),
            yast.Leg(config_U1, s=1, t=(-2, -1, 0, 1, 2), D=(6, 5, 4, 3, 2)),
            yast.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))]
    a = yast.rand(config=config_U1, n=3, legs=legs)

    U1, S1, V1 = yast.linalg.svd(a, axes=((0, 1), (2, 3)), nU=True, sU=1)
    US1 = yast.tensordot(U1, S1, axes=(2, 0))
    USV1 = yast.tensordot(US1, V1, axes=(2, 0))
    assert yast.norm(a - USV1) < tol  # == 0.0
    assert U1.struct.n == (3,)
    assert V1.struct.n == (0,)

    U2, S2, V2 = yast.linalg.svd(a, axes=((0, 1), (2, 3)), nU=False, sU=1)
    US2 = yast.tensordot(U2, S2, axes=(2, 0))
    USV2 = yast.tensordot(US2, V2, axes=(2, 0))
    assert yast.norm(a - USV2) < tol  # == 0.0
    assert U2.struct.n == (0,)
    assert V2.struct.n == (3,)


@pytest.mark.skipif(config_dense.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_svd_backward_basic():
    import torch
    # U1
    for dtype in ["float64", "complex128"]:
        for p in ['inf', 'fro']:
            a = yast.rand(config=config_U1, s=(-1, -1, 1, 1),
                    t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                    D=[(2, 3), (4, 5), (4, 3), (2, 1)], dtype=dtype)

            def test_f(data):
                a._data=data
                U, S, V = a.svd(axes=([0, 1], [2, 3]))
                return S.norm(p=p)

            op_args = (torch.randn_like(a.data,requires_grad=True), )
            test = torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)
            assert test


@pytest.mark.skipif(config_dense.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_svd_backward_truncate():
    import torch
    # U1
    for dtype in ["float64", "complex128"]:
        a = yast.rand(config=config_U1, s=(-1, -1, 1, 1),
                      t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                      D=[(2, 3), (4, 5), (4, 3), (2, 1)], dtype=dtype)

        b = yast.rand(config=config_U1, s=(-1, -1),
                      t=[(0, 1), (0, 1)],
                      D=[(4, 3), (2, 1)], dtype=dtype)

        c = yast.rand(config=config_U1, s=(1, 1),
                      t=[(0, 1), (0, 1)],
                      D=[(2, 3), (4, 5)], dtype=dtype)

        def test_f(data):
            a._data=data
            U, S, Vh = a.svd_with_truncation(axes=([0, 1], [2, 3]), D_total=10)
            r = c.tensordot(U, axes=([0, 1], [0, 1]))
            r = r.tensordot(S, axes=(0, 0))
            r = r.tensordot(Vh, axes=(0, 0))
            r = r.tensordot(b, axes=([0, 1], [0, 1]))
            return r.to_number()

        op_args = (torch.randn_like(a.data, requires_grad=True),)
        test = torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)
        assert test


def test_svd_exceptions():
    """ raising exceptions by svd(), and some corner cases. """
    legs = [yast.Leg(config_U1, s=1, t=(0, 1), D=(5, 6)),
            yast.Leg(config_U1, s=1, t=(-1, 0), D=(5, 6)),
            yast.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4))]
    a = yast.rand(config=config_U1, legs=legs)
    
    with pytest.raises(yast.YastError):
        _ = yast.svd(a, axes=((0, 1), 2), policy='wrong_policy')
        # svd policy should be one of (`lowrank`, `fullrank`)
    with pytest.raises(yast.YastError):
        _ = yast.svd(a, axes=((0, 1), 2), policy='lowrank')
        # lowrank policy in svd requires passing argument D_block
    
    _, S, _ = yast.svd(a, axes=((0, 1), 2))
    with pytest.raises(yast.YastError):
        _ = yast.truncation_mask(1j * S, tol=1e-10)
        # Truncation_mask requires S to be real and diagonal
    with pytest.raises(yast.YastError):
        _ = yast.truncation_mask_multiplets(1j * S, tol=1e-10)
        # Truncation_mask requires S to be real and diagonal




if __name__ == '__main__':
    test_svd_basic()
    test_svd_Z3()
    test_svd_sparse()
    test_svd_complex()
    test_svd_truncate()
    test_svd_tensor_charge_division()
    test_svd_multiplets()
    test_svd_exceptions()
    test_svd_backward_basic()
    test_svd_backward_truncate()
