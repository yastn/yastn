""" yast.linalg.svd() and truncation of its singular values """
import pytest
import numpy as np
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-10  #pylint: disable=invalid-name


def svd_combine(a):
    """ decompose and contracts tensor using svd decomposition """
    U, S, V = yast.linalg.svd_pure(a, axes=((3, 1), (2, 0)), sU=-1)
    US = yast.tensordot(U, S, axes=(2, 0))
    USV = yast.tensordot(US, V, axes=(2, 0))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert yast.norm(a - USV) < tol  # == 0.0
    assert a.is_consistent()
    assert U.is_consistent()
    assert S.is_consistent()
    assert V.is_consistent()

    # changes signature of new leg; and position of new leg
    U, S, V = yast.linalg.svd_pure(a, axes=((3, 1), (2, 0)), sU=1, Uaxis=0, Vaxis=-1)
    US = yast.tensordot(S, U, axes=(0, 0))
    USV = yast.tensordot(US, V, axes=(0, 2))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert yast.norm(a - USV) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()
    assert V.is_consistent()


def test_svd_basic():
    """ test svd decomposition for various symmetries """
    # dense
    a = yast.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    svd_combine(a)

    # U1
    a = yast.rand(config=config_U1, s=(-1, -1, 1, 1), n=1,
                  t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                  D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    svd_combine(a)

    # Z2xU1
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.ones(config=config_Z2xU1, s=(-1, -1, 1, 1),
                  t=[t1, t1, t1, t1],
                  D=[(2, 3, 4, 5), (5, 4, 3, 2), (3, 4, 5, 6), (1, 2, 3, 4)])
    svd_combine(a)

def test_svd_complex():
    """ test svd decomposition for various symmetries """
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
    if SS.config.backend.BACKEND_ID == 'torch':
        with pytest.raises(RuntimeError):
            US = yast.tensordot(U, SS, axes=(2, 0))  # here tensordot uses @ in the backend
            # torch throws an error for: complex128 @ float64
        SS = SS.to(dtype='complex128')
    US = yast.tensordot(U, SS, axes=(2, 0))
    USV = yast.tensordot(US, V, axes=(2, 0))
    assert yast.norm(a - USV) < tol  # == 0.0


def test_svd_sparse():
    a = yast.Tensor(config=config_U1, s=(-1, -1, -1, 1, 1, 1), n=0)
    a.set_block(ts=(1, 0, 0, 1, 0, 0), Ds=(2, 2, 2, 2, 2, 2), val='rand')
    a.set_block(ts=(0, 1, 0, 0, 1, 0), Ds=(2, 2, 2, 2, 2, 2), val='rand')
    a.set_block(ts=(0, 0, 1, 0, 0, 1), Ds=(2, 2, 2, 2, 2, 2), val='rand')
    axes = ((1, 2, 0), (5, 3, 4))
    U, S, V = yast.linalg.svd(a, axes)
    assert U.is_consistent()
    assert S.is_consistent()
    assert V.is_consistent()

    ll = len(axes[0])
    US = yast.tensordot(U, S, axes=(ll, 0))
    USV = yast.tensordot(US, V, axes=(ll, 0))
    b = a.transpose(axes=axes[0] + axes[1])
    assert yast.norm(b - USV) < tol  # == 0.0


def test_svd_truncate():
    a = yast.rand(config=config_U1, s=(1, 1, -1, -1), n=1,
                  t=[(0, 1), (-1, 0), (-1, 0, 1), (-1, 0, 1)],
                  D=[(5, 6), (5, 6), (2, 3, 4), (2, 3, 4)])
    U, S, V = yast.linalg.svd(a, axes=((0, 1), (2, 3)), sU=-1)

    # fixing singular values for testing
    S.set_block(ts=(-2, -2), Ds=4, val=np.array([2**(-ii - 6) for ii in range(4)]))
    S.set_block(ts=(-1, -1), Ds=12, val=np.array([2**(-ii - 2) for ii in range(12)]))
    S.set_block(ts=(0, 0), Ds=25, val=np.array([2**(-ii - 1) for ii in range(25)]))

    a = yast.ncon([U, S, V], [(-1, -2, 1), (1, 2), (2, -3, -4)])

    opts = {'tol': 0.01, 'D_block': 100, 'D_total': 12}
    _, S1, _, uS = yast.linalg.svd(a, axes=((0, 1), (2, 3)), sU=-1, **opts, untruncated_S=True)
    assert S1.get_shape() == (12, 12)
    assert all(pytest.approx(sum(uS[t]).item(), rel=tol) == sum(S[t + t]).item() for t in [(-2,), (-1,), (0,)])

    try:
        _, S2, _ = yast.linalg.svd_lowrank(a, axes=((0, 1), (2, 3)), sU=-1, **opts)
        assert S2.get_shape() == (12, 12)
        assert yast.norm(S1 - S2) < tol
    except NameError:
        pass


def test_svd_multiplets():
    a = yast.rand(config=config_U1, s=(1, 1, -1, -1), n=0,
                  t=[(-1, 0, 1), (-1, 0, 1), (-1, 0, 1), (-1, 0, 1)],
                  D=[(2, 3, 2), (3, 4, 3), (4, 5, 4), (5, 6, 5)])
    U, S, V = yast.linalg.svd(a, axes=((0, 1), (2, 3)))

    # fixing singular values for testing
    v00 = np.array([1, 1, 0.1001, 0.1000, 0.1000, 0.0999, 0.001001, 0.001000] + [0] * 16)
    S.set_block(ts=(0, 0), Ds=24, val=v00)

    v11 = np.array([1, 1, 0.1001, 0.1000, 0.0999, 0.001000, 0.000999] + [0] * 10)
    S.set_block(ts=(1, 1), Ds=17, val=v11)
    S.set_block(ts=(-1, -1), Ds=17, val=v11)

    v22 = np.array([1, 1, 0.1001, 0.1000, 0.001000, 0])
    S.set_block(ts=(2, 2), Ds=6, val=v22)
    S.set_block(ts=(-2, -2), Ds=6, val=v22)

    a = yast.ncon([U, S, V], [(-1, -2, 1), (1, 2), (2, -3, -4)])

    opts = {'tol': 0.0001, 'D_block': 7, 'D_total': 30}
    _, S1, _ = yast.linalg.svd(a, axes=((0, 1), (2, 3)), **opts)
    print(sorted(np.diag(S1.to_numpy())))
    assert S1.get_shape() == (30, 30)

    opts = {'tol': 0.00001, 'D_block': 7, 'D_total': 30, 'keep_multiplets': True, 'eps_multiplet': 0.001}
    _, S1, _ = yast.linalg.svd(a, axes=((0, 1), (2, 3)), **opts)
    print(sorted(np.diag(S1.to_numpy())))
    assert S1.get_shape() == (24, 24)


def test_svd_tensor_charge_division():
    a = yast.rand(config=config_U1, s=(-1, -1, 1, 1), n=3,
                  t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                  D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
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
def test_svd_backward_1():
    import torch

    # U1
    for dtype in ["float64", "complex128"]:
        a= yast.rand(config=config_U1, s=(-1, -1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(2, 3), (4, 5), (4, 3), (2, 1)], dtype=dtype)

        def test_f(data):
            a._data=data
            U,S,V= a.svd(axes=([0,1],[2,3]))
            return S.norm(p='inf')

        op_args = (torch.randn_like(a.data,requires_grad=True), )
        test = torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)
        assert test

@pytest.mark.skipif(config_dense.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_svd_backward_full():
    import torch

    # U1
    for dtype in ["float64", "complex128"]:
        a= yast.rand(config=config_U1, s=(-1, -1, 1, 1),
                      t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                      D=[(2, 3), (4, 5), (4, 3), (2, 1)], dtype=dtype)

        def test_f(data):
            a._data=data
            U,S,V= a.svd(axes=([0,1],[2,3]))
            return S.norm(p='fro')

        op_args = (torch.randn_like(a.data,requires_grad=True), )
        test = torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)
        assert test

@pytest.mark.skipif(config_dense.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_svd_backward_truncate():
    import torch

    # U1
    for dtype in ["float64", "complex128"]:
        a= yast.rand(config=config_U1, s=(-1, -1, 1, 1),
                      t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                      D=[(2, 3), (4, 5), (4, 3), (2, 1)], dtype=dtype)

        b= yast.rand(config=config_U1, s=(-1, -1),
                      t=[(0, 1), (0, 1)],
                      D=[(4, 3), (2, 1)], dtype=dtype)

        c= yast.rand(config=config_U1, s=(1, 1),
                      t=[(0, 1), (0, 1)],
                      D=[(2, 3), (4, 5)], dtype=dtype)

        def test_f(data):
            a._data=data
            U,S,Vh= a.svd(axes=([0,1],[2,3]), D_total=10)
            r= c.tensordot(U,([0,1],[0,1]))
            r= r.tensordot(S,(0,0))
            r= r.tensordot(Vh,([0],[0]))
            r= r.tensordot(b,([0,1],[0,1]))
            return r.to_number()

        op_args = (torch.randn_like(a.data,requires_grad=True), )
        test = torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)
        assert test

if __name__ == '__main__':
    test_svd_basic()
    test_svd_sparse()
    test_svd_complex()
    test_svd_truncate()
    test_svd_tensor_charge_division()
    test_svd_multiplets()
    test_svd_backward_1()
    test_svd_backward_full()
    test_svd_backward_truncate()
