try:
    import yast
except ModuleNotFoundError:
    import fix_path
    import yast
import config_U1_R
import numpy as np

# import yast.backend.backend_torch as backend
# config_U1_R.backend = backend

tol = 1e-12


def test_svd_sparse():
    a = yast.Tensor(config=config_U1_R, s=(-1, -1, -1, 1, 1, 1), n=0)
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
    assert b.norm_diff(USV) < tol  # == 0.0


def test_svd_truncate():
    a = yast.rand(config=config_U1_R, s=(1, 1, -1, -1), n=1,
                  t=[(0, 1), (-1, 0), (-1, 0, 1), (-1, 0, 1)],
                  D=[(5, 6), (5, 6), (2, 3, 4), (2, 3, 4)])
    U, S, V = yast.linalg.svd(a, axes=((0, 1), (2, 3)), sU=-1)

    # fixing singular values for testing
    S.set_block(ts=(-2, -2), Ds=4, val=np.array([2**(-ii - 6) for ii in range(4)]))
    S.set_block(ts=(-1, -1), Ds=12, val=np.array([2**(-ii - 2) for ii in range(12)]))
    S.set_block(ts=(0, 0), Ds=25, val=np.array([2**(-ii - 1) for ii in range(25)]))

    a = yast.ncon([U, S, V], [(-1, -2, 1), (1, 2), (2, -3, -4)])

    opts = {'tol': 0.01, 'D_block': 100, 'D_total': 12}
    U1, S1, V1 = yast.linalg.svd(a, axes=((0, 1), (2, 3)), sU=-1, **opts)
    assert S1.get_shape() == (12, 12)
    try:
        U1, S1, V1 = yast.linalg.svd_lowrank(a, axes=((0, 1), (2, 3)), sU=-1, **opts)
        assert S1.get_shape() == (12, 12)
    except NameError:
        pass


def test_svd_multiplets():
    a = yast.rand(config=config_U1_R, s=(1, 1, -1, -1), n=0,
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
    U1, S1, V1 = yast.linalg.svd(a, axes=((0, 1), (2, 3)), **opts)
    print(sorted(np.diag(S1.to_numpy())))
    assert S1.get_shape() == (30, 30)

    opts = {'tol': 0.00001, 'D_block': 7, 'D_total': 30, 'keep_multiplets': True, 'eps_multiplet': 0.0001}
    U1, S1, V1 = yast.linalg.svd(a, axes=((0, 1), (2, 3)), **opts)
    print(sorted(np.diag(S1.to_numpy())))
    assert S1.get_shape() == (24, 24)


def test_svd_n_division():
    a = yast.rand(config=config_U1_R, s=(-1, -1, 1, 1), n=3,
                  t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                  D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    U1, S1, V1 = yast.linalg.svd(a, axes=((0, 1), (2, 3)), nU=True, sU=1)
    US1 = yast.tensordot(U1, S1, axes=(2, 0))
    USV1 = yast.tensordot(US1, V1, axes=(2, 0))
    assert a.norm_diff(USV1) < tol  # == 0.0
    assert abs(U1.n - 3) < tol  # == 0.0
    assert abs(V1.n - 0) < tol  # == 0.0

    U2, S2, V2 = yast.linalg.svd(a, axes=((0, 1), (2, 3)), nU=False, sU=1)
    US2 = yast.tensordot(U2, S2, axes=(2, 0))
    USV2 = yast.tensordot(US2, V2, axes=(2, 0))
    assert a.norm_diff(USV2) < tol  # == 0.0
    assert U2.n == 0
    assert V2.n == 3


if __name__ == '__main__':
    test_svd_sparse()
    test_svd_truncate()
    test_svd_n_division()
    test_svd_multiplets()
