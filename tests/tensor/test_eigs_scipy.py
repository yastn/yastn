import numpy as np
import pytest
from scipy.sparse.linalg import eigs, LinearOperator
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-10  #pylint: disable=invalid-name


@pytest.mark.skipif(config_U1.backend.BACKEND_ID=="torch", reason="uses scipy procedures for raw data")
def test_eigs_simple():
    a = yast.rand(config=config_U1, s=(1, 1, -1), n=0,
                  t=[(-1, 0, 1), (0, 1), (-1, 0, 1)],
                  D=[(2, 3, 4), (2, 3), (2, 3, 4)])

    # dense transfer matrix build from a
    tm = yast.ncon([a, a], [(-1, 1, -3), (-2, 1, -4)], conjs=(0, 1))
    tm.fuse_legs(axes=((0, 1), (2, 3)), inplace=True)
    tmn = tm.to_numpy()

    wn, vn = eigs(tmn, k=9, which='LM')
    # print(wn)

    ## initializing random tensor matching TM, with 3-rd leg extra carrying charges -1, 0, 1
    vv = yast.randR(config=a.config, legs=[(a, 2, 'flip_s'), (a, 2), {'s':1, -1:1, 0:1, 1:1}])
    r1d, meta = yast.compress_to_1d(vv)

    def f(x):  # change all that into a wraper around ncon part?
        t = yast.decompress_from_1d(x, config_U1, meta)
        t2 = yast.ncon([a, a, t], [(-1, 1, 2), (-2, 1, 3), (2, 3, -3)], conjs=(0, 1, 0))
        t3, _ = yast.compress_to_1d(t2, meta=meta)
        return t3

    ff = LinearOperator(shape=(len(r1d), len(r1d)), matvec=f, dtype=np.float64)

    # eigs going though yast.tensor
    wy, vy1d = eigs(ff, v0=r1d, k=9, which='LM', tol=1e-10)
    # print(wy)  # eigenvalues

    # transform eigenvectors into yast tensors
    vy = [yast.decompress_from_1d(x, config_U1, meta) for x in vy1d.T]
    # remove zero blocks
    vyr = [yast.remove_zero_blocks(a) for a in vy]
    for a, b in zip(vy, vyr):
        assert a.are_independent(b)
    assert all((yast.norm(x - y) < tol for x, y in zip(vy, vyr)))
    # display charges of eigenvectors (only charge on last leg) -- now there is superposition between +1 and -1
    # print([x.get_leg_structure(axis=2) for x in vyr])


@pytest.mark.skipif(config_U1.backend.BACKEND_ID=="torch", reason="uses scipy procedures for raw data")
def test_eigs_exception():
    a = yast.rand(config=config_U1, s=(1, 1, -1), n=0,
                  t=[(-2, -1, 0, 1), (0, 1), (-1, 0, 1, 2)],
                  D=[(1, 2, 3, 4), (2, 3), (2, 3, 4, 5)])

    # dense transfer matrix build from a -- here a has some un-matching blocks between first and last legs
    tm = yast.ncon([a, a], [(-1, 1, -3), (-2, 1, -4)], conjs=(0, 1))
    tm.fuse_legs(axes=((0, 1), (2, 3)), inplace=True)
    # make sure to fill-in zero blocks
    ls0 = tm.get_leg_structure(axis=0)
    ls1 = tm.get_leg_structure(axis=1)
    tmn = tm.to_numpy(leg_structures={0: ls1, 1: ls0})

    wn, vn = eigs(tmn, k=9, which='LM')
    # print(wn)

    ## initializing random tensor matching TM, with 3-rd leg extra carrying charges -1, 0, 1
    vv = yast.randR(config=a.config, legs=[(a, 2, 'flip_s', a, 0), (a, 2, a, 0, 'flip_s'), {'s':1, (-1,):1, (0,):1, (1,):1}])
    r1d, meta = yast.compress_to_1d(vv)

    def f(x):  # change all that into a wraper aorund ncon part?
        t = yast.decompress_from_1d(x, config_U1, meta)
        t2 = yast.ncon([a, a, t], [(-1, 1, 2), (-2, 1, 3), (2, 3, -3)], conjs=(0, 1, 0))
        t3, _ = yast.compress_to_1d(t2, meta=meta)
        return t3
    ff = LinearOperator(shape=(len(r1d), len(r1d)), matvec=f, dtype=np.float64)

    # eigs going though yast.tensor
    wy, vy1d = eigs(ff, v0=r1d, k=9, which='LM', tol=1e-10)
    # print(wy)  # eigenvalues


    # for tm with fused legs
    vv2 = yast.randR(config=a.config, legs=[(tm, 1, 'flip', tm, 0), {'s':1, -1:1, 0:1, 1:1}])
    r1d2, meta2 = yast.compress_to_1d(vv2)

    def f2(x):
        t = yast.decompress_from_1d(x, config_U1, meta2)
        t2 = yast.ncon([tm, t], [(-1, 1), (1, -2)], conjs=(0, 0))
        t3, _ = yast.compress_to_1d(t2, meta=meta2)
        return t3

    ff2 = LinearOperator(shape=(len(r1d2), len(r1d2)), matvec=f2, dtype=np.float64)

    wy, vy2d = eigs(ff2, v0=r1d2, k=9, which='LM', tol=1e-10)
    # print(wy)  # eigenvalues


    # transform eigenvectors into yast tensors
    vy = [yast.decompress_from_1d(x, config_U1, meta) for x in vy1d.T]
    # remove zero blocks
    vyr = [yast.remove_zero_blocks(a, rtol=1e-12) for a in vy]
    assert all((yast.norm(x - y) < tol for x, y in zip(vy, vyr)))
    # display charges of eigenvectors (only charge on last leg) -- now there is superposition between +1 and -1
    # print([x.get_leg_structure(axis=2) for x in vyr])

if __name__ == '__main__':
    test_eigs_simple()
    test_eigs_exception()
