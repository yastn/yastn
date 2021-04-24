try:
    import yast
except ModuleNotFoundError:
    import fix_path
    import yast
from yast.initialize import randR
import config_U1_R
import numpy as np
from scipy.sparse.linalg import eigs, LinearOperator
# import torch

tol = 1e-12

# import yast.backend.backend_torch as backend
# config_U1_R.backend = backend

def test_eigs_1():
    a = yast.rand(config=config_U1_R, s=(1, 1, -1), n=0,
                  t=[(-1, 0, 1), (0, 1), (-1, 0, 1)],
                  D=[(2, 3, 4), (2, 3), (2, 3, 4)])

    tm = yast.ncon([a, a], [(-1, 1, -3), (-2, 1, -4)], conjs=(0, 1))
    tm.fuse_legs(axes=((0, 1), (2, 3)), inplace=True)
    tmn = tm.to_numpy()
    wn , vn = eigs(tmn, k=9, which='LM')
    print(wn)

    v0 = yast.match_legs([a, a], (2, 2), (0, 1), val='randR')
    r1d, meta = yast.compress_to_1d(v0)

    # v1 = yast.match_legs([a, a], (2, 2), (0, 1), val='randR', n=1)
    # r1d, meta = yast.compress_to_1d(v1)


    def f(x):
        t = yast.decompress_from_1d(x, config_U1_R, meta)
        t2 = yast.ncon([a, a, t], [(-1, 1, 2), (-2, 1, 3), (2, 3)], conjs=(0, 1, 0))
        t3, _ = yast.compress_to_1d(t2, meta=meta)
        return t3

    
    # ff = LinearOperator(shape=(len(r1d), len(r1d)), matvec=f, dtype=np.float64)
    # wy , vy = eigs(ff, v0=r1d, k=9, which='LM', tol=1e-10)
    # print(wy)


if __name__ == '__main__':
    test_eigs_1()
