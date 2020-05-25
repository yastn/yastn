import yamps.mps as mps
import yamps.ops.settings_Z2 as settings
from yamps.ops.settings_Z2 import tensor
import numpy as np


def mps_random(N=2, Dmax=2, total_parity=0, dtype='float64'):
    psi = mps.Mps(N, nr_phys=1)
    tc = (0, 1)
    Dc = (1, 1)
    for n in range(N):
        tr = (0, 1) if n < N - 1 else (total_parity, )
        Dr = (Dmax, Dmax) if n < N - 1 else (1,)
        tl = (0, 1) if n > 0 else (0,)
        Dl = (Dmax, Dmax) if n > 0 else (1,)
        psi.A[n] = tensor.rand(settings=settings, s=(1, 1, -1), t=[tl, tc, tr], D=[Dl, Dc, Dr], dtype=dtype)
    return psi


def mpo_random(N=2, Dmax=2, total_parity=0, t_out=None, t_in=(0, 1), dtype='float64'):
    psi = mps.Mps(N, nr_phys=2)
    if t_out is None:
        t_out = t_in
    Din = (1,) * len(t_in)
    Dout = (1,) * len(t_out)
    D0 = Dmax // 2
    D1 = Dmax - D0
    for n in range(N):
        tr = (0, 1) if n < N - 1 else (total_parity, )
        Dr = (D0, D1) if n < N - 1 else (1,)
        tl = (0, 1) if n > 0 else (0,)
        Dl = (D0, D1) if n > 0 else (1,)
        psi.A[n] = tensor.rand(settings=settings, s=(1, 1, -1, -1), t=[tl, t_out, t_in, tr], D=[Dl, Dout, Din, Dr], dtype=dtype)
    return psi


def mpo_XX_model(N, t, mu):
    H = mps.Mps(N, nr_phys=2)
    m = mu
    w = t
    for n in H.g.sweep(to='last'):
        H.A[n] = tensor.Tensor(settings=settings, s=[1, 1, -1, -1], n=0)
        if n == H.g.first:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=np.array([0, 1]), Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=np.array([m, 1]), Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, 1), val=np.array([w, 0]), Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=np.array([0, w]), Ds=(1, 1, 1, 2))
        elif n == H.g.last:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=np.array([1, 0]), Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=np.array([1, m]), Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(1, 1, 0, 0), val=np.array([1, 0]), Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=np.array([0, 1]), Ds=(2, 1, 1, 1))
        else:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=np.array([[1, 0], [0, 1]]), Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=np.array([[1, 0], [m, 1]]), Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, 1), val=np.array([[0, 0], [w, 0]]), Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=np.array([[0, 0], [0, w]]), Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(1, 1, 0, 0), val=np.array([[1, 0], [0, 0]]), Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=np.array([[0, 0], [1, 0]]), Ds=(2, 1, 1, 2))
    return H
