import yamps.mps as mps
import yamps.yast as yast
import numpy as np
import config_U1_R as config


def mps_random(N=2, Dblocks=[2], total_charge=1):
    Dblocks = tuple(Dblocks)
    psi = mps.Mps(N, nr_phys=1)
    tc = (0, 1)
    Dc = (1, 1)
    nb = len(Dblocks)
    for n in range(N):
        tl = (n * total_charge) // N
        tl = tuple(tl + ii for ii in range((-nb + 1) // 2, (nb + 1) // 2)) if n > 0 else (0,)
        Dl = Dblocks if n > 0 else (1,)
        tr = ((n + 1) * total_charge) // N
        tr = tuple(tr + ii for ii in range((-nb + 1) // 2, (nb + 1) // 2)) if n < N - 1 else (total_charge,)
        Dr = Dblocks if n < N - 1 else (1,)
        psi.A[n] = yast.rand(config=config, s=(1, 1, -1), t=[tl, tc, tr], D=[Dl, Dc, Dr])
    return psi


def mpo_random(N=2, Dblocks=[2], total_charge=1, t_out=None, t_in=(0, 1), dtype='float64'):
    psi = mps.Mps(N, nr_phys=2)
    Dblocks = tuple(Dblocks)
    nb = len(Dblocks)
    if t_out is None:
        t_out = t_in
    Din = (1,) * len(t_in)
    Dout = (1,) * len(t_out)
    for n in range(N):
        tl = (n * total_charge) // N
        tl = tuple(tl + ii for ii in range((-nb + 1) // 2, (nb + 1) // 2)) if n > 0 else (0,)
        Dl = Dblocks if n > 0 else (1,)
        tr = ((n + 1) * total_charge) // N
        tr = tuple(tr + ii for ii in range((-nb + 1) // 2, (nb + 1) // 2)) if n < N - 1 else (total_charge,)
        Dr = Dblocks if n < N - 1 else (1,)
        psi.A[n] = yast.rand(config=config, s=(1, 1, -1, -1), t=[tl, t_out, t_in, tr], D=[Dl, Dout, Din, Dr])
    return psi


def mpo_XX_model(N, t, mu):
    H = mps.Mps(N, nr_phys=2)
    m = mu
    w = t
    for n in H.g.sweep(to='last'):
        H.A[n] = yast.Tensor(config=config, s=[1, 1, -1, -1], n=0)
        if n == H.g.first:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[mu, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, -1), val=[w], Ds=(1, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[w], Ds=(1, 1, 1, 1))
        elif n == H.g.last:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[1, mu], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(-1, 1, 0, 0), val=[1], Ds=(1, 1, 1, 1))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[1], Ds=(1, 1, 1, 1))
        else:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[[1, 0], [mu, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, -1), val=[0, w], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[0, w], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(-1, 1, 0, 0), val=[1, 0], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[1, 0], Ds=(1, 1, 1, 2))
    return H

def mpo_occupation(N):
    H = mps.Mps(N, nr_phys=2)
    for n in H.g.sweep(to='last'):
        H.A[n] = yast.Tensor(config=config, s=[1, 1, -1, -1], n=0)
        if n == H.g.first:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[1, 1], Ds=(1, 1, 1, 2))
        elif n == H.g.last:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[1, 1], Ds=(2, 1, 1, 1))
        else:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[[1, 0], [1, 1]], Ds=(2, 1, 1, 2))
    return H