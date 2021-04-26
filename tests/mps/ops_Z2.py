from context import yast
from context import config_Z2


def mps_random(N=2, Dblock=2, total_parity=0):
    psi = yast.mps.Mps(N, nr_phys=1)
    tc = (0, 1)
    Dc = (1, 1)
    for n in range(N):
        tr = (0, 1) if n < N - 1 else (total_parity, )
        Dr = (Dblock, Dblock) if n < N - 1 else (1,)
        tl = (0, 1) if n > 0 else (0,)
        Dl = (Dblock, Dblock) if n > 0 else (1,)
        psi.A[n] = yast.rand(config=config_Z2, s=(1, 1, -1), t=[tl, tc, tr], D=[Dl, Dc, Dr])
    return psi


def mpo_random(N=2, Dblock=2, total_parity=0, t_out=None, t_in=(0, 1)):
    psi = yast.mps.Mps(N, nr_phys=2)
    if t_out is None:
        t_out = t_in
    Din = (1,) * len(t_in)
    Dout = (1,) * len(t_out)
    for n in range(N):
        tr = (0, 1) if n < N - 1 else (total_parity, )
        Dr = (Dblock, Dblock) if n < N - 1 else (1,)
        tl = (0, 1) if n > 0 else (0,)
        Dl = (Dblock, Dblock) if n > 0 else (1,)
        psi.A[n] = yast.rand(config=config_Z2, s=(1, 1, -1, -1), t=[tl, t_out, t_in, tr], D=[Dl, Dout, Din, Dr])
    return psi


def mpo_XX_model(N, t, mu):
    H = yast.mps.Mps(N, nr_phys=2)
    for n in H.g.sweep(to='last'):
        H.A[n] = yast.Tensor(config=config_Z2, s=[1, 1, -1, -1], n=0)
        if n == H.g.first:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[mu, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, 1), val=[t, 0], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[0, t], Ds=(1, 1, 1, 2))
        elif n == H.g.last:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[1, mu], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(1, 1, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[0, 1], Ds=(2, 1, 1, 1))
        else:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[[1, 0], [mu, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, 1), val=[[0, 0], [t, 0]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[[0, 0], [0, t]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(1, 1, 0, 0), val=[[1, 0], [0, 0]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[[0, 0], [1, 0]], Ds=(2, 1, 1, 2))
    return H

def mpo_occupation(N):
    H = yast.mps.Mps(N, nr_phys=2)
    for n in H.g.sweep(to='last'):
        H.A[n] = yast.Tensor(config=config_Z2, s=[1, 1, -1, -1], n=0)
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
