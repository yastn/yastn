import numpy as np
import yast
import yamps
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1


def random_seed(seed):
    config_U1.backend.random_seed(seed)


def mps_random(N=2, Dblocks=(2,), total_charge=1, dtype='float64'):
    Dblocks = tuple(Dblocks)
    psi = yamps.Mps(N, nr_phys=1)
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
        psi.A[n] = yast.rand(config=config_U1, s=(1, 1, -1), t=[tl, tc, tr], D=[Dl, Dc, Dr], dtype=dtype)
    return psi


def mpo_random(N=2, Dblocks=(2,), total_charge=1, t_out=None, t_in=(0, 1)):
    psi = yamps.Mps(N, nr_phys=2)
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
        psi.A[n] = yast.rand(config=config_U1, s=(1, 1, -1, -1), t=[tl, t_out, t_in, tr], D=[Dl, Dout, Din, Dr])
    return psi


def mpo_XX_model(N, t, mu):
    H = yamps.Mps(N, nr_phys=2)
    for n in H.sweep(to='last'):
        H.A[n] = yast.Tensor(config=config_U1, s=[1, 1, -1, -1], n=0)
        if n == H.first:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[mu, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, -1), val=[t], Ds=(1, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[t], Ds=(1, 1, 1, 1))
        elif n == H.last:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[1, mu], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(-1, 1, 0, 0), val=[1], Ds=(1, 1, 1, 1))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[1], Ds=(1, 1, 1, 1))
        else:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[[1, 0], [mu, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 0, 1, -1), val=[0, t], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[0, t], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(-1, 1, 0, 0), val=[1, 0], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(1, 0, 1, 0), val=[1, 0], Ds=(1, 1, 1, 2))
    return H


def mpo_occupation(N):
    H = yamps.Mps(N, nr_phys=2)
    for n in H.sweep(to='last'):
        H.A[n] = yast.Tensor(config=config_U1, s=[1, 1, -1, -1], n=0)
        if n == H.first:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[1, 1], Ds=(1, 1, 1, 2))
        elif n == H.last:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[1, 1], Ds=(2, 1, 1, 1))
        else:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[[1, 0], [1, 1]], Ds=(2, 1, 1, 2))
    return H


def mpo_gen_XX(chain, t, mu):
    Ds, s = (1, 1), (1, -1)

    C = yast.Tensor(config=config_U1, s=s, n=-1)
    C.set_block(Ds=Ds, val=1, ts=(0, 1))

    CP = yast.Tensor(config=config_U1, s=s, n=1)
    CP.set_block(Ds=Ds, val=1, ts=(1, 0))

    NN = yast.Tensor(config=config_U1, s=s, n=0)
    NN.set_block(Ds=Ds, val=1, ts=(1, 1))

    Z = yast.Tensor(config=config_U1, s=s, n=0)
    Z.set_block(Ds=Ds, val=1, ts=(0, 0))
    Z.set_block(Ds=Ds, val=-1, ts=(1, 1))

    EE = yast.Tensor(config=config_U1, s=s, n=0)
    EE.set_block(Ds=Ds, val=1, ts=(0, 0))
    EE.set_block(Ds=Ds, val=1, ts=(1, 1))

    B = np.diag([mu] * chain) + np.diag([t] * (chain - 1), 1) + np.diag([t] * (chain - 1), -1)

    from_it, to_it = B.nonzero()
    amplitude = B[B.nonzero()]
    L = len(from_it)

    permute_amp = [-1] * L

    Tensor_from, Tensor_to, Tensor_conn, Tensor_other = [], [], [], []
    for n in range(L):
        if from_it[n] == to_it[n]:
            Tensor_other.append(EE)
            Tensor_from.append(NN)
            Tensor_conn.append(None)
            Tensor_to.append(None)
        else:
            Tensor_other.append(EE)
            Tensor_from.append(CP)
            Tensor_conn.append(Z)
            Tensor_to.append(C)

    N, nr_phys, common_legs = chain, 2, (0, 1)
    return yamps.automatic_Mps(amplitude, from_it, to_it, permute_amp, Tensor_from, Tensor_to, Tensor_conn, Tensor_other, N, nr_phys, common_legs, opts={'tol': 1e-14})
