import numpy as np
import yast
import yamps
try:
    from .configs import config_U1, config_U1_fermionic
except ImportError:
    from configs import config_U1, config_U1_fermionic


def random_seed(seed):
    config_U1.backend.random_seed(seed)


def mps_random(N=2, Dblocks=(2,), total_charge=1, dtype='float64'):
    Dblocks = tuple(Dblocks)
    psi = yamps.Mps(N)
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

def mps_random_fermionic(N=2, Dblocks=(2,), total_charge=1, dtype='float64'):
    Dblocks = tuple(Dblocks)
    psi = yamps.Mps(N)
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
        psi.A[n] = yast.rand(config=config_U1_fermionic, s=(1, 1, -1), t=[tl, tc, tr], D=[Dl, Dc, Dr], dtype=dtype)
    return psi


def mpo_random(N=2, Dblocks=(2,), total_charge=1, t_out=None, t_in=(0, 1)):
    psi = yamps.Mpo(N)
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
    H = yamps.Mpo(N)
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
    H = yamps.Mpo(N)
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
    # prepare generator
    gen = yamps.GenerateOpEnv(N=chain, config=config_U1_fermionic)
    gen.use_default()
    ## generate directly, most general
    #H = gen.sum([yamps.mpo_term(mu, (n, n), ('cp', 'c')) for n in range(gen.N)]) \
    #    + gen.sum([yamps.mpo_term(t, (n, n+1), ('cp', 'c')) for n in range(gen.N-1)]) \
    #    + gen.sum([yamps.mpo_term(t, (n+1, n), ('cp', 'c')) for n in range(gen.N-1)])
    # generate Mpo from string, user firiendly but not very good
    parameters = {"t": t, "mu": mu}
    H_str = "\sum_{j=0}^{"+str(chain-1)+"} mu*cp_{j}.c_{j} + \sum_{j=0}^{"+str(chain-2)+"} t*cp_{j}.c_{j+1} + \sum_{j=0}^{"+str(chain-2)+"} t*cp_{j+1}.c_{j}"
    H = gen.latex2yamps(H_str, parameters)
    return H

