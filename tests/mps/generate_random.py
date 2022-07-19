import yast
import yamps

def random_seed(config, seed):
    config.backend.random_seed(seed)

####### mps_random ##########
def mps_random_dense(config, N, Dmax, d, dtype):
    if isinstance(d, int):
        d = [d]
    d *= (N + len(d) - 1) // len(d)

    psi = yamps.Mps(N)
    Dl, Dr = 1, Dmax
    for n in range(N):
        Dr = Dmax if n < N - 1 else 1
        Dl = Dmax if n > 0 else 1
        psi.A[n] = yast.rand(config=config, s=(1, 1, -1), D=[Dl, d[n], Dr], dtype=dtype)
    return psi


def mps_random_Z2(config, N, Dblock, total_parity, dtype):
    psi = yamps.Mps(N)
    tc = (0, 1)
    Dc = (1, 1)
    for n in range(N):
        tr = (0, 1) if n < N - 1 else (total_parity, )
        Dr = (Dblock, Dblock) if n < N - 1 else (1,)
        tl = (0, 1) if n > 0 else (0,)
        Dl = (Dblock, Dblock) if n > 0 else (1,)
        psi.A[n] = yast.rand(config=config, s=(1, 1, -1), t=[tl, tc, tr], D=[Dl, Dc, Dr], dtype=dtype)
    return psi

def mps_random_U1(config, N, Dblocks, total_charge, dtype):
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
        psi.A[n] = yast.rand(config=config, s=(1, 1, -1), t=[tl, tc, tr], D=[Dl, Dc, Dr], dtype=dtype)
    return psi


def mps_random(config, N=2, Dmax=2, d=2, Dblocks=(2,), total_charge=1, Dblock=2, total_parity=0, dtype='float64'):
    if config.sym.SYM_ID == 'dense':
        return mps_random_dense(config, N, Dmax, d, dtype)
    elif config.sym.SYM_ID == 'Z2':
        return mps_random_Z2(config, N, Dblock, total_parity, dtype)
    elif config.sym.SYM_ID == 'U(1)':
        return mps_random_U1(config, N, Dblocks, total_charge, dtype)

####### mpo_random ##########

def mpo_random_dense(config, N, Dmax, d_out, d):
    if d_out is None:
        d_out = d
    if isinstance(d, int):
        d = [d]
    d *= ((N + len(d) - 1) // len(d))
    if isinstance(d_out, int):
        d_out = [d_out]
    d_out *= ((N + len(d_out) - 1) // len(d_out))

    psi = yamps.Mpo(N)
    Dl, Dr = 1, Dmax
    for n in range(N):
        Dr = Dmax if n < N - 1 else 1
        Dl = Dmax if n > 0 else 1
        psi.A[n] = yast.rand(config=config, s=(1, 1, -1, -1), D=[Dl, d_out[n], d[n], Dr])
    return psi

def mpo_random_Z2(config, N, Dblock, total_parity, t_out, t_in):
    psi = yamps.Mpo(N)
    if t_out is None:
        t_out = t_in
    Din = (1,) * len(t_in)
    Dout = (1,) * len(t_out)
    for n in range(N):
        tr = (0, 1) if n < N - 1 else (total_parity, )
        Dr = (Dblock, Dblock) if n < N - 1 else (1,)
        tl = (0, 1) if n > 0 else (0,)
        Dl = (Dblock, Dblock) if n > 0 else (1,)
        psi.A[n] = yast.rand(config=config, s=(1, 1, -1, -1), t=[tl, t_out, t_in, tr], D=[Dl, Dout, Din, Dr])
    return psi

def mpo_random_U1(config, N, Dblocks, total_charge, t_out, t_in):
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
        psi.A[n] = yast.rand(config=config, s=(1, 1, -1, -1), t=[tl, t_out, t_in, tr], D=[Dl, Dout, Din, Dr])
    return psi

def mpo_random(config, N=2, Dmax=2, d_out=None, d=2, t_out=None, t_in=(0, 1), Dblocks=(2,), total_charge=1, Dblock=2, total_parity=0):
    if config.sym.SYM_ID == 'dense':
        return mpo_random_dense(config, N, Dmax, d_out, d)
    elif config.sym.SYM_ID == 'Z2':
        return mpo_random_Z2(config, N, Dblock, total_parity, t_out, t_in)
    elif config.sym.SYM_ID == 'U(1)':
        return mps_random_U1(config, N, Dblocks, total_charge, t_out, t_in)

