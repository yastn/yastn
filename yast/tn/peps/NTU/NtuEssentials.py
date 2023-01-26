"""
Routines for NTU timestep on a checkerboard lattice.
NTU supports fermions though application of swap-gates.
PEPS tensors have 5 legs: (top, left, bottom, right, system)
In case of purification, system leg is a fusion of (ancila, system)
"""
import logging
import yast
from yast.tn.peps.operators.gates import trivial_tensor, match_ancilla_1s, match_ancilla_2s
from yast import tensordot, vdot, svd_with_truncation, svd, qr, swap_gate, fuse_legs, ncon, eigh_with_truncation, eye

def ntu_machine(Gamma, bd, GA, GB, Ds, truncation_mode, step, fix_bd, flag):
    # step can be svd-step, one-step or two-step
    # application of nearest neighbor gate and subsequent optimization of peps tensor using NTU
    QA, QB, RA, RB = single_bond_nn_update(Gamma, bd, GA, GB, flag)

    if step == "svd-update":
        MA, MB = truncation_step(RA, RB, Ds, normalize=True)
        Gamma[bd.site_0], Gamma[bd.site_1] = form_new_peps_tensors(QA, QB, MA, MB, bd)
        info = {}
        return Gamma, info
    else:
        g = env_NTU(Gamma, bd, QA, QB, dirn=bd.dirn)
        info={}
        MA, MB, ntu_error, optim, svd_error = truncate_and_optimize(g, RA, RB, Ds, fix_bd, truncation_mode)
        if step == 'two-step':  # else 'one-step'
            MA_int, MB_int, _, _, _ = truncate_and_optimize(g, RA, RB, int(2*Ds), fix_bd, truncation_mode)
            MA_2, MB_2, ntu_error_2, optim_2, svd_error_2 = truncate_and_optimize(g, MA_int, MB_int, Ds, fix_bd, truncation_mode)
            if ntu_error < ntu_error_2:
                logging.info("1-step NTU; ntu errors 1-and 2-step %0.5e,  %0.5e; svd error %0.5e,  %0.5e" % (ntu_error, ntu_error_2, svd_error, svd_error_2))
            else:
                MA, MB = MA_2, MB_2
                logging.info("2-step NTU; ntu errors 1-and 2-step %0.5e,  %0.5e; svd error %0.5e,  %0.5e " % (ntu_error, ntu_error_2, svd_error, svd_error_2))
                ntu_error, optim, svd_error = ntu_error_2, optim_2, svd_error_2
        Gamma[bd.site_0], Gamma[bd.site_1] = form_new_peps_tensors(QA, QB, MA, MB, bd)
        info.update({'ntu_error': ntu_error, 'optimal_cutoff': optim, 'svd_error': svd_error})

        return Gamma, info

############################
##### gate application  ####
############################

def single_bond_local_update(Gamma, G_loc, flag):
    """ apply local gates on PEPS tensors """

    target_site = (round((Gamma.Nx-1)*0.5), round((Gamma.Ny-1)*0.5)) # center of an odd x odd lattice
    for ms in Gamma.sites():
        if ms != target_site:
            G_l = match_ancilla_1s(G_loc, Gamma[ms])
            Gamma[ms] = tensordot(Gamma[ms], G_l, axes=(2, 1)) # [t l] [b r] [s a]
        elif ms == target_site: # target site can have a string attached to ancilla when hole is created at the center
            if flag == 'hole':
                Gamma[ms] =  Gamma[ms].unfuse_legs(axes=2).unfuse_legs(axes=3)  # [t l] [b r] s a str
                Gamma[ms] =  Gamma[ms].fuse_legs(axes=(0, 1, (2, 3), 4)).fuse_legs(axes=(0, 1, 3, 2)) # [t l] [b r] str [s a]
                G_l = match_ancilla_1s(G_loc, Gamma[ms])
                Gamma[ms] = tensordot(Gamma[ms], G_l, axes=(3, 1))
                Gamma[ms] = Gamma[ms].fuse_legs(axes=(0, 1, 3, 2)).unfuse_legs(axes=(2)).fuse_legs(axes=(0, 1, 2, (3, 4))).fuse_legs(axes=(0, 1, (2, 3))) # [t l] [b r] [s [a string]]
            else:
                G_l = match_ancilla_1s(G_loc, Gamma[ms])
                Gamma[ms] = tensordot(Gamma[ms], G_l, axes=(2, 1)) # [t l] [b r] [s a]

    return Gamma # local update all sites at once
    

def single_bond_nn_update(Gamma, bd, GA, GB, flag):

    """ apply nn gates on PEPS tensors. """
    target_site = (round((Gamma.Nx-1)*0.5), round((Gamma.Ny-1)*0.5)) # center of an odd x odd lattice
    A, B = Gamma[bd.site_0], Gamma[bd.site_1]  # A = [t l] [b r] s
    
    dirn = bd.dirn
    if dirn == "h":  # Horizontal Gate        
        if bd.site_0 != target_site:
            GA_an = match_ancilla_2s(GA, A, dir='l') 
            int_A = tensordot(A, GA_an, axes=(2, 1)) # [t l] [b r] s c
        elif bd.site_0 == target_site:
            if flag == 'hole':
                A =  A.unfuse_legs(axes=2).unfuse_legs(axes=3) # [t l] [b r] s a str 
                A =  A.fuse_legs(axes=(0, 1, (2, 3), 4)).fuse_legs(axes=(0, 1, 3, 2)) # [t l] [b r] str [s a]
                GA_an = match_ancilla_2s(GA, A, dir='l')
                A = tensordot(A, GA_an, axes=(3, 1)) # [t l] [b r] str [s a] c 
                A = A.unfuse_legs(axes=3) # [t l] [b r] str s a c
                A = A.fuse_legs(axes=(0, 1, 3, (4, 2), 5)) # [t l] [b r] s [a str] c
                int_A = A.fuse_legs(axes=(0, 1, (2, 3), 4)) # [t l] [b r] [s [a str]] c
            else:
                GA_an = match_ancilla_2s(GA, A, dir='l') 
                int_A = tensordot(A, GA_an, axes=(2, 1)) # [t l] [b r] s c
        int_A = int_A.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] s] [b r] c
        int_A = int_A.unfuse_legs(axes=1)  # [[t l] s] b r c
        int_A = int_A.swap_gate(axes=(1, 3))  # b X c
        int_A = int_A.fuse_legs(axes=((0, 1), (2, 3)))  # [[[t l] s] b] [r c]
        QA, RA = qr(int_A, axes=(0, 1), sQ=-1)  # [[[t l] s] b] rr @ rr [r c]

        if bd.site_1 != target_site:
            GB_an = match_ancilla_2s(GB, B, dir='r')
            int_B = tensordot(B, GB_an, axes=(2, 1)) # [t l] [b r] s c
        elif bd.site_1 == target_site:
            if flag == 'hole':
                B =  B.unfuse_legs(axes=2).unfuse_legs(axes=3) # [t l] [b r] s a str 
                B =  B.fuse_legs(axes=(0, 1, (2, 3), 4)).fuse_legs(axes=(0, 1, 3, 2)) # [t l] [b r] str [s a]
                GB_an = match_ancilla_2s(GB, B, dir='r')
                B = tensordot(B, GB_an, axes=(3, 1)) # [t l] [b r] str [s a] c 
                B = B.unfuse_legs(axes=3) # [t l] [b r] str s a c
                B = B.fuse_legs(axes=(0, 1, 3, (4, 2), 5)) # [t l] [b r] s [a str] c
                int_B = B.fuse_legs(axes=(0, 1, (2, 3), 4)) # [t l] [b r] [s [a str]] c
            else:
                GB_an = match_ancilla_2s(GB, B, dir='r')
                int_B = tensordot(B, GB_an, axes=(2, 1)) # [t l] [b r] s c
        int_B = int_B.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] s] c
        int_B = int_B.unfuse_legs(axes=0)  # t l [[b r] s] c
        int_B = int_B.fuse_legs(axes=((0, 2), (1, 3)))  # [t [[b r] s]] [l c]
        QB, RB = qr(int_B, axes=(0, 1), sQ=1, Qaxis=0, Raxis=-1)  # ll [t [[b r] s]]  @  [l c] ll       

    elif dirn == "v":  # Vertical Gate
        if bd.site_0 != target_site:
            GA_an = match_ancilla_2s(GA, A, dir='l') 
            int_A = tensordot(A, GA_an, axes=(2, 1)) # [t l] [b r] s c
        elif bd.site_0 == target_site:
            if flag == 'hole':
                A =  A.unfuse_legs(axes=2).unfuse_legs(axes=3) # [t l] [b r] s a str 
                A =  A.fuse_legs(axes=(0, 1, (2, 3), 4)).fuse_legs(axes=(0, 1, 3, 2)) # [t l] [b r] str [s a]
                GA_an = match_ancilla_2s(GA, A, dir='l')
                A = tensordot(A, GA_an, axes=(3, 1)) # [t l] [b r] str [s a] c 
                A = A.unfuse_legs(axes=3) # [t l] [b r] str s a c
                A = A.fuse_legs(axes=(0, 1, 3, (4, 2), 5)) # [t l] [b r] s [a str] c
                int_A = A.fuse_legs(axes=(0, 1, (2, 3), 4)) # [t l] [b r] [s [a str]] c
            else:
                GA_an = match_ancilla_2s(GA, A, dir='l') 
                int_A = tensordot(A, GA_an, axes=(2, 1)) # [t l] [b r] s c
        int_A = int_A.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] s] [b r] c
        int_A = int_A.unfuse_legs(axes=1)  # [[t l] s] b r c
        int_A = int_A.fuse_legs(axes=((0, 2), (1, 3)))  # [[[t l] s] r] [b c]
        QA, RA = qr(int_A, axes=(0, 1), sQ=1)  # [[[t l] s] r] bb  @  bb [b c]

        if bd.site_1 != target_site:
            GB_an = match_ancilla_2s(GB, B, dir='r')
            int_B = tensordot(B, GB_an, axes=(2, 1)) # [t l] [b r] s c
        elif bd.site_1 == target_site:
            if flag == 'hole':
                B =  B.unfuse_legs(axes=2).unfuse_legs(axes=3) # [t l] [b r] s a str 
                B =  B.fuse_legs(axes=(0, 1, (2, 3), 4)).fuse_legs(axes=(0, 1, 3, 2)) # [t l] [b r] str [s a]
                GB_an = match_ancilla_2s(GB, B, dir='r')
                B = tensordot(B, GB_an, axes=(3, 1)) # [t l] [b r] str [s a] c 
                B = B.unfuse_legs(axes=3) # [t l] [b r] str s a c
                B = B.fuse_legs(axes=(0, 1, 3, (4, 2), 5)) # [t l] [b r] s [a str] c
                int_B = B.fuse_legs(axes=(0, 1, (2, 3), 4)) # [t l] [b r] [s [a str]] c
            else:
                GB_an = match_ancilla_2s(GB, B, dir='r')
                int_B = tensordot(B, GB_an, axes=(2, 1)) # [t l] [b r] s c
        int_B = int_B.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] s] c
        int_B = int_B.unfuse_legs(axes=0)  # t l [[b r] s] c
        int_B = int_B.swap_gate(axes=(1, 3))  # l X c
        int_B = int_B.fuse_legs(axes=((1, 2), (0, 3)))  # [l [[b r] s]] [t c]
        QB, RB = qr(int_B, axes=(0, 1), sQ=-1, Qaxis=0, Raxis=-1)  # tt [l [[b r] s]]  @  [t c] tt

    return QA, QB, RA, RB


def truncation_step(RA, RB, Ds, normalize=False):
    """ svd truncation of central tensor """
    theta = RA @ RB
    if isinstance(Ds, dict):
        Ds = sum(Ds.values())
    UA, S, UB = svd_with_truncation(theta, sU=RA.get_signature()[1], tol_block=1e-15, D_total=Ds)
    if normalize:
        S = S / S.norm(p='inf')
    sS = S.sqrt()
    MA = sS.broadcast(UA, axis=1)
    MB = sS.broadcast(UB, axis=0)
    return MA, MB


def form_new_peps_tensors(QA, QB, MA, MB, bd):
    """ combine unitaries in QA with optimized MA to form new peps tensors. """
    if bd.dirn == "h":
        A = QA @ MA  # [[[[t l] s] b] r
        A = A.unfuse_legs(axes=0)  # [[t l] s] b r
        A = A.fuse_legs(axes=(0, (1, 2)))  # [[t l] s] [b r]
        A = A.unfuse_legs(axes=0)  # [t l] s [b r]
        A = A.transpose(axes=(0, 2, 1))  # [t l] [b r] s

        B = MB @ QB  # l [t [[b r] s]]
        B = B.unfuse_legs(axes=1)  # l t [[b r] s]
        B = B.fuse_legs(axes=((1, 0), 2))  # [t l] [[b r] s]
        B = B.unfuse_legs(axes=1)  # [t l] [b r] s
    elif bd.dirn == "v":
        A = QA @ MA  # [[[t l] s] r] b
        A = A.unfuse_legs(axes=0)  # [[t l] s] r b
        A = A.fuse_legs(axes=(0, (2, 1)))  # [[t l] s] [b r]
        A = A.unfuse_legs(axes=0)  # [t l] s [b r]
        A = A.transpose(axes=(0, 2, 1))  # [t l] [b r] s

        B = MB @ QB  # t [l [[b r] s]]
        B = B.unfuse_legs(axes=1)  # t l [[b r] s]
        B = B.fuse_legs(axes=((0, 1), 2))  # [t l] [[b r] s]
        B = B.unfuse_legs(axes=1)  # [t l] [b r] s
    return A, B


###################################
##### creating ntu environment ####
###################################

def env_NTU(Gamma, bd, QA, QB, dirn):
    """ calculate metric g """
  
    env = Gamma.tensors_NtuEnv(bd)
    G={}
    for ms in env.keys():
        if env[ms] is None:
            leg = Gamma[(0, 0)].get_legs(axis=-1)
            leg, _ = yast.leg_undo_product(leg) # last leg of A should be fused
            fid = yast.eye(config=Gamma[(0,0)].config, legs=[leg, leg.conj()]).diag()
            G[ms] = trivial_tensor(fid)
        else:
            G[ms] = Gamma[env[ms]]

    if dirn == "h":
        m_tl, m_l, m_bl = con_tl(G['tl']), con_l(G['l']), con_bl(G['bl'])
        m_tr, m_r, m_br = con_tr(G['tr']), con_r(G['r']), con_br(G['br'])
        env_l = con_Q_l(QA, m_l)  # [t t'] [b b'] [rr rr']
        env_r = con_Q_r(QB, m_r)  # [ll ll'] [t t'] [b b']
        env_t = m_tl @ m_tr  # [tl tl'] [tr tr']
        env_b = m_br @ m_bl  # [br br'] [bl bl']
        g = ncon((env_t, env_l, env_r, env_b), ((1, 4), (1, 3, -1), (-2, 4, 2), (2, 3)))  # [ll ll'] [rr rr']
    elif dirn == "v":
        m_tl, m_t, m_tr = con_tl(G['tl']), con_t(G['t']), con_tr(G['tr'])
        m_bl, m_b, m_br = con_bl(G['bl']), con_b(G['b']), con_br(G['br'])
        env_t = con_Q_t(QA, m_t)  # [l l'] [r r'] [bb bb']
        env_b = con_Q_b(QB, m_b)  # [tt tt'] [l l'] [r r']
        env_l = m_bl @ m_tl  # [bl bl'] [tl tl']
        env_r = m_tr @ m_br  # [tr tr'] [br br']
        g = ncon((env_l, env_t, env_b, env_r), ((4, 1), (1, 3, -1), (-2, 4, 2), (3, 2)))  # [tt tt'] [bb bb']
    return g.unfuse_legs(axes=(0, 1))


def con_tl(A):  # A -> [t l] [b r] s
    """ top-left env tensor """  
    A = fuse_legs(A, axes=((0, 2), 1))  # [[t l] s] [b r]
    m_tl = tensordot(A, A, axes=(0, 0), conj=(0, 1))  # [b r] [b' r']
    m_tl = m_tl.unfuse_legs(axes=(0, 1))  # b r b' r'
    m_tl = m_tl.swap_gate(axes=((0, 2), 3))  # b b' X r'
    m_tl = m_tl.fuse_legs(axes=((0, 2), (1, 3)))  # [b b'] [r r']
    return m_tl


def con_tr(A):  # A = [t l] [b r] s
    """ top-right env tensor """
    A = A.unfuse_legs(axes=(0, 1))
    A = swap_gate(A, axes=(0, 1, 2, 3))  # t X l, b X r
    A = fuse_legs(A, axes=((1, 2), (0, 3, 4)))  # [l b] [t r s]
    m_tr = tensordot(A, A, axes=(1, 1), conj=(0, 1))  # [l b] [l' b']
    m_tr = m_tr.unfuse_legs(axes=(0, 1))  # l b l' b'
    m_tr = m_tr.fuse_legs(axes=((0, 2), (1, 3)))  # [l l'] [b b']
    return m_tr


def con_br(A):  # A = [t l] [b r] s
    """ bottom-right env tensor """
    A = fuse_legs(A, axes=(0, (1, 2)))  # [t l] [[b r] s]
    m_br = tensordot(A, A, axes=(1, 1), conj=(0, 1))  # [t l] [t' l']
    m_br = m_br.unfuse_legs(axes=(0, 1))  # t l t' l'
    m_br = m_br.swap_gate(axes=((1, 3), 2))  # l l' X t'
    m_br = m_br.fuse_legs(axes=((0, 2), (1, 3)))  # [t t'] [l l']
    return m_br


def con_bl(A):  # A = [t l] [b r] s
    """ bottom-left env tensor """
    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    A = fuse_legs(A, axes=((0, 3), (1, 2, 4)))  # [t r] [b l s]
    m_bl = tensordot(A, A, axes=(1, 1), conj=(0, 1))  # [t r] [t' r']
    m_bl = m_bl.unfuse_legs(axes=(0, 1))  # t r t' r'
    m_bl = m_bl.fuse_legs(axes=((1, 3), (0, 2)))  # [r r'] [t t']
    return m_bl


def con_l(A):  # A = [t l] [b r] s
    """ left env tensor """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    A = A.fuse_legs(axes=((0, 1, 3), 2))  # [[t l] b s] r
    m_l = tensordot(A, A, axes=(0, 0), conj=(1, 0))  # r' r
    return m_l


def con_r(A):  # A = [t l] [b r] s
    """ right env tensor """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    A = A.fuse_legs(axes=(1, (0, 2, 3)))  # l [t [b r] s]
    m_r = tensordot(A, A, axes=(1, 1), conj=(1, 0))  # [l' l]
    return m_r


def con_b(A):  # A = [t l] [b r] s
    """ bottom env tensor """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    A = A.fuse_legs(axes=(0, (1, 2, 3)))  # t [l [b r] s]
    m_b = tensordot(A, A, axes=(1, 1), conj=(1, 0))  # [t' t]
    return m_b


def con_t(A):
    """ top env tensor """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    A = A.fuse_legs(axes=((0, 2, 3), 1))  # [[t l] r s] b
    m_t = tensordot(A, A, axes=(0, 0), conj=(1, 0))  # b' b
    return m_t



def con_Q_l(QA, m_l):  # QA -> [[[t l] s] b] rr
    """ env_l where left legs of double-layer Q tensors are contracted with m_l """
    QA = QA.unfuse_legs(axes=0)  # [[t l] s] b rr
    QA = QA.swap_gate(axes=(1, 2))  # b X rr
    QA = QA.fuse_legs(axes=(0, (1, 2)))   # [[t l] s] [b rr]
    QA = QA.unfuse_legs(axes=0)  # [t l] s [b rr]
    QA = QA.unfuse_legs(axes=0)  # t l s [b rr]
    QA = QA.fuse_legs(axes=(1, 2, (0, 3)))  # l s [t [b rr]]
    mlQA = m_l @ QA  # l' s [t [b rr]]
    env_l = tensordot(mlQA, QA, axes=((0, 1), (0, 1)), conj=(0, 1))  # [t [b rr]] [t' [b' rr']]
    env_l = env_l.unfuse_legs(axes=(0, 1))  # t [b rr] t' [b' rr']
    env_l = env_l.fuse_legs(axes=((0, 2), 1, 3))  # [t t'] [b rr] [b' rr']
    env_l = env_l.unfuse_legs(axes=(1, 2))  # [t t'] b rr b' rr'
    env_l = env_l.swap_gate(axes=(1, (2, 4)))  # b x rr rr'
    env_l = env_l.fuse_legs(axes=(0, (1, 3), (2, 4)))  # [t t'] [b b'] [rr rr']
    return env_l


def con_Q_r(QB, m_r):  # QB -> ll [t [[b r] s]]
    """ env_r where right legs of double-layer Q tensors are contracted with m_r """
    QB = QB.unfuse_legs(axes=1)  # ll t [[b r] s]
    QB = QB.fuse_legs(axes=((0, 1), 2))  # [ll t] [[b r] s]
    QB = QB.unfuse_legs(axes=1)  # [ll t] [b r] s
    QB = QB.unfuse_legs(axes=1)  # [ll t] b r s
    QB = QB.swap_gate(axes=(1, 2))  # b X r
    QB = QB.fuse_legs(axes=(2, 3, (0, 1)))  # r s [[ll t] b]
    mrsQB = m_r @ QB  # r' s [ll [t b]]
    env_r = tensordot(mrsQB, QB, axes=((0, 1), (0, 1)), conj=(0, 1))  # [[ll t] b] [[ll' t'] b']
    env_r = env_r.unfuse_legs(axes=(0, 1))  # [ll t] b [ll' t'] b'
    env_r = env_r.fuse_legs(axes=(0, 2, (1, 3)))  # [ll t] [ll' t'] [b b']
    env_r = env_r.unfuse_legs(axes=(0, 1))  # ll t ll' t' [b b']
    env_r = env_r.swap_gate(axes=((0, 2), 3))  # ll ll' X t'
    env_r = env_r.fuse_legs(axes=((0, 2), (1, 3), 4))  # [ll ll'] [t t'] [b b']
    return env_r


def con_Q_t(QA, m_t):  # QA -> [[[t l] s] r] bb
    """ env_t where top legs of double-layer Q tensors are contracted with m_t """
    QA = QA.unfuse_legs(axes=0)  # [[t l] s] r bb
    QA = QA.fuse_legs(axes=(0, (1, 2)))  # [[t l] s] [r bb]
    QA = QA.unfuse_legs(axes=0)  # [t l] s [r bb]
    QA = QA.unfuse_legs(axes=0)  # t l s [r bb]
    QA = QA.swap_gate(axes=(0, 1))  # t X l
    QA = QA.fuse_legs(axes=(0, 2, (1, 3)))  # t s [l [r bb]]
    mtsQA = m_t @ QA  # t' s [l [r bb]]
    env_t = tensordot(mtsQA, QA, axes=((0, 1), (0, 1)), conj=(0, 1))  # [l [r bb]] [l' [r' bb']]
    env_t = env_t.unfuse_legs(axes=(0, 1))  # l [r bb] l' [r' bb']
    env_t = env_t.fuse_legs(axes=((0, 2), 1, 3))  # [l l'] [r bb] [r' bb']
    env_t = env_t.unfuse_legs(axes=(1, 2))  # [l l'] r bb r' bb'
    env_t = env_t.swap_gate(axes=(3, (2, 4)))  # r' X bb bb'
    env_t = env_t.fuse_legs(axes=(0, (1, 3), (2, 4))) # [l l'] [r r'] [bb bb']
    return env_t


def con_Q_b(QB, m_b):  # QB -> t [l [[b r] s]]
    """ env_t where top legs of double-layer Q tensors are contracted with m_t """
    QB = QB.unfuse_legs(axes=1)  # tt l [[b r] s]
    QB = QB.swap_gate(axes=(0, 1))  # tt X l
    QB = QB.fuse_legs(axes=((0, 1), 2))  # [tt l] [[b r] s]
    QB = QB.unfuse_legs(axes=1)  # [tt l] [b r] s
    QB = QB.unfuse_legs(axes=1)  # [tt l] b r s
    QB = QB.fuse_legs(axes=(1, 3, (0, 2)))  # b s [[tt l] r]
    mbQB = m_b @ QB  # b' s [[tt l] r]
    env_b = tensordot(mbQB, QB, axes=((0, 1), (0, 1)), conj=(0, 1))  # [[tt l] r] [[tt' l'] r']
    env_b = env_b.unfuse_legs(axes=(0, 1))  # [tt l] r [tt' l'] r'
    env_b = env_b.fuse_legs(axes=(0, 2, (1, 3)))  # [tt l] [tt' l'] [r r']
    env_b = env_b.unfuse_legs(axes=(0, 1))  # tt l tt' l' [r r']
    env_b = env_b.swap_gate(axes=((0, 2), 1)) # tt tt' X l
    env_b = env_b.fuse_legs(axes=((0, 2), (1, 3), 4))  # [tt tt'] [l l'] [r r']
    return env_b


def forced_sectorial_truncation(U, L, V, Ds):
    # truncates bond dimensions of different symmetry sectors according to agiven distribution Ds
    F = eye(config=U.config, legs=U.get_legs(1).conj())
    discard_block_weight= {}
    for k in L.get_leg_structure(axis=0).keys():
        if k in Ds.keys():
            v = Ds.get(k)
            discard_block_weight[k] = sum(L.A[k+k][v:]**2)
        elif k not in Ds.keys():
            discard_block_weight[k] = sum(L.A[k+k]**2)
    for k, v in Ds.items():
        if k in F.get_leg_structure(axis=0).keys():
            F.A[k+k][v:] = 0
    for k, v in F.get_leg_structure(axis=0).items():
        if k not in Ds.keys(): 
            F.A[k+k][0:v] = 0
    U = U.mask(F, axis=1)
    new_L = L.mask(F, axis=0)
    V = V.mask(F, axis=0)
    return U, new_L, V, discard_block_weight


def environment_aided_truncation_step(g, gRR, fgf, fgRAB, RA, RB, Ds, fix_bd, truncation_mode):

    if truncation_mode == 'optimal':
        G = ncon((g, RA, RB, RA, RB), ([1, 2, 3, 4], [1, -1], [-3, 3], [2, -2], [-4, 4]), conjs=(0, 0, 0, 1, 1))
        [ul, _, vr] = svd_with_truncation(G, axes=((0, 1), (2, 3)), tol_block=1e-15, D_total=1)
        ul = ul.remove_leg(axis=2)
        vr = vr.remove_leg(axis=0)
        GL, GR = ul.transpose(axes=(1, 0)), vr
        _, SL, UL = svd(GL)
        UR, SR, _ = svd(GR)
        XL, XR = SL.sqrt() @ UL, UR @ SR.sqrt()
        XRRX = XL @ XR
        if fix_bd == 1:
            U, L, V = svd_with_truncation(XRRX, sU=RA.get_signature()[1], tol_block=1e-15) 
            U, L, V, discard_block_weight = forced_sectorial_truncation(U, L, V, Ds)
            mA, mB = U @ L.sqrt(), L.sqrt() @ V
            MA, MB, svd_error, _ = optimal_initial_pinv(mA, mB, RA, RB, gRR, SL, UL, SR, UR, fgf, fgRAB)
            return MA, MB, svd_error
        elif fix_bd == 0:
            if isinstance(Ds, dict):
                Dn = sum(Ds.values())
            else:
                Dn = Ds
            U, L, V = svd_with_truncation(XRRX, sU=RA.get_signature()[1], D_total=Dn, tol_block=1e-15)
            mA, mB = U @ L.sqrt(), L.sqrt() @ V
            MA, MB, svd_error, _ = optimal_initial_pinv(mA, mB, RA, RB, gRR, SL, UL, SR, UR, fgf, fgRAB)
            return MA, MB, svd_error

    elif truncation_mode == 'normal':

        if fix_bd == 0:
            MA, MB = truncation_step(RA, RB, Ds)
            MAB = MA @ MB
            MAB = MAB.fuse_legs(axes=[(0, 1)])
            gMM = vdot(MAB, fgf @ MAB).item()
            gMR = vdot(MAB, fgRAB).item()
            svd_error = abs((gMM + gRR - gMR - gMR.conjugate()) / gRR)
        elif fix_bd == 1:
            XRX = RA @ RB
            U, L, V = svd_with_truncation(XRX, sU=RA.get_signature()[1], tol_block=1e-15)
            U, L, V, discard_block_weight = forced_sectorial_truncation(U, L, V, Ds)
            MA, MB = U @ L.sqrt(), L.sqrt() @ V
            MAB = MA @ MB
            MAB = MAB.fuse_legs(axes=[(0, 1)])
            gMM = vdot(MAB, fgf @ MAB).item()
            gMR = vdot(MAB, fgRAB).item()
            svd_error = abs((gMM + gRR - gMR - gMR.conjugate()) / gRR)
         
        return MA, MB, svd_error


def optimal_initial_pinv(mA, mB, RA, RB, gRR, SL, UL, SR, UR, fgf, fgRAB):

    cutoff_list = [10**n for n in range(-14, -5)]
    results = []
    for c_off in cutoff_list:
        XL_inv, XR_inv = tensordot(UL.conj(), SL.sqrt().reciprocal(cutoff=c_off), axes=(0, 0)), tensordot(SR.sqrt().reciprocal(cutoff=c_off), UR.conj(), axes=(1, 1)) 
        pA, pB = XL_inv @ mA, mB @ XR_inv
        MA, MB = RA @ pA, pB @ RB
        MAB = MA @ MB
        MAB = MAB.fuse_legs(axes=[(0, 1)])
        gMM = vdot(MAB, fgf @ MAB).item()
        gMR = vdot(MAB, fgRAB).item()
        svd_error = abs((gMM + gRR - gMR - gMR.conjugate()) / gRR)
        results.append((svd_error, c_off, MA, MB))
    svd_error, c_off, MA, MB = min(results, key=lambda x: x[0])
    return MA, MB, svd_error, c_off


def ntu_single_optimization(MA, MB, gRAB, gf, gRR, svd_error, max_iter):
    MA_guess, MB_guess = MA, MB
    ntu_errorA_old, ntu_errorB_old = 0, 0
    epsilon1, epsilon2 = 0, 0

    for _ in range(max_iter):
        # fix MB and optimize MA
        j_A = tensordot(gRAB, MB, axes=(1, 1), conj=(0, 1)).fuse_legs(axes=[(0, 1)])
        g_A = tensordot(MB, gf, axes=(1, 1), conj=(1, 0)).fuse_legs(axes=((1, 0), 2))
        g_A = g_A.unfuse_legs(axes=1)
        g_A = tensordot(g_A, MB, axes=(2, 1)).fuse_legs(axes=(0, (1, 2)))
        ntu_errorA, optimal_cf, MA = optimal_pinv(g_A, j_A, gRR)

        epsilon1 = abs(ntu_errorA_old - ntu_errorA)
        ntu_errorA_old = ntu_errorA
        count1, count2 = 0, 0
        if abs(ntu_errorA) >= abs(svd_error):
            MA = MA_guess
            epsilon1, ntu_errorA_old = 0, 0
            count1 += 1

         # fix MA and optimize MB
        j_B = tensordot(MA, gRAB, axes=(0, 0), conj=(1, 0)).fuse_legs(axes=[(0, 1)])
        g_B = tensordot(MA, gf, axes=(0, 0), conj=(1, 0)).fuse_legs(axes=((0, 1), 2))
        g_B = g_B.unfuse_legs(axes=1)
        g_B = tensordot(g_B, MA, axes=(1, 0)).fuse_legs(axes=(0, (2, 1)))
        ntu_errorB, optimal_cf, MB = optimal_pinv(g_B, j_B, gRR)

        epsilon2 = abs(ntu_errorB_old - ntu_errorB)
        ntu_errorB_old = ntu_errorB

        if abs(ntu_errorB) >= abs(svd_error):
            MB = MB_guess
            epsilon2, ntu_errorB_old = 0, 0
            count2 += 1

        count = count1 + count2
        if count > 0:
            break

        epsilon = max(epsilon1, epsilon2)
        if epsilon < 1e-14: ### convergence condition
            break

    return MA, MB, ntu_errorB, optimal_cf

###############################################################################
########### given the environment and the seeds, optimization #################
########################## of MA and MB #######################################
###############################################################################

def truncate_and_optimize(g, RA, RB, Ds, fix_bd, truncation_mode):
    """ optimize truncated MA and MB tensors, using NTU metric. """
    max_iter = 1000 # max no of NTU optimization loops
    assert (g.fuse_legs(axes=((0, 2), (1, 3))) - g.fuse_legs(axes=((0, 2), (1, 3))).conj().transpose(axes=(1, 0))).norm() < 1e-14 * g.fuse_legs(axes=((0, 2), (1, 3))).norm()
    
    RAB = RA @ RB
    RAB = RAB.fuse_legs(axes=[(0, 1)])
    gf = g.fuse_legs(axes=(1, 3, (0, 2)))
    fgf = gf.fuse_legs(axes=((0, 1), 2))
    fgRAB = fgf @ RAB
    gRAB =  gf @ RAB
    gRR = vdot(RAB, fgRAB).item()
    
    MA, MB, svd_error = environment_aided_truncation_step(g, gRR, fgf, fgRAB, RA, RB, Ds, fix_bd, truncation_mode)
    MA, MB, ntu_errorB, optimal_cf  = ntu_single_optimization(MA, MB, gRAB, gf, gRR, svd_error, max_iter)
    MA, MB = truncation_step(MA, MB, Ds, normalize=True)
    return MA, MB, ntu_errorB, optimal_cf, svd_error

def optimal_pinv(gg, J, gRR):
    """ solve pinv(gg) * J, optimizing pseudoinverse cutoff for NTU error. """
    
    assert (gg - gg.conj().transpose(axes=(1, 0))).norm() < 1e-12 * gg.norm()
    S, U = eigh_with_truncation(gg, axes=(0, 1), tol=1e-14)
    UdJ = tensordot(J, U, axes=(0, 0), conj=(0, 1))
    cutoff_list = [10**n for n in range(-14, -5)]
    results = []
    for c_off in cutoff_list:
        Sd = S.reciprocal(cutoff=c_off)
        SdUdJ = Sd.broadcast(UdJ, axis=0)
        # Mnew = tensordot(SdUdJ, V, axes=(0, 0), conj=(0, 1))
        Mnew = U @ SdUdJ

        # calculation of errors with respect to metric
        met_newA = vdot(Mnew, gg @ Mnew).item()
        met_mixedA = vdot(Mnew, J).item()
        ntu_error = abs((met_newA + gRR - met_mixedA - met_mixedA.conjugate()) / gRR)
        results.append((ntu_error, c_off, Mnew))

    ntu_error, c_off, Mnew = min(results, key=lambda x: x[0])

    Mnew = Mnew.unfuse_legs(axes=0)
    return ntu_error, c_off, Mnew
