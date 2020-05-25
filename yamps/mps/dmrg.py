from yamps.mps import Env3
from yamps.tensor import eigs

#################################
#           dmrg                #
#################################


def dmrg_sweep_1site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, tol=1e-14):
    """ Assume psi is in left cannonical form (=Q-R-)
    """
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last'):
        psi.absorb_central(towards=psi.g.last)
        init = psi.A[n]
        val, vec, happy = eigs(Av=lambda v: env.Heff1(v, n), init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        psi.A[n] = vec[val.index(min(val))]
        psi.orthogonalize_site(n, towards=psi.g.last)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep(to='first'):
        psi.absorb_central(towards=psi.g.first)
        init = psi.A[n]
        val, vec, happy = eigs(Av=lambda v: env.Heff1(v, n), init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        psi.A[n] = vec[val.index(min(val))]
        psi.orthogonalize_site(n, towards=psi.g.first)
        env.update(n, towards=psi.g.first)

    return env  # can be used in the next sweep
