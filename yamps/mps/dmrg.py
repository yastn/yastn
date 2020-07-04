from yamps.mps import Env3
from yamps.tensor import eigs

#################################
#           dmrg                #
#################################


def dmrg_sweep_1site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, tol=1e-14):
    """
    Assume psi is in the left cannonical form.
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
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep(to='first'):
        psi.absorb_central(towards=psi.g.first)
        init = psi.A[n]
        val, vec, happy = eigs(Av=lambda v: env.Heff1(v, n), init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        psi.A[n] = vec[val.index(min(val))]
        psi.orthogonalize_site(n, towards=psi.g.first)
        env.clear_site(n)
        env.update(n, towards=psi.g.first)

    return env  # can be used in the next sweep


def dmrg_sweep_2site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, tol=1e-14, opts_svd={}):
    """
    Assume psi is in the left cannonical form.
    """
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        val, vec, happy = eigs(Av=lambda v: env.Heff2(v, n), init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        out = vec[val.index(min(val))]
        A1, S, A2 = out.split_svd(axes=((0, 1), (2, 3)), sU=-1, **opts_svd)
        psi.A[n] = A1
        psi.A[n1] = A2.dot_diag(S, axis=0)
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep(to='first', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        val, vec, happy = eigs(Av=lambda v: env.Heff2(v, n), init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        out = vec[val.index(min(val))]
        A1, S, A2 = out.split_svd(axes=((0, 1), (2, 3)), sU=-1, **opts_svd)
        psi.A[n] = A1.dot_diag(S, axis=2)
        psi.A[n1] = A2
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n1, towards=psi.g.first)
    env.update(n, towards=psi.g.first)

    return env  # can be used in the next sweep


def dmrg_sweep_2site_group(psi, H, env=None, dtype='complex128', hermitian=True, k=4, tol=1e-14, opts_svd={}):
    """
    Assume psi is in the left cannonical form.
    """
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
        val, vec, happy = eigs(Av=lambda v: env.Heff2_group(v, n), init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        out = vec[val.index(min(val))]
        out = out.ungroup_leg(axis=1, leg_order=leg_order)
        A1, S, A2 = out.split_svd(axes=((0, 1), (2, 3)), sU=-1, **opts_svd)
        psi.A[n] = A1
        psi.A[n1] = A2.dot_diag(S, axis=0)
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep(to='first', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
        val, vec, happy = eigs(Av=lambda v: env.Heff2_group(v, n), init=[init], tol=tol, k=k, hermitian=True, dtype=dtype)
        val = list(val)
        out = vec[val.index(min(val))]
        out = out.ungroup_leg(axis=1, leg_order=leg_order)
        A1, S, A2 = out.split_svd(axes=((0, 1), (2, 3)), sU=-1, **opts_svd)
        psi.A[n] = A1.dot_diag(S, axis=2)
        psi.A[n1] = A2
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n1, towards=psi.g.first)
    env.update(n, towards=psi.g.first)

    return env  # can be used in the next sweep
