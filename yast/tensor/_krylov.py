""" Building Krylov space. """
from ._contractions import vdot


def _expand_krylov_space(f, tol, ncv, hermitian, V, H=None, info=None):
    if H is None:
        H = {}
    happy = False
    for j in range(len(V)-1, ncv):
        w = f(V[-1])
        if info is not None:
            info['krylov_steps'] += 1
        if not hermitian:  # Arnoldi
            for i in range(j + 1):
                H[(i, j)] = vdot(V[i], w)
                w = w.apxb(V[i], x=-H[(i, j)])
        else:  # Lanczos
            if j > 0:
                H[(j - 1, j)] = H[(j, j - 1)]
                w = w.apxb(V[j - 1], x=-H[(j - 1, j)])
            H[(j, j)] = vdot(V[j], w)
            w = w.apxb(V[j], x=-H[(j, j)])
        H[(j + 1, j)] = w.norm()
        if H[(j + 1, j)] < tol:
            happy = True
            H.pop((j + 1, j))
            break
        V.append(w / H[(j + 1, j)])
    return V, H, happy
