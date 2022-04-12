import numpy as np
import pytest
import yast
try:
    from .configs import config_U1xU1_fermionic
except ImportError:
    from configs import config_U1xU1_fermionic
tol = 1e-12  #pylint: disable=invalid-name


def tensordot_vs_numpy(a, b, axes, conj, policy=None):
    outa = tuple(ii for ii in range(a.ndim) if ii not in axes[0])
    outb = tuple(ii for ii in range(b.ndim) if ii not in axes[1])
    tDs = {nn: a.get_leg_structure(ii) for nn, ii in enumerate(outa)}
    tDs.update({nn + len(outa): b.get_leg_structure(ii) for nn, ii in enumerate(outb)})
    tDsa = {ia: b.get_leg_structure(ib) for ia, ib in zip(*axes)}
    tDsb = {ib: a.get_leg_structure(ia) for ia, ib in zip(*axes)}
    na = a.to_numpy(tDsa)
    nb = b.to_numpy(tDsb)
    if conj[0]:
        na = na.conj()
    if conj[1]:
        nb = nb.conj()
    nab = np.tensordot(na, nb, axes)

    c = yast.tensordot(a, b, axes, conj, policy=policy)

    nc = c.to_numpy(tDs)
    assert c.is_consistent()
    assert a.are_independent(c)
    assert c.are_independent(b)
    assert np.linalg.norm(nc - nab) < tol  # == 0.0
    return c


def test_dot():
    """ test tensordot for different symmetries. """
   

    # U1xU1
    t1 = [(0, -1), (0, 1), (0, 0), (1, -1), (1, 1), (1, 0), (-1, -1), (-1, 1), (-1, 0)]
    D1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    a = yast.randC(config=config_U1xU1_fermionic, s=(1, 1),
                  t=(t1, t1),
                  D=(D1, D1))
    b = yast.randC(config=config_U1xU1_fermionic, s=(-1, 1),
                  t=(t1, t1),
                  D=(D1, D1))

    X = a.tensordot(b.transpose((1,0)), (1,0), conj=(0,1))
    Y = a.tensordot(b, (1,1), conj=(0,1))
    print((X-Y).norm())
    print(a.dtype)

if __name__ == '__main__':
    test_dot()