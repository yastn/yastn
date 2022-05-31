""" yast.trace() """
import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def trace_vs_numpy(a, axes):
    """ Compares yast.trace vs dense operations in numpy. Return traced yast tensor. """
    if isinstance(axes[0], int):
        axes = ((axes[0],), (axes[1],))
    out = tuple(i for i in range(a.ndim) if i not in axes[0] + axes[1])

    if not (a.isdiag or len(axes[0]) == 0):
        ma = a.fuse_legs(axes=axes+out, mode='meta')
        tDin = {0: ma.get_legs(1).conj(), 1: ma.get_legs(0).conj()}
        na = ma.to_numpy(tDin)  # to_numpy() with 2 matching axes to be traced
    else:
        na = a.to_numpy() # no trace is axes=((),())

    nat = np.trace(na) if len(axes[0]) > 0 else na
    c = yast.trace(a, axes)

    assert c.is_consistent()
    if len(axes[0]) > 0:
        assert a.are_independent(c)

    legs_out = {nn: a.get_legs(ii) for nn, ii in enumerate(out)}
    # trace might have removed some charges on remaining legs
    nc = c.to_numpy(legs=legs_out) # for comparison they have to be filled in.
    assert np.linalg.norm(nc - nat) < tol
    return c


def test_trace_basic():
    """ test trace for different symmetries. """
    # dense
    a = yast.ones(config=config_dense, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))
    b = trace_vs_numpy(a, axes=(0, 2))
    c = trace_vs_numpy(b, axes=(1, 0))
    assert pytest.approx(c.item(), rel=tol) == 10.

    a = yast.eye(config=config_dense, D=5)
    b = trace_vs_numpy(a, axes=((), ()))
    assert yast.norm(a - b) < tol
    c = trace_vs_numpy(a, axes=(0, 1))
    assert pytest.approx(c.item(), rel=tol) == 5.

    # U1
    t1, D1, D2, D3 = (0, 1), (2, 3), (4, 5), (6, 7)
    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                t=[t1, t1, t1, t1, t1, t1], D=[D1, D2, D3, D3, D2, D1])
    b = trace_vs_numpy(a, axes=(0, 5))
    b = trace_vs_numpy(b, axes=((), ())) # no trace
    b = trace_vs_numpy(b, axes=(3, 0))
    b = trace_vs_numpy(b, axes=(0, 1))
    assert pytest.approx(b.item(), rel=tol) == 5 * 9 * 13

    a = yast.eye(config=config_U1, t=(1, 2, 3), D=(3, 4, 5))
    b = trace_vs_numpy(a, axes=((), ())) # no trace
    b = trace_vs_numpy(a, axes=(0, 1))

    a = yast.ones(config=config_U1, s=(-1, -1, 1),
                  t=[(1,), (1,), (2,)], D=[(2,), (2,), (2,)])
    b = trace_vs_numpy(a, axes=(0, 2))
    assert b.norm() < tol  # == 0

    # Z2xU1
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    D1, D2 = (6, 4, 9, 6), (20, 16, 25, 20)
    a = yast.randC(config=config_Z2xU1, s=(-1, -1, 1, 1),
                t=[t1, t1, t1, t1], D=[D1, D2, D2, D1])
    b = trace_vs_numpy(a, axes=(0, 3))
    b = trace_vs_numpy(b, axes=(1, 0))
    c = a.trace(axes=((0, 1), (3, 2)))
    assert pytest.approx(c.item(), rel=tol) == b.item()

    a = yast.ones(config=config_Z2xU1, isdiag=True,
                  t=[((0, 0), (1, 1), (0, 2))], D=[(2, 2, 2)])
    b = trace_vs_numpy(a, axes=((), ())) # no trace
    assert yast.norm(a - b) < tol
    b = trace_vs_numpy(a, axes=(0, 1))
    assert pytest.approx(b.item(), rel=tol) == 6


def test_trace_fuse_meta():
    """ test trace of meta-fused tensor. """
    t1 = (-1, 1, 2)
    D1, D2 = (1, 2, 3), (4, 5, 6)
    a = yast.randR(config=config_U1, s=(-1, 1, 1, -1, 1, -1),
                  t=(t1, t1, t1, t1, t1, t1), D=(D1, D2, D1, D2, D1, D2))

    af = yast.fuse_legs(a, axes=((1, 2), (3, 0), (4, 5)), mode='meta')
    b = trace_vs_numpy(a, axes=((1, 2), (3, 0)))
    bf = trace_vs_numpy(af, axes=(0, 1)).unfuse_legs(axes=0)
    assert yast.norm(bf - b) < tol


def test_trace_fuse_hard():
    """ test trace of hard-fused tensor. """
    # to_numpy() does not handle fixing mismatches between hard-fused legs,
    # so here tests do not use function trave_vs_numpy
    t1 = (-1, 1, 2)
    D1, D2 = (1, 2, 3), (4, 5, 6)
    a = yast.randC(config=config_U1, s=(-1, 1, 1, -1, 1, -1),
                  t=(t1, t1, t1, t1, t1, t1), D=(D1, D2, D1, D2, D1, D2))

    af = yast.fuse_legs(a, axes=((1, 2), (3, 0), (4, 5)), mode='hard')
    b = trace_vs_numpy(a, axes=((1, 2), (3, 0)))
    bf = trace_vs_numpy(af, axes=(0, 1)).unfuse_legs(axes=0)
    assert yast.norm(bf - b) < tol

    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1, 1),  # TODO
                  t=((-1, 1), (-1, 2), (-1, 1), (1, 2), (0, 1)),  # change last to (1, 2) to trigger exception in merge -- fix
                  D=((1, 2), (4, 6), (1, 2), (5, 6), (3, 4)))
    af = yast.fuse_legs(a, axes=((1, 2), (3, 0), 4), mode='hard')
    tra = yast.trace(a, axes=((1, 2), (3, 0)))
    traf = yast.trace(af, axes=(0, 1))
    assert yast.norm(tra - traf) < tol

    a = yast.Tensor(config=config_U1, s=(1, -1, 1, 1, -1, -1))
    a.set_block(ts=(1, 2, 0, 2, 1, 0), Ds=(2, 3, 4, 3, 2, 4), val='rand')
    a.set_block(ts=(2, 1, 1, 1, 2, 1), Ds=(6, 5, 4, 5, 6, 4), val='rand')
    a.set_block(ts=(3, 2, 1, 2, 2, 2), Ds=(1, 3, 4, 3, 6, 2), val='rand')

    af = yast.fuse_legs(a, axes=((0, 1), 2, (4, 3), 5), mode='hard')
    tra = yast.trace(a, axes=((0, 1), (4, 3)))
    traf = yast.trace(af, axes=(0, 2))
    assert yast.norm(tra - traf) < tol

    aff = yast.fuse_legs(af, axes=((0, 1), (2, 3)), mode='hard')
    tra = yast.trace(a, axes=((0, 1, 2), (4, 3, 5)))
    traff = yast.trace(aff, axes=(0, 1))

    a = yast.Tensor(config=config_U1, s=(1, -1, 1, -1, 1, -1, 1, -1))
    a.set_block(ts=(1, 1, 2, 2, 0, 0, 1, 1), Ds=(1, 1, 2, 2, 4, 4, 1, 1), val='rand')
    a.set_block(ts=(2, 1, 1, 2, 0, 0, 1, 1), Ds=(2, 1, 1, 2, 4, 4, 1, 1), val='rand')
    a.set_block(ts=(3, 1, 1, 2, 1, 1, 0, 1), Ds=(3, 1, 1, 2, 1, 1, 4, 1), val='rand')
    af = yast.fuse_legs(a, axes=((0, 2), (1, 3), (4, 6), (5, 7)), mode='hard')
    aff = yast.fuse_legs(af, axes=((0, 2), (1, 3)), mode='hard')
    tra = yast.trace(a, axes=((0, 2, 4, 6), (1, 3, 5, 7)))
    traf = yast.trace(af, axes=((0, 2), (1, 3)))
    traff = yast.trace(aff, axes=(0, 1))
    assert yast.norm(tra - traf) < tol
    assert yast.norm(tra - traff) < tol

    b = yast.trace(a, axes=((0, 2), (1, 3)))
    bf = yast.trace(af, axes=(0, 1))
    bf = bf.unfuse_legs(axes=(0, 1)).transpose(axes=(0, 2, 1, 3))
    assert yast.norm(b - bf) < tol


def test_trace_exceptions():
    """ test trigerring some expections """
    t1, D1, D2 = (0, 1), (2, 3), (4, 5)
    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                t=[t1, t1, t1, t1, t1, t1], D=[D1, D2, D2, D2, D2, D2])
    with pytest.raises(yast.YastError):
        a.trace(axes=(0, 6))  # Error in trace: axis outside of tensor ndim
    with pytest.raises(yast.YastError):
        a.trace(axes=((0, 1, 2), (2, 3, 4)))  # Error in trace: repeated axis in axes
    with pytest.raises(yast.YastError):
        a.trace(axes=((0, 1, 2), (2, 3, 3)))  # Error in trace: repeated axis in axes
    with pytest.raises(yast.YastError):
        a.trace(axes=((0, 1, 2), (3, 4)))  # Error in trace: unmatching number of axes to trace.
    with pytest.raises(yast.YastError):
        a.trace(axes=((1, 3), (2, 4)))  # Error in trace: signatures do not match.
    with pytest.raises(yast.YastError):
        a.trace(axes=((0, 1, 2), (3, 4, 5)))  # Error in trace: bond dimensions of traced legs do not match.
    b = a.fuse_legs(axes=((0, 1, 2), (3, 4), 5), mode='meta')
    with pytest.raises(yast.YastError):
        b.trace(axes=(0, 1))  # Error in trace: meta-fusions of traced axes do not match.
    b = a.fuse_legs(axes=(0, (1, 3), (2, 4), 5), mode='meta')
    with pytest.raises(yast.YastError):
        b.trace(axes=(1, 2))  # Error in trace: signatures do not match.
    b = a.fuse_legs(axes=((0, 1, 2), (3, 4), 5), mode='hard')
    with pytest.raises(yast.YastError):
        b.trace(axes=(0, 1))  # Error in trace: hard fusions of legs 0 and 1 are not compatible.
    b = a.fuse_legs(axes=(0, (1, 3), (4, 5), 2), mode='hard')
    with pytest.raises(yast.YastError):
        b.trace(axes=(1, 2))  # Error in trace: hard fusions of legs 1 and 2 are not compatible.


if __name__ == '__main__':
    test_trace_basic()
    test_trace_fuse_meta()
    test_trace_fuse_hard()
    test_trace_exceptions()
