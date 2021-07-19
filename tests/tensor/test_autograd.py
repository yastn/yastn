import pytest
import yast
try:
    from .configs import config_dense, config_U1
except ImportError:
    from configs import config_dense, config_U1

tol = 1e-12  #pylint: disable=invalid-name


@pytest.mark.skipif(config_dense.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_trace_0():
    H= yast.Tensor(config=config_dense, s=(-1, -1, 1, 1))
    H.set_block(Ds=(2, 2, 2, 2), val='zeros')
    inds= [(0, 0, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)]
    vals= [0.25, -0.25, 0.5, 0.5, -0.25, 0.25]
    for t, val in zip(inds, vals):
        H.A[()][t] = val
    v = yast.Tensor(config=config_dense, s=(-1, -1))
    v.set_block(Ds=(2, 2), val='zeros')
    inds_v= [(0, 0), (0, 1), (1, 0), (1, 1)]
    vals_v= [0.1, 0.9,-0.9, 0.1]
    for t, val in zip(inds_v, vals_v):
        v.A[()][t] = val

    assert v.requires_grad is False
    v.requires_grad_()
    assert v.requires_grad is True

    Hv= yast.tensordot(H,v,([2,3],[0,1]))
    vHv= yast.tensordot(v.conj(),Hv,([0,1],[0,1]))
    vv= yast.vdot(v,v)
    vHv_vv= vHv/vv
    loss= vHv_vv.to_number()

    loss.backward()
    expected_grad= [[0.12046400951814398, -0.013384889946460365],
                    [0.013384889946460365, 0.12046400951814398]]
    g = yast.Tensor(config=config_dense, s=(-1, -1))
    g.set_block(Ds=(2, 2), val=expected_grad)

    for t in inds_v:
        assert pytest.approx( v.A[()].grad[t].item(), rel=tol ) == g.A[()][t]


@pytest.mark.skipif(config_U1.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_trace_1():
    H = yast.Tensor(config=config_U1, s=(-1, -1, 1, 1))
    ta= [(-1, -1, -1, -1), (-1, 1, -1, 1), (-1, 1, 1, -1), (1, -1, -1, 1), (1, -1, 1, -1), (1, 1, 1, 1)]
    ba= [0.25, -0.25, 0.5, 0.5, -0.25, 0.25]
    for t, b in zip(ta, ba):
        H.set_block(ts=t, Ds=(1, 1, 1, 1), val=b)
    v = yast.Tensor(config=config_U1, s=(-1, -1, 1))
    tv= [(-1, -1, -2), (-1, 1, 0), (1, -1, 0), (1, 1, 2)]
    bv= [0.1, 0.9,-0.9, 0.1]
    for t, b in zip(tv, bv):
        v.set_block(ts=t, Ds=(1, 1, 1), val=b)

    assert v.requires_grad is False
    v.requires_grad_()
    assert v.requires_grad is True

    Hv= yast.tensordot(H,v,([2, 3], [0, 1]))
    vHv= yast.tensordot(v.conj(),Hv,([0, 1, 2], [0, 1, 2]))
    vv= yast.vdot(v,v)
    vHv_vv= vHv/vv
    loss= vHv_vv.to_number()

    loss.backward()
    expected_grad= [0.12046400951814398, -0.013384889946460365, 0.013384889946460365,\
        0.12046400951814398]
    for t, g in zip(tv, expected_grad):
        assert pytest.approx(v.A[t].grad.item(), rel=tol) == g

if __name__ == '__main__':
    test_trace_0()
    test_trace_1()
