""" test yast.block """
import pytest
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_block_U1():
    """ test yast.block() to create Hamiltonian MPO U1-tensor. """
    w = 0.6  # hopping
    mu = -0.4  # chemical potential

    H0 = yast.Tensor(config=config_U1, s=(1, 1, -1, -1))  
    H0.set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
    H0.set_block(ts=(0, 1, 1, 0), val=[[1, 0], [mu, 1]], Ds=(2, 1, 1, 2))
    H0.set_block(ts=(0, 0, 1, -1), val=[0, w], Ds=(2, 1, 1, 1))
    H0.set_block(ts=(0, 1, 0, 1), val=[0, w], Ds=(2, 1, 1, 1))
    H0.set_block(ts=(-1, 1, 0, 0), val=[1, 0], Ds=(1, 1, 1, 2))
    H0.set_block(ts=(1, 0, 1, 0), val=[1, 0], Ds=(1, 1, 1, 2))
    # H0 is an MPO tensor for fermions in a chain with nearest-neighbor hopping w and chemical potential mu.

    II = yast.ones(config=config_U1, t=(0, (0, 1), (0, 1), 0), D=(1, (1, 1), (1, 1), 1), s=(1, 1, -1, -1))
    nn = yast.ones(config=config_U1, t=(0, 1, 1, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    c01 = yast.ones(config=config_U1, t=(0, 0, 1, -1), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    cp01 = yast.ones(config=config_U1, t=(0, 1, 0, 1), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    cp10 = yast.ones(config=config_U1, t=(1, 0, 1, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    c10 = yast.ones(config=config_U1, t=(-1, 1, 0, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))

    #
    # dict keys give relative coordinates of blocks in a new super-tensor
    # common_legs are excluded from provided coordinates
    H1 = yast.block({(0, 0): II, (1, 0): cp10, (2, 0): c10, (3, 0): mu * nn, (3, 1): w * c01, (3, 2): w * cp01, (3, 3): II}, common_legs=(1, 2))
    H2 = yast.block({(1, 1): II, (3, 1): cp10, (5, 1): c10, (7, 1): mu * nn, (7, 3): w * c01, (7, 5): w * cp01, (7, 9): II}, common_legs=(1, 2))
    H3 = yast.block({(0, 0, 0, 0): II, (1, 0, 0, 0): cp10, (2, 0, 0, 0): c10, (3, 0, 0, 0): mu * nn, (3, 0, 0, 1): w * c01, (3, 0, 0, 2): w * cp01, (3, 0, 0, 3): II})
    # those tree give the same output

    assert all(x.is_consistent() for x in (H0, H1, H2, H3))
    assert all(yast.norm(H0 - x) < tol for x in (H0, H1, H2, H3))  # == 0.0
    assert H1.get_shape() == (4, 2, 2, 4)

    

def test_block_exceptions():
    """ raising exceptions while using yast.block()"""
    II = yast.ones(config=config_U1, t=(0, (0, 1), (0, 1), 0), D=(1, (1, 1), (1, 1), 1), s=(1, 1, -1, -1))
    nn = yast.ones(config=config_U1, t=(0, 1, 1, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    # MPO tensor for chemical potential
    H = yast.block({(0, 0): II,  (1, 0): nn, (1, 1): II}, common_legs=(1, 2))

    with pytest.raises(yast.YastError):
        _ = yast.block({(0, ): II,  (1, 0): nn, (1, 1): II}, common_legs=(1, 2))
        # Wrong number of coordinates encoded in tensors.keys()
    with pytest.raises(yast.YastError):
        nnc = yast.ones(config=config_U1, t=(0, 1, 1, 0), D=(1, 1, 1, 1), s=(-1, -1, 1, 1))
        _ = yast.block({(0, 0): II,  (1, 0): nnc, (1, 1): II}, common_legs=(1, 2))
        # Signatues of blocked tensors are inconsistent.
    with pytest.raises(yast.YastError):
        nnn = yast.ones(config=config_U1, t=(0, 1, 1, 1), D=(1, 1, 1, 1), s=(1, 1, -1, -1), n=1)
        _ = yast.block({(0, 0): II,  (1, 0): nnn, (1, 1): II}, common_legs=(1, 2))
        # Tensor charges of blocked tensors are inconsistent.
    with pytest.raises(yast.YastError):
        e1 = yast.eye(config=config_U1, t=[(0, 1), (0, 1)], D=[(2, 2), (2, 2)])
        e2 = yast.eye(config=config_U1, t=[(0, 2), (0, 2)], D=[(2, 2), (2, 2)])
        _ = yast.block({(0, 0): e1, (1, 1): e2})
        # Block does not support diagonal tensors. Use .diag() first.
    with pytest.raises(yast.YastError):
        fII = II.fuse_legs(axes=(0, (1, 2, 3)), mode='meta')
        fnn = nn.fuse_legs(axes=(0, 1, (2, 3)), mode='meta')
        fnn = fnn.fuse_legs(axes=(0, (1, 2)), mode='meta')
        _ = yast.block({(0, 0): fII,  (1, 0): fnn, (1, 1): fII}, common_legs=())
        # Meta fusions of blocked tensors are inconsistent.
    with pytest.raises(yast.YastError):
        fII = II.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        fnn = nn.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        _ = yast.block({(0, 0): fII,  (1, 0): fnn, (1, 1): fII}, common_legs=())
        # Blocking of hard-fused legs is currently not supported. Go through meta-fusion. Only common_legs can be hard-fused.
    with pytest.raises(yast.YastError):
        nnn = yast.ones(config=config_U1, t=(0, 1, 1, 0), D=(1, 2, 1, 1), s=(1, 1, -1, -1))
        _ = yast.block({(0, 0): II,  (1, 0): nnn, (1, 1): II}, common_legs=(1, 2))
        # Dimensions of blocked tensors are not consistent

def test_block_fuse():
    leg1 = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(3, 1, 2))
    leg2 = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 1))
    leg3 = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(3, 4, 5))
    leg4 = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(4, 5, 6))

    # THIS WOULD GENERATE ERROR
    # BLOCKING DO NOT REMEMBER HISTORY OF BLOCKING WHICH CAN LEAD TO ERRORS
    # HAS TO BE USED WITH CAUCIOUS
    # leg1 = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(3, 1, 2))
    # leg2 = yast.Leg(config_U1, s=1, t=(0, 1, 2), D=(2, 3, 1))
    # leg3 = yast.Leg(config_U1, s=1, t=(0, 1, 2), D=(3, 4, 5))
    # leg4 = yast.Leg(config_U1, s=1, t=(-1, 0, 1), D=(4, 5, 6))

    legs_l = {1: [leg1, leg2.conj()], 2: [leg2, leg3.conj()]}
    legs_r = {1: [leg3, leg4.conj()], 2: [leg4, leg1.conj()]}

    l = {i: yast.rand(config=config_U1, legs=legs_l[i]) for i in (1, 2)}
    r = {j: yast.rand(config=config_U1, legs=legs_r[j]) for j in (1, 2)}
    c = {(i, j): yast.rand(config=config_U1, legs=legs_l[i] + legs_r[j]).conj() for i, j in ((1, 1), (1, 2), (2, 2))}
    s = {(i, j): yast.einsum('kl,klmn,mn->', l[i], c[i, j], r[j]) for i, j in ((1, 1), (1, 2), (2, 2))}

    mode = 'meta'
    fl = {i: l[i].fuse_legs(axes=[(0, 1)], mode=mode) for i in (1, 2)}
    fr = {j: r[j].fuse_legs(axes=[(0, 1)], mode=mode) for j in (1, 2)}
    fc = {(i, j): c[i, j].fuse_legs(axes=[(0, 1), (2, 3)], mode=mode) for i, j in ((1, 1), (1, 2), (2, 2))}

    bfl = yast.block({1: fl[1], 2: fl[2]})
    bfr = yast.block({1: fr[1], 2: fr[2]})
    bfc = yast.block({(1, 1): fc[1, 1], (2, 2): fc[2, 2], (1, 2): fc[1, 2]})

    # this explains blocking of meta-fused tensors
    bc = yast.block({(1, 1, 1, 1): c[1, 1], (2, 2, 2, 2): c[2, 2], (1, 1, 2, 2): c[1, 2]})
    fbc = bc.fuse_legs(axes=((0, 1), (2, 3)), mode=mode)
    assert yast.norm(bfc - fbc) < tol

    bfs = yast.ncon([bfl, bfc, bfr], [[1], [1, 2], [2]])
    assert yast.norm(bfs - s[1, 1] - s[2, 2] - s[1, 2]) < tol




if __name__ == "__main__":
    test_block_U1()
    test_block_exceptions()
    test_block_fuse()
