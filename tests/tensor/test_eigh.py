""" yast.linalg.eigh() """
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-10  #pylint: disable=invalid-name


def eigh_combine(a):
    """ decompose and contracts hermitian tensor using eigh decomposition """
    a2 = yast.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))
    S, U = yast.linalg.eigh(a2, axes=((0, 1), (2, 3)))
    US = yast.tensordot(U, S, axes=(2, 0))
    USU = yast.tensordot(US, U, axes=(2, 2), conj=(0, 1))
    assert yast.norm(a2 - USU) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()

    # changes signature of new leg; and position of new leg
    S, U = yast.eigh(a2, axes=((0, 1), (2, 3)), Uaxis=0, sU=-1)
    US = yast.tensordot(S, U, axes=(0, 0))
    USU = yast.tensordot(US, U, axes=(0, 0), conj=(0, 1))
    assert yast.norm(a2 - USU) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()


def test_eigh_basic():
    """ test eigh decomposition for various symmetries """
    # dense
    a = yast.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    eigh_combine(a)

    # U1
    a = yast.rand(config=config_U1, s=(-1, -1, 1, 1), n=1,
                  t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                  D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    eigh_combine(a)

    # Z2xU1
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.ones(config=config_Z2xU1, s=(-1, -1, 1, 1),
                  t=[t1, t1, t1, t1],
                  D=[(2, 3, 4, 5), (5, 4, 3, 2), (3, 4, 5, 6), (1, 2, 3, 4)])
    eigh_combine(a)


if __name__ == '__main__':
    test_eigh_basic()
