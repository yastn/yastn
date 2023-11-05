""" examples for addition of the Mps-s """
import pytest
import numpy as np
import yastn
import yastn.tn.mps as mps
try:
    from .configs import config_dense as cfg
    # pytest modifies cfg to inject different backends and devices during tests
except ImportError:
    from configs import config_dense as cfg

tol = 1e-12


def build_spin1_aklt_state(N=5, lvec=(1, 0), rvec=(0, 1)):
    """
    Initialize MPS tensors by hand. Example for Spin-1 AKLT state of N sites.

    Allow inputing boundary vectors for an MPS with open boundary conditions.
    """
    # Prepare rank-3 on-site tensor with virtual dimensions 2
    # and physical dimension dim(Spin-1)=3
    #                _
    # dim(left)=2 --|A|-- dim(right)=2
    #                |
    #           dim(Spin-1)=3
    #
    A = yastn.Tensor(config=cfg, s=(-1, 1, 1))
    s13, s23 = np.sqrt(1. / 3), np.sqrt(2. / 3)
    A.set_block(Ds=(2, 3, 2), val=[[[0, s23], [-s13, 0], [0, 0]],
                                   [[0, 0], [0, s13], [-s23, 0]]])
    #
    # as well as left and right boundar vectors
    #
    LV = yastn.Tensor(config=cfg, s=(-1, 1))
    LV.set_block(Ds=(1, 2), val=lvec)
    RV = yastn.Tensor(config=cfg, s=(-1, 1))
    RV.set_block(Ds=(2, 1), val=rvec)
    #
    # We initialize empty MPS for N sites, and assign its on-site tensors one-by-one.
    #
    psi = mps.Mps(N)
    for n in range(N):
        psi[n] = A.copy()
    #
    # Due to open boundary conditions, the first and the last
    # MPS tensors have left and right virtual indices projected to dimension 1.
    #
    psi[psi.first] = LV @ psi[psi.first]
    psi[psi.last] = psi[psi.last] @ RV
    #
    # The resulting MPS is not normalized, nor in any canonical form.
    #
    return psi


def test_measure_mps():
    """ Test measuring MPS expectation values in AKLT state"""
    #
    # AKLT state with open boundary conditions and N = 31 sites.
    #
    N = 31
    psi = build_spin1_aklt_state(N=N, lvec=(1, 0), rvec=(1, 0))
    #
    # we chack transfer matrix constructed in the middle of AKLT state
    #
    A = psi[N // 2]  # read MPS tensor
    T = yastn.ncon((A, A.conj()), ((-0, 1, -2), (-1, 1, -3)))
    T = T.fuse_legs(axes=((0, 1), (2, 3)))
    Tref = np.array([[1, 0, 0, 2], [0, -1, 0, 0], [0, 0, -1, 0], [2, 0, 0, 1]]) / 3
    assert np.allclose(T.to_numpy(), Tref, atol=tol)
    #
    # psi1 is not normalized, but for this system size, it is very close to sqrt(0.5)
    #
    norm = psi.norm()  # this command uses canonization to calculate norm
    norm2 = mps.vdot(psi, psi)  # and this directly contracts mps squared
    #
    assert pytest.approx(norm.item(), rel=tol) == np.sqrt(0.5)
    assert pytest.approx(norm2.item(), rel=tol) == 0.5
    #
    # To calculate expectation values, we initialize dense Spin1 operators
    #
    ops = yastn.operators.Spin1(sym='dense', backend=cfg.backend, default_device=cfg.default_device)
    #
    # measure Sz at the first and last sites
    #
    ez = mps.measure_1site(psi, {0: ops.sz(), N-1: ops.sz()}, psi)
    assert len(ez) == 2 and isinstance(ez, dict)
    assert pytest.approx(ez[0], rel=tol) == 1. / 3  # state psi1 is not normalized
    assert pytest.approx(ez[N-1], rel=tol) == -1. / 3
    #
    psin = psi / norm
    ez = mps.measure_1site(psin, ops.sz(), psin)
    assert len(ez) == N and isinstance(ez, dict) # calculate Sz on all sites
    assert pytest.approx(ez[0], rel=tol) == 2. / 3  # state psi1 is normalized
    #
    # calculate <Sz_i Sz_j> for all pairs i < j
    ezz = mps.measure_2site(psin, ops.sz(), ops.sz(), psin)
    assert len(ezz) == N * (N - 1) // 2 and isinstance(ez, dict)
    assert pytest.approx(ezz[N // 2, N // 2 + 1], rel=tol) == -4. / 9
    #
    #  measure string operator; exp (i pi * Sz) = I - 2 * abs(Sz)
    #
    esz = ops.I() - 2 * abs(ops.sz())
    Hterm = mps.Hterm(amplitude=1,
                      positions=(10, 11, 12, 13, 14, 15),
                      operators=(ops.sz(), esz, esz, esz, esz, ops.sz()))
    I = mps.product_mpo(ops.I(), N)
    Ostring = mps.generate_mpo(I, [Hterm])  # generate MPO for a correlator with string.
    Estring = mps.vdot(psin, Ostring, psin)
    assert pytest.approx(Estring.item(), rel=tol) == -4. / 9


def test_mps_spectrum():
    """ Test measuring Schmidt spectrum and entropy of mps. """
    N = 8
    for sym in ["dense", "Z3", "U1"]:
        # consider 3-dimensional local Hilbert space
        ops = yastn.operators.Spin1(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        #
        # take 3 orthonormal product states
        #
        psi1 = mps.product_mps(ops.vec_z(val=+0), N)
        psi2 = mps.product_mps([ops.vec_z(val=+1), ops.vec_z(val=-1)], N)
        psi3 = mps.product_mps([ops.vec_z(val=-1), ops.vec_z(val=+1)], N)
        #
        # and their sum; for Z2 and U1, it has a well-defined global charge only for even N
        #
        psi = mps.add(psi1, psi2, psi3, amplitudes=(0.5, 0.25, -0.25))
        #
        # state psi is not normalized
        #
        assert pytest.approx(psi.norm(), rel=tol) == np.sqrt(0.375)
        #
        # calculate Schmidt values; Schmidt values are properly normalized
        #
        schmidt_spectra = psi.get_Schmidt_values()
        assert len(schmidt_spectra) == N + 1 and isinstance(schmidt_spectra, list)
        assert all(abs(sv.norm() - 1) < tol for sv in schmidt_spectra)
        #
        bds = (1,) + (3,) * (N-1) + (1,)  # (1, 3, 3, 3, 3, 3, 3, 3, 1) for N = 8
        assert psi.get_bond_dimensions() == bds
        # schmidt values are stored as diagonal yastn.Tensors
        assert all(sv.isdiag and sv.get_shape() == (bd, bd) for sv, bd in zip(schmidt_spectra, bds))
        #
        # mps.get_Schmidt_values is used by mps.get_entropy
        #
        entropies = psi.get_entropy()
        assert len(entropies) == N + 1 and isinstance(entropies, list)
        #
        # by default calculate von Neuman entropy (base-2 log)
        #
        assert all(abs(ent - (np.log2(6) - 4/3)) < tol for ent in entropies[1:-1])
        assert abs(entropies[0]) < tol and abs(entropies[-1]) < tol
        #
        # Renyi entropy with alpha=2 (and base-2 log)
        #
        entropies2 = psi.get_entropy(alpha=2)
        assert all(abs(ent - 1.) < tol for ent in entropies2[1:-1])
        assert abs(entropies2[0]) < tol and abs(entropies2[-1]) < tol
        #
        # the state psi is not affected by calculations
        #
        assert pytest.approx(psi.norm(), rel=tol) == np.sqrt(0.375)
        #
        # similar thing can be done for MPO
        #


def test_mpo_spectrum():
    """ Test measuring Schmidt spectrum and entropy of mps. """
    N = 8
    for sym in ["dense", "Z3", "U1"]:
        # consider 3-dimensional local Hilbert space
        ops = yastn.operators.Spin1(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        #
        # take 3 orthogonal operator-product states
        #
        psi1 = mps.product_mpo(ops.sz(), N)
        psi2 = mps.product_mpo([ops.sm(), ops.sp()], N)
        psi3 = mps.product_mpo([ops.sp(), ops.sm()], N)
        #
        # they are not normalized
        assert abs(psi1.norm() - 2. ** (N / 2)) < tol
        assert abs(psi2.norm() - 2. ** N) < tol
        assert abs(psi3.norm() - 2. ** N) < tol
        #
        # their sum; for Z2 and U1, it has a well-defined global charge only for even N
        #
        psi = mps.add(psi1, psi2, psi3, amplitudes=(2. ** (N / 2), 1, 1))
        #
        # state psi is not normalized
        #
        assert pytest.approx(psi.norm(), rel=tol) == np.sqrt(3) * 2. ** N
        #
        # calculate Schmidt values; Schmidt values are properly normalized
        #
        schmidt_spectra = psi.get_Schmidt_values()
        assert len(schmidt_spectra) == N + 1 and isinstance(schmidt_spectra, list)
        assert all(abs(sv.norm() - 1) < tol for sv in schmidt_spectra)
        #
        bds = (1,) + (3,) * (N - 1) + (1,)  # (1, 3, 3, 3, 3, 3, 3, 3, 1) for N = 8
        assert psi.get_bond_dimensions() == bds
        # schmidt values are stored as diagonal yastn.Tensors
        assert all(sv.isdiag and sv.get_shape() == (bd, bd) for sv, bd in zip(schmidt_spectra, bds))
        #
        # mps.get_Schmidt_values is used by mps.get_entropy
        # Renyi entropy with alpha=0.5 (and base-2 log)
        # #
        entropies = psi.get_entropy(alpha=0.5)
        # here all Schmidt valus are the same
        assert all(abs(ent - np.log2(3)) < tol for ent in entropies[1:-1])
        assert abs(entropies[0]) < tol and abs(entropies[-1]) < tol


if __name__ == "__main__":
    test_measure_mps()
    test_mps_spectrum()
    test_mpo_spectrum()
