""" examples for addition of the Mps-s """
import pytest
import numpy as np
import yastn
import yastn.tn.mps as mps
try:
    from .configs import config_dense as cfg
except ImportError:
    from configs import config_dense as cfg
# pytest modifies cfg to inject different backends and devices during tests


def build_aklt_state_manually(N=5, lvec=(1, 0), rvec=(0, 1), config=None):
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
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    config = yastn.make_config(**opts_config)
    A = yastn.Tensor(config, s=(-1, 1, 1))
    s13, s23 = np.sqrt(1. / 3), np.sqrt(2. / 3)
    A.set_block(Ds=(2, 3, 2), val=[[[0, s23], [-s13, 0], [0, 0]],
                                   [[0, 0], [0, s13], [-s23, 0]]])
    #
    # as well as left and right boundar vectors
    #
    LV = yastn.Tensor(config, s=(-1, 1))
    LV.set_block(Ds=(1, 2), val=lvec)
    RV = yastn.Tensor(config, s=(-1, 1))
    RV.set_block(Ds=(2, 1), val=rvec)
    #
    # We initialize empty MPS for N sites
    # and assign its on-site tensors one-by-one.
    #
    psi = mps.Mps(N)
    for n in range(N):
        psi[n] = A.copy()
    #
    # Due to open boundary conditions, the first and the last
    # MPS tensors have left and right virtual indices projected to dimension 1
    #
    psi[psi.first] = LV @ psi[psi.first]
    psi[psi.last] = psi[psi.last] @ RV
    #
    # The resulting MPS is not normalized, nor in any canonical form.
    #
    return psi


@pytest.mark.parametrize('kwargs', [{'config': cfg}])
def test_measure_mps_aklt(kwargs):
    measure_mps_aklt(**kwargs, tol=1e-12)

def measure_mps_aklt(config=None, tol=1e-12):
    """ Test measuring MPS expectation values in AKLT state"""
    #
    # AKLT state with open boundary conditions and N = 31 sites.
    #
    N = 31
    psi = build_aklt_state_manually(N=N, lvec=(1, 0), rvec=(1, 0), config=config)
    #
    # We verify transfer matrix in the middle of AKLT state
    # to specify the effect of lvec and rvec
    #
    A = psi[N // 2]  # read MPS tensor
    T = yastn.ncon((A, A.conj()), ((-0, 1, -2), (-1, 1, -3)))
    T = T.fuse_legs(axes=((0, 1), (2, 3)))
    Tref = np.array([[1,  0,  0,  2],
                     [0, -1,  0,  0],
                     [0,  0, -1,  0],
                     [2,  0,  0,  1]])
    assert np.allclose(T.to_numpy(), Tref / 3, atol=tol)
    #
    # psi1 is not normalized.
    # Norm is very close to sqrt(0.5) for this system size.
    #
    norm = psi.norm()  # use canonization to calculate norm
    norm2 = mps.vdot(psi, psi)  # directly contract mps squared
    #
    assert abs(norm - np.sqrt(0.5)) < tol
    assert abs(norm2 - 0.5) < tol
    #
    # Initialize dense Spin1 operators for expectation values calculations
    #
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.Spin1(sym='dense', **opts_config)
    #
    # Measure Sz at the first and last sites
    #
    ez = mps.measure_1site(psi, {0: ops.sz(), N-1: ops.sz()}, psi)
    assert len(ez) == 2 and isinstance(ez, dict)
    assert abs(ez[0] - 1./3) < tol  # psi is not normalized
    assert abs(ez[N - 1] + 1./3) < tol
    #
    psin = psi / norm
    ez = mps.measure_1site(psin, ops.sz(), psin)  # calculate sz on all sites
    assert len(ez) == N and isinstance(ez, dict)
    assert abs(ez[0] - 2. / 3) < tol  # psin is normalized
    #
    # Calculate <Sz_i Sz_j> for all pairs i < j
    ezz = mps.measure_2site(psin, ops.sz(), ops.sz(), psin)
    assert len(ezz) == N * (N - 1) // 2 and isinstance(ez, dict)
    assert abs(ezz[N // 2, N // 2 + 1] + 4. / 9) < tol
    #
    #  Measure string operator; exp (i pi * Sz) = I - 2 * abs(Sz)
    #
    esz = ops.I() - 2 * abs(ops.sz())
    Hterm = mps.Hterm(amplitude=1,
                      positions=(10, 11, 12, 13, 14, 15),
                      operators=(ops.sz(), esz, esz, esz, esz, ops.sz()))
    I = mps.product_mpo(ops.I(), N)
    Ostring = mps.generate_mpo(I, [Hterm])  # MPO a string correlator
    Estring = mps.vdot(psin, Ostring, psin)
    assert abs(Estring + 4. / 9) < tol


@pytest.mark.parametrize('kwargs', [{'sym': 'dense', 'config': cfg},
                                  {'sym': 'Z3', 'config': cfg},
                                  {'sym': 'U1', 'config': cfg}])
def test_mps_spectrum_ghz(kwargs):
    mps_spectrum_ghz(**kwargs)

def mps_spectrum_ghz(sym='dense', config=None, tol=1e-12):
    """ Test measuring Schmidt spectrum and entropy of mps. """
    N = 8
    # consider 3-dimensional local Hilbert space
    opts_config = {} if config is None else \
                    {'backend': config.backend,
                    'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.Spin1(sym=sym, **opts_config)
    #
    # take 3 orthonormal product states
    #
    psi1 = mps.product_mps(ops.vec_z(val=+0), N)
    psi2 = mps.product_mps([ops.vec_z(val=+1), ops.vec_z(val=-1)], N)
    psi3 = mps.product_mps([ops.vec_z(val=-1), ops.vec_z(val=+1)], N)
    #
    # For Z3 and U1, their sum has a well-defined
    # global charge only for even N
    #
    psi = mps.add(psi1, psi2, psi3, amplitudes=(0.5, 0.25, -0.25))
    #
    # state psi is not normalized
    #
    assert abs(psi.norm() - np.sqrt(0.375)) < tol
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
    assert all(sv.isdiag and sv.get_shape() == (bd, bd)
               for sv, bd in zip(schmidt_spectra, bds))
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
    assert abs(psi.norm() - np.sqrt(0.375)) < tol
    #
    # similar thing can be done for MPO


@pytest.mark.parametrize("sym, config", [('dense', cfg), ('Z3', cfg), ('U1', cfg)])
def test_mpo_spectrum(sym, config, tol=1e-12):
    """ Test measuring Schmidt spectrum and entropy of mps. """
    N = 8
    opts_config = {} if config is None else \
                    {'backend': config.backend,
                    'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.Spin1(sym=sym, **opts_config)
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
    # For Z2 and U1, their sum has a well-defined
    # global charge only for even N
    #
    psi = mps.add(psi1, psi2, psi3, amplitudes=(2. ** (N / 2), 1, 1))
    #
    # state psi is not normalized
    #
    assert abs(psi.norm() - np.sqrt(3) * 2. ** N) < tol * 2. ** N
    #
    # calculate Schmidt values; Schmidt values are properly normalized
    #
    schmidt_spectra = psi.get_Schmidt_values()
    assert len(schmidt_spectra) == N + 1 and isinstance(schmidt_spectra, list)
    assert all(abs(sv.norm() - 1) < tol for sv in schmidt_spectra)
    #
    bds = (1,) + (3,) * (N - 1) + (1,)  # (1, 3, 3, 3, 3, 3, 3, 3, 1) for N = 8
    assert psi.get_bond_dimensions() == bds
    #
    # schmidt values are stored as diagonal yastn.Tensors
    #s
    assert all(sv.isdiag and sv.get_shape() == (bd, bd) for sv, bd in zip(schmidt_spectra, bds))
    #
    # mps.get_Schmidt_values is used by mps.get_entropy
    # Renyi entropy with alpha=0.5 (and base-2 log)
    #
    entropies = psi.get_entropy(alpha=0.5)
    #
    # here all Schmidt values are the same
    #
    assert all(abs(ent - np.log2(3)) < tol for ent in entropies[1:-1])
    assert abs(entropies[0]) < tol and abs(entropies[-1]) < tol


def test_measurment_raise(config=cfg):
    opts_config = {} if config is None else \
                    {'backend': config.backend,
                    'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.Spin1(sym='dense', **opts_config)
    #
    # take 3 orthogonal operator-product states
    #
    H7 = mps.product_mpo(ops.sz(), N=7)
    psi7 = mps.product_mps(ops.vec_z(), N=7)
    psi8 = mps.product_mps(ops.vec_z(), N=8)

    with pytest.raises(yastn.YastnError):
        mps.vdot(psi7, psi8)
        # MpsMpo for bra and ket should have the same number of sites.
    with pytest.raises(yastn.YastnError):
        mps.vdot(psi7, H7)
        # MpsMpo for bra and ket should have the same number of physical legs.


# def correlation_matrix(psi, ops):
#     """ Calculate correlation matrix for Mps psi  C[m,n] = <c_n^dag c_m>"""
#     assert pytest.approx(psi.norm().item()) == 1
#     N = psi.N
#     # first approach: directly act with c operators on state psi
#     I = mps.product_mpo(ops.I(), N)
#     cns = [mps.generate_mpo(I, [mps.Hterm(1, [n], [ops.c()])]) for n in range(N)]
#     ps = [cn @ psi for cn in cns]
#     C = np.zeros((N, N), dtype=np.complex128)
#     for m in range(N):
#         for n in range(N):
#             C[m, n] = mps.vdot(ps[n], ps[m])

#     # second approach: use measure_1site() and measure_2site()
#     occs = mps.measure_1site(psi, ops.n(), psi)
#     cpc = mps.measure_2site(psi, ops.cp(), ops.c(), psi)
#     C2 = np.zeros((N, N), dtype=np.complex128)
#     for n, v in occs.items():
#         C2[n, n] = v
#     for (n1, n2), v in cpc.items():
#         C2[n2, n1] = v
#         C2[n1, n2] = v.conj()
#     assert np.allclose(C, C2)
#     return C


if __name__ == "__main__":
    measure_mps_aklt()
    test_measurment_raise()
    for sym in ['dense', 'Z3', 'U1']:
        mps_spectrum_ghz(sym=sym)
        test_mpo_spectrum(sym=sym, config=cfg)
