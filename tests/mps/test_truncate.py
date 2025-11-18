# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Truncation of Mps. """
import pytest
import yastn
import yastn.tn.mps as mps


def test_truncate(config_kwargs, tol=1e-12):
    """ Test mps.truncate_ on random input states. """

    # initialize random MPO
    N = 14
    Di = 35  # initial D
    Df, svd_tol = 15, 6e-2  # truncation parameters
    for sym, n, seed in [("dense", (), 0), ("Z2", (0,), 1), ("Z2", (1,), 2)]:
        ops = yastn.operators.Spin12(sym=sym, **config_kwargs)
        I = mps.product_mpo(ops.I(), N)
        ops.random_seed(seed=seed)
        #
        #  truncate MPO keeping the norm;
        #
        psi = 2 * mps.random_mpo(I, D_total=Di)  # extra factor
        assert psi.get_bond_dimensions() == (1, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
        psipsi = mps.vdot(psi, psi)
        #
        # canonize psi to the last site; retains the norm in psi.factor for the record
        #
        psi.canonize_(to='last', normalize=False)
        # assert psi.get_bond_dimensions() == (1, 4, 16, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
        assert abs(mps.vdot(psi, psi) / psipsi - 1) < tol
        assert psi.is_canonical(to='last', tol=tol)
        #
        # truncate phi while canonizing to the first site;
        # keep the norm of the original state psi in phi.factor
        #
        phi = psi.shallow_copy()
        discarded = phi.truncate_(to='first', opts_svd={'D_total': Df}, normalize=False)
        assert phi.get_bond_dimensions() == (1, 4, Df, Df, Df, Df, Df, Df, Df, Df, Df, Df, Df, 4, 1)
        #
        phiphi = mps.vdot(phi, phi)
        phipsi = mps.vdot(phi, psi)
        #
        assert abs(phiphi / phipsi - 1) < tol
        assert abs(phipsi / psipsi + discarded ** 2 - 1) < tol
        #
        # norm of phi is phi.factor; individual tensors are properly canonized
        assert phi.is_canonical(to='first', tol=tol)

        #
        # truncate MPS forgetting the norm;
        #
        psi = 0.5j * mps.random_mps(I, D_total=Di, n=n)  #extra factor
        assert psi.get_bond_dimensions() == (1, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, Di, 1)
        psipsi = mps.vdot(psi, psi)
        #
        # canonize psi to the first site; keep track of the norm
        #
        psi.canonize_(to='first', normalize=False)
        assert abs(mps.vdot(psi, psi) / psipsi - 1) < tol
        assert psi.is_canonical(to='first', tol=tol)
        #
        #  truncate phi while canonizing to the last site; truncate based on tolerance
        #
        phi = psi.shallow_copy()
        discarded = phi.truncate_(to='last', opts_svd={'tol': svd_tol})  # normalize=True
        #
        svals = phi.get_Schmidt_values()
        for sval in svals:
            assert min(sval._data) / max(sval.data) > svd_tol * 0.8  # 0.8 as local truncations affect each other
        #
        phiphi = mps.vdot(phi, phi)
        phipsi = mps.vdot(phi, psi)
        assert abs(mps.vdot(phi, phi) - 1) < tol
        assert abs(phipsi ** 2 / psipsi + discarded ** 2 - 1) < tol


def test_zipper(config_kwargs, tol=1e-12):
    """ Test mps.zipper on random input states. """
    N = 13
    for sym, n, seed in [("dense", (), 0), ("Z2", (0,), 1), ("Z2", (1,), 2)]:
        #
        #  zipper for MPO @ MPO
        #
        Dai, Dbi = 13, 8  # initial D
        ops = yastn.operators.Spin12(sym=sym, **config_kwargs)
        I = mps.product_mpo(ops.I(), N)
        ops.random_seed(seed=seed)
        a = mps.random_mpo(I, D_total=Dai)
        b = mps.random_mpo(I, D_total=Dbi)
        ab = a @ b
        vabab = mps.vdot(ab, ab)
        #
        Df = 31  # final D
        c, discarded = mps.zipper(a, b, opts_svd={'D_total': Df}, normalize=False, return_discarded=True)
        #
        vcc = mps.vdot(c, c)
        vcab = mps.vdot(c, ab)
        assert c.is_canonical(to='first', tol=tol)
        assert abs(vcab / vcc - 1) < tol
        assert abs(vcab / vabab + discarded**2 - 1)  <  0.1  # discarded is only an estimate
        assert c.get_bond_dimensions() == (1, Df, Df, Df, Df, Df, Df, Df, Df, Df, Df, 16, 4, 1)

        #
        #  zipper for MPO @ MPS; forgetting the norm
        #
        Dai, Dbi = 17, 11  # initial D
        a = mps.random_mpo(I, D_total=Dai).canonize_(to='last', normalize=False)
        b = mps.random_mps(I, D_total=Dbi, n=n).canonize_(to='first', normalize=False)
        # some random canonization to move norm to .factor
        ab = a @ b
        vabab = mps.vdot(ab, ab)
        #
        c, discarded = mps.zipper(a, b, opts_svd={'tol': 1e-1}, return_discarded=True)  # normalize=True,
        #
        assert c.is_canonical(to='first', tol=tol)
        assert abs(mps.vdot(c, c) - 1) < tol
        #
        vcab = mps.vdot(c, ab)
        assert abs(vcab ** 2 / vabab + discarded**2 - 1)  <  0.1  # discarded is only an estimate


def test_compression(config_kwargs, tol=1e-12):
    """ Test mps.compression on random input states. """
    N = 13
    for sym, n, seed in [("dense", (), 0), ("Z2", (0,), 1), ("Z2", (1,), 2)]:
        #
        #  compression for MPO @ MPO; initialize with zipper
        #
        Dai, Dbi = 13, 8  # initial D
        ops = yastn.operators.Spin12(sym=sym, **config_kwargs)
        I = mps.product_mpo(ops.I(), N)
        ops.random_seed(seed=seed)
        a = 2.0 * mps.random_mpo(I, D_total=Dai)  # add extra factors
        b = 0.5 * mps.random_mpo(I, D_total=Dbi)
        ab = a @ b
        vabab = mps.vdot(ab, ab)
        #
        Df = 29  # final D
        c = mps.zipper(a, b, opts_svd={'D_total': Df})
        #
        #  optimize overlap
        #
        out = mps.compression_(c, (a, b), method='1site', overlap_tol=1e-4, Schmidt_tol=1e-4,
                         max_sweeps=100, normalize=False)
        #
        vcc = mps.vdot(c, c)
        vcab = mps.vdot(c, ab)
        assert c.is_canonical(to='first', tol=tol)
        assert abs(vcab / vcc - 1) < tol
        assert c.get_bond_dimensions() == (1, 4, 16, Df, Df, Df, Df, Df, Df, Df, Df, 16, 4, 1)
        # converged within desired tolerance
        assert out.doverlap < 1e-4 and out.max_dSchmidt < 1e-4 and out.sweeps < 100
        # absolut convergence is still far from perfect for this Df.
        assert 0.4 < vcab / vabab < 1

        #
        # compression for MPO @ MPS; initialize with zipper
        #
        Dai, Dbi = 17, 11  # initial D
        a = 3.0 * mps.random_mpo(I, D_total=Dai).canonize_(to='last', normalize=False)
        b = mps.random_mps(I, D_total=Dbi, n=n).canonize_(to='first', normalize=False)
        # some random canonization to move norm to .factor
        ab = a @ b
        vabab = mps.vdot(ab, ab)
        #
        c = mps.zipper(a, b, opts_svd={'D_total': 13})  # normalize=True
        #
        #  optimize overlap, testing iterator
        #
        overlap_old, doverlap_old = 0, 1
        for out in mps.compression_(c, (a, b),
                                    method='2site', opts_svd={'D_total': 13},
                                    max_sweeps = 2, iterator=True):  # normalize=True
            assert doverlap_old > out.doverlap and overlap_old < out.overlap
            overlap_old, doverlap_old = out.overlap, out.doverlap
            assert out.max_discarded_weight > 0
        #
        assert c.is_canonical(to='first', tol=tol)
        assert abs(mps.vdot(c, c) - 1) < tol

        #
        # compression for MPS; initialize with truncate_
        #
        Dai = 57
        a = mps.random_mps(I, D_total=Dai, n=n).canonize_(to='last', normalize=False)
        c = a.shallow_copy()
        c.truncate_(to='first', opts_svd={'D_total': 13}, normalize=False)
        #
        #  optimize overlap
        #
        out = mps.compression_(c, (a,), method='2site', opts_svd={'D_total': 13},
                               Schmidt_tol=1e-3, max_sweeps = 20, normalize=False)
        #
        assert out.max_dSchmidt < 1e-3  # is converged based on Schmidt values
        vaa, vca, vcc = mps.vdot(a, a), mps.vdot(c, a), mps.vdot(c, c)
        assert c.is_canonical(to='first', tol=tol)
        assert abs(vca / vcc - 1) < tol
        assert 0.6 < vca / vaa  < 1


@pytest.mark.skipif( "not config.getoption('long_tests')", reason="long duration tests are skipped" )
def test_compression_sum(config_kwargs, tol=1e-6):
    """ Test various combinations of targets and environments. """
    ops = yastn.operators.SpinlessFermions(sym='Z2', **config_kwargs)
    N = 7
    I = mps.product_mpo(ops.I(), N)
    terms = []
    for i in range(N):
        tmp = [mps.Hterm(1.0, [i, (i+1) % N], [ops.cp(), ops.c()]),
               mps.Hterm(1.0, [(i+1) % N, i], [ops.cp(), ops.c()]),
               mps.Hterm(0.5, [i, (i+2) % N], [ops.cp(), ops.c()]),
               mps.Hterm(0.5, [(i+2) % N, i], [ops.cp(), ops.c()]),
               mps.Hterm(0.2, [i,], [ops.n()])]
        terms.extend(tmp)
    H = mps.generate_mpo(I, terms)

    Dmax = 8
    opts_svd = {'D_total': Dmax}
    E0s = {0: -4.24891733952, 1: -3.27046940557}

    for n, E0 in E0s.items():
        psi = mps.random_mps(H, n=n, D_total=Dmax)
        step = mps.dmrg_(psi, H, method='2site', max_sweeps=10, energy_tol=1e-11, opts_svd=opts_svd)
        assert abs(step.energy - E0) < tol

        phi = mps.random_mps(H, n=n, D_total=Dmax)
        target = [[H, psi], [2j * psi], [[-H, 3 * H], psi]]
        mps.compression_(phi, target, method='2site', max_sweeps=2, opts_svd=opts_svd, normalize=False)
        mps.compression_(phi, target, method='1site', max_sweeps=10, normalize=False)
        assert abs(phi.norm() - abs(3 * E0 + 2j)) < tol
        #
        # for variance
        psi0 = mps.zipper(H - E0 * I, psi, opts_svd=opts_svd, normalize=False)
        assert mps.vdot(psi0, psi0) < tol
        H2ref = (H - E0 * I) @ (H - E0 * I)
        assert mps.vdot(psi, H2ref, psi) < tol
        #
        for method in ['1site', '2site']:
            opts_svd = {'D_total': 8}
            mps.compression_(psi0, [H - E0 * I, psi], normalize=False,
                            method=method, max_sweeps=10, opts_svd=opts_svd)
            assert mps.vdot(psi0, psi0) < tol
            #
            mps.compression_(psi0, [[H, psi], [-E0 * psi]], normalize=False,
                            method=method, max_sweeps=10, opts_svd=opts_svd)
            assert mps.vdot(psi0, psi0) < tol
            #
            mps.compression_(psi0, [[H, psi], [-E0 * psi]], normalize=False,
                            method=method, max_sweeps=10, opts_svd=opts_svd)
            assert mps.vdot(psi0, psi0) < tol
            #
            opts_svd = {'D_total': 30}
            H2 = mps.zipper(H - E0 * I, H - E0 * I, opts_svd=opts_svd)
            mps.compression_(H2, [H - E0 * I, H - E0 * I], normalize=False,
                                method=method, max_sweeps=10, opts_svd=opts_svd)
            assert mps.vdot(psi, H2, psi) < tol
            assert (H2 - H2ref).norm() < tol
            #
            mps.compression_(H2, [[H, H], [-2 * E0 * H], [E0 * E0 * I]], normalize=False,
                                method=method, max_sweeps=10, opts_svd=opts_svd)
            assert mps.vdot(psi, H2, psi) < tol
            assert (H2 - H2ref).norm() < tol
            #
            mps.compression_(H2, [[H.on_bra(), H], [-2 * E0 * H], [E0 * E0 * I]], normalize=False,
                                method=method, max_sweeps=10, opts_svd=opts_svd)
            assert mps.vdot(psi, H2, psi) < tol
            assert (H2 - H2ref).norm() < tol


    HP = mps.Mpo(N, periodic=True)
    for i in range(N):
        HP[(i + N // 2) % N] = H[i].copy()
    #
    #  here, we cannot mix HP and H (due to messed-up fermionic in the above construction of HP)
    #
    for n, E0 in E0s.items():
        psi = mps.random_mps(HP, n=n, D_total=Dmax)
        step = mps.dmrg_(psi, HP, method='2site', max_sweeps=10, energy_tol=1e-11, opts_svd=opts_svd)
        assert abs(step.energy - E0) < tol

        phi = mps.random_mps(HP, n=n, D_total=Dmax)
        target = [[HP, psi], [2j * psi], [[-HP, 3 * HP], psi]]
        mps.compression_(phi, target, method='2site', max_sweeps=2, opts_svd=opts_svd, normalize=False)
        mps.compression_(phi, target, method='1site', max_sweeps=10, normalize=False)
        assert abs(phi.norm() - abs(3 * E0 + 2j)) < tol


def test_comression_raise(config_kwargs):
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    N = 7
    I = mps.product_mpo(ops.I(), N=N)
    H = mps.random_mpo(I, D_total=5)
    psi0 = mps.random_mpo(I, D_total=4)
    psi1 = mps.random_mpo(I, D_total=4)

    with pytest.raises(yastn.YastnError):
        psi1.truncate_()
        # truncate_: provide opts_svd.
    with pytest.raises(yastn.YastnError):
        mps.compression_(psi1, [H, psi0], method='1site', Schmidt_tol=-1)
        # Compression: Schmidt_tol has to be positive or None.
    with pytest.raises(yastn.YastnError):
        mps.compression_(psi1, [H, psi0], method='1site', overlap_tol=-1)
        # Compression: overlap_tol has to be positive or None.
    with pytest.raises(yastn.YastnError):
        mps.compression_(psi1, [H, psi0], method='one-site')
        # Compression: method one-site not recognized.
    with pytest.raises(yastn.YastnError):
        mps.compression_(psi1, [H, psi0], method='2site')
        # Compression: provide opts_svd for 2site method.
    with pytest.raises(yastn.YastnError):
        psi_Np1 = mps.Mps(N=N+1)
        mps.zipper(H, psi_Np1)
        #  Zipper: Mpo and Mpo/Mps must have the same number of sites to be multiplied.
    with pytest.raises(yastn.YastnError):
        H_pbc = mps.Mpo(N, periodic=True)
        mps.zipper(H_pbc, psi0)
        # Zipper: Application of MpoPBC on Mpo is currently not supported. Contact developers to add this functionality.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
