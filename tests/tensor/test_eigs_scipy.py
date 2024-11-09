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
""" yastn.compress_to_1d() yastn.decompress_from_1d()  in combination with scipy LinearOperator and eigs """
import numpy as np
import pytest
from scipy.sparse.linalg import eigs, LinearOperator
import yastn

tol = 1e-8  #pylint: disable=invalid-name

numpy_test = pytest.mark.skipif("'np' not in config.getoption('--backend')",
                                reason="using scipy procedures for raw data requires np")


@numpy_test
def test_eigs_simple(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 2)),
            yastn.Leg(config_U1, s=1, t=(0, 1), D=(1, 1)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 2))]
    a = yastn.rand(config=config_U1, legs=legs)  # could be mps tensor
    a, _ = yastn.qr(a, axes=((0, 1), 2), sQ=-1)  # orthonormalize

    # dense transfer matrix build from a; reference solution
    tm = yastn.ncon([a, a.conj()], [(-1, 1, -3), (-2, 1, -4)])
    tm = tm.fuse_legs(axes=((2, 3), (0, 1)), mode='hard')
    tmn = tm.to_numpy()
    w_ref, v_ref = eigs(tmn, k=1, which='LM')  # use scipy.sparse.linalg.eigs

    # initializing random tensor matching tm from left
    # we add an extra 3-rd leg carrying charges -1, 0, 1
    # to calculate eigs over those 3 subspaces in one go
    legs = [a.get_legs(0).conj(),
            a.get_legs(0),
            yastn.Leg(a.config, s=1, t=(-1, 0, 1), D=(1, 1, 1))]
    v0 = yastn.rand(config=a.config, legs=legs)
    # define a wrapper that goes r1d -> yastn.tensor -> tm @ yastn.tensor -> r1d
    r1d, meta = yastn.compress_to_1d(v0)
    def f(x):
        t = yastn.decompress_from_1d(x, meta=meta)
        t2 = yastn.ncon([t, a, a.conj()], [(1, 3, -3), (1, 2, -1), (3, 2, -2)])
        t3, _ = yastn.compress_to_1d(t2, meta=meta)
        return t3
    ff = LinearOperator(shape=(len(r1d), len(r1d)), matvec=f, dtype=np.float64)
    # scipy.sparse.linalg.eigs that goes though yastn symmetric tensor
    wa, va1d = eigs(ff, v0=r1d, k=1, which='LM', tol=1e-10)
    # transform eigenvectors into yastn tensors
    va = [yastn.decompress_from_1d(x, meta) for x in va1d.T]
    # we can remove zero blocks now, as there are eigenvectors with well defined charge
    # (though we might get superposition of symmetry sectors in case of degeneracy)
    va = [x.remove_zero_blocks() for x in va]

    # we can also limit ourselves directly to eigenvectors with desired charge, here 0.
    legs = [a.get_legs(0).conj(),
            a.get_legs(0)]
    v0 = yastn.rand(config=a.config, legs=legs, n=0)
    r1d, meta = yastn.compress_to_1d(v0)
    def f(x):
        t = yastn.decompress_from_1d(x, meta=meta)
        t2 = yastn.ncon([t, a, a.conj()], [(1, 3), (1, 2, -1), (3, 2, -2)])
        t3, _ = yastn.compress_to_1d(t2, meta=meta)
        return t3
    ff = LinearOperator(shape=(len(r1d), len(r1d)), matvec=f, dtype=np.float64)
    wb, vb1d = eigs(ff, v0=r1d, k=1, which='LM', tol=1e-10)  # scipy.sparse.linalg.eigs
    vb = [yastn.decompress_from_1d(x, meta) for x in vb1d.T]  # eigenvectors as yastn tensors

    # dominant eigenvalue should have amplitude 1 (likely degenerate in our example)
    assert all(pytest.approx(abs(x), rel=tol) == 1.0 for x in (w_ref, wa, wb))
    print("va -> ", va.pop())
    print("vb -> ", vb.pop())


@numpy_test
def test_eigs_mismatches(config_kwargs):
    #
    # here define a problem in a way that there are some mismatches in legs to be resolved
    #
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg0 = yastn.Leg(config_U1, s=1, t=(-2, -1, 0, 1), D=(1, 2, 3 ,4))
    leg1 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))
    leg2 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1, 2), D=(2, 3 ,4, 5))

    a = yastn.rand(config=config_U1, legs=(leg0, leg1, leg2.conj()), n=0)
    # will be treated as mps tensor

    # dense transfer matrix build from a -- here a has some un-matching blocks between first and last legs
    tm = yastn.ncon([a, a], [(-1, 1, -3), (-2, 1, -4)], conjs=(0, 1))
    tm = tm.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
    # make sure to fill-in zero blocks, as in this example tm is not a square matrix
    legs_for_tm = {0: tm.get_legs(1).conj(), 1: tm.get_legs(0).conj()}
    tmn = tm.to_numpy(legs=legs_for_tm)
    wn, vn = eigs(tmn, k=5, which='LM')  # scipy

    ## initializing random tensor matching TM, with 3-rd leg extra carrying charges -1, 0, 1
    leg02 = yastn.leg_union(leg0, leg2)
    leg_aux = yastn.Leg(a.config, s=1, t=(-1, 0, 1), D=(1, 1, 1))
    vv = yastn.rand(config=a.config, legs=(leg02, leg02.conj(), leg_aux), dtype='float64')
    r1d, meta = yastn.compress_to_1d(vv)

    def f(x):  # change all that into a wrapper around ncon part?
        t = yastn.decompress_from_1d(x, meta)
        t2 = yastn.ncon([a, a, t], [(-1, 1, 2), (-2, 1, 3), (2, 3, -3)], conjs=(0, 1, 0))
        t3, _ = yastn.compress_to_1d(t2, meta=meta)
        return t3
    ff = LinearOperator(shape=(len(r1d), len(r1d)), matvec=f, dtype=np.float64)

    # eigs going though yastn.tensor
    wy1, vy1d = eigs(ff, v0=r1d, k=5, which='LM', tol=1e-10)  # scipy going though yastn.tensor

    # transform eigenvectors into yastn tensors
    vy = [yastn.decompress_from_1d(x, meta) for x in vy1d.T]
    # remove zero blocks and checks if that was correct
    vyr = [yastn.remove_zero_blocks(a, rtol=1e-12) for a in vy]
    assert all((yastn.norm(x - y) < tol for x, y in zip(vy, vyr)))
    # display charges of eigenvectors (only charge on last leg)
    print(vy[0].get_legs(2))
    print(vyr[0].get_legs(2))
    # for others there might be superposition between +1 and -1


@pytest.mark.skipif("'np' not in config.getoption('--backend')",
                    reason="torch TODO, some problem to identify")  # TODO
def test_eigs_temp(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    config_U1.backend.random_seed(seed=0)  # fix seed for testing

    legs = [yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=1, t=(0, 1), D=(1, 1)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4))]
    a = yastn.rand(config=config_U1, legs=legs)  # could be an MPS tensor

    tm = yastn.ncon([a, a.conj()], [(-1, 1, -3), (-2, 1, -4)])
    tm = tm.fuse_legs(axes=((2, 3), (0, 1)), mode='hard')
    tmn = tm.to_numpy()
    f = lambda t: yastn.ncon([t, a, a.conj()], [(1, 3, -3), (1, 2, -1), (3, 2, -2)])

    legs = [a.get_legs(0).conj(),
            a.get_legs(0),
            yastn.Leg(a.config, s=1, t=(-1, 0, 1, -2, 2), D=(1, 1, 1, 1, 1))]

    for which in ('LM', 'LR', 'SR'):
        w_ref, _ = eigs(tmn, k=1, which=which)  # use scipy.sparse.linalg.eigs
        v0 = yastn.randC(config=a.config, legs=legs)
        v0 = [v0 / v0.norm()]
        w_old = 100
        for ii in range(100):  # no restart in yastn.eigs
            w, v0 = yastn.eigs(f, v0=v0[0], k=1, which=which, ncv=4, hermitian=False)
            if abs(w - w_old) < tol / 10:
                break
            w_old = w
        print(which, ii, abs(w_ref - w.item()))
        assert abs(w_ref - w.item()) < tol

    tmn = tmn + tmn.T
    f = lambda t: yastn.ncon([t, a, a.conj()], [(1, 3, -3), (1, 2, -1), (3, 2, -2)]) + \
                  yastn.ncon([t, a.conj(), a], [(1, 3, -3), (-1, 2, 1), (-2, 2, 3)])

    for which in ('LM', 'LR', 'SR'):
        w_ref, _ = eigs(tmn, k=1, which=which)  # use scipy.sparse.linalg.eigs
        v0 = yastn.randC(config=a.config, legs=legs)
        v0 = [v0 / v0.norm()]
        w_old = 100
        for ii in range(100):  # no restart in yastn.eigs
            w, v0 = yastn.eigs(f, v0=v0[0], k=1, which=which, ncv=4, hermitian=True)
            if abs(w - w_old) < tol / 10:
                break
            w_old = w
        print(which,  ii, abs(w_ref - w.item()))
        assert abs(w_ref - w.item()) < tol


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
