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
""" Test PEPS measurments with MpsBoundary in a product state. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps

tol = 1e-12  #pylint: disable=invalid-name


def init_peps(config_kwargs, Dphys=(), boundary='infinite'):
    """ initialized PEPS with mixed bond dimensions for testing. """
    geometry = fpeps.SquareLattice(dims=(2, 3), boundary=boundary)
    psi = fpeps.Peps(geometry)
    s = (-1, 1, 1, -1) + (1,) * len(Dphys)
    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=5)
    psi[0, 0] = yastn.rand(config, s=s, D=(2, 3, 4, 5) + Dphys, dtype='complex128')
    psi[1, 0] = yastn.rand(config, s=s, D=(4, 3, 2, 4) + Dphys, dtype='complex128')
    psi[0, 1] = yastn.rand(config, s=s, D=(3, 5, 5, 2) + Dphys, dtype='complex128')
    psi[1, 1] = yastn.rand(config, s=s, D=(5, 4, 3, 3) + Dphys, dtype='complex128')
    psi[0, 2] = yastn.rand(config, s=s, D=(2, 2, 3, 3) + Dphys, dtype='complex128')
    psi[1, 2] = yastn.rand(config, s=s, D=(3, 3, 2, 3) + Dphys, dtype='complex128')
    return psi


def test_window_shapes(config_kwargs):
    """ Initialize a product PEPS and perform a set of measurment. """
    for Dphys in [(), (2,)]:  # Dphys = () gives single-layer PEPS; (2,) gives double-layer PEPS
        psi = init_peps(config_kwargs, Dphys)
        #
        env_ctm = fpeps.EnvCTM(psi, init='eye')
        #
        #  test contractions of < mps | mpo | mps > in different configurations
        #
        for xrange in [(0, 1), (1, 2), (2, 3), (0, 3)]:
            for yrange in [(0, 5), (1, 3), (2, 6)]:
                env_win = fpeps.EnvWindow(env_ctm, xrange=xrange, yrange=yrange)
                for ix in range(*xrange):
                    top = env_win[ix, 't']
                    TMh = env_win[ix, 'h']
                    btm = env_win[ix, 'b']
                    mps.vdot(btm.conj(), TMh, top)
        #
        for yrange in [(0, 1), (1, 2), (2, 3), (1, 4)]:
            for xrange in [(0, 5), (1, 3), (2, 6)]:
                env_win = fpeps.EnvWindow(env_ctm, xrange=xrange, yrange=yrange)
                for iy in range(*yrange):
                    rht = env_win[iy, 'r']
                    TMv = env_win[iy, 'v']
                    lft = env_win[iy, 'l']
                    mps.vdot(rht.conj(), TMv, lft)

    with pytest.raises(yastn.YastnError):
        env_win[10, 'l']
        # n=10 not within self.yrange=(0, 3)
    with pytest.raises(yastn.YastnError):
        env_win[-2, 't']
        # n=-2 not within self.xrange=(0, 4)
    with pytest.raises(yastn.YastnError):
        env_win[2, 'none']
        # dirn='none' not recognized. Should be 't', 'h' 'b', 'r', 'v', or 'l'.
    with pytest.raises(yastn.YastnError):
        psi = init_peps(config_kwargs, Dphys=(), boundary='obc')
        env_ctm = fpeps.EnvCTM(psi, init='eye')
        env_win = fpeps.EnvWindow(env_ctm, xrange=(1, 5), yrange=(1, 5))
        # Window range xrange=(1, 5), yrange=(1, 5) does not fit within the lattice.


def test_window_measure(config_kwargs):
    """ checks syntax of sample and measure_2site"""
    # for Dphys = 2
    psi = init_peps(config_kwargs, Dphys=(2,))
    D_total = 15
    opts_svd = {'D_total': D_total, 'tol': 1e-10}
    env_ctm = fpeps.EnvCTM(psi, init='eye')
    #
    info = env_ctm.ctmrg_(opts_svd, max_sweeps=20, corner_tol=1e-4)
    print(info)  # did not converge
    #
    env_win = fpeps.EnvWindow(env_ctm, xrange=(0, 4), yrange=(0, 3))
    #
    # test sample
    #
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    vecs = [ops.vec_z(val=v) for v in [-1, 1]]
    #
    number = 4
    out, probs = env_win.sample(vecs, number=number, return_probabilities=True, progressbar=True)
    assert len(out) == 12
    for ny in range(0, 3):
        for nx in range(0, 4):
            assert len(out[nx, ny]) == number
            assert all(x in [0, 1] for x in out[nx, ny])

    vecs = {k: v for k, v in zip('tb', vecs)}
    out = env_win.sample(vecs, number=number)
    assert len(out) == 12
    for ny in range(0, 3):
        for nx in range(0, 4):
            assert len(out[nx, ny]) == number
            assert all(x in 'tb' for x in out[nx, ny])
    #
    with pytest.raises(yastn.YastnError):
        env_win.sample(projectors={(0, 0): vecs, (1, 0): vecs})
        # Projectors not defined for some sites in xrange=(0, 4), yrange=(0, 3).
    #
    # test measure_2site
    #
    out = env_win.measure_2site(ops.z(), ops.z(), site0='corner')
    sites = env_win.sites()
    assert len(sites) == 3 * 4
    assert all(((0, 0), site) in out for site in sites)
    #
    # here we can check some values
    #
    outv = env_ctm.measure_2site(ops.z(), ops.z(), xrange=(1, 5), yrange=(0, 1))
    ev = [env_ctm.measure_line(ops.z(), ops.z(), sites=((1, 0), (n, 0))) for n in [2, 3, 4,]]
    for n, ref in zip([1, 2, 3, 4], [1] + ev):
        assert abs(outv[(1, 0), (n, 0)] - ref) / abs(ref) < 1e-2
    #
    outh = env_ctm.measure_2site(ops.z(), ops.z(), xrange=(2, 5), yrange=(2, 3), site0='row')
    eh = [env_ctm.measure_line(ops.z(), ops.z(), sites=((2, 2), (n, 2))) for n in [3, 4]]
    for n, ref in zip([2, 3, 4], [1] + eh):
        assert abs(outh[(2, 2), (n, 2)] - ref) / abs(ref) < 1e-5


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
