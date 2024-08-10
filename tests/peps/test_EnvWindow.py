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
try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg

tol = 1e-12

def test_window_shapes():
    """ Initialize a product PEPS and perform a set of measurment. """

    # initialized PEPS with mixed bond dimensions
    geometry = fpeps.SquareLattice(dims=(2, 3))  # boundary='infinite'
    psi = fpeps.Peps(geometry)
    #
    for sphys, Dphys in [[(), ()], [(1,), (2,)]]:
        # sphys, Dphys = (), () gives single-layer PEPS
        # sphys, Dphys = (1,), (2,) gives double-layer PEPS
        s = (-1, 1, 1, -1) + sphys
        psi[0, 0] = yastn.rand(cfg, s=s, D=(2, 3, 4, 5) + Dphys)
        psi[1, 0] = yastn.rand(cfg, s=s, D=(4, 6, 2, 4) + Dphys)
        psi[0, 1] = yastn.rand(cfg, s=s, D=(3, 5, 5, 2) + Dphys)
        psi[1, 1] = yastn.rand(cfg, s=s, D=(5, 4, 3, 6) + Dphys)
        psi[0, 2] = yastn.rand(cfg, s=s, D=(2, 2, 3, 3) + Dphys)
        psi[1, 2] = yastn.rand(cfg, s=s, D=(3, 6, 2, 6) + Dphys)
        #
        opts_svd = {'D_total': 20, 'tol': 1e-10}
        env_ctm = fpeps.EnvCTM(psi, init='rand')
        #
        for _ in range(1):
            env_ctm.update_(opts_svd=opts_svd)
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
                    mps.vdot(btm, TMh, top)
        #
        for yrange in [(0, 1), (1, 2), (2, 3), (1, 4)]:
            for xrange in [(0, 5), (1, 3), (2, 6)]:
                env_win = fpeps.EnvWindow(env_ctm, xrange=xrange, yrange=yrange)
                for iy in range(*yrange):
                    rht = env_win[iy, 'r']
                    TMv = env_win[iy, 'v']
                    lft = env_win[iy, 'l']
                    mps.vdot(rht, TMv, lft)

    ops = yastn.operators.Spin12(sym='dense')
    env_win = fpeps.EnvWindow(env_ctm, xrange=(0, 4), yrange=(0, 3))
    out = env_win.measure_2site(ops.sz(), ops.sz(), opts_svd={'D_total':2})

    pr = [yastn.tensordot(ops.vec_z(val=v), ops.vec_z(val=v).conj(), axes=((), ())) for v in [-1, 1]]
    prs = {(nx, ny): pr[:] for nx in range(0, 4) for ny in range(0, 3)}
    smpl = env_win.sample(prs)
    print(smpl)


def test_window_raises():
    ops = yastn.operators.Spin12(sym='Z2', backend=cfg.backend, default_device=cfg.default_device)
    geometry = fpeps.SquareLattice(dims=(3, 5), boundary='obc')
    psi = fpeps.product_peps(geometry, ops.I())
    env_ctm = fpeps.EnvCTM(psi, init='eye')
    env_win = fpeps.EnvWindow(env_ctm, xrange=(0, 4), yrange=(0, 3))

    with pytest.raises(yastn.YastnError):
        env_win[10, 'l']
        # n=10 not within self.yrange=(0, 3)
    with pytest.raises(yastn.YastnError):
        env_win[-2, 't']
        # n=-2 not within self.xrange=(0, 4)
    with pytest.raises(yastn.YastnError):
        env_win[2, 'none']
        # dirn='none' not recognized. Should be 't', 'h' 'b', 'r', 'v', or 'l'.


if __name__ == '__main__':
    test_window_shapes()
    test_window_raises()
