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
"""
Test ctmrg on 2D Classical Ising model.
Calculate expectation values using ctm for analytical dense peps tensors
of 2D Ising model with zero transverse field (Onsager solution)
"""
import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps
try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg

tol_exp = 1e-6

def mean(xs):
    return sum(xs) / len(xs)

def max_dev(xs):
    mx = mean(xs)
    return max(abs(x - mx) for x in xs)


def create_Ising_tensor(si, sz, beta):
    """ Creates peps tensor for given beta. """
    szz = yastn.ncon((sz, sz), ((-0, -1), (-2, -3)))
    sii = yastn.ncon((si, si), ((-0, -1), (-2, -3)))
    G = np.cosh(beta / 2) * sii + np.sinh(beta / 2) * szz
    U, S, V = yastn.svd_with_truncation(G, axes = ((0, 1), (2, 3)), sU = -1, tol = 1e-15, Uaxis=1, Vaxis=1)
    S = S.sqrt()
    GA = S.broadcast(U, axes=1)
    GB = S.broadcast(V, axes=1)
    T = yastn.tensordot(GA, GB, axes=(2, 0))
    T = yastn.tensordot(T, GB, axes=(3, 0))
    T = yastn.tensordot(T, GA, axes=(4, 0))
    T = T.fuse_legs(axes=(1, 2, 3, 4, (0, 5)))
    return T


def gauges_random(leg):
    """ Returns a d0 x d1 dense random matrix and its inverse """
    a = yastn.rand(config=cfg, legs=[leg.conj(), leg])
    inv_tu = np.linalg.inv(a.to_numpy())
    b = yastn.Tensor(config=cfg, s=(-leg.s, leg.s))
    b.set_block(val=inv_tu, Ds=a.get_shape()[::-1])
    return a, b


def create_Ising_peps(ops, beta, lattice='checkerboard', dims=(2, 2), boundary='infinite', gauges=False):
    if lattice == 'checkerboard':
        geometry = fpeps.CheckerboardLattice()
    else: # lattice == 'square':
        geometry = fpeps.SquareLattice(dims=dims, boundary=boundary)

    psi = fpeps.Peps(geometry)
    T = create_Ising_tensor(ops.I(), ops.z(), beta)
    for site in psi.sites():
        psi[site] = T

    if not gauges:
        return psi

    for bond in psi.bonds():
        dirn, l_order = psi.nn_bond_type(bond)
        s0, s1 = bond if l_order else bond[::-1]
        T0 = psi[s0].unfuse_legs(axes=(0, 1))
        T1 = psi[s1].unfuse_legs(axes=(0, 1))
        if dirn == 'h':
            leg = T0.get_legs(axes=3)
            g0, g1 = gauges_random(leg)
            psi[s0] = yastn.ncon([T0, g0], [(-0, -1, -2, 1, -4), (1, -3)])
            psi[s1] = yastn.ncon([g1, T1], [(-1, 1), (-0, 1, -2, -3, -4)])
        else: # dirn == 'v':
            leg = T0.get_legs(axes=3)
            g0, g1 = gauges_random(leg)
            psi[s0] = yastn.ncon([g0, T0], [(-2, 1), (-0, -1, 1, -3, -4)])
            psi[s1] = yastn.ncon([T1, g1], [(1, -1, -2, -3, -4), (1, -0)])
    return psi


def run_ctm(psi, ops, D=8, init='eye', method='2site'):
    """ Compares ctm expectation values with analytical result. """
    opts_svd = {'D_total': D, 'tol': 1e-10}
    Z_old, ZZ_old = 0, 0
    env = fpeps.EnvCTM(psi, init=init)
    for _ in range(6):
        env.update_(opts_svd=opts_svd, method='2site')
    out = env.ctmrg_(max_sweeps=1000, opts_svd=opts_svd, method=method, corner_tol=tol_exp)
    return env


def check_Z(env, ops, Z_exact, site=None, tol_ev = 1e-3):
    Z = env.measure_1site(ops.z(), site=site)
    if site is None:
        Znn1 = env.measure_nn(ops.z(), ops.I())
        Znn2 = env.measure_nn(ops.I(), ops.z())
        Zs = [*Z.values(), *Znn1.values(), *Znn2.values()]
        Z = mean(Zs)
        print(f"max dev Z ={max_dev(Zs):0.6f}")
        assert max_dev(Zs) < tol_ev
    print(f"{Z=:0.6f}")
    assert abs(abs(Z) - Z_exact) < tol_ev


def check_ZZ(env, ops, ZZ_exact, bond=None, tol_ev = 1e-3):
    ZZ = env.measure_nn(ops.z(), ops.z(), bond=bond)
    if bond is None:
        ZZ = mean([*ZZ.values()])
    print(f"{ZZ=:0.6f}")
    assert abs(ZZ - ZZ_exact) < tol_ev


def test_ctm_ising():
    """ Calculate magnetization for classical 2D Ising model and compares with the exact result. """
    # beta_c = ln(1 + sqrt(2)) / 2 = 0.44068679350977147
    # spontanic magnetization = (1 - sinh(2 * beta) ** -4) ** 0.125
    Z_exact  = {0.3: 0.000000, 0.5: 0.911319, 0.6: 0.973609, 0.75: 0.993785}
    ZZ_exact = {0.3: 0.352250, 0.5: 0.872783, 0.6: 0.954543, 0.75: 0.988338}
    #
    ops = yastn.operators.Spin12(sym='dense', backend=cfg.backend, default_device=cfg.default_device)
    ops.random_seed(seed=0)
    #
    method = '1site'
    beta = 0.6
    print(f"Lattice: checkerboard infinite; gauges= False; {beta=}; {method=}")
    psi = create_Ising_peps(ops, beta, lattice='checkerboard', dims=(2, 2), boundary='infinite', gauges=False)
    env = run_ctm(psi, ops, init='rand', method=method) # eye
    check_Z(env, ops, Z_exact[beta])
    check_ZZ(env, ops, ZZ_exact[beta])
    #
    method = '2site'
    beta = 0.3
    print(f"Lattice = square infinite, gauges = False; {beta=}; {method=}")
    psi = create_Ising_peps(ops, beta, lattice='square', dims=(2, 3), boundary='infinite', gauges=True)
    env = run_ctm(psi, ops, init='eye', method=method)
    check_Z(env, ops, Z_exact[beta])
    check_ZZ(env, ops, ZZ_exact[beta])
    #
    method = '1site'
    beta = 0.6  # CTM should not be really used with "cylinder"
    print(f"Lattice = square cylinder, gauges = False; {beta=}; {method=}")
    psi = create_Ising_peps(ops, beta, lattice='square', dims=(3, 17), boundary='cylinder', gauges=False)
    env = run_ctm(psi, ops, D=4, init='rand', method=method) # eye
    check_Z(env, ops, Z_exact[beta], site=(1, 6))
    check_Z(env, ops, Z_exact[beta], site=(0, 6))
    check_ZZ(env, ops, ZZ_exact[beta], bond=((1, 8), (0, 8)))
    check_ZZ(env, ops, ZZ_exact[beta], bond=((0, 8), (2, 8)))


def test_ctm_save_load_copy():
    ops = yastn.operators.Spin12(sym='dense', backend=cfg.backend, default_device=cfg.default_device)

    psi = create_Ising_peps(ops, beta=0.5, lattice='checkerboard', dims=(2, 2), boundary='infinite', gauges=False)
    env = run_ctm(psi, ops, init='eye', method='2site')

    d = env.save_to_dict()

    env_save = fpeps.load_from_dict(ops.config, d)
    env_copy = env.copy()

    for site in env.sites():
        for dirn in  ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r']:
            ten0 = getattr(env[site], dirn)
            ten1 = getattr(env_save[site], dirn)
            ten2 = getattr(env_copy[site], dirn)

            yastn.are_independent(ten0, ten1)
            yastn.are_independent(ten0, ten2)
            assert (ten0 - ten1).norm() < 1e-14
            assert (ten0 - ten2).norm() < 1e-14

if __name__ == '__main__':
    test_ctm_ising()
    test_ctm_save_load_copy()
