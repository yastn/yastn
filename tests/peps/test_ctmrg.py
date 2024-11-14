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

tol_exp = 1e-6  #pylint: disable=invalid-name

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


def gauges_random(leg, config):
    """ Returns a d0 x d1 dense random matrix and its inverse """
    a = yastn.rand(config=config, legs=[leg.conj(), leg])
    inv_tu = np.linalg.inv(a.to_numpy())
    b = yastn.Tensor(config=config, s=(-leg.s, leg.s))
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
            g0, g1 = gauges_random(leg, ops.config)
            psi[s0] = yastn.ncon([T0, g0], [(-0, -1, -2, 1, -4), (1, -3)])
            psi[s1] = yastn.ncon([g1, T1], [(-1, 1), (-0, 1, -2, -3, -4)])
        else: # dirn == 'v':
            leg = T0.get_legs(axes=3)
            g0, g1 = gauges_random(leg, ops.config)
            psi[s0] = yastn.ncon([g0, T0], [(-2, 1), (-0, -1, 1, -3, -4)])
            psi[s1] = yastn.ncon([T1, g1], [(1, -1, -2, -3, -4), (1, -0)])
    return psi


def run_ctm(psi, D=8, init='eye', method='2site'):
    """ Compares ctm expectation values with analytical result. """
    opts_svd = {'D_total': D, 'tol': 1e-10}
    env = fpeps.EnvCTM(psi, init=init)
    for _ in range(6):
        env.update_(opts_svd=opts_svd, method='2site')
    env.ctmrg_(max_sweeps=1000, opts_svd=opts_svd, method=method, corner_tol=tol_exp)
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


def test_ctm_ising(config_kwargs):
    """ Calculate magnetization for classical 2D Ising model and compares with the exact result. """
    # beta_c = ln(1 + sqrt(2)) / 2 = 0.44068679350977147
    # spontanic magnetization = (1 - sinh(2 * beta) ** -4) ** 0.125
    Z_exact  = {0.3: 0.000000, 0.5: 0.911319, 0.6: 0.973609, 0.75: 0.993785}
    ZZ_exact = {0.3: 0.352250, 0.5: 0.872783, 0.6: 0.954543, 0.75: 0.988338}
    #
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    ops.random_seed(seed=0)
    #
    method = '1site'
    beta = 0.6
    print(f"Lattice: checkerboard infinite; gauges= False; {beta=}; {method=}")
    psi = create_Ising_peps(ops, beta, lattice='checkerboard', dims=(2, 2), boundary='infinite', gauges=False)
    env = run_ctm(psi, init='rand', method=method) # eye
    check_Z(env, ops, Z_exact[beta])
    check_ZZ(env, ops, ZZ_exact[beta])
    #
    method = '2site'
    beta = 0.3
    print(f"Lattice = square infinite, gauges = False; {beta=}; {method=}")
    psi = create_Ising_peps(ops, beta, lattice='square', dims=(2, 3), boundary='infinite', gauges=True)
    env = run_ctm(psi, init='eye', method=method)
    check_Z(env, ops, Z_exact[beta])
    check_ZZ(env, ops, ZZ_exact[beta])
    #
    method = '1site'
    beta = 0.6  # CTM should not be really used with "cylinder"
    print(f"Lattice = square cylinder, gauges = False; {beta=}; {method=}")
    psi = create_Ising_peps(ops, beta, lattice='square', dims=(3, 17), boundary='cylinder', gauges=False)
    env = run_ctm(psi, D=4, init='rand', method=method) # eye
    check_Z(env, ops, Z_exact[beta], site=(1, 6))
    check_Z(env, ops, Z_exact[beta], site=(0, 6))
    check_ZZ(env, ops, ZZ_exact[beta], bond=((1, 8), (0, 8)))
    check_ZZ(env, ops, ZZ_exact[beta], bond=((0, 8), (2, 8)))


def test_ctm_save_load_copy(config_kwargs):
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)

    psi = create_Ising_peps(ops, beta=0.5, lattice='checkerboard', dims=(2, 2), boundary='infinite', gauges=False)
    env = run_ctm(psi, init='eye', method='2site')

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



def test_ctmrg_onsager_dense(config_kwargs):
    r"""
    Calculate magnetization for classical 2D Ising model and compares with the analytical result.
    """
    beta = 0.5
    chi = 8
    nn_exact = {0.3: 0.352250, 0.5: 0.872783, 0.6: 0.954543, 0.75: 0.988338}
    local_exact  = {0.3: 0.000000, 0.5: 0.911319, 0.6: 0.973609, 0.75: 0.993785}

    config = yastn.make_config(sym='none', **config_kwargs)
    config.backend.random_seed(seed=0)


    leg = yastn.Leg(config, s=1, D=[2])
    TI = yastn.zeros(config, legs=[leg, leg, leg.conj(), leg.conj()])
    TI[()][0, 0, 0, 0] = 1
    TI[()][1, 1, 1, 1] = 1

    TX = yastn.zeros(config, legs=[leg, leg, leg.conj(), leg.conj()])
    TX[()][0, 0, 0, 0] = 1
    TX[()][1, 1, 1, 1] = -1

    B = yastn.zeros(config, legs=[leg, leg.conj()])
    B.set_block(ts=(), val=[[np.exp(beta), np.exp(-beta)],
                            [np.exp(-beta), np.exp(beta)]])

    TI = yastn.ncon([TI, B, B], [(-0, -1, 2, 3), [2, -2], [3, -3]])
    TX = yastn.ncon([TX, B, B], [(-0, -1, 2, 3), [2, -2], [3, -3]])

    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    psi = fpeps.Peps(geometry=geometry, tensors={(0, 0): TI})

    env = fpeps.EnvCTM(psi, init='rand')
    opts_svd = {"D_total": chi}
    info = env.ctmrg_(opts_svd=opts_svd, max_sweeps=100, corner_tol=1e-8)
    print(info)

    ev_XX = env.measure_nn(TX, TX)
    for val in ev_XX.values():
        assert abs(val - nn_exact[beta]) < 1e-5
        print(val)

    ev_X = env.measure_1site(TX)
    assert abs(abs(ev_X[(0 ,0)]) - local_exact[beta]) < 1e-5



def test_ctmrg_onsager_Z2(config_kwargs):
    r"""
    Use CTMRG to calculate some expectation values in 2D Ising model.
    Compare with analytical results.

    Here, we enforce Z2 symmetry of the model,
    which prevents spontaneous symmetry breaking.
    """
    beta = 0.5
    chi = 16
    XX_exact = {0.3: 0.352250, 0.5: 0.872783, 0.6: 0.954543, 0.75: 0.988338}
    local_exact  = {0.3: 0.000000, 0.5: 0.911319, 0.6: 0.973609, 0.75: 0.993785}

    config = yastn.make_config(sym='Z2', **config_kwargs)

    leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
    TI = yastn.ones(config, legs=[leg, leg, leg.conj(), leg.conj()], n=0)
    TX = yastn.ones(config, legs=[leg, leg, leg.conj(), leg.conj()], n=1)

    B = yastn.zeros(config, legs=[leg, leg.conj()])
    B.set_block(ts=(0, 0), val=np.cosh(beta))
    B.set_block(ts=(1, 1), val=np.sinh(beta))

    TI = yastn.ncon([TI, B, B], [(-0, -1, 2, 3), [2, -2], [3, -3]])
    TX = yastn.ncon([TX, B, B], [(-0, -1, 2, 3), [2, -2], [3, -3]])

    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    psi = fpeps.Peps(geometry=geometry, tensors={(0, 0): TI})

    env = fpeps.EnvCTM(psi, init='eye')
    opts_svd = {"D_total": chi}
    info = env.ctmrg_(opts_svd=opts_svd, max_sweeps=100, corner_tol=1e-8)
    print(info)

    # ev_XX = env.measure_nn(TX, TX)
    # for val in ev_XX.values():
    #     assert abs(val - XX_exact[beta]) < 1e-5


    ev_h = env.measure_line(TX, TX, sites=[(0, 0), (0, 20)])
    ev_v = env.measure_line(TX, TX, sites=[(0, 0), (20, 0)])

    print(ev_h)
    print(ev_v)
    print(local_exact[beta] ** 2)


if __name__ == '__main__':
    config_kwargs = {}
    test_ctmrg_onsager_Z2(config_kwargs)
    # test_ctmrg_onsager_dense(config_kwargs)
    #pytest.main([__file__, "-vs", "--durations=0"])
