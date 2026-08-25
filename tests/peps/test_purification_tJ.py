# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps
import logging

def exact_diagonalization_tJ(config_kwargs, Nx, Ny, t, J, mu, beta, sym="U1xU1xZ2"):
    net = fpeps.SquareLattice((Nx, Ny), "obc")
    N = len(net.sites())
    s2i = {s0: i for i, s0 in enumerate(net.sites())}
    #
    ops = yastn.operators.SpinfulFermions_tJ(sym=sym, **config_kwargs)
    I = ops.I()
    c_up, c_dn = ops.c(spin='u'), ops.c(spin='d')
    cdag_up, cdag_dn = ops.cp(spin='u'), ops.cp(spin='d')
    n_up, n_dn = ops.n(spin='u'), ops.n(spin='d')
    n = n_up + n_dn
    Sz, Sp, Sm = ops.Sz(), ops.Sp(), ops.Sm()
    #
    terms = []
    for (s0, s1) in net.bonds():
        terms.append(mps.Hterm(-t, (s2i[s0], s2i[s1]), (cdag_up, c_up)))
        terms.append(mps.Hterm(-t, (s2i[s0], s2i[s1]), (cdag_dn, c_dn)))
        terms.append(mps.Hterm(-t, (s2i[s1], s2i[s0]), (cdag_up, c_up)))
        terms.append(mps.Hterm(-t, (s2i[s1], s2i[s0]), (cdag_dn, c_dn)))
        terms.append(mps.Hterm(J, (s2i[s0], s2i[s1]), (Sz, Sz)))
        terms.append(mps.Hterm(J / 2, (s2i[s0], s2i[s1]), (Sp, Sm)))
        terms.append(mps.Hterm(J / 2, (s2i[s0], s2i[s1]), (Sm, Sp)))
        terms.append(mps.Hterm(-J / 4, (s2i[s0], s2i[s1]), (n, n)))
    for s0 in net.sites():
        terms.append(mps.Hterm(-mu, (s2i[s0],), (n,)))

    H = mps.generate_mpo(I, terms, N=N).to_matrix()
    S, U = yastn.eigh(H, axes=(0, 1))
    eS = yastn.exp(-beta * S)
    eS = eS / yastn.trace(eS).to_number()
    rho = U @ eS @ U.H
    #
    obs = {}
    obs['n_up'] = {s0: mps.generate_mpo(I, [mps.Hterm(1, [s2i[s0]], [n_up])], N=N).to_matrix() for s0 in net.sites()}
    obs['n_dn'] = {s0: mps.generate_mpo(I, [mps.Hterm(1, [s2i[s0]], [n_dn])], N=N).to_matrix() for s0 in net.sites()}
    obs['cdagc_up'] = {(s0, s1): mps.generate_mpo(I, [mps.Hterm(1, [s2i[s0], s2i[s1]], [cdag_up, c_up])], N=N).to_matrix() for s0, s1 in net.bonds()}
    obs['cdagc_dn'] = {(s0, s1): mps.generate_mpo(I, [mps.Hterm(1, [s2i[s0], s2i[s1]], [cdag_dn, c_dn])], N=N).to_matrix() for s0, s1 in net.bonds()}
    obs['ccdag_up'] = {(s0, s1): mps.generate_mpo(I, [mps.Hterm(1, [s2i[s0], s2i[s1]], [c_up, cdag_up])], N=N).to_matrix() for s0, s1 in net.bonds()}
    obs['ccdag_dn'] = {(s0, s1): mps.generate_mpo(I, [mps.Hterm(1, [s2i[s0], s2i[s1]], [c_dn, cdag_dn])], N=N).to_matrix() for s0, s1 in net.bonds()}
    obs['SmSp'] = {(s0, s1): mps.generate_mpo(I, [mps.Hterm(1, [s2i[s0], s2i[s1]], [Sm, Sp])], N=N).to_matrix() for s0, s1 in net.bonds()}
    obs['SpSm'] = {(s0, s1): mps.generate_mpo(I, [mps.Hterm(1, [s2i[s0], s2i[s1]], [Sp, Sm])], N=N).to_matrix() for s0, s1 in net.bonds()}
    obs['SzSz'] = {(s0, s1): mps.generate_mpo(I, [mps.Hterm(1, [s2i[s0], s2i[s1]], [Sz, Sz])], N=N).to_matrix() for s0, s1 in net.bonds()}
    obs['nn'] = {(s0, s1): mps.generate_mpo(I, [mps.Hterm(1, [s2i[s0], s2i[s1]], [n, n])], N=N).to_matrix() for s0, s1 in net.bonds()}
    #
    return {kobs: {k: yastn.trace(rho @ op).to_number() for k, op in oo.items()} for kobs, oo in obs.items()}


def peps_purification_tJ(config_kwargs, Nx, Ny, t, J, mu, beta, dbeta, which, D, sym="U1xU1xZ2"):
    #
    ops = yastn.operators.SpinfulFermions_tJ(sym = sym, **config_kwargs)
    I = ops.I()
    c_up, c_dn = ops.c(spin='u'), ops.c(spin='d')
    cdag_up, cdag_dn = ops.cp(spin='u'), ops.cp(spin='d')
    n_up, n_dn = ops.n(spin='u'), ops.n(spin='d')
    n = n_up + n_dn
    Sz, Sp, Sm = ops.Sz(), ops.Sp(), ops.Sm()
    #
    net = fpeps.SquareLattice((Nx, Ny), "obc")
    psi = fpeps.product_peps(net, I)
    #
    num_steps = round(beta / dbeta)
    dbeta = beta / num_steps
    #
    coef = 0.25  # evolution gate has a step dbeta * coef
    gates = [fpeps.gates.gate_nn_tJ(J, t, t, mu / 4, mu / 4, mu / 4, mu / 4, dbeta * coef, I, c_up, cdag_up, c_dn, cdag_dn, bond)
             for bond in net.bonds()]
    #
    # correct boundary terms with local chemical potential to have uniform mu
    for nx in range(Nx):
        gates.append(fpeps.gates.gate_local_occupation(mu / 4, dbeta * coef, I, n, site=(nx, 0)))
        gates.append(fpeps.gates.gate_local_occupation(mu / 4, dbeta * coef, I, n, site=(nx, Ny - 1)))
    for ny in range(Ny):
        gates.append(fpeps.gates.gate_local_occupation(mu / 4, dbeta * coef, I, n, site=(0, ny)))
        gates.append(fpeps.gates.gate_local_occupation(mu / 4, dbeta * coef, I, n, site=(Nx - 1, ny)))
    #
    # symmetrize
    gates = gates + gates[::-1]
    #
    if 'BP' in which:
        env_evol = fpeps.EnvBP(psi, which=which)
        env_evol.iterate_(max_sweeps=100, diff_tol=1e-8)
    else:
        env_evol = fpeps.EnvNTU(psi, which=which)

    opts_svd = {"D_total": D, 'tol': 1e-10}

    beta = 0
    for _ in range(num_steps):
        beta += dbeta
        print(f"{beta=:0.3f}")
        fpeps.evolution_step_(env_evol, gates, opts_svd=opts_svd)
        if 'BP' in which:
            env_evol.iterate_(max_sweeps=100, diff_tol=1e-8)

    # calculate observables with ctm
    env = fpeps.EnvCTM(psi, init="eye")
    opts_svd_ctm = {'D_total': 2 * D}
    info = env.ctmrg_(max_sweeps=32, method="2x2", opts_svd=opts_svd_ctm, corner_tol=1e-7)
    print(info)
    #
    # calculate expectation values
    out = {}
    out['cdagc_up'] = env.measure_nn(cdag_up, c_up)  # calculate for all unique bonds
    out['ccdag_up'] = env.measure_nn(c_up, cdag_up)  # -> {bond: value}
    out['cdagc_dn'] = env.measure_nn(cdag_dn, c_dn)
    out['ccdag_dn'] = env.measure_nn(c_dn, cdag_dn)
    out['SpSm'] = env.measure_nn(Sp, Sm)
    out['SmSp'] = env.measure_nn(Sm, Sp)
    out['SzSz'] = env.measure_nn(Sz, Sz)
    out['nn'] = env.measure_nn(n, n)
    out['n_up'] = env.measure_1site(n_up)
    out['n_dn'] = env.measure_1site(n_dn)
    return out


@pytest.mark.skipif("not config.getoption('long_tests')", reason="long duration tests are skipped")
def test_purification_tJ(config_kwargs):
    #
    Nx, Ny = 3, 2
    sym = "U1xU1xZ2"
    t = 1
    J = 0.5
    mu = 1.7
    beta = 1.0
    #
    # peps simulations
    dbeta = 0.1
    which = 'NN+BP'
    D = 10
    out = peps_purification_tJ(config_kwargs, Nx, Ny, t, J, mu, beta, dbeta, which, D, sym=sym)
    #
    # reference from exact diagonalization
    ed = exact_diagonalization_tJ(config_kwargs, Nx, Ny, t, J, mu, beta, sym=sym)
    #
    for val in ['n_up', 'n_dn']:
        for k in out[val]:
            print(val, k, ed[val][k], out[val][k], abs(ed[val][k] - out[val][k]))

    diffs = {(val, k): abs(ed[val][k] - out[val][k]) for val in ed.keys() for k in ed[val].keys()}
    (val, k) = max(diffs, key=diffs.get)
    err = diffs[val, k]
    print(val, k, ed[val][k], out[val][k], err)
    assert err < 1e-3


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", '--long_tests'])
