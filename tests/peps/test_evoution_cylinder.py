""" Real-time evolution of spinless fermions on a cylinder. """
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps

try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg


def evolve_correlation_matrix(Js, mus, occs0, s2i, t):
    """
    Initialize correlation matrix in a product state with occupation occ0.
    Evolve it by time t, with hopping Js and chemical potentials mus.
    s2i gives a map between sites and matrix indices.
    Return initial and final correlation matrices.
    """
    N = len(s2i)
    Ci = np.zeros((N, N))
    for st, oc in occs0.items():
        Ci[s2i[st], s2i[st]] = oc

    Hs = np.zeros((N, N))
    for (s0, s1), v in Js.items():
        Hs[s2i[s0], s2i[s1]] = -v
        Hs[s2i[s1], s2i[s0]] = -v
    for s0, v in mus.items():
        Hs[s2i[s0], s2i[s0]] = -v

    # U = expm(1j * t * Hs)
    D, V = np.linalg.eigh(Hs)
    U = V @ np.diag(np.exp(1j * t * D)) @ V.T.conj()
    Cf = U.conj().T @ Ci @ U

    return Ci, Cf


def test_evol_cylinder():
    """ Simulate purification of spinful fermions in a small finite system """
    print(" Simulating spinful fermions in a small finite system. ")

    Nx, Ny = 5, 1
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary='cylinder')

    bonds = geometry.bonds()
    sites = geometry.sites()
    assert len(bonds) == Nx * Ny + Nx * (Ny - 1)
    assert len(sites) == Nx * Ny

    # random couplings, chemical potentials and initial occupation
    np.random.seed(seed=0)
    # Js = dict(zip(bonds, 2 * np.random.rand(len(bonds)) - 1))
    # mus = dict(zip(sites, np.random.rand(len(sites)) / 20 - 0.025))

    # occs0 = dict(zip(sites, np.random.rand(len(sites)) > 0.5))
    # occs1 = dict(zip(sites, np.random.rand(len(sites)) > 0.5))

    Js = dict(zip(bonds, [1, 1, 1, 1, 1]))
    mus = dict(zip(sites, [0, 0, 0, 0, 0]))

    occs0 = dict(zip(sites, [1, 0, 1, 0, 1]))
    occs1 = dict(zip(sites, [0, 1, 0, 1, 0]))

    i2s = dict(enumerate(sites))
    s2i = {s: i for i, s in i2s.items()}  # 1d order of sites

    tf = 0.6
    Ci0, Cf0 = evolve_correlation_matrix(Js, mus, occs0, s2i, tf)
    Ci1, Cf1 = evolve_correlation_matrix(Js, mus, occs1, s2i, tf)
    # print(np.diag(Ci0))
    # print(np.diag(Cf0))

    D, dt = 6, 0.05
    steps = round(tf / dt)
    dt = tf / steps

    # prepare evolution gates
    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    I = ops.I()
    c_up, cdag_up, n_up = ops.c(spin='u'), ops.cp(spin='u'), ops.n(spin='u')
    c_dn, cdag_dn, n_dn = ops.c(spin='d'), ops.cp(spin='d'), ops.n(spin='d')

    gates_nn = []
    for bond, t in Js.items():
        gt = fpeps.gates.gate_nn_hopping(t, 1j * dt / 2, I, c_up, cdag_up)
        gates_nn.append(gt._replace(bond=bond))
        gt = fpeps.gates.gate_nn_hopping(t, 1j * dt / 2, I, c_dn, cdag_dn)
        gates_nn.append(gt._replace(bond=bond))
    gates_local = []
    for site, mu in mus.items():
        gt = fpeps.gates.gate_local_occupation(mu, 1j * dt / 2, I, n_up)
        gates_local.append(gt._replace(site=site))
        gt = fpeps.gates.gate_local_occupation(mu, 1j * dt / 2, I, n_dn)
        gates_local.append(gt._replace(site=site))

    gates = fpeps.Gates(gates_nn, gates_local)

    # initialized product state

    occs = {s: ops.vec_n(val=(occs0[s], occs1[s])) for s in occs0}
    psi = fpeps.product_peps(geometry, occs)

    # time-evolve initial state
    env = fpeps.EnvNTU(psi, which='NN++')
    opts_svd = {"D_total": D, 'tol_block': 1e-15}
    for step in range(steps):
        print(f"t = {(step + 1) * dt:0.2f}" )
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd)


    opts_svd_mps = {'D_total': 4 * D, 'tol': 1e-10}
    env = fpeps.EnvBoundaryMps(psi, opts_svd=opts_svd_mps, setup='lr')
    occf = env.measure_1site(n_up)
    for k, v in sorted(occf.items()):
        print(k, v.real, Cf0[s2i[k], s2i[k]].real, v.real - Cf0[s2i[k], s2i[k]].real)

    occf = env.measure_1site(n_dn)
    for k, v in sorted(occf.items()):
        print(k, v.real, Cf1[s2i[k], s2i[k]].real, v.real - Cf1[s2i[k], s2i[k]].real)


if __name__ == '__main__':
    test_evol_cylinder()
