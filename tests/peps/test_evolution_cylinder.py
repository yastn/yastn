""" Real-time evolution of spinless fermions on a cylinder. """
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps

try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg


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

    Nx, Ny = 3, 1
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary='cylinder')

    bonds = geometry.bonds()
    sites = geometry.sites()
    assert len(bonds) == Nx * Ny + Nx * (Ny - 1)
    assert len(sites) == Nx * Ny

    Js = {bond: 1 for bond in bonds}
    ms = dict(zip(sites, [0.0, 2.0, 0.0]))

    i2s = dict(enumerate(sites))
    s2i = {s: i for i, s in i2s.items()}  # 1d order of sites for free-fermions

    occs = {'u': dict(zip(sites, [1, 0, 0])),
            'd': dict(zip(sites, [0, 1, 0]))}

    tf = 0.3
    Ci, Cf = {}, {}
    for spin in 'ud':
        Ci[spin], Cf[spin] = evolve_correlation_matrix(Js, ms, occs[spin], s2i, tf)

    D, dt = 8, 0.05
    steps = round(tf / dt)
    dt = tf / steps

    # prepare evolution gates
    ops = yastn.operators.SpinfulFermions(sym='U1xU1', backend=cfg.backend, default_device=cfg.default_device)
    I = ops.I()
    gates_nn = []
    gates_local = []
    for spin in 'ud':
        for bond, t in Js.items():
            gt = fpeps.gates.gate_nn_hopping(t, 1j * dt / 2, I, ops.c(spin=spin), ops.cp(spin=spin))
            gates_nn.append(gt._replace(bond=bond))
        for site, mu in ms.items():
            gt = fpeps.gates.gate_local_occupation(mu, 1j * dt / 2, I, ops.n(spin=spin))
            gates_local.append(gt._replace(site=site))
    gates = fpeps.Gates(gates_nn, gates_local)
    #
    # initialized product state
    psi = fpeps.product_peps(geometry, {s: ops.vec_n(val=(occs['u'][s], occs['d'][s])) for s in sites})
    #
    # time-evolve initial state
    env = fpeps.EnvNTU(psi, which='NN')
    opts_svd = {"D_total": D, 'tol': 1e-12}
    for step in range(steps):
        print(f"t = {(step + 1) * dt:0.3f}" )
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd)

    opts_svd_mps = {'D_total': D, 'tol': 1e-10}
    env = fpeps.EnvBoundaryMps(psi, opts_svd=opts_svd_mps, setup='lr')
    for spin in 'ud':
        print(f"{spin=}")
        occf = env.measure_1site(ops.n(spin=spin))
        for k, v in sorted(occf.items()):
            print(f"{k}, {v.real:0.7f}, {Cf[spin][s2i[k], s2i[k]].real:0.7f}, {v.real - Cf[spin][s2i[k], s2i[k]].real:0.2e}")
            assert abs(v - Cf[spin][s2i[k], s2i[k]]) < 5e-4

if __name__ == '__main__':
    test_evol_cylinder()
    # revise U1xU1xxZ2
