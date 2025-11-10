""" Test the excited state expectation values for spin-1/2 triangular lattice systems. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import TriangularLattice
import itertools
import numpy as np

import torch
from scipy.linalg import eig


def _cast_to_real(t):
    return t.real if t.is_complex() else t


def init_peps(config_kwargs):
    """ initialized PEPS with mixed bond dimensions for testing. """
    geometry = fpeps.TriangularLattice(full_patch=False)
    psi = fpeps.Peps(geometry)
    # s = (-1, 1, 1, -1) + (1,)
    config = yastn.make_config(sym='none', **config_kwargs)
    up_state = yastn.Tensor(config=config, s=(-1, 1, 1, -1, 1))
    up_state.set_block(val=[[[[1, 0]]]], Ds=(1, 1, 1, 1, 2))
    down_state = yastn.Tensor(config=config, s=(-1, 1, 1, -1, 1))
    down_state.set_block(val=[[[[0, 1]]]], Ds=(1, 1, 1, 1, 2))
    psi[0, 0] = up_state
    psi[0, 1] = up_state
    psi[0, 2] = down_state
    return psi


def test_exci_sma_UUD(config_kwargs):
    """ Compute excited state expectation values of a UUD state on a 3x3 finite patch. """
    print(" Compute excited state expectation values of a UUD state on a 3x3 finite patch. ")

    geometry = TriangularLattice(full_patch=False) # \sqrt{3}\times\sqrt{3}
    psi = init_peps(config_kwargs)
    # excited state basis
    exci_basis = np.array([[0, 1], [0, 1], [1, 0]])
    env_ctm = fpeps.EnvCTM(psi, init='eye')
    env_opts_svd = {
        "D_total": 20,
    }
    for out in env_ctm.ctmrg_(opts_svd=env_opts_svd, max_sweeps=30, iterator_step=1, corner_tol=1e-12):
        # print(out)
        continue

    config = yastn.make_config(sym='none', **config_kwargs)
    s12 = yastn.operators.Spin12(**config._asdict())
    Id, Sz, Sx, Sy, Sp, Sm = s12.I(), s12.sz(), s12.sx(), s12.sy(), s12.sp(), s12.sm()

    Jzz = 2.98
    Jxy = 0.21
    hz = 3.16

    ### Compute ground state energy
    e0s = {}
    for site in psi.sites():
        e0s[f"{tuple(site)}"] = - hz * _cast_to_real(env_ctm.measure_1site(Sz, site=site))
        # horizontal bond
        site_r = psi.nn_site(site, "r")
        h_bond = fpeps.Bond(site, site_r)
        e0s[f"{tuple(site)}, {tuple(site_r)}"] = Jzz * _cast_to_real(env_ctm.measure_nn(Sz, Sz, bond=h_bond))
        e0s[f"{tuple(site)}, {tuple(site_r)}"] += 0.5 * Jxy * _cast_to_real(env_ctm.measure_nn(Sp, Sm, bond=h_bond))
        e0s[f"{tuple(site)}, {tuple(site_r)}"] += 0.5 * Jxy * _cast_to_real(env_ctm.measure_nn(Sm, Sp, bond=h_bond))

        # vertical bond
        site_b = psi.nn_site(site, "b")
        v_bond = fpeps.Bond(site, site_b)
        e0s[f"{tuple(site)}, {tuple(site_b)}"] = Jzz * _cast_to_real(env_ctm.measure_nn(Sz, Sz, bond=v_bond))
        e0s[f"{tuple(site)}, {tuple(site_b)}"] += 0.5 * Jxy * _cast_to_real(env_ctm.measure_nn(Sp, Sm, bond=v_bond))
        e0s[f"{tuple(site)}, {tuple(site_b)}"] += 0.5 * Jxy * _cast_to_real(env_ctm.measure_nn(Sm, Sp, bond=v_bond))
            
        # anti-diagonal bond
        e0s[f"{tuple(site_b)}, {tuple(site_r)}"] = Jzz * _cast_to_real(env_ctm.measure_2x2(Sz, Sz, sites=[site_b, site_r]))
        e0s[f"{tuple(site_b)}, {tuple(site_r)}"] += 0.5 * Jxy * _cast_to_real(env_ctm.measure_2x2(Sp, Sm, sites=[site_b, site_r]))
        e0s[f"{tuple(site_b)}, {tuple(site_r)}"] += 0.5 * Jxy * _cast_to_real(env_ctm.measure_2x2(Sm, Sp, sites=[site_b, site_r]))
    e_gs = sum([v for v in e0s.values()]).numpy()
    
    ### Compute excited state energy
    lp = 3 # patch size
    bonds = [
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 0)),
    ]
    def _shift_coord(coord, dx, dy):
        return (coord[0] + dx, coord[1] + dy)

    def _convert_tensor(t_val, config):
        A = yastn.Tensor(config=config, s=(-1, 1, 1, -1, 1))
        A.set_block(val=t_val, Ds=t_val.shape)
        return A

    def compute_exci(env_ctm, *operators, sites_op, exci_bra, exci_ket, site_bra, site_ket, compute_grad=False):
        sites = sites_op + [site_bra, site_ket]
        xrange = (min(site[0] for site in sites), max(site[0] for site in sites) + 1)
        yrange = (min(site[1] for site in sites), max(site[1] for site in sites) + 1)
        exci_bra.requires_grad_(requires_grad=True)
        if exci_bra._data.grad is not None:
            exci_bra._data.grad.zero_()
        env_exci = fpeps.EnvExciSMA(env_ctm, xrange, yrange)
        bond_val_exci = env_exci.measure_exci(*operators, exci_bra=exci_bra, exci_ket=exci_ket, site_bra=site_bra, site_ket=site_ket, sites_op=sites_op)
        
        grad_dbra = None
        if compute_grad:
            grad_dbra = torch.autograd.grad(bond_val_exci, exci_bra.data, grad_outputs=torch.ones_like(bond_val_exci), retain_graph=True)[0]

        return bond_val_exci, grad_dbra

    es_exci = {bond: np.zeros((lp, lp, lp, lp, 3), dtype=np.complex128) for bond in bonds}
    ns_exci = np.zeros((lp, lp, 3), dtype=np.complex128)
    computed_norm = False
    for bond in bonds:
        for i_sl in range(3):
            cx, cy = tuple(psi.sites()[i_sl])
            shift_bond = [_shift_coord(site, cx, cy) for site in bond]
            min_x = cx - (lp - 1)//2
            max_x = cx + lp // 2
            min_y = cy - (lp - 1)//2
            max_y = cy + lp // 2
            patches = [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
            for (lx_b, ly_b), (lx_k, ly_k) in itertools.combinations_with_replacement(patches, 2):
                site_ket = (lx_k, ly_k)
                site_bra = (lx_b, ly_b)
                ket_np = exci_basis[psi.site2index(site_ket)].reshape(1,1,1,1,2)
                bra_np = exci_basis[psi.site2index(site_bra)].reshape(1,1,1,1,2)
                exci_ket = _convert_tensor(ket_np, config)
                exci_bra = _convert_tensor(bra_np, config)

                # compute excited energies
                if len(bond) == 1:
                    Hhz_val, dHhz_dbra = compute_exci(env_ctm, Sz, sites_op=shift_bond, exci_bra=exci_bra, exci_ket=exci_ket, site_bra=site_bra, site_ket=site_ket)
                    es_exci[bond][lx_b-min_x, ly_b-min_y, lx_k-min_x, ly_k-min_y, i_sl] = -hz * Hhz_val
                else:
                    H_zz_val, dHzz_dbra = compute_exci(env_ctm, Sz, Sz, sites_op=shift_bond, exci_bra=exci_bra, exci_ket=exci_ket, site_bra=site_bra, site_ket=site_ket)
                    H_xx_val, dHxx_dbra = compute_exci(env_ctm, Sx, Sx, sites_op=shift_bond, exci_bra=exci_bra, exci_ket=exci_ket, site_bra=site_bra, site_ket=site_ket)
                    H_yy_val, dHyy_dbra = compute_exci(env_ctm, Sy, Sy, sites_op=shift_bond, exci_bra=exci_bra, exci_ket=exci_ket, site_bra=site_bra, site_ket=site_ket)
                    # es_exci[bond][:, i_basis, lx_b, ly_b, lx_k, ly_k, i_sl] = exci_basis[state.site2index(site_bra)].T.conj() @ dHzz_dbra
                    # es_exci[bond][:, i_basis, lx_b, ly_b, lx_k, ly_k, i_sl] += exci_basis[state.site2index(site_bra)].T.conj() @ dHxx_dbra
                    # es_exci[bond][:, i_basis, lx_b, ly_b, lx_k, ly_k, i_sl] += exci_basis[state.site2index(site_bra)].T.conj() @ dHyy_dbra
                    es_exci[bond][lx_b-min_x, ly_b-min_y, lx_k-min_x, ly_k-min_y, i_sl] = Jzz * H_zz_val + Jxy * (H_xx_val + H_yy_val)

                # compute excited norms
                if not computed_norm and lx_b == min_x and ly_b == min_y:
                    site_c = (cx, cy)
                    bra_c_np = exci_basis[psi.site2index(site_c)].reshape(1,1,1,1,2)
                    exci_bra_c = _convert_tensor(bra_c_np, config)

                    N_val, dN_dbra = compute_exci(env_ctm, sites_op=[], exci_bra=exci_bra_c, exci_ket=exci_ket, site_bra=site_c, site_ket=site_ket)
                    # ns_exci[:, i_basis, lx_k, ly_k, i_sl] = exci_basis[state.site2index(site_bra)].T.conj() @ dN_dbra
                    ns_exci[lx_k-min_x, ly_k-min_y, i_sl] = N_val
        computed_norm = True


    # ground state expectation values
    print("Ground state values: ")
    print(f"<-hz.S^z(0, 0)>={e0s['(0, 0)']}", f"<-hz.S^z(0, 1)>={e0s['(0, 1)']}", f"<-hz.S^z(0, 2)>={e0s['(0, 2)']}")
    print(f"<J S(0,0).S(0,1)>={e0s['(0, 0), (0, 1)']}", f"<J S(0,1).S(0,2)>={e0s['(0, 1), (0, 2)']}", f"<J S(0,2).S(0,3)>={e0s['(0, 2), (0, 3)']}")
    print(f"<J S(0,0).S(1,0)>={e0s['(0, 0), (1, 0)']}", f"<J S(0,1).S(1,1)>={e0s['(0, 1), (1, 1)']}", f"<J S(0,2).S(1,2)>={e0s['(0, 2), (1, 2)']}")

    # excited state expectation values
    print(f"<B(0, 0)|-hz.S^z(0, 0)|B(0, 0)>={es_exci[((0, 0),)][1, 1, 1, 1, 0]:.5f}")
    print(f"<B(0, 1)|-hz.S^z(0, 1)|B(0, 1)>={es_exci[((0, 0),)][1, 1, 1, 1, 1]:.5f}")
    print(f"<B(0, 2)|-hz.S^z(0, 2)|B(0, 2)>={es_exci[((0, 0),)][1, 1, 1, 1, 2]:.5f}")

    print(f"<B(0, 0)|J S(0,0).S(0,1)|B(0, 0)>={es_exci[((0, 0), (0, 1))][1, 1, 1, 1, 0]:.5f}")
    print(f"<B(0, 1)|J S(0,1).S(0,2)|B(0, 1)>={es_exci[((0, 0), (0, 1))][1, 1, 1, 1, 1]:.5f}")
    print(f"<B(0, 2)|J S(0,2).S(0,3)|B(0, 2)>={es_exci[((0, 0), (0, 1))][1, 1, 1, 1, 2]:.5f}")

    # Momentum space
    # compute effective H and effective N from real space components
    def lin_ex(s, e, n):
        return np.linspace(s, e, n)[:-1]

    def phase(kx, ky, x0, x1, y0, y1):
        return np.exp(-1j * (kx * (x1-x0) + ky * (y1 - y0)))

    def _3sublat(coord):
        x = coord[0]
        y = coord[1]
        return (y-x) % 3

    from math import pi
    from math import sqrt

    kxs = np.concatenate(
        [
            lin_ex(0, 2 * pi / sqrt(3), 10),
            lin_ex(2 * pi / sqrt(3), pi / sqrt(3), 5),
            np.linspace(pi / sqrt(3), 0, 8),
        ]
    )
    kys = np.concatenate(
        [
            lin_ex(0, 2 * pi / 3, 10),
            lin_ex(2 * pi / 3, pi, 5),
            np.linspace(pi, 0, 8),
        ]
    )
    distort_trgl2sqr = np.array([[np.sqrt(3)/2, 1/2], [np.sqrt(3)/2, -1/2]])

    kxs_ = np.zeros(kxs.shape)
    kys_ = np.zeros(kys.shape)
    for i in range(len(kxs)):
        new_k = distort_trgl2sqr@np.array([kxs[i], kys[i]])
        kxs_[i] = new_k[0]
        kys_[i] = new_k[1]

    k_labels = {
        "ticks": [0, 9, 13, 20],
        "labels": [
            r"$\Gamma(0,0)$",
            r"$K(\frac{2\pi}{\sqrt{3}},\frac{2\pi}{3})$",
            r"$M(\frac{\pi}{\sqrt{3}},\pi)$",
            r"$\Gamma(0,0)$",
        ],
    }

    # compute each k 
    evs = np.zeros((len(kxs), 3)) # only 3 modes
    for ik, kx in enumerate(kxs_):
        ky = kys_[ik]

        h_eff = np.zeros((3, 3), dtype=np.complex128) 
        n_eff = np.zeros((3, 3), dtype=np.complex128)
        computed_norm = False
        for bond in bonds:
            for i_sl in range(3):
                cx, cy = tuple(psi.sites()[i_sl])
                shift_bond = [_shift_coord(site, cx, cy) for site in bond]
                min_x = cx - (lp - 1)//2
                max_x = cx + lp // 2
                min_y = cy - (lp - 1)//2
                max_y = cy + lp // 2
                patches = [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
                for (lx_b, ly_b), (lx_k, ly_k) in itertools.combinations_with_replacement(patches, 2):
                    sl_ket = _3sublat((lx_k, ly_k))
                    sl_bra = _3sublat((lx_b, ly_b))
                    tmp_h = phase(ky, kx, lx_b, lx_k, ly_b, ly_k) * es_exci[bond][lx_b-min_x, ly_b-min_y, lx_k-min_x, ly_k-min_y, i_sl]
                    h_eff[sl_bra, sl_ket] += tmp_h
                    if lx_b != lx_k or ly_b != ly_k:
                        h_eff[sl_ket, sl_bra] += tmp_h.conj()

                    if not computed_norm and lx_b == min_x and ly_b == min_y:
                        sl_c = i_sl
                        tmp_n = phase(ky, kx, cx, lx_k, cy, ly_k) * ns_exci[lx_k-min_x, ly_k-min_y, i_sl]
                        n_eff[sl_c, sl_ket] += tmp_n / (2)
                        n_eff[sl_ket, sl_c] += tmp_n.conj() / (2)
            computed_norm = True
        
        # subtract ground state energy
        h_eff = h_eff - e_gs * n_eff * 3

        # solve the generalized eigenvalue problems
        ev_N, P = np.linalg.eig(n_eff)
        idx = ev_N.real.argsort()[::-1]
        ev_N = ev_N[idx]
        # selected = (ev_N / ev_N.max()) > 1e-3
        P = P[:, idx]

        ev, vectors = eig(h_eff, n_eff)
        ixs = np.argsort(ev)
        ev = ev[ixs]
        vectors = vectors[:, ixs]
        evs[ik, :] = ev

    print(evs.T)
    
    # Reference values from CTM summation (3, 21) 
    # [[2.84500011, 2.8618264 , 2.90943498, 2.97813477, 3.0430191 , 3.0430191 , 2.97813477, 2.90943498, 2.8618264 , 2.84500011,
    #   2.86621718, 2.92521297, 3.0052694 , 3.05500011, 3.07579664, 3.13406725, 3.1017295 , 3.00827071, 2.92406725, 2.86579664, 2.84500011],
    #  [3.4750001 , 3.45817381, 3.41056523, 3.34186544, 3.27698111, 3.27698111, 3.34186544, 3.41056523, 3.45817381, 3.4750001 ,
    #   3.45378303, 3.39478724, 3.31473081, 3.2650001 , 3.24420357, 3.18593296, 3.21827071, 3.3117295 , 3.39593296, 3.45420357, 3.4750001 ],
    #  [5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 
    #   5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011, 5.78000011]]


if __name__ == '__main__':
    # pytest.main([__file__, "-vs", "--durations=0", "--long_tests"])
    yastn_config = {"backend": "torch",
                    "fermionic": False,
                    "default_dtype": "complex128",
                    "default_device": 'cpu',
                    "tensordot_policy": "no_fusion"
                    }

    # yastn.make_config(
    #     backend='torch',
    #     sym='dense',
    #     fermionic=False,
    #     default_device='cpu',
    #     default_dtype="complex128",
    #     tensordot_policy="no_fusion",
    # )
    test_exci_sma_UUD(yastn_config)