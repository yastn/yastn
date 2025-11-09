""" Test the excited state expectation values for spin-1/2 triangular lattice systems. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import TriangularLattice
import itertools
print(yastn.tn.fpeps.__file__)
import numpy as np

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

    lp = 3 # patch size
    bonds = [
        ((0, 0),),
        ((0, 0), (0, 1)), 
        ((0, 0), (1, 0)), 
        # ((0, 1), (1, 0)), 
    ]
    def _shift_coord(coord, dx, dy):
        return (coord[0] + dx, coord[1] + dy)
    
    def _convert_tensor(t_val, config):
        A = yastn.Tensor(config=config, s=(-1, 1, 1, -1, 1))
        A.set_block(val=t_val, Ds=t_val.shape)
        return A
    
    def compute_exci(env_ctm, *operators, sites_op, exci_bra, exci_ket, site_bra, site_ket):
        sites = sites_op + [site_bra, site_ket]
        xrange = (min(site[0] for site in sites), max(site[0] for site in sites) + 1)
        yrange = (min(site[1] for site in sites), max(site[1] for site in sites) + 1)    
        # exci_bra.requires_grad_(requires_grad=True)
        # if exci_bra._data.grad is not None:
        #     exci_bra._data.grad.zero_()
        env_exci = fpeps.EnvExciSMA(env_ctm, xrange, yrange)
        # env_exci = yastn.tn.fpeps.envs._env_exci_sma.EnvExciSMA(env_ctm, xrange, yrange)
        bond_val_exci = env_exci.measure_exci(*operators, exci_bra=exci_bra, exci_ket=exci_ket, site_bra=site_bra, site_ket=site_ket, sites_op=sites_op)
        return bond_val_exci

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
                    H_val = -hz * compute_exci(env_ctm, Sz, sites_op=shift_bond, exci_bra=exci_bra, exci_ket=exci_ket, site_bra=site_bra, site_ket=site_ket)                    
                    es_exci[bond][lx_b-min_x, ly_b-min_y, lx_k-min_x, ly_k-min_y, i_sl] = H_val
                else:
                    H_val = Jzz * compute_exci(env_ctm, Sz, Sz, sites_op=shift_bond, exci_bra=exci_bra, exci_ket=exci_ket, site_bra=site_bra, site_ket=site_ket)
                    # es_exci[bond][:, i_basis, lx_b, ly_b, lx_k, ly_k, i_sl] = exci_basis[state.site2index(site_bra)].T @ dH_dbra
                    es_exci[bond][lx_b-min_x, ly_b-min_y, lx_k-min_x, ly_k-min_y, i_sl] = H_val

                # compute excited norms
                if not computed_norm and lx_b == min_x and ly_b == min_y:
                    site_c = (cx, cy)
                    bra_c_np = exci_basis[psi.site2index(site_c)].reshape(1,1,1,1,2)
                    exci_bra_c = _convert_tensor(bra_c_np, config)

                    N_val = compute_exci(env_ctm, sites_op=[], exci_bra=exci_bra_c, exci_ket=exci_ket, site_bra=site_c, site_ket=site_ket)
                    # ns_exci[:, i_basis, lx_k, ly_k, i_sl] = exci_basis[state.site2index(site_bra)].T @ dN_dbra                                
                    ns_exci[lx_k-min_x, ly_k-min_y, i_sl] = N_val
        computed_norm = True


    # ground state expectation values
    print("Ground state values: ")
    print("<-hz.S^z(0, 0)>=-1.54", "<-hz.S^z(0, 1)>=-1.54", "<-hz.S^z(0, 2)>=+1.54")
    print("<Jzz.Sz(0,0).Sz(0,1)>=0.754", "<Jzz.Sz(0,1).Sz(0,2)>=-0.754", "<Jzz.Sz(0,2).Sz(0,3)>=-0.754")
    print("<Jzz.Sz(0,0).Sz(1,0)>=-0.754", "<Jzz.Sz(0,1).Sz(1,1)>=0.754", "<Jzz.Sz(0,2).Sz(1,2)>=-0.754")

    # excited state expectation values
    print(f"<B(0, 0)|-hz.S^z(0, 0)|B(0, 0)>={es_exci[(0, 0)][1, 1, 1, 1, 0]:.5f}")
    print(f"<B(0, 1)|-hz.S^z(0, 1)|B(0, 1)>={es_exci[(0, 0)][1, 1, 1, 1, 1]:.5f}")
    print(f"<B(0, 2)|-hz.S^z(0, 2)|B(0, 2)>={es_exci[(0, 0)][1, 1, 1, 1, 2]:.5f}")

    print(f"<B(0, 0)|Jzz.Sz(0,0).Sz(0,1)|B(0, 0)>={es_exci[((0, 0), (0, 1))][1, 1, 1, 1, 0]:.5f}")
    print(f"<B(0, 1)|Jzz.Sz(0,1).Sz(0,2)|B(0, 1)>={es_exci[((0, 0), (0, 1))][1, 1, 1, 1, 1]:.5f}")
    print(f"<B(0, 2)|Jzz.Sz(0,2).Sz(0,3)|B(0, 2)>={es_exci[((0, 0), (0, 1))][1, 1, 1, 1, 2]:.5f}")

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