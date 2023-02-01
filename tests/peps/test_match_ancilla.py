import logging
import pytest
import yast
import yast.tn.peps as peps
from yast.tn.peps import initialize_Neel_spinfull, initialize_peps_purification, initialize_vacuum
from yast.tn.peps.operators.gates import gate_local_Hubbard, match_ancilla_1s, match_ancilla_2s, gates_hopping

try:
    from .configs import config_U1_R_fermionic, config_U1xU1_R_fermionic
except ImportError:
    from configs import config_U1_R_fermionic, config_U1xU1_R_fermionic

def test_match_ancilla():
    """ initialize vacuum state and check the functions match_ancilla_1s and match_ancilla_2s """

    net = peps.Lattice(lattice='rectangle',dims=(3,3),boundary='finite')  
    opt = yast.operators.SpinfulFermions(sym='U1xU1xZ2',backend=config_U1xU1_R_fermionic.backend,default_device=config_U1xU1_R_fermionic.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')
    gamma = initialize_vacuum(fid, net)   # initializing vacuum state; has no ancilla (so 5 legs when unfused : t l b r s)

    for ms in gamma.sites():
        assert(gamma[ms].unfuse_legs(axes=(0, 1, 2)).ndim==5)
 
    cdag_up = match_ancilla_1s(fcdag_up, gamma[0,0]) # creation operator of up polarization; adds ancilla index when operated on vacuum
    cdag_dn = match_ancilla_1s(fcdag_dn, gamma[0,0]) # creation operator of down polarization; adds ancilla index when operated on vacuum

    ############ next we add creation operators of alternate (up and down) polarizations to alternate lattice sites
    ############ in a checkerboard fashion to create a Néel state; this also adds ancilla to each lattice site to preserve parity

    m = list([cdag_up, cdag_dn])
    i=0
    for x in range(net.Nx):
        for y in range(net.Ny)[::2]:
            gamma[x, y] = yast.tensordot(gamma[x,y], m[i], axes=(2,1))  # spin-up polarization
        for y in range(net.Ny)[1::2]:
            gamma[x, y] = yast.tensordot(gamma[x,y], m[(i+1)%2], axes=(2,1)) # spin-down polarization
        i = (i+1)%2

    ######## check if all sites have ancilla leg by asserting their unfused forms have dimension 6
    for ms in gamma.sites():
        assert(gamma[ms].unfuse_legs(axes=(0, 1, 2)).ndim==6)

    ###### now we check what happens if we apply an annihilation operator of the same polarization 
    #######  as is present at say the first site (1,1). It should create a hole there.
    G_before = gamma[0,0]
    an_loc_up = match_ancilla_1s(fc_up, G_before) # creating a local annihilation operator for application at the first site
    G_after = yast.tensordot(G_before, an_loc_up, axes=(2, 1))
    assert(list(G_before.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(1,0,1)]) # checks if the spin leg has charge corresponding to spin-up
    assert(list(G_after.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(0,0,0)]) # checks if the spin leg has charge corresponding to hole


    ###### if we apply a an annihilation operator of the opposite polarization 
    #######  at the same site, the site should vanish.
    G_before = gamma[0,0]
    an_loc_dn = match_ancilla_1s(fc_dn, G_before) # creating a local annihilation operator for application at the first site
    G_after = yast.tensordot(G_before, an_loc_dn, axes=(2, 1))
    assert(list(G_before.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(1,0,1)]) # checks if the spin leg has charge corresponding to spin-up
    assert(list(G_after.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[]) # checks if the spin leg has no charge

    ###### if we apply a creation operator of the opposite polarization 
    #######  at the same site, then there would be double occupancy
    G_before = gamma[0,0]
    cr_loc_dn = match_ancilla_1s(fcdag_dn, G_before) # creating a local annihilation operator for application at the first site
    G_after = yast.tensordot(G_before, cr_loc_dn, axes=(2, 1))
    assert(list(G_before.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(1,0,1)]) # checks if the spin leg has charge corresponding to spin-up
    assert(list(G_after.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(1,1,0)]) # checks if the spin leg has double occupancy
    # next we transfer the spin-up electron from central site (2,2) to the site to the right (2,3)
    # we acheieve this with help of 2-site gate c_up(2,2)cdag_up(3,3) 

    c1 = fc_up.add_leg(s=1).swap_gate(axes=(1, 2))
    c2dag = fcdag_up.add_leg(s=-1)
    cc =  yast.ncon([c1, c2dag], ((-0, -1, 1) , (-2, -3, 1)))
    A = gamma[1,1]
    B = gamma[1,2]
    U, S, V = yast.svd_with_truncation(cc, axes = ((0, 1), (2, 3)), sU = -1, tol = 1e-15, Vaxis=2)
    S = S.sqrt()
    GA = S.broadcast(U, axis=2)
    GB = S.broadcast(V, axis=2)
    ### test for match_ancilla_2s
    GA_an = match_ancilla_2s(GA, A, dir='l')  
    GB_an = match_ancilla_2s(GB, B, dir='r')
    
    int_A = yast.tensordot(A, GA_an, axes=(2, 1)) # [t l] [b r] s c
    int_B = yast.tensordot(B, GB_an, axes=(2, 1)) # [t l] [b r] s c

    assert(int_A.unfuse_legs(axes=2).get_leg_structure(axis=2).keys()==([0,0,0])) # checking if the central site has up-spin fermion trasferred
    assert(int_B.unfuse_legs(axes=2).get_leg_structure(axis=2).keys()==([1,1,0])) # to the site to the right


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    test_match_ancilla()

