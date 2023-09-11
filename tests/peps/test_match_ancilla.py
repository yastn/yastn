import logging
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.operators.gates import match_ancilla_1s, match_ancilla_2s

try:
    from .configs import config_U1xU1_R_fermionic
except ImportError:
    from configs import config_U1xU1_R_fermionic

def test_match_ancilla():
    """ initialize vacuum state and check the functions match_ancilla_1s and match_ancilla_2s """

    net = fpeps.Lattice(lattice='square',dims=(3,3),boundary='obc')  
    opt = yastn.operators.SpinfulFermions(sym='U1xU1xZ2',backend=config_U1xU1_R_fermionic.backend,default_device=config_U1xU1_R_fermionic.default_device)
    fid, fc_up, fc_dn, fcdag_up, fcdag_dn = opt.I(), opt.c(spin='u'), opt.c(spin='d'), opt.cp(spin='u'), opt.cp(spin='d')

    # initialize vacuum state peps with lattice specifications
    A = yastn.Leg(fid.config, t= ((0,0,0),), D=((1),))
    A = yastn.ones(fid.config, legs=[A])
    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)

    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    peps = fpeps.Lattice(net.lattice, net.dims, net.boundary)

    for ms in net.sites():
        peps[ms] = A
    ################################################################

    for ms in peps.sites():
        assert(peps[ms].unfuse_legs(axes=(0, 1, 2)).ndim==5)  # asserting each peps tensor has 5 legs in its unfused form
 
    cdag_up = match_ancilla_1s(fcdag_up, peps[0,0]) # creation operator of up polarization; adds ancilla index when operated on vacuum
    cdag_dn = match_ancilla_1s(fcdag_dn, peps[0,0]) # creation operator of down polarization; adds ancilla index when operated on vacuum

    ############ next we add creation operators of alternate (up and down) polarizations to alternate lattice sites
    ############ in a checkerboard fashion to create a Néel state; this also adds ancilla to each lattice site to preserve parity

    m = list([cdag_up, cdag_dn])
    i=0
    for x in range(net.Nx):
        for y in range(net.Ny)[::2]:
            peps[x, y] = yastn.tensordot(peps[x,y], m[i], axes=(2,1))  # spin-up polarization
        for y in range(net.Ny)[1::2]:
            peps[x, y] = yastn.tensordot(peps[x,y], m[(i+1)%2], axes=(2,1)) # spin-down polarization
        i = (i+1)%2

    ######## check if all sites have ancilla leg by asserting their unfused forms have dimension 6
    for ms in peps.sites():
        assert(peps[ms].unfuse_legs(axes=(0, 1, 2)).ndim==6)

    ###### now we check what happens if we apply an annihilation operator of the same polarization 
    #######  as is present at say the first site (1,1). It should create a hole there.
    G_before = peps[0,0]
    an_loc_up = match_ancilla_1s(fc_up, G_before) # creating a local annihilation operator for application at the first site
    G_after = yastn.tensordot(G_before, an_loc_up, axes=(2, 1))
    assert(list(G_before.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(1,0,1)]) # checks if the spin leg has charge corresponding to spin-up
    assert(list(G_after.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(0,0,0)]) # checks if the spin leg has charge corresponding to hole


    ###### if we apply a an annihilation operator of the opposite polarization 
    #######  at the same site, the site should vanish.
    G_before = peps[0,0]
    an_loc_dn = match_ancilla_1s(fc_dn, G_before) # creating a local annihilation operator for application at the first site
    G_after = yastn.tensordot(G_before, an_loc_dn, axes=(2, 1))
    assert(list(G_before.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(1,0,1)]) # checks if the spin leg has charge corresponding to spin-up
    assert(list(G_after.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[]) # checks if the spin leg has no charge

    ###### if we apply a creation operator of the opposite polarization 
    #######  at the same site, then there would be double occupancy
    G_before = peps[0,0]
    cr_loc_dn = match_ancilla_1s(fcdag_dn, G_before) # creating a local annihilation operator for application at the first site
    G_after = yastn.tensordot(G_before, cr_loc_dn, axes=(2, 1))
    assert(list(G_before.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(1,0,1)]) # checks if the spin leg has charge corresponding to spin-up
    assert(list(G_after.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(1,1,0)]) # checks if the spin leg has double occupancy
    # next we transfer the spin-up electron from central site (2,2) to the site to the right (2,3)
    # we acheieve this with help of 2-site gate c_up(2,2)cdag_up(3,3) 

    c1 = fc_up.add_leg(s=1).swap_gate(axes=(1, 2))
    c2dag = fcdag_up.add_leg(s=-1)
    cc =  yastn.ncon([c1, c2dag], ((-0, -1, 1) , (-2, -3, 1)))
    A = peps[1,1]
    B = peps[1,2]
    U, S, V = yastn.svd_with_truncation(cc, axes = ((0, 1), (2, 3)), sU = -1, tol = 1e-15, Vaxis=2)
    S = S.sqrt()
    GA = S.broadcast(U, axes=2)
    GB = S.broadcast(V, axes=2)
    ### test for match_ancilla_2s
    GA_an = match_ancilla_2s(GA, A, dir='l')  
    GB_an = match_ancilla_2s(GB, B, dir='r')
    
    int_A = yastn.tensordot(A, GA_an, axes=(2, 1)) # [t l] [b r] s c
    int_B = yastn.tensordot(B, GB_an, axes=(2, 1)) # [t l] [b r] s c
    assert(list(int_A.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(0,0,0)]) # checking if the central site has up-spin fermion trasferred
    assert(list(int_B.unfuse_legs(axes=2).get_leg_structure(axis=2).keys())==[(1,1,0)]) # to the site to the right


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    test_match_ancilla()

