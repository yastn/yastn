import numpy as np
import yamps as yamps
import settings_transport_vectorization_full as settings
from settings_transport_vectorization_full import backend
from settings_transport_vectorization_full import Tensor
from settings_transport_vectorization_full import main
import yamps.mps as mps

print('\nMake full Lindblad evolution')
dtype = 'complex128'  # 'float64'
sgn = 1j # imaginary time evolution for sgn==1j
tmax = 2. * sgn
dt = .125 * sgn
Dt = dt

tol = 1e-6

ID = 0
ver = False

Lval = [4]
D = 16
w0 = 1
v = .5
v0 = 0
wS = +1
io = [.5]

mu = 0.5
dV = 0
fermionic = False

gval = [.1]
Tval = [10]

ordered = True
dmrg = False
version = '1site' 
for im, L in enumerate(Lval):
    for itemp, temp in enumerate(Tval):
        LS = 1
        N = 2 * L + LS
        tempL = temp
        tempR = temp

        opts = {'tol': tol, 'D_block': D}
        for ig, gamma in enumerate(gval):
            # Prepare Hamiltonian, Kraus operator and state
            iterations = 100
            
            #psi = main.random_state(N, D)
            LL, LSR, ww, vk, tp, LdagL = main.H_M_psi_1im_mixed(L=L , v=v, mu=mu, w0=w0, wS=wS, gamma=gamma, ordered=ordered, dV=dV, tempL=tempL, tempR=tempR, dt=dt, AdagA=True)
            psi = main.thermal_state(L, io, ww, tp)
            curr_LS = main.current(LSR, vk, cut = 'LS')
            curr_SR = main.current(LSR, vk, cut = 'SR')
            occ_NS = main.occupancy(N, L)
            
            if ordered:
                name = '21-sorted_ID-' + str(ID) + '_purification_L_' + str(L) + '_temp_' + str(tempL) + '_mu_' + str(mu) + '_v_' + str(v) + '_dV_' + str(dV) + '_gamma_' + str(gamma) + '_D_' + str(D) + '_DK_' + '_dt_' + str(dt) + '.txt'
            else:
                name = '21-unsorted_ID-' + str(ID) + '_purification_L_' + str(L) + '_temp_' + str(tempL) + '_mu_' + str(mu) + '_v_' + str(v) + '_dV_' + str(dV) + '_gamma_' + str(gamma) + '_D_' + str(D) + '_DK_' + '_dt_' + str(dt) + '.txt'
            
            # expactation value of Lindbladian sqaured: L^\dagger L
            LdagL = LdagL#main.stack_MPOs(LL, LL, [1,1], [1,0])
            
            psi.canonize_left(normalize=False)
            JLS = yamps.Envs([curr_LS, psi])    
            JSR = yamps.Envs([curr_SR, psi])    
            NS = yamps.Envs([occ_NS, psi])    
            
            if dmrg:
                print('DMRG')
                out = yamps.dmrg_OBC(psi=psi, H=LdagL, cutoff_sweep=iterations, cutoff_dE=-1, k=10, measure_O=None, hermitian=True, tol=tol, dtype=dtype, version='2site', opts=opts)
            else:
                print('TDVP')
                out=[None]
                for iter in range(iterations):
                    Dt = dt
                    env, E, dE, out = mps.tdvp.tdvp_OBC(psi=psi, tmax=Dt, dt=dt, H=LdagL, measure_O=None, version=version, opts_svd=opts_svd)
                    JLS.setup(direction=-1)
                    JSR.setup(direction=-1)
                    NS.setup(direction=-1)
                    print('Current - LS:', 2*np.pi*JLS.F[(0,-1)].to_number().real,' SR:', 2*np.pi*JSR.F[(0,-1)].to_number().real,' NS:', NS.F[(0,-1)].to_number().real)
            
            
            break