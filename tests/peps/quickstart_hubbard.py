""" Test the expectation values of spin 1/2 fermions with analytical values of fermi sea """
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps

opt = yastn.operators.SpinfulFermions(sym='U1xU1xZ2')

mu = 0
t = 1
U=0
dbeta = 0.01
geometry = fpeps.SquareLattice(lattice = 'checkerboard')
step = "one-step"
tr_mode = 'optimal'
Nx=2
Ny=2
D = 12
opts_svd = {'D_total': D, 'tol_block': 1e-15}

gate_hopping_u = fpeps.operators.gates.gates_hopping(dbeta*0.5, t, opt.I(), opt.c(spin='u'), opt.cp(spin='u'))
gate_hopping_d = fpeps.operators.gates.gates_hopping(dbeta*0.5, t, opt.I(), opt.c(spin='d'), opt.cp(spin='d'))
gate_loc_Hubbard = fpeps.operators.gates.gate_local_Hubbard(dbeta*0.5, mu, mu, U, opt.I(), opt.n(spin='u'), opt.n(spin='d'))
gates = fpeps.evolution.gates_homogeneous(geometry, nn_gates=[gate_hopping_u, gate_hopping_d], loc_gates=gate_loc_Hubbard)

psi = fpeps.product_peps(geometry=geometry, vectors = opt.I() / 2)

beta = 0.1
steps = int((beta / 2) / dbeta)

opts_evol = {"D_total": D, "gradual_truncation": "two-step", "initialization": "EAT"} # initialization is "EAT" or "normal"
# contatins the state psi and information how to calculate metric tensor for truncation; here we use nearest tensor clusters (NTU) environment
env = fpeps.EnvCluster(psi, depth=1)
opts_svd_ntu = {'D_total': D, 'tol_block': 1e-15}


for step in range(steps):
    beta = (step + 1) * dbeta
    print("beta = %0.3f" % beta)
    env = fpeps.evolution.evolution_step_(env, gates, opts_evol, opts_svd)
    # psi is updated in place inside the environment




# convergence criteria for CTM based on total energy
chi = 80 # environmental bond dimension
tol = 1e-10 # truncation of singular values of CTM projectors
max_sweeps=50 
tol_exp = 1e-7   # difference of some observable must be lower than tolernace

ops = {'cdagc_up': {'l': opt.cp(spin='u'), 'r': opt.c(spin='u')},
       'ccdag_up': {'l': opt.c(spin='u'), 'r': opt.cp(spin='u')},
       'cdagc_dn': {'l': opt.cp(spin='d'), 'r': opt.c(spin='d')},
       'ccdag_dn': {'l': opt.c(spin='d'), 'r':  opt.cp(spin='d')}}

cf_energy_old = 0
opts_svd_ctm = {'D_total': chi, 'tol': tol}

for step in fpeps.ctm.ctmrg(psi, max_sweeps, iterator_step=2, AAb_mode=0, opts_svd=opts_svd_ctm):
    
    assert step.sweeps % 2 == 0 # stop every 2nd step as iteration_step=2

    obs_hor, obs_ver =  fpeps.ctm.nn_exp_dict(psi, step.env, ops)

    cdagc_up = (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) + 
            sum(abs(val) for val in obs_ver.get('cdagc_up').values()))

    ccdag_up = (sum(abs(val) for val in obs_hor.get('ccdag_up').values()) + 
            sum(abs(val) for val in obs_ver.get('ccdag_up').values()))

    cdagc_dn = (sum(abs(val) for val in obs_hor.get('cdagc_dn').values()) + 
            sum(abs(val) for val in obs_ver.get('cdagc_dn').values()))

    ccdag_dn = (sum(abs(val) for val in obs_hor.get('ccdag_dn').values()) + 
            sum(abs(val) for val in obs_ver.get('ccdag_dn').values()))


    cf_energy = - (cdagc_up + ccdag_up +cdagc_dn + ccdag_dn) / 16

    print("energy : ", cf_energy)
    if abs(cf_energy - cf_energy_old) < tol_exp:
        break # here break if the relative differnece is below tolerance
    cf_energy_old = cf_energy

print("Energy per bond per correlator:", cf_energy)


nn_exact = 0.02481459 # analytical nn fermionic correlator at beta = 0.1 for 2D infinite lattice with checkerboard ansatz



