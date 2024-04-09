""" Test the expectation values of spin 1/2 fermions with analytical values of fermi sea """
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps

geometry = fpeps.CheckerboardLattice()

mu = 0
t_up = 1
t_dn = 1
mu_up = 0
mu_dn = 0
beta = 0.2
U = 8

dbeta = 0.01
D = 16

ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2')
fid = ops.I()
fc_up, fc_dn, fcdag_up, fcdag_dn = ops.c(spin='u'), ops.c(spin='d'), ops.cp(spin='u'), ops.cp(spin='d')
n_up, n_dn =  ops.n(spin='u'), ops.n(spin='d')
n_int = n_up @ n_dn
g_hop_u = fpeps.gates.gate_nn_hopping(t_up, dbeta / 2, fid, fc_up, fcdag_up)
g_hop_d = fpeps.gates.gate_nn_hopping(t_dn, dbeta / 2, fid, fc_dn, fcdag_dn)
g_loc = fpeps.gates.gate_local_Coulomb(mu_up, mu_dn, U, dbeta / 2, fid, n_up, n_dn)
gates = fpeps.gates.distribute(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_loc)

psi = fpeps.product_peps(geometry=geometry, vectors = ops.I())
env = fpeps.EnvNTU(psi, which='NN')
# contatins the state psi and information how to calculate metric tensor for truncation;
# here we use nearest tensor clusters (NTU) environment

opts_svd = {'D_total': D, 'tol_block': 1e-15}
steps = np.rint((beta / 2) / dbeta).astype(int)
for step in range(steps):
    print(f"beta = {(step + 1) * dbeta}" )
    evolution_results = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="EAT")
    print(f"Error after optimization for all gates: {evolution_results.truncation_error}")

# convergence criteria for CTM based on total energy
chi = 80  # environmental bond dimension
tol = 1e-10  # truncation of singular values of CTM projectors
max_sweeps = 50
tol_exp = 1e-7  # difference of some observable must be lower than tolernace


energy_old, tol_exp = 0, 1e-7

opts_svd_ctm = {'D_total': 40, 'tol': 1e-10}

env_ctm = fpeps.EnvCTM(psi)

for i in range(50):
    env_ctm.update_(opts_svd=opts_svd_ctm)  # single CMTRG sweep

    # calculate expectation values
    d_oc = env_ctm.measure_1site(n_int)
    cdagc_up = env_ctm.measure_nn(fcdag_up, fc_up)  # calculate for all unique bonds
    cdagc_dn = env_ctm.measure_nn(fcdag_dn, fc_dn)  # -> {bond: value}
    PEn = U * np.mean([*d_oc.values()]) 
    KEn = -2 * np.sum([*cdagc_up.values(), *cdagc_dn.values()])

    energy = PEn + KEn
    print(f"Energy after iteration {i+1}: ", energy)
    if abs(energy - energy_old) < tol_exp:
        break
    energy_old = energy

print("Final Energy:", energy)
