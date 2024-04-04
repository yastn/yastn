""" Test the expectation values of spin 1/2 fermions with analytical values of fermi sea """
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps

geometry = fpeps.CheckerboardLattice()

mu = 0
t = 1
U = 0
beta = 0.1

dbeta = 0.01
D = 12

ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2')
g_hop_u = fpeps.gates.gate_nn_hopping(t, dbeta / 2, ops.I(), ops.c(spin='u'), ops.cp(spin='u'))
g_hop_d = fpeps.gates.gate_nn_hopping(t, dbeta / 2, ops.I(), ops.c(spin='d'), ops.cp(spin='d'))
g_Hubbard = fpeps.gates.gate_local_Coulomb(dbeta / 2, mu, mu, U, ops.I(), ops.n(spin='u'), ops.n(spin='d'))
gates = fpeps.gates.distribute(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_Hubbard)

psi = fpeps.product_peps(geometry=geometry, vectors = ops.I())

env = fpeps.EnvNTU(psi, which='NN')
# contatins the state psi and information how to calculate metric tensor for truncation;
# here we use nearest tensor clusters (NTU) environment

opts_svd = {'D_total': D, 'tol_block': 1e-15}
steps = round((beta / 2) / dbeta)
for step in range(steps):
    print(f"beta = {(step + 1) * dbeta}" )
    out = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="EAT")



# convergence criteria for CTM based on total energy
energy_old, tol_exp = 0, 1e-7

env = fpeps.EnvCTM(psi)
opts_svd_ctm = {'D_total': 40, 'tol': 1e-10}

for _ in range(50):
    env.update_(opts_svd=opts_svd_ctm)
    cdagc_up = env.measure_nn(ops.cp(spin='u'), ops.c(spin='u'))
    cdagc_dn = env.measure_nn(ops.cp(spin='d'), ops.c(spin='d'))

    cdagc_up.values()
    energy =  - sum(cdagc_up.values()) - sum(cdagc_dn.values())

    print("Energy: ", energy)
    if abs(energy - energy_old) < tol_exp:
        break
    energy_old = energy



print("Energy per bond:", energy)
# analytical nn fermionic correlator at beta = 0.1 for 2D infinite lattice with checkerboard ansatz
nn_exact = 0.02481459
print("Exact nn hopping:", nn_exact)
