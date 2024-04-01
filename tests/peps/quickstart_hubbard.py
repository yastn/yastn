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
gates = fpeps.gates_homogeneous(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_Hubbard)

psi = fpeps.product_peps(geometry=geometry, vectors = ops.I())

env = fpeps.EnvNTU(psi, which='NN')
# contatins the state psi and information how to calculate metric tensor for truncation;
# here we use nearest tensor clusters (NTU) environment

opts_svd = {'D_total': D, 'tol_block': 1e-15}
steps = np.rint((beta / 2) / dbeta).astype(int)
for step in range(steps):
    print(f"beta = {(step + 1) * dbeta}" )
    out = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="EAT")

# convergence criteria for CTM based on total energy
chi = 80  # environmental bond dimension
tol = 1e-10  # truncation of singular values of CTM projectors
max_sweeps = 50
tol_exp = 1e-7  # difference of some observable must be lower than tolernace

ops = {'cdagc_up': {'l': ops.cp(spin='u'), 'r': ops.c(spin='u')},
       'ccdag_up': {'l': ops.c(spin='u'),  'r': ops.cp(spin='u')},
       'cdagc_dn': {'l': ops.cp(spin='d'), 'r': ops.c(spin='d')},
       'ccdag_dn': {'l': ops.c(spin='d'),  'r': ops.cp(spin='d')}}

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

    cf_energy = -(cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) / 16

    print("Energy : ", cf_energy)
    if abs(cf_energy - cf_energy_old) < tol_exp:
        break # here break if the relative differnece is below tolerance
    cf_energy_old = cf_energy

print("Energy per bond:", cf_energy)
# analytical nn fermionic correlator at beta = 0.1 for 2D infinite lattice with checkerboard ansatz
nn_exact = 0.02481459
print("Exact nn hopping:", nn_exact)
