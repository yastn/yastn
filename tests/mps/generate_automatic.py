
import yamps

def mpo_occupation(config, N):
    gen = yamps.GenerateOpEnv(N, config=config)
    gen.use_default()
    H_str = "\sum_{j=0}^{"+str(N-1)+"} cp_{j}.c_{j}"
    H = gen.latex2yamps(H_str)
    return H

def mpo_XX_model(config, N, t, mu):
    gen = yamps.GenerateOpEnv(N, config=config)
    gen.use_default()
    parameters = {"t": t, "mu": mu}
    H_str = "\sum_{j=0}^{"+str(N-1)+"} mu*cp_{j}.c_{j} + \sum_{j=0}^{"+str(N-2)+"} cp_{j}.c_{j+1} + \sum_{j=0}^{"+str(N-2)+"} t*cp_{j+1}.c_{j}"
    H = gen.latex2yamps(H_str, parameters)
    return H

def mpo_Ising_model(config, N, Jij, gi):
    """ 
    MPO for Hamiltonian sum_i>j Jij Zi Zj + sum_i Jii Zi - sum_i gi Xi.
    For now only nearest neighbour coupling -- # TODO make it general
    """
    gen = yamps.GenerateOpEnv(N, config=config)
    gen.use_default(basis_type='pauli_matrices')
    parameters = {"J": Jij, "g": -gi}
    H_str = "\sum_{j=0}^{"+str(N-1)+"} g*x_{j} +\sum_{j=0}^{"+str(N-1)+"} J*z_{j} + \sum_{j=0}^{"+str(N-2)+"} J*z_{j}.z_{j+1}"
    H = gen.latex2yamps(H_str, parameters)
    return H
