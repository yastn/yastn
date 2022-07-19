
import yamps

def mpo_occupation(config, N):
    gen = yamps.GenerateOpEnv(N, config=config)
    gen.use_default()
    H_str = "\sum_{j=0}^{"+str(N-1)+"} cp_{j}.c_{j}"
    H = gen.latex2yamps(H_str)
    return H

def mpo_XX_model(config, N, t, mu):
    """ 
    MPO for Hamiltonian sum_j t (cp_j c_{j+1}+cp_{j+1} c_j) + sum_j mu cp_j c_j
    """
    #
    # First we initialise the generator for MPO of legth N and some config.
    #
    gen = yamps.GenerateOpEnv(N, config=config)
    #
    # Generator gen has to be supplied with the basis which will be later 
    # used to construct the MPO. 
    #
    # You can do it by hand by giving a dictionary of basis elements, e.g.:
    #
    # dict_basis = {"c": <yast Tensor with two physical legs only>, ...}
    # gen.use_basis(dict_basis)
    #
    # We can also use predefined operators by doing, e.g.:
    #
    # basis_type = 'creation_annihilation'
    # gen.use_default(basis_type)
    #
    # here for short we can do,
    gen.use_default()
    #
    # There are a couple different ways to use our generator. 
    # 
    # The simplest one is by constructing class mpo_term and using it generator. For example to 
    # create an MPO element t cp_{j} c_{j+1}:
    #
    # sum_element = yamps.mpo_term(amplitude = t, position=(j, j+1), operator=("cp", "c"))
    # MPO_element = gen.generate(sum_element)
    #
    # or for many sum_element-s:
    # MPO_element = gen.sum(sum_elements)
    #
    # Another way to generate MPO is to use custom latex2yamps converter.
    #
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
