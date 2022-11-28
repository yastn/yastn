import pytest
import numpy as np
import yast
import yast.tn.mps as mps
try:
    from .configs import config_dense as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_dense as cfg

tol = 1e-12


####### MPO for XX model ##########
def mpo_XX_model_dense(config, N, t, mu):
    # Initialize MPO tensor by tensor. Example for NN-hopping model
    # TODO ref ?

    # Define basic rank-2 blocks (matrices) of on-site tensors
    #
    cp = np.array([[0, 0], [1, 0]])
    c = np.array([[0, 1], [0, 0]])
    nn = np.array([[0, 0], [0, 1]])
    ee = np.array([[1, 0], [0, 1]])
    oo = np.array([[0, 0], [0, 0]])

    # Build empty MPO for system of N sites
    # 
    H = mps.Mpo(N)

    # Depending on the site position, define elements of on-site tensor
    #
    for n in H.sweep(to='last'):  # empty tensors
        if n == H.first:
            tmp = np.block([[mu * nn, t * cp, t * c, ee]])
            tmp = tmp.reshape((1, 2, 4, 2))
            Ds = (1, 2, 2, 4)
        elif n == H.last:
            tmp = np.block([[ee], [c], [cp], [mu * nn]])
            tmp = tmp.reshape((4, 2, 1, 2))
            Ds = (4, 2, 2, 1)
        else:
            tmp = np.block([[ee, oo, oo, oo],
                            [c, oo, oo, oo],
                            [cp, oo, oo, oo],
                            [mu * nn, t * cp, t * c, ee]])
            tmp = tmp.reshape((4, 2, 4, 2))
            Ds = (4, 2, 2, 4)
        # tmp = np.transpose(tmp, (0, 1, 3, 2))
        #
        # We chose signature convention for indices of the MPO tensor as follows
        #         
        #          | 
        #          V(+1)
        #          | 
        # (+1) ->-|T|->-(-1)
        #          |
        #          V(-1)
        #          |
        #
        on_site_t = yast.Tensor(config=config, s=(1, 1, -1, -1))
        on_site_t.set_block(val=tmp, Ds=Ds)

        # Set n-th on-site tensor of MPO
        H[n]= on_site_t
    return H


def mpo_XX_model_Z2(config, N, t, mu):
    # Initialize MPO tensor by tensor. Example for NN-hopping model, 
    # using explicit Z2 symmetry of the model.
    # TODO ref ?

    # Build empty MPO for system of N sites
    #
    H = mps.Mpo(N)

    # Depending on the site position, define elements of on-site tensor
    #
    for n in H.sweep(to='last'):
        #
        # Define empty yast.Tensor as n-th on-site tensor
        # We chose signature convention for indices of the MPO tensor as follows
        #         
        #          | 
        #          V(+1)
        #          | 
        # (+1) ->-|T|->-(-1)
        #          |
        #          V(-1)
        #          |
        #
        H[n] = yast.Tensor(config=config, s=[1, 1, -1, -1], n=0)
        
        # set blocks, indexed by tuple of Z2 charges, of on-site tensor at n-th position
        #
        if n == H.first:
            H[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 2, 1))
            H[n].set_block(ts=(0, 1, 0, 1), val=[mu, 1], Ds=(1, 1, 2, 1))
            H[n].set_block(ts=(0, 0, 1, 1), val=[t, 0], Ds=(1, 1, 2, 1))
            H[n].set_block(ts=(0, 1, 1, 0), val=[0, t], Ds=(1, 1, 2, 1))
        elif n == H.last:
            H[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H[n].set_block(ts=(0, 1, 0, 1), val=[1, mu], Ds=(2, 1, 1, 1))
            H[n].set_block(ts=(1, 1, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H[n].set_block(ts=(1, 0, 0, 1), val=[0, 1], Ds=(2, 1, 1, 1))
        else:
            H[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 2, 1))
            H[n].set_block(ts=(0, 1, 0, 1), val=[[1, 0], [mu, 1]], Ds=(2, 1, 2, 1))
            H[n].set_block(ts=(0, 0, 1, 1), val=[[0, 0], [t, 0]], Ds=(2, 1, 2, 1))
            H[n].set_block(ts=(0, 1, 1, 0), val=[[0, 0], [0, t]], Ds=(2, 1, 2, 1))
            H[n].set_block(ts=(1, 1, 0, 0), val=[[1, 0], [0, 0]], Ds=(2, 1, 2, 1))
            H[n].set_block(ts=(1, 0, 0, 1), val=[[0, 0], [1, 0]], Ds=(2, 1, 2, 1))
    return H


def mpo_XX_model_U1(config, N, t, mu):
    # Initialize MPO tensor by tensor. Example for NN-hopping model, 
    # using explicit U(1) symmetry of the model.
    # TODO ref ?

    # Build empty MPO for system of N sites
    #
    H = mps.Mpo(N)

    # Depending on the site position, define elements of on-site tensor
    #
    for n in H.sweep(to='last'):
        #
        # Define empty yast.Tensor as n-th on-site tensor
        # We chose signature convention for indices of the MPO tensor as follows
        #         
        #          | 
        #          V(+1)
        #          | 
        # (+1) ->-|T|->-(-1)
        #          |
        #          V(-1)
        #          |
        #
        H.A[n] = yast.Tensor(config=config, s=[1, 1, -1, -1], n=0)

        # set blocks, indexed by tuple of U(1) charges, of on-site tensor at n-th position
        #
        if n == H.first:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 2, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[mu, 1], Ds=(1, 1, 2, 1))
            H.A[n].set_block(ts=(0, 0, -1, 1), val=[t], Ds=(1, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[t], Ds=(1, 1, 1, 1))
        elif n == H.last:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[1, mu], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(-1, 1, 0, 0), val=[1], Ds=(1, 1, 1, 1))
            H.A[n].set_block(ts=(1, 0, 0, 1), val=[1], Ds=(1, 1, 1, 1))
        else:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 2, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[[1, 0], [mu, 1]], Ds=(2, 1, 2, 1))
            H.A[n].set_block(ts=(0, 0, -1, 1), val=[0, t], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[0, t], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(-1, 1, 0, 0), val=[1, 0], Ds=(1, 1, 2, 1))
            H.A[n].set_block(ts=(1, 0, 0, 1), val=[1, 0], Ds=(1, 1, 2, 1))
    return H


def mpo_XX_model(config, N, t, mu):
    if config.sym.SYM_ID == 'dense':
        return mpo_XX_model_dense(config, N, t, mu)
    elif config.sym.SYM_ID == 'Z2':
        return mpo_XX_model_Z2(config, N, t, mu)
    elif config.sym.SYM_ID == 'U(1)':
        return mpo_XX_model_U1(config, N, t, mu)



def test_generator_mpo():
    N = 5
    t = 1
    mu = 0.2
    operators = yast.operators.SpinlessFermions(sym='Z2', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N, operators)
    generate.random_seed(seed=0)
    parameters = {"t": lambda j: t, "mu": lambda j: mu, "range1": range(N), "range2": range(1, N-1)}
    H_str = "\sum_{j \in range2} t ( cp_{j} c_{j+1} + cp_{j+1} c_{j} ) + \sum_{j\in range1} mu cp_{j} c_{j} + ( cp_{0} c_{1} + 1*cp_{1} c_{0} )*t "
    H_ref = mpo_XX_model(generate.config, N=N, t=t, mu=mu)
    H = generate.mpo(H_str, parameters)
    psi = generate.random_mps(D_total=8, n=0) + generate.random_mps( D_total=8, n=1)
    x_ref = mps.measure_mpo(psi, H_ref, psi).item()
    x = mps.measure_mpo(psi, H, psi).item()
    assert abs(x_ref - x) < tol

    psi.canonize_(to='first')
    psi.canonize_(to='last')
    x_ref = mps.measure_mpo(psi, H_ref, psi).item()
    x = mps.measure_mpo(psi, H, psi).item()
    assert abs(x_ref - x) < tol

def mpo_Ising_model():
    op = yast.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    gn = mps.Generator(10, op)
    HT1 = mps.Hterm(1, (4,), (op.c(),))
    HT2 = mps.Hterm(1, (4,), (op.cp(),))
    H = mps.generate_mpo(gn.I(), [HT1, HT2])
    print(H.get_bond_dimensions())
    print(H.virtual_leg('first'))

    p1 = gn.random_mps(n=2, D_total=8)
    p2 = gn.random_mps(n=3, D_total=8)
    pp = p1 + p2
    qq = p1 + p2
    
    print(pp.get_bond_dimensions())
    qq.canonize_(to='first', normalize=False)
    qq.canonize_(to='last', normalize=False)
    print(qq.get_bond_dimensions())
   
    print(mps.measure_overlap(pp + -1 * qq, pp + -1 * qq))


if __name__ == "__main__":
    test_generator_mpo()
    mpo_Ising_model()
