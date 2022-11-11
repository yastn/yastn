# This tests the expectation values with CTM_dense using exact analytic peps tensors for
# 2D Ising model with 0 transverse field (Onsager solution) 
import logging
import numpy as np
import pytest
import yast
from yast.operators._spin12 import Spin12
try:
    from .configs import config_dense as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_dense as cfg


def create_ZZ_ten(betas):
    # creates the exact peps tensor for a certain beta
    iden = np.array(([1, 0], [0, 1]))
    sz = np.array(([1, 0], [0, -1]))
    spin = 2
    ancilla = 2
    D = 2  # bond dimension
    L = np.zeros((ancilla, D, spin))

    L[:, 0, :] = iden * np.sqrt(np.cosh(0.5 * betas))
    L[:, 1, :] = sz * np.sqrt(np.sinh(0.5 * betas))

    T = np.tensordot(L, L, axes=(2, 0))
    T = np.tensordot(T, L, axes=(3, 0))
    T = np.tensordot(T, L, axes=(4, 0))
    T = np.transpose(T, axes=(1, 2, 3, 4, 5, 0))
    Dt = (D, D, D, D, spin, ancilla)
    strt =  yast.Tensor(config=cfg, s=(-1, 1, 1, -1, -1, 1))
    strt.set_block(val=T, Ds=Dt)
    return strt


def matrix_inverse_random(n):
    """ Returns a n*n random matrix and its inverse """
    tu = np.random.rand(n,n)
    inv_tu = np.linalg.inv(tu)
    a = yast.Tensor(config=cfg, s=(1, -1))
    a.set_block(val=tu, Ds=(2, 2))
    b = yast.Tensor(config=cfg, s=(1, -1))
    b.set_block(val=inv_tu, Ds=(2, 2))
    return a, b


def CTM_for_Onsager(peps, Z_exact):
    opt = Spin12(sym='dense', backend=cfg.backend, default_device=cfg.default_device)
    opts = {'chi': 16, 'cutoff': 1e-10, 'nbitmax': 400, 'prec' : 1e-7, 'tcinit' : (0,), 'Dcinit' : (1,)}
    sz, one = opt.z(), opt.I()
    ops = {'magA1': {'l': sz, 'r': one},
           'magB1': {'l': one, 'r': sz}}

    opt.random_seed(0)  # fix random seed in the backend

    EVs, _ = CTM_EVs(A=peps, ops=ops, **opts) 
    cf = 0.5 * (abs(EVs.get('magA1')) + abs(EVs.get('magB1')))
    assert pytest.approx(cf, rel=1e-3) == Z_exact


def test_CTM_loop_1():
    """ Calculate magnetization for classical 2D Ising model and compares with the exact result. """
    beta = 0.8  # check for a certain inverse temperature
    Z_exact = 0.99602 # analytical value of magnetization up to 4 decimal places for beta = 0.8 (2D Classical Ising)
    peps = [create_ZZ_ten(beta), create_ZZ_ten(beta)]
    CTM_for_Onsager(peps, Z_exact)


def test_CTM_loop_2():
    """ Calculate magnetization for classical 2D Ising model and compares with the exact result. """
    beta = 0.8 # check for a ceratin inverse temperature
    Z_exact = 0.99602 # analytical value of magnetization up to 4 decimal places for beta = 0.8 (2D Classical Ising)

    A = create_ZZ_ten(beta)
    B = create_ZZ_ten(beta)
    [h_rg, inv_h_rg] = matrix_inverse_random(2)
    [v_rg, inv_v_rg] = matrix_inverse_random(2)
    A = yast.ncon((A, h_rg), ((-0, -1, -2, 1, -4, -5), (1, -3)))
    B = yast.ncon((inv_h_rg, B), ((-1, 1), (-0, 1, -2, -3, -4, -5)))
    A = yast.ncon((A, v_rg), ((1, -1, -2, -3, -4, -5), (1, -0)))
    B = yast.ncon((inv_v_rg, B), ((-2, 1), (-0, -1, 1, -3, -4, -5)))
    peps = [A, B]
    CTM_for_Onsager(peps, Z_exact)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_CTM_loop_1()
    test_CTM_loop_2()
