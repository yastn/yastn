"""
Test ctmrg on 2D Classical Ising model.
Calculate expectation values using ctm for analytical dense peps tensors
of 2D Ising model with zero transverse field (Onsager solution)
"""
import logging
import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.ctm import nn_exp_dict, ctmrg

try:
    from .configs import config_dense as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_dense as cfg


def create_Ising_tensor(sz, beta):
    """ Creates peps tensor for given beta. """
    L = yastn.ncon((sz, sz), ((-0, -2), (-1, -3)))
    L = L.fuse_legs(axes=((0, 1), (2, 3)))
    D, S = yastn.eigh(L, axes=(0, 1))
    D = yastn.exp(D, step=beta / 2)
    U = yastn.ncon((S, D, S), ((-1, 1), (1, 2), (-3, 2)), conjs=(0, 0, 1))
    U = U.unfuse_legs(axes=(0, 1))
    U, S, V = yastn.svd_with_truncation(U, axes = ((0, 2), (1, 3)), sU = -1, tol = 1e-15, Uaxis=1, Vaxis=1)
    S = S.sqrt()
    GA = S.broadcast(U, axes=1)
    GB = S.broadcast(V, axes=1)

    T = yastn.tensordot(GA, GB, axes=(2, 0))
    T = yastn.tensordot(T, GB, axes=(3, 0))
    T = yastn.tensordot(T, GA, axes=(4, 0))
    T = T.fuse_legs(axes=(1, 2, 3, 4, (0, 5)))
    return T


def gauges_random():
    """ Returns a 2 x 2 dense random matrix and its inverse """
    ss = (1, -1)
    a = yastn.rand(config=cfg, s=ss, D=(2, 2))
    inv_tu = np.linalg.inv(a.to_numpy())
    b = yastn.Tensor(config=cfg, s=ss)
    b.set_block(val=inv_tu, Ds=(2, 2))
    return a, b


def ctm_for_Onsager(peps, ops, Z_exact):
    """ Compares ctm expectation values with analytical result. """

    chi = 8 # max environmental bond dimension
    tol = 1e-10 # singular values of svd truncation of projectors
    tol_exp = 1e-5 # tolerance for expectation values
    max_sweeps = 100

    cf_old = 0
    opts_svd = {'D_total': chi, 'tol': tol}

    ops = {'magA1': {'l': ops.z(), 'r': ops.I()},
           'magB1': {'l': ops.I(), 'r': ops.z()}}

    for step in ctmrg(peps, max_sweeps, iterator_step=4, AAb_mode=0, opts_svd=opts_svd):
        assert step.sweeps % 4 == 0 # stop every 4th step as iteration_step=4

        ob_hor, ob_ver = nn_exp_dict(peps, step.env, ops)
        cf =  0.125 * (sum([abs(val) for val in ob_hor.get('magA1').values()]) +
                sum([abs(val) for val in ob_hor.get('magB1').values()]) +
                sum([abs(val) for val in ob_ver.get('magA1').values()]) +
                sum([abs(val) for val in ob_ver.get('magB1').values()]))
        print("expectation value: ", cf)
        if abs(cf - cf_old) < tol_exp:
            break # here break if the relative differnece is below tolerance
        cf_old = cf
    assert pytest.approx(cf, rel=1e-3) == Z_exact


def test_ctm_loop():  ###high temperature
    """ Calculate magnetization for classical 2D Ising model and compares with the exact result. """
    beta = 0.8 # check for a certain inverse temperature
    Z_exact = 0.99602 # analytical value of magnetization up to 4 decimal places for beta = 0.7 (2D Classical Ising)

    ops = yastn.operators.Spin12(sym='dense', backend=cfg.backend, default_device=cfg.default_device)
    ops.random_seed(seed=0)

    T = create_Ising_tensor(ops.z(), beta)
    geometry = fpeps.CheckerboardLattice()
    psi = fpeps.Peps(geometry)
    for site in psi.sites():
        psi[site] = T
    ctm_for_Onsager(psi, ops, Z_exact)

    h_rg1, inv_h_rg1 = gauges_random()
    h_rg2, inv_h_rg2 = gauges_random()
    v_rg1, inv_v_rg1 = gauges_random()
    v_rg2, inv_v_rg2 = gauges_random()
    TA = yastn.ncon((T, h_rg1), ((-0, -1, -2, 1, -4), (1, -3)))
    TB = yastn.ncon((inv_h_rg1, T), ((-1, 1), (-0, 1, -2, -3, -4)))
    TA = yastn.ncon((TA, h_rg2), ((-0, 1, -2, -3, -4), (-1, 1)))
    TB = yastn.ncon((inv_h_rg2, TB), ((1, -3), (-0, -1, -2, 1, -4)))
    TA = yastn.ncon((TA, v_rg1), ((1, -1, -2, -3, -4), (1, -0)))
    TB = yastn.ncon((inv_v_rg1, TB), ((-2, 1), (-0, -1, 1, -3, -4)))
    TA = yastn.ncon((TA, v_rg2), ((-0, -1, 1, -3, -4), (-2, 1)))
    TB = yastn.ncon((inv_v_rg2, TB), ((1, -0), (1, -1, -2, -3, -4)))

    for site in psi.sites():
        psi[site] = TA if sum(site) % 2 == 0 else TB
    ctm_for_Onsager(psi, ops, Z_exact)


if __name__ == '__main__':
    test_ctm_loop()