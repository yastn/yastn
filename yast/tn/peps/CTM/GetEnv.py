""" Functions performing many CTMRG steps until convergence and return of CTM environment tensors for mxn lattice. """
import logging
import yast.tn.peps as peps
import numpy as np
from itertools import chain
from .CtmIterationRoutines import CTM_it
from .CtmIterationRoutines import fPEPS_2layers, fPEPS_fuse_layers, check_consistency_tensors
from .CtmEnv import CtmEnv, init_rand

def GetEnv(fid, A, net, chi=32, cutoff=1e-10, prec=1e-7, nbitmax=10, tcinit=(), Dcinit=(1,), init_env='rand', AAb_mode=0):

    r"""
    Based on lattice structure create converged CTM environments using functionalities from CTM_rountines.py

    Input
    -----

    A: peps tensors on mxn lattice
    lattice: structure of lattice; can be checkerboard or rectangle(finite/infinite) or brickwall 
    chi: maximal CTM bond dimension
    cutoff: controls removal of singular values smaller than cutoff during CTM projector construction
    prec: stop execution when 2 consecutive iterations give difference of a function used to evaluate conv smaller than prec
    nbitmmax: stop execution when more than nbitmax iterations performed
    tcinit: symmetry sectors of initial corners legs
    Dcinit: dimensions of initial corner legs
    AAb = 0 (no double-peps tensors; 1 = double pepes tensors with identity; 2 = for all
    """

    A = check_consistency_tensors(A, net=net) # to check if A has the desired fused form of legs i.e. t l b r [s a]
    list_sites = net.sites()
    AAb = {m: fPEPS_2layers(fid, A._data[m]) for m in list_sites}

    if AAb_mode >= 1:
        fPEPS_fuse_layers(AAb)

    if init_env == 'rand':
        net = peps.Peps(lattice=net.lattice, dims=net.dims, boundary='infinite')
        env = init_rand(A, tcinit, Dcinit, net)  # random initialization

    for ctr in range(nbitmax):
        logging.info('CTM iteration: %2d', ctr+1)
        env, proj_hor, proj_ver = CTM_it(net, env, AAb, chi, cutoff)
    
    logging.info('CTM completed sucessfully')

    return env



