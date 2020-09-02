import numpy as np
import yamps.mps as mps
import yamps.tensor as tensor
from yamps.tensor.ncon import ncon
import AIM_transport.transport_vectorization as main_fun


def get_tensor_type(basis):
    if basis == 'Majorana':
        return ('full', 'float64')
    elif basis == 'Dirac':
        return ('full', 'complex128')


def thermal_state(tensor_type, LSR, io, ww, temp, basis):
    return main_fun.thermal_state(tensor_type, basis, LSR, io, ww, temp)


def vectorized_Lindbladian(tensor_type, LSR, temp, dV, gamma, corr, basis, corr_U=False, AdagA=False, compress=True, tol=1e-14):
    H_U, HdagH = None, None
    if basis == 'Dirac':
        H = main_fun.vectorized_Lindbladian_general(
            tensor_type, LSR, temp, dV, gamma, corr, basis)
        if not isinstance(corr_U, bool):
            H_U = main_fun.Liouville_AIM_Coulomb_general(
                tensor_type, LSR, temp, dV, gamma, corr_U, basis)
            H = main_fun.add_MPOs(H, H_U)
    elif basis == 'Majorana':
        H = main_fun.vectorized_Lindbladian_real(
            tensor_type, LSR, temp, dV, gamma, corr, basis)
        if not isinstance(corr_U, bool):
            H_U = main_fun.Liouville_AIM_Coulomb_real(
                tensor_type, LSR, temp, dV, gamma, corr_U, basis)
            H = main_fun.add_MPOs(H, H_U)    
    if compress:
        H.canonize_sweep(to='last', normalize=False)
        H.sweep_truncate(to='first', opts={'tol': tol}, normalize=False)    
    if AdagA:
        HdagH = main_fun.stack_MPOs(H, H, transpose=[0,1], conj=[0,1])
        if compress:
            HdagH.canonize_sweep(to='last', normalize=False)
            HdagH.sweep_truncate(to='first', opts={'tol': tol}, normalize=False)
    return H, HdagH


def current(tensor_type, sign_from, id_to, sign_list, vk, direction, basis):
    if basis == 'Dirac':
        return main_fun.current_ccp(tensor_type, sign_from, id_to, sign_list, vk, direction, basis)
    elif basis == 'Majorana':
        return main_fun.current_XY(tensor_type, sign_from, id_to, sign_list, vk, direction, basis)


def measure_Op(tensor_type, N, id, Op, basis):
    return main_fun.measure_Op(tensor_type, N, id, Op, basis)


def measure_sumOp(tensor_type, choice, LSR, basis, Op):
    return main_fun.measure_sumOp(tensor_type, choice, LSR, basis, Op)


def identity(tensor_type, N, basis):
    return main_fun.identity(tensor_type, N, basis)
