import numpy as np
import yamps.mps as mps
import yamps.tensor as tensor
from yamps.tensor.ncon import ncon
import transport_vectorization_general as general
import transport_vectorization as main_fun


def get_tensor_type(basis):
    return ('U1', 'complex128')


def thermal_state(tensor_type, LSR, io, ww, temp, basis):
    return main_fun.thermal_state(tensor_type, basis, LSR, io, ww, temp)


def Lindbladian_1AIM_mixed(tensor_type, NL, LSR, wk, temp, vk, dV, gamma, basis, AdagA=False):
    if basis == 'Majorana':
        return main_fun.Lindbladian_1AIM_mixed_real(tensor_type, NL, LSR, wk, temp, vk, dV, gamma, basis, AdagA=AdagA)
    elif basis == 'Dirac':
        return main_fun.Lindbladian_1AIM_mixed(tensor_type, NL, LSR, wk, temp, vk, dV, gamma, basis, AdagA=AdagA)


def current(tensor_type, LSR, vk, cut, basis):
    if basis == 'Majorana':
        return main_fun.current_XY(tensor_type, LSR, vk, cut, basis)
    elif basis == 'Dirac':
        return main_fun.current_ccp(tensor_type, LSR, vk, cut, basis)


def measure_Op(tensor_type, N, id, Op, basis):
    return main_fun.measure_Op(tensor_type, N, id, Op, basis)


def measure_sumOp(tensor_type, choice, LSR, basis, Op):
    return main_fun.measure_sumOp(tensor_type, choice, LSR, basis, Op)


def identity(tensor_type, N, basis):
    return main_fun.identity(tensor_type, N, basis)
