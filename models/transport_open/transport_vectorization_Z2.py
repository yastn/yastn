import numpy as np
import yamps.mps as mps
import yamps.tensor as tensor
from yamps.tensor.ncon import ncon
import transport_vectorization_general as general
import transport_vectorization as main_fun


def thermal_state(LSR, io, ww, temp, basis):
    return main_fun.thermal_state('Z2', basis, LSR, io, ww, temp)


def Lindbladian_1AIM_mixed(NL, LSR, wk, temp, vk, dV, gamma, basis, AdagA=False):
    return main_fun.Lindbladian_1AIM_mixed('Z2', NL, LSR, wk, temp, vk, dV, gamma, basis, AdagA=AdagA)


def current(LSR, vk, cut, basis):
    return main_fun.current('Z2', LSR, vk, cut, basis)


def measure_Op(N, id, Op, basis):
    return main_fun.measure_Op('Z2', N, id, Op, basis)


def measure_sumOp(choice, LSR, basis, Op):
    return main_fun.measure_sumOp('Z2', choice, LSR, basis, Op)


def identity(N, basis):
    return main_fun.identity('Z2', N, basis)
