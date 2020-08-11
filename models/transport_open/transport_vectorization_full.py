import numpy as np
import yamps.mps as mps
import yamps.tensor as tensor
from yamps.tensor.ncon import ncon
import transport_vectorization_general as general
import transport_vectorization as main_fun


def thermal_state(LSR, io, ww, temp, basis):
    return main_fun.thermal_state('full', basis, LSR, io, ww, temp)


def Lindbladian_1AIM_mixed(NL, LSR, wk, temp, vk, dV, gamma, basis, AdagA=False):
    return main_fun.Lindbladian_1AIM_mixed('full', NL, LSR, wk, temp, vk, dV, gamma, basis, AdagA=AdagA)


def current(LSR, vk, cut, basis):
    return main_fun.current('full', LSR, vk, cut, basis)


def measure_Op(N, id, Op, basis):
    return main_fun.measure_Op('full', N, id, Op, basis)


def measure_sumOp(choice, LSR, basis, Op):
    return main_fun.measure_sumOp('full', choice, LSR, basis, Op)


def identity(N, basis):
    return main_fun.identity('full', N, basis)
