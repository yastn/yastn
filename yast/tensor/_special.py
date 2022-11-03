from ._contractions import ncon

def _attach_01(M, T):
    return ncon([T, M], ((-0, 1, 2, -2), (2, 1, -1, -3)))

def _attach_23(M, T):
    return ncon([T, M], ((-0, 1, 2, -2), (-1, -3, 2, 1)))
