from ._DoublePepsTensor import DoublePepsTensor, append_a_tl

def _attach_01(M, T):
    if isinstance(T, DoublePepsTensor):
        return T.append_a_tl(M)

def _attach_23(M, T):
    if isinstance(T, DoublePepsTensor):
        return T.append_a_br(M)