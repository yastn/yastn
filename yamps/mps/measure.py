import logging
from .env2 import Env2
from .env3 import Env3


class FatalError(Exception):
    pass


logger = logging.getLogger('yamps.mps.env2')



def measure_overlap(bra, ket):
    r"""
    Calculate overlap <bra|ket>

    Returns
    -------
    overlap : float or complex
    """
    env = Env2(bra=bra, ket=ket)
    return env.overlap()


def measure_mpo(bra, op, ket):
    r"""
    Calculate overlap <bra|ket>

    Returns
    -------
    overlap : float or complex
    """
    env = Env3(bra=bra, op=op, ket=ket)
    return env.overlap()
