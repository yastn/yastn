from typing import NamedTuple
from ._tests import YastError

__all__ = ['Leg']


class _Leg(NamedTuple):
    s: int = 1  # leg signature in (1, -1)
    t: tuple = ()  # leg charges
    D: tuple = ()  # and their dimensions

    def conj(self):
        return self._replace(s=-self.s)


def Leg(**kwargs):
    """ Test input of Leg. """
    if 's' in kwargs and kwargs['s'] not in (-1, 1):
        raise YastError('Signature of Leg should be 1 or -1')
    if 't' not in kwargs:
        kwargs['t'] = ()
    if 'D' not in kwargs:
        kwargs['D'] = ()
    try:   # simplify syntax for a single charge with nsym == 1
        kwargs['t'] = tuple(kwargs['t'])
    except TypeError:
        kwargs['t'] = (kwargs['t'],)
    try: 
        kwargs['D'] = tuple(kwargs['D'])
    except TypeError:
        kwargs['D'] = (kwargs['D'],)
    if len(kwargs['D']) != len(kwargs['t']):
        raise YastError('Charges t and their dimensions D do not match')
    return _Leg(**kwargs)
