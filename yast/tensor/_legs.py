from typing import NamedTuple
from ._tests import YastError

__all__ = ['Leg']


class _Leg(NamedTuple):
    s: int = 1  # leg signature equall 1 or -1
    t: tuple = ()  # leg charges
    D: tuple = ()  # and their dimensions

    def conj(self):
        return self._replace(s=-self.s)


def Leg(**kwargs):
    """ Test input of Leg. """
    if 's' in kwargs and kwargs['s'] not in (-1, 1):
        raise YastError('Signature of Leg should be 1 or -1')
    if 't' in kwargs or 'D' in kwargs:
        if 't' not in kwargs or 'D' not in kwargs:
            raise YastError('Charges t and their dimensions D do not match')
        try: 
            kwargs['t'] = tuple(kwargs['t'])
        except TypeError:
             kwargs['t'] = (kwargs['t'],)
        try: 
            kwargs['D'] = tuple(kwargs['D'])
        except TypeError:
             kwargs['D'] = (kwargs['D'],)
        # if not all(all(isinstance(x, int) for x in kwargs[field]) for field in ('t', 'D')):
        #     raise YastError('Charges and dimensions should be int.')
        # if any(x <= 0 for x in kwargs['D']):
        #     raise YastError('Dimensions should be positive.')
        if len(kwargs['D']) != len(kwargs['t']):
            raise YastError('Charges t and their dimensions D do not match')
    return _Leg(**kwargs)
