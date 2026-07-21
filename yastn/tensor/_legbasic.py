# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" class yastn.Leg """
from __future__ import annotations

from typing import NamedTuple

__all__ = ['LegBasic', 'legs_from_dict_v2']

class LegBasic(NamedTuple):
    s: int = 1  # leg signature in (1, -1)
    t: tuple = ()  # leg charges
    D: tuple = ()  # and their dimensions

    def conj(self):
        return LegBasic(s=-self.s, t=self.t, D=self.D)

    def conj_charges(self, sym):
        tD = sorted((sym.conj_charge(tt), DD) for tt, DD in zip(self.t, self.D))
        return LegBasic(s=-self.s, t=tuple(x[0] for x in tD), D=tuple(x[1] for x in tD))

    def __getitem__(self, t) -> int:
        r"""
        Size of a charge sector.

        Parameters
        ----------
        t : int | Sequence[int]
            selected charge sector
        """
        return self.D[self.t.index(t)]

    def add_charge(self, t, D) -> LegBasic:
        r"""
        Add a space of charge t and dimension D.

        Parameters
        ----------
        t : int | Sequence[int]
            selected charge sector
        """
        t = tuple(t)
        di = 1 if t in self.t else 0
        ind = sum(x < t for x in self.t)
        newt = self.t[:ind] + (t,) + self.t[ind+di:]
        newD = self.D[:ind] + (D,) + self.D[ind+di:]
        return LegBasic(s=self.s, t=newt, D=newD)

    def __contains__(self, t) -> bool:
        r"""
        Test if charge sector is in the leg.

        Parameters
        ----------
        t : int | Sequence[int]
            selected charge sector
        """
        return t in self.t

    def __str__(self):
        return (f"LegBasic(s={self.s}, t={self.t}, D={self.D})")

    def __repr__(a) -> str:
        return str(a)

    @property
    def tD(self) -> dict[tuple, int]:
        r"""
        Return charge sectors `t` and their sizes `D` as a dictionary ``{t: D}``.
        """
        return dict(zip(self.t, self.D))

    def are_consistent(self, other, sgn=-1) -> bool:
        tD0, tD1 = self.tD, other.tD
        return not (self.s != sgn * other.s or any(tD0[k] != tD1[k] for k in tD0.keys() & tD1.keys()))

    def union(self, other, isdiag=False) -> LegBasic:
        if not isdiag and not self.are_consistent(other, sgn=1):
            raise ValueError("Cannot take an union of inconsistent legs")
        tD0, tD1 = self.tD, other.tD
        if isdiag:  # takes larger dimension for common charge
            tD = [(k, max(tD0.get(k, 0), tD1.get(k, 0))) for k in sorted(tD0.keys() | tD1.keys())]
        else:
            tD = sorted({**tD0, **tD1}.items())
        return LegBasic(s=self.s, t=tuple(x[0] for x in tD), D=tuple(x[1] for x in tD))

    def intersection(self, other) -> LegBasic:
        if not self.are_consistent(other, sgn=1):
            raise ValueError("Cannot take an intersection of inconsistent legs")
        tD0, tD1 = self.tD, other.tD
        t = tuple(sorted(tD0.keys() & tD1.keys()))
        return LegBasic(s=self.s, t=t, D=tuple(tD0[k] for k in t))

    def trim(self, tsub) -> LegBasic:
        tD = self.tD
        return LegBasic(s=self.s, t=tuple(tsub), D=tuple(tD[k] for k in tsub))


def legs_from_dict_v2(struct):
    nsym = len(struct['n'])
    ndim = len(struct['s'])
    legs = []
    for i in range(ndim):
        tDn = {tn[i * nsym: (i + 1) * nsym]: Dn[i] for tn, Dn in zip(struct['t'], struct['D'])}
        tDn = dict(sorted(tDn.items()))
        leg = LegBasic(s=struct['s'][i], t=tuple(tDn.keys()), D=tuple(tDn.values()))
        legs.append(leg)
    return tuple(legs)
