# Copyright 2024 The YASTN Authors. All Rights Reserved.
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
""" Auxiliary functions supporting operator-related operations. """
import numpy as np

def swap_charges(charges_0, charges_1, fss) -> int:
    """ Calculates a sign accumulated while swapping lists of charges. """
    if not fss:
        return 1
    t0 = np.array(charges_0, dtype=np.int64)
    t1 = np.array(charges_1, dtype=np.int64)
    if fss is True:
        return 1 - 2 * (np.sum(t0 * t1, dtype=np.int64).item() % 2)
    return 1 - 2 * (np.sum((t0 * t1)[:, fss], dtype=np.int64).item() % 2)


def sign_canonical_order(*operators, sites=None, tn='fpeps') -> int:
    """
    Calculates a sign corresponding to the commutation of operators into canonical order,
    where the corresponding sites get ordered according to fermionic order.
    In the canonical ordering, the operators at sites appearing
    later in the fermionic order are applied first.

    For instance, consider operators O, P at sites=(s0, s1),
    which corresponds to a product operator Q = O_s0 P_s1.
    If s0 <= s1 in fermionic order, then the sign is 1.
    If s1 < s0 in fermionic order, Q = sign * P_s1 O_s0,
    where the sign follows from swapping the charges carried by O and P.
    Operators at the same site are not swapped.

    Parameters
    ----------
    operators: Sequence[yastn.Tensor]
        List of local operators to calculate <O_s0 Ps_s1 ...>.

    sites: Sequence[Sites]
        A list of sites [s0, s1, ...] matching the operators.

    tn: str
        type of lattice: 'peps' or 'mps', informing about the fermionic order of sites.
    """
    fss = operators[0].config.fermionic
    if not fss:
        return 1

    if tn == 'fpeps':
        key = lambda x: x[::-1]
    elif tn == 'mps':
        key = lambda x: x

    osites = sorted(sites, key=key)
    sites = list(sites)
    charges = [op.n for op in operators]

    charges_0, charges_1 = [], []

    for site in osites:
        ind = sites.index(site)
        sites.pop(ind)
        c1 = charges.pop(ind)
        for c0 in charges[:ind]:
            charges_0.append(c0)
            charges_1.append(c1)

    if len(charges_0) == 0:
        return 1
    return swap_charges(charges_0, charges_1, fss)
