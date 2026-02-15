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
""" Predefined spinless fermion operators. """
import pytest
import yastn
import yastn.tn.fpeps as peps
from yastn.tensor import sign_canonical_order

tol = 1e-12  #pylint: disable=invalid-name


def test_ordering_sign(config_kwargs):
    """ Generate standard operators in two-dimensional Hilbert space for various symmetries. """

    #  sites in canonical order
    s0, s1, s2, s3 = peps.Site(0, 0), peps.Site(1, 0), peps.Site(0, 1), peps.Site(1, 1)
    net = peps.SquareLattice()
    assert net.f_ordered(s0, s1)
    assert net.f_ordered(s1, s2)
    assert net.f_ordered(s2, s3)
    #
    # spinless fermions
    #
    ops = yastn.operators.SpinlessFermions(sym='Z2', **config_kwargs)
    c, cp, n = ops.c(), ops.cp(), ops.n()
    assert  1 == sign_canonical_order(c, cp, sites=(s0, s1), f_ordered=net.f_ordered)
    assert -1 == sign_canonical_order(c, cp, sites=(s2, s1), f_ordered=net.f_ordered)
    assert  1 == sign_canonical_order(c, cp, cp, cp, sites=(s3, s3, s2, s2), f_ordered=net.f_ordered)
    assert -1 == sign_canonical_order(c, cp, n, cp, sites=(s3, s2, s1, s0), f_ordered=net.f_ordered)
    # for mps
    assert -1 == sign_canonical_order(c, cp, n, cp, sites=(3, 2, 1, 0), f_ordered=lambda s0, s1: s0 <= s1)
    assert 1 == sign_canonical_order(c, cp, n, cp, sites=(3, 3, 1, 1), f_ordered=lambda s0, s1: s0 <= s1)
    #
    # spinful fermions with anticommuting spiecies (using U1xU1xZ2 symmetry)
    #
    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **config_kwargs)
    cu, cpu, nu = ops.c(spin='u'), ops.cp(spin='u'), ops.n(spin='u')
    cd, cpd, nd = ops.c(spin='d'), ops.cp(spin='d'), ops.n(spin='d')
    assert  1 == sign_canonical_order(cu, cpd, sites=(s0, s1), f_ordered=net.f_ordered)
    assert -1 == sign_canonical_order(cd, cpu, sites=(s2, s1), f_ordered=net.f_ordered)
    assert  1 == sign_canonical_order(cd, nu, cd, cu, sites=(s2, s1, s0, s0), f_ordered=net.f_ordered)
    assert -1 == sign_canonical_order(cu, nd, nu, cpd, sites=(s1, s3, s2, s0), f_ordered=net.f_ordered)
    assert  1 == sign_canonical_order(cu, cpd, cu, cpu, sites=(s3, s3, s2, s2), f_ordered=net.f_ordered)
    #
    # spinful fermions with commuting (!) spiecies (using U1xU1 symmetry)
    #
    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
    cu, cpu, nu = ops.c(spin='u'), ops.cp(spin='u'), ops.n(spin='u')
    cd, cpd, nd = ops.c(spin='d'), ops.cp(spin='d'), ops.n(spin='d')
    assert  1 == sign_canonical_order(cu, cpd, sites=(s0, s1), f_ordered=net.f_ordered)
    assert  1 == sign_canonical_order(cd, cpu, sites=(s2, s1), f_ordered=net.f_ordered)
    assert -1 == sign_canonical_order(cd, nu, cd, cu, sites=(s2, s1, s0, s0), f_ordered=net.f_ordered)
    assert  1 == sign_canonical_order(cu, nd, nu, cpd, sites=(s1, s3, s2, s0), f_ordered=net.f_ordered)
    assert  1 == sign_canonical_order(cu, cpd, cu, cpu, sites=(s3, s3, s2, s2), f_ordered=net.f_ordered)
    #
    # fermionically-trivial spin operators
    #
    ops = yastn.operators.Spin1(sym='U1', **config_kwargs)
    sp, sm = ops.sp(), ops.sm()
    assert 1 == sign_canonical_order(sp, sm, sites=(s2, s1), f_ordered=net.f_ordered)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
