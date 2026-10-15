# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
""" tensor.to_number() """
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def run_to_number(a, b):
    ax = tuple(range(a.ndim))  # here a.ndim == b.ndim
    t0 = yastn.tensordot(a, b, axes=(ax, ax), conj=(1, 0))  # 0-dim tensor with 1 element, i.e., a number

    nb0 = t0.to_number()  # this is backend-type number
    it0 = t0.item()  # this is python float (or int)

    legs_for_b = {ii: leg for ii, leg in enumerate(a.get_legs())}  # info on charges and dimensions on all legs
    legs_for_a = {ii: leg for ii, leg in enumerate(b.get_legs())}
    na = a.to_numpy(legs_for_a)  # use tDb to fill in missing zero blocks to make sure that na and nb match
    nb = b.to_numpy(legs_for_b)
    ns = na.conj().reshape(-1) @ nb.reshape(-1)  # this is numpy scalar

    assert pytest.approx(it0, rel=tol) == ns
    assert pytest.approx(it0, rel=tol) == it0
    assert isinstance(it0, (float, int))  # in the examples it is real
    assert type(it0) is not type(nb0)


def test_to_number_basic(config_kwargs):
    """ test yastn.to_number() for various symmetries"""
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    a = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    b = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    run_to_number(a, b)

    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 1, 0), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9)),
            yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))]
    a = yastn.rand(config=config_U1, legs=legs)
    b = yastn.rand(config=config_U1, legs=legs)

    legs[0] = yastn.Leg(config_U1, s=-1, t=(-2, 2), D=(1, 3))
    c = yastn.rand(config=config_U1, legs=legs)

    run_to_number(a, b)
    run_to_number(a, c)
    run_to_number(b, c)


def test_to_number_exceptions(config_kwargs):
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    a = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    with pytest.raises(yastn.YastnError):
        a.to_number()
        # Only single-element (symmetric) Tensor can be converted to scalar
    with pytest.raises(yastn.YastnError):
        a.item()
        # Only single-element (symmetric) Tensor can be converted to scalar


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
