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
""" yastn.make_config(); catching some config mismatches between tensors. """
import pytest
import yastn


def test_config_exceptions(config_kwargs):
    """ Handling mismatches of tensor configs when combining two tensors. """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    config_Z2 = yastn.make_config(sym='Z2', **config_kwargs)
    config_Z2_f = yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)

    leg_U1 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(1, 2))
    leg_Z2 = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(1, 2))
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_U1, legs=[leg_U1, leg_U1, leg_U1.conj()])
        b = yastn.rand(config=config_Z2, legs=[leg_Z2, leg_Z2, leg_Z2.conj()])
        _ = a + b
        # Two tensors have different symmetry rules.
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_Z2, legs=[leg_Z2, leg_Z2, leg_Z2.conj()])
        # leg do not depend on fermionic statistics so the next line is fine
        b = yastn.rand(config=config_Z2_f, legs=[leg_Z2, leg_Z2, leg_Z2.conj()])
        _ = a + b
        # Two tensors have different assignment of fermionic statistics.


def test_config_exceptions_torch(config_kwargs):
    """ Mismatches requiring different backends or devices. """
    config = yastn.make_config(sym='U1', **config_kwargs)
    config_np = yastn.make_config(sym='U1', backend='np')

    if config.backend != config_np.backend:
        a = yastn.rand(config=config, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
        b = yastn.rand(config=config_np, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
        with pytest.raises(yastn.YastnError):
            _ = a + b
            # Two tensors have different backends.


def test_make_config(config_kwargs):
    """ Parameters in yastn.make_config(). """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    assert config_U1.sym.SYM_ID == 'U1'

    with pytest.raises(yastn.YastnError, match="sym encoded as string only supports:"):
        yastn.make_config(sym="random_name")
        # sym encoded as string only supports: names listed in yastn.sym.SYMMETRIES
    for sym in (None, 3, object()):
        with pytest.raises(yastn.YastnError, match="sym should be a str or a symmetry class"):
            yastn.make_config(sym=sym)
            # sym should be a str or a symmetry class defining SYM_ID, NSYM and fuse; see yastn.sym.sym_abelian.
    assert yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs).sym.SYM_ID == 'Z2xU1'

    with pytest.raises(yastn.YastnError):
        yastn.make_config(backend="random_name")
        # backend encoded as string only supports: 'np', 'torch'

    # fermionic is a bool, or a tuple with one bool per symmetry component
    assert yastn.make_config(sym='U1', fermionic=True, **config_kwargs).fermionic is True
    config = yastn.make_config(sym='U1xU1xU1xZ2', fermionic=(False, False, False, True), **config_kwargs)
    assert config.fermionic == (False, False, False, True)

    with pytest.raises(yastn.YastnError, match="sym.NSYM=4 bools"):
        yastn.make_config(sym='U1xU1xU1xZ2', fermionic=(False, False, True), **config_kwargs)
        # fermionic given as a tuple should contain sym.NSYM=4 bools.
    with pytest.raises(yastn.YastnError, match="sym.NSYM=3 bools"):
        yastn.make_config(sym='U1xU1xZ2', fermionic=(False, False, 1), **config_kwargs)
        # fermionic given as a tuple should contain sym.NSYM=3 bools.
    with pytest.raises(yastn.YastnError, match="bool or a tuple of bools"):
        yastn.make_config(sym='U1xU1', fermionic=[True, True], **config_kwargs)
        # fermionic should be a bool or a tuple of bools.
    with pytest.raises(yastn.YastnError, match="bool or a tuple of bools"):
        yastn.make_config(sym='U1', fermionic=1, **config_kwargs)
        # fermionic should be a bool or a tuple of bools.


if __name__ == '__main__':
    # pytest.main([__file__, "-vs", "--durations=0"])
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--device", "cuda"])
