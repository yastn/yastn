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
import re
from dataclasses import fields
import pytest
import yastn
import yastn.tn.fpeps as fpeps


def test_dataclasses(config_kwargs):
    r"""
    Test auxilliary dataclasses for local environments, projectors, etc.
    """
    #
    config = yastn.make_config(sym='U1', **config_kwargs)
    leg = yastn.Leg(config, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    T = yastn.rand(config, legs=[leg, leg, leg.conj()])
    C = yastn.rand(config, legs=[leg, leg.conj()])
    #
    # all Tensors given
    ctm = fpeps.envs.EnvCTM_local(tl=C, t=T, tr=C, r=T, br=C, b=T, bl=C, l=T)
    pro = fpeps.envs.EnvCTM_projectors(hlt=T, hlb=T, hrt=T, hrb=T, vtl=T, vtr=T, vbl=T, vbr=T)
    bp = fpeps.envs.EnvBP_local(t=C, l=C, b=C, r=C)
    #
    # some fields are None
    ctm2 = fpeps.envs.EnvCTM_local(tl=C, t=T, tr=C, r=T, br=C, b=None, bl=None, l=None)
    pro2 = fpeps.envs.EnvCTM_projectors(hlt=None, hlb=None, hrt=T, hrb=T, vtl=T, vtr=T, vbl=T, vbr=T)
    bp2 = fpeps.envs.EnvBP_local(t=C, l=C, b=C, r=None)
    #
    # all fields are None
    ctm3 = fpeps.envs.EnvCTM_local()
    pro3 = fpeps.envs.EnvCTM_projectors()
    bp3 = fpeps.envs.EnvBP_local()

    for a in [ctm, pro, bp, ctm2, pro2, bp2, ctm3, pro3, bp3]:
        #
        for b in [ctm, pro, bp, ctm2, pro2, bp2, ctm3, pro3, bp3]:
            assert a.allclose(b) == (a is b)
        #
        a_copy = a.copy()
        a_clone = a.clone()
        a_shallow = a.shallow_copy()
        a_shallow.detach_()
        a_detach = a.detach()
        a_dict_level_0 = a.from_dict(a.to_dict(level=0))
        a_dict_level_1 = a.from_dict(a.to_dict(level=1))
        a_dict_level_2 = a.from_dict(a.to_dict(level=2))

        for b, ind in zip([a_copy, a_clone, a_shallow, a_detach, a_dict_level_0, a_dict_level_1, a_dict_level_2],
                          [True, True, False, False, False, False, True]):
            assert a.allclose(b)
            assert b is not a
            assert a.are_independent(b, independent=ind)


    with pytest.raises(yastn.YastnError,
                       match=re.escape("EnvBP_local does not match d['type'] == EnvCTM_local")):
        d = ctm.to_dict()
        fpeps.envs.EnvBP_local.from_dict(d)


def test_lattice(config_kwargs):
    """ Test operations of Lattice"""
    geometries = [fpeps.SquareLattice(dims=(2, 2), boundary='obc'),
                  fpeps.CheckerboardLattice(),
                  fpeps.RectangularUnitcell(dims=(2, 2), pattern=[[1, 0], [1, 0]]),
                  fpeps.TriangularLattice(),]


    config = yastn.make_config(sym='U1', **config_kwargs)
    leg = yastn.Leg(config, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    T = yastn.rand(config, legs=[leg, leg, leg.conj()])
    C = yastn.rand(config, legs=[leg, leg.conj()])

    objs = [T,
            fpeps.envs.EnvCTM_local(tl=C, t=T, tr=C, r=T, br=C, b=T, bl=C, l=None),
            fpeps.envs.EnvCTM_projectors(hlt=T, hlb=T, hrt=T, hrb=T, vtl=T, vtr=T, vbl=T, vbr=None),
            fpeps.envs.EnvBP_local(t=C, l=C, b=C, r=None)]
    #
    for geometry in geometries:
        for cls in [fpeps.Lattice, fpeps.Peps]:
            net = cls(geometry)
            for obj in objs:
                net[0, 0] = net[0, 1] = obj
                net_copy = net.copy()
                net_clone = net.clone()
                net_shallow = net.shallow_copy()
                net_shallow.detach_()
                net_detach = net.detach()
                net_dict_level_0 = cls.from_dict(net.to_dict(level=0))
                net_dict_level_1 = cls.from_dict(net.to_dict(level=1))
                net_dict_level_2 = cls.from_dict(net.to_dict(level=2))
                #
                for nn, ind in zip([net_copy, net_clone, net_shallow, net_detach, net_dict_level_0, net_dict_level_1, net_dict_level_2],
                                    [True, True, False, False, False, False, True]):
                    assert net.allclose(nn)
                    assert nn is not net
                    assert net.are_independent(nn, independent=ind)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
