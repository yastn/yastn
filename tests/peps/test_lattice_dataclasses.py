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
import re
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
    ctm_c4v = fpeps.envs.EnvCTM_c4v_local(tl=C, t=T)
    pro_c4v = fpeps.envs.EnvCTM_c4v_projectors(vtl=T, vtr=T)
    gau = fpeps.envs.Gauge(t=C, l=C, b=C, r=C)
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
    #
    for a in [ctm, pro, bp, ctm_c4v, pro_c4v, gau, ctm2, pro2, bp2, ctm3, pro3, bp3]:
        #
        for b in [ctm, pro, bp, ctm_c4v, pro_c4v, gau, ctm2, pro2, bp2, ctm3, pro3, bp3]:
            assert a.allclose(b) == (a is b)
        #
        a_copy = a.copy()
        a_clone = a.clone()
        a_shallow = a.shallow_copy()
        a_shallow.detach_()
        a_detach = a.detach()

        dicts = [a.to_dict(level) for level in [0, 1, 2]]
        a_dict = [a.from_dict(d) for d in dicts]
        a_split = [a.from_dict(yastn.combine_data_and_meta(*yastn.split_data_and_meta(d))) for d in dicts]

        for b, ind in zip([a_copy, a_clone, a_shallow, a_detach, *a_dict, *a_split],
                          [True, True, False, False, False, False, True, False, False, True]):
            assert a.allclose(b)
            assert b is not a
            assert a.are_independent(b, independent=ind)

    #
    # testing fields
    assert sorted(ctm.fields()) == ['b', 'bl', 'br', 'l', 'r', 't', 'tl', 'tr']
    assert sorted(pro.fields()) == ['hlb', 'hlt', 'hrb', 'hrt', 'vbl', 'vbr', 'vtl', 'vtr']
    assert sorted(bp.fields()) == ['b', 'bR', 'l', 'lR', 'r', 'rR', 't', 'tR']
    assert sorted(gau.fields()) == ['b', 'l', 'r', 't']
    assert sorted(ctm_c4v.fields()) == ['t', 'tl']
    assert sorted(pro_c4v.fields()) == ['vtl', 'vtr']
    #
    # testing field selection
    assert sorted(ctm_c4v.fields(among=['tl', 'tr', 'bl', 'br'])) == ['tl']
    assert sorted(ctm.fields(among=['b', 'tl'])) == ['b', 'tl']
    #
    # testing symmetry of ctm_c4v
    assert ctm_c4v.t is ctm_c4v.b is ctm_c4v.r is ctm_c4v.l
    assert ctm_c4v.tl is ctm_c4v.tr is ctm_c4v.bl is ctm_c4v.br
    #
    # test automatic convertion of EnvBP_local  t = tR.H @ tR
    C0 = yastn.rand(config, legs=[leg, leg.conj()])
    C1 = yastn.rand(config, legs=[leg, leg.conj()])
    bp = fpeps.envs.EnvBP_local(t=C0)
    assert (bp.t - C0).norm() < 1e-12
    bp.t = None
    assert bp.t is None
    bp.tR = C1
    assert (bp.t - C1.H @ C1).norm() < 1e-12
    #
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
