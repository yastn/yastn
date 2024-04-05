""" Test operation of peps.Lattice and peps.Peps that inherits Lattice"""
import pytest
import yastn
import yastn.tn.fpeps as fpeps


tol = 1e-12

def test_dpt():
    """ Generate a few lattices veryfing expected output of some functions. """
    ops = yastn.operators.SpinlessFermions(sym='U1')

    # single peps tensor
    leg0 = yastn.Leg(ops.config, s=-1, t=(-2, 0, 2), D=(1, 3, 1))
    leg1 = yastn.Leg(ops.config, s=1,  t=(-2, 0, 2), D=(1, 2, 1))
    leg2 = yastn.Leg(ops.config, s=1,  t=(-2, 0), D=(2, 1))
    leg3 = yastn.Leg(ops.config, s=-1, t=(0, 2), D=(1, 1))
    leg4 = ops.space()

    A = yastn.rand(ops.config, legs=[leg0, leg1, leg2, leg3, leg4])
    A = A.fuse_legs(axes=((0, 1), (2, 3), 4))
    # we keep the legs [t l] [b r] fused for efficiency of higher symmetries

    # creat double peps tensor
    T0123 = fpeps.DoublePepsTensor(top=A, btm=A)

    assert T0123.get_shape() == (25, 16, 9, 4)
    T1230 = T0123.transpose(axes=(1, 2, 3, 0))
    assert T1230.get_shape() == (16, 9, 4, 25)
    T2301 = T1230.transpose(axes=(1, 2, 3, 0))
    assert  T2301.get_shape() == (9, 4, 25, 16)
    T3012 = T1230.transpose(axes=(2, 3, 0, 1))
    assert  T3012.get_shape() == (4, 25, 16, 9)

    T0321 = T1230.transpose(axes=(3, 2, 1, 0))
    assert T0321.get_shape(axes=1) == 4
    assert T0321.get_legs(axes=0).t == ((-4,), (-2,), (0,), (2,), (4,))
    T3210 = T3012.transpose(axes=(0, 3, 2, 1))
    assert  T3210.get_shape() == (4, 9, 16, 25)
    T2103 = T0321.transpose(axes=(2, 3, 0, 1))
    assert  T2103.get_shape() == (9, 16, 25, 4)
    T1032 = T0123.transpose(axes=(1, 0, 3, 2))
    assert  T1032.get_shape() == (16, 25, 4, 9)


    # prepare legs for boundary vectors to attach
    legf0, legf1, legf2, legf3 = T0123.get_legs()

    t01 = yastn.rand(ops.config, legs=[leg1, legf1.conj(), legf0.conj(), leg0])
    assert t01.get_shape() == (4, 16, 25, 5)
    tt01a = T0123._attach_01(t01)
    tt01b = T2301._attach_23(t01)
    assert all(tmp.get_shape() == (4, 9, 5, 4) for tmp in [tt01a, tt01b])
    t10 = yastn.rand(ops.config, legs=[leg1, legf0.conj(), legf1.conj(), leg0])
    tt10a = T1032._attach_01(t10)
    tt10b = T3210._attach_23(t10)
    assert all(tmp.get_shape() == (4, 4, 5, 9) for tmp in [tt10a, tt10b])

    t23 = yastn.rand(ops.config, legs=[leg1, legf3.conj(), legf2.conj(), leg0])
    assert t23.get_shape() == (4, 4, 9, 5)
    tt23a = T0123._attach_23(t23)
    tt23b = T2301._attach_01(t23)
    assert all(tmp.get_shape() == (4, 25, 5, 16) for tmp in [tt23a, tt23b])
    t32 = yastn.rand(ops.config, legs=[leg1, legf2.conj(), legf3.conj(), leg0])
    tt32a = T1032._attach_23(t32)
    tt32b = T3210._attach_01(t32)
    assert all(tmp.get_shape() == (4, 16, 5, 25) for tmp in [tt32a, tt32b])

    t12 = yastn.rand(ops.config, legs=[leg1, legf2.conj(), legf1.conj(), leg0])
    assert t12.get_shape() == (4, 9, 16, 5)
    tt12a = T1230._attach_01(t12)
    tt12b = T3012._attach_23(t12)
    assert all(tmp.get_shape() == (4, 4, 5, 25) for tmp in [tt12a, tt12b])
    t21 = yastn.rand(ops.config, legs=[leg1, legf1.conj(), legf2.conj(), leg0])
    tt21a = T2103._attach_01(t21)
    tt21b = T0321._attach_23(t21)
    assert all(tmp.get_shape() == (4, 25, 5, 4) for tmp in [tt21a, tt21b])

    t30 = yastn.rand(ops.config, legs=[leg1, legf0.conj(), legf3.conj(), leg0])
    assert t30.get_shape() == (4, 25, 4, 5)
    tt30a = T3012._attach_01(t30)
    tt30b = T1230._attach_23(t30)
    assert all(tmp.get_shape() == (4, 16, 5, 9) for tmp in [tt30a, tt30b])
    t03 = yastn.rand(ops.config, legs=[leg1, legf3.conj(), legf0.conj(), leg0])
    tt03a = T0321._attach_01(t03)
    tt03b = T2103._attach_23(t03)
    assert all(tmp.get_shape() == (4, 9, 5, 16) for tmp in [tt03a, tt03b])


    with pytest.raises(yastn.YastnError):
        T0123.transpose(axes=(1, 0, 2, 3))
        # DoublePEPSTensor only supports permutations that retain legs' ordering.
    with pytest.raises(yastn.YastnError):
        fpeps.DoublePepsTensor(top=A, btm=A, transpose=(1, 2, 0, 3))
        # DoublePEPSTensor only supports permutations that retain legs' ordering.

if __name__ == '__main__':
    test_dpt()
