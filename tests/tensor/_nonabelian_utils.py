import yastn


def four_leg_tensor(config, D=2):
    """Invariant four-spin tensor with two fusion channels."""
    charge = (1,) if config.sym.SYM_ID == 'SU2' else ((1, 2),)
    leg = yastn.Leg(config, s=1, t=charge, D=(D,))
    tensor = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()))
    assert tensor.nblocks > 1  # two independent singlet fusion channels
    return tensor


def matrix_tensor(config, D=(2, 3, 2)):
    charges = ((0,), (1,), (2,)) if config.sym.SYM_ID == 'SU2' else (
        (0, 0), (1, 0), (2, 0))
    leg = yastn.Leg(config, s=1, t=charges, D=D)
    tensor = yastn.rand(config, legs=(leg.conj(), leg))
    assert tensor.nblocks == 3
    return tensor
