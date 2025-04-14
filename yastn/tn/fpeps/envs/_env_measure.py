""" Common measure functions for EnvCTM and EnvBoudndaryMPS """
from ... import mps
from .... import YastnError
from ....operators import sign_canonical_order


def _measure_nsite(env, *operators, sites=None, dirn='tb', opts_svd=None, opts_var=None) -> float:
    r"""
    Calculate expectation value of a product of local operators.

    dirn == 'lr' or 'tb'
    """
    if sites is None or len(operators) != len(sites):
        raise YastnError("Number of operators and sites should match.")

    sign = sign_canonical_order(*operators, sites=sites, f_ordered=env.psi.f_ordered)
    ops = {}
    for n, op in zip(sites, operators):
        ops[n] = ops[n] @ op if n in ops else op

    if opts_var is None:
        opts_var = {'max_sweeps': 2}
    if opts_svd is None:
        rr = env.yrange if dirn == 'lr' else env.xrange
        D_total = max(max(env[i, d].get_bond_dimensions()) for i in range(*rr) for d in dirn)
        opts_svd = {'D_total': D_total}

    if dirn == 'lr':
        i0, i1 = env.yrange[0], env.yrange[1] - 1
        bra = env[i1, 'r'].conj()
        tms = {ny: env[ny, 'v'] for ny in range(*env.yrange)}
        ket = env[i0, 'l']
        dx = env.xrange[0] - env.offset
        tens = {(nx, ny): tm[nx - dx] for ny, tm in tms.items() for nx in range(*env.xrange)}
    else:
        i0, i1 = env.xrange[0], env.xrange[1] - 1
        bra = env[i1, 'b'].conj()
        tms = {nx: env[nx, 'h'] for nx in range(*env.xrange)}
        ket = env[i0, 't']
        dy = env.yrange[0] - env.offset
        tens = {(nx, ny): tm[ny - dy] for nx, tm in tms.items() for ny in range(*env.yrange)}

    val_no = contract_window(bra, tms, ket, i0, i1, opts_svd, opts_var)

    nx0, ny0 = env.xrange[0], env.yrange[0]
    for (nx, ny), op in ops.items():
        tens[nx, ny].set_operator_(op)
        tens[nx, ny].add_charge_swaps_(op.n, axes=('b0' if nx == nx0 else 'k1'))
        for ii in range(nx0 + 1, nx):
            tens[ii, ny].add_charge_swaps_(op.n, axes=['k1', 'k4', 'b3'])
        if nx > nx0:
            tens[nx0, ny].add_charge_swaps_(op.n, axes=['b0', 'k4', 'b3'])
        for jj in range(ny0, ny):
            tens[nx0, jj].add_charge_swaps_(op.n, axes=['b0', 'k2', 'k4'])

    val_op = contract_window(bra, tms, ket, i0, i1, opts_svd, opts_var)
    return sign * val_op / val_no


def contract_window(bra, tms, ket, i0, i1, opts_svd, opts_var):
    """ Helper funcion performing mps contraction of < mps0 | mpo mpo ... | mps1 >. """
    vec = ket
    for ny in range(i0, i1):
        vec_next = mps.zipper(tms[ny], vec, opts_svd=opts_svd)
        mps.compression_(vec_next, (tms[ny], vec), method='1site', normalize=False, **opts_var)
        vec = vec_next
    return mps.vdot(bra, tms[i1], vec)
