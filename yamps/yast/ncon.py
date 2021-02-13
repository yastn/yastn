""" ncon routine for automatic contraction of a list of tensors. """
from .yast import YastError


def ncon(ts, inds, conjs=None):
    """Execute series of tensor contractions"""
    if len(ts) != len(inds):
        raise YastError('Wrong number of tensors')
    for ii, ind in enumerate(inds):
        if ts[ii].ldim() != len(ind):
            raise YastError('Wrong number of legs in %02d-th tensors.' % ii)

    ts = {ind: val for ind, val in enumerate(ts)}
    cutoff = 512
    cutoff2 = 2 * cutoff
    edges = [(order, leg, ten) if order >= 0 else (-order + cutoff2, leg, ten) for ten, el in enumerate(inds) for leg, order in enumerate(el)]

    edges.append((cutoff, cutoff, cutoff))
    conjs = [0] * len(inds) if conjs is None else list(conjs)
    edges = sorted(edges, reverse=True, key=lambda x: x[0])  # order of contraction with info on tensor and axis

    order1, leg1, ten1 = edges.pop()
    ax1, ax2 = [], []
    while order1 != cutoff:  # tensordot two tensors; or trace one tensor
        order2, leg2, ten2 = edges.pop()
        if (order1 != order2):
            raise YastError('Contracted legs do not match')
        if ten1 < ten2:
            (t1, t2) = (ten1, ten2)
            ax1.append(leg1)
            ax2.append(leg2)
        else:
            (t1, t2) = (ten2, ten1)
            ax1.append(leg2)
            ax2.append(leg1)
        if edges[-1][0] == cutoff or min(edges[-1][2], edges[-2][2]) != t1 or max(edges[-1][2], edges[-2][2]) != t2:
            # execute contraction
            if t1 == t2:  # trace
                ts[t1] = ts[t1].trace(axes=(ax1, ax2))
                ax12 = ax1 + ax2
                for ii, (order, leg, ten) in enumerate(edges):
                    if (ten == t1):
                        edges[ii] = (order, leg - sum(ii < leg for ii in ax12), ten)
            else:  # tensordot
                ts[t1] = ts[t1].dot(ts[t2], axes=(ax1, ax2), conj=(conjs[t1], conjs[t2]))
                conjs[t1], conjs[t2] = 0, 0
                del ts[t2]
                lt1 = sum(ii[2] == t1 for ii in edges)  # legs of t1
                for ii, (order, leg, ten) in enumerate(edges):
                    if ten == t1:
                        edges[ii] = (order, leg - sum(ii < leg for ii in ax1), ten)
                    elif ten == t2:
                        edges[ii] = (order, lt1 + leg - sum(ii < leg for ii in ax2), t1)
            ax1, ax2 = [], []
        order1, leg1, ten1 = edges.pop()

    if edges:
        while len(ts) > 1:
            edges = sorted(edges, key=lambda x: x[2])
            t1 = edges[0][2]
            t2 = [key for key in ts.keys() if key != t1][0]
            ts[t1] = ts[t1].dot(ts[t2], axes=((), ()), conj=(conjs[t1], conjs[t2]))
            conjs[t1], conjs[t2] = 0, 0
            lt1 = sum(ii[2] == t1 for ii in edges)
            for ii, (order, leg, ten) in enumerate(edges):
                if ten == t2:
                    edges[ii] = (order, leg + lt1, t1)
            del ts[t2]
        order = [ed[1] for ed in sorted(edges)]
        _, result = ts.popitem()
        return result.transpose(axes=order, inplace=True)
    result = 1
    for num in ts.values():
        result *= num.to_number()
    return result
