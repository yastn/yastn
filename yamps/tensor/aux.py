r"""
Functions creating tensor.
"""


def block(td, common_legs, ndim):
    """ Assemble new tensor by blocking a set of tensors.

        Parameters
        ----------
        td : dict
            dictionary of tensors {(k,l): tensor at position k,l}.
            Length of tuple should be equall to tensor.ndim - len(common_legs)

        common_legs : list
            Legs which are not blocked

        ndim : int
            All tensor should have the same rank ndim
    """
    try:
        ls = len(common_legs)
        out_s = tuple(common_legs)
    except TypeError:
        out_s = (common_legs,)  # indices going u
        ls = 1

    out_m = tuple(ii for ii in range(ndim) if ii not in out_s)
    out_ma = np.array(out_m, dtype=int)
    li = ndim - ls
    pos = []
    newdtype = 'float64'
    for ind, ten in td.items():
        if li != len(ind):
            raise TensorShapeError('block: wrong tensors rank or placement')
        pos.append(ind)
        if ten.dtype == 'complex128':
            newdtype = 'complex128'
    pos.sort()

    # all charges and bond dimensions
    tlist, Dlist = {}, {}
    for ind in pos:
        tt, DD = td[ind].get_tD_list()
        tlist[ind] = tt
        Dlist[ind] = DD

    # combinations of charges on legs to merge
    t_out_m = [np.unique(td[ind].tset[:, out_ma], axis=0) for ind in pos]
    t_out_unique = np.unique(np.vstack(t_out_m), axis=0)

    # positions including those charges
    t_out_pos = []
    for tt in t_out_unique:
        t_out_pos.append([ind for ind, tm in zip(pos, t_out_m) if not np.any(np.sum(np.abs(tt - tm), axis=1))])

    # print(t_out_m)
    # print(t_out_unique)
    # print(t_out_pos)

    for tt, pos_tt in zip(t_out_unique, t_out_pos):
        for ind in pos_tt:
            for kk in td[ind].tset:
                if np.all(kk[out_ma] == tt):
                    pass
        # pos_tt

    posa = np.array(pos, dtype=int)
    legs_ind = []  # indices on specific legs
    legs_D = []  # and corresponding keys
    kk = -1
    for ii in range(ndim):
        if ii in out_m:
            kk += 1
            x, y = np.unique(posa[:, kk], return_index=True)
            legs_ind.append(list(x))
            legs_D.append([td[pos[ll]].get_shape()[ii] for ll in y])
        else:
            legs_D.append([td[pos[0]].get_shape()[ii]])

    Ad = {key: td[key].A for key in pos}
    to_execute = [(0, pos, legs_ind, legs_D)]

    c = Tensor(td[pos[0]].settings, dtype=newdtype)
    c.A = c.backend.block(Ad, to_execute)
    return c
