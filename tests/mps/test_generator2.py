from math import gamma
from os import remove
from tracemalloc import get_tracemalloc_memory
import numpy as np
import pytest
import yast
import yamps

from itertools import compress, product
import re

tol = 1e-12

def test_generator_mps():
    N = 10
    D_total = 16
    bds = (1,) + (D_total,) * (N - 1) + (1,)

    for sym, nn in (('Z2', (0,)), ('Z2', (1,)), ('U1', (N // 2,))):
        operators = yast.operators.SpinlessFermions(sym=sym)
        generate = yamps.Generator(N, operators)
        I = generate.I()
        assert pytest.approx(yamps.measure_overlap(I, I).item(), rel=tol) == 2 ** N
        O = I @ I + (-1 * I)
        assert pytest.approx(yamps.measure_overlap(O, O).item(), abs=tol) == 0
        psi = generate.random_mps(D_total=D_total, n = nn)
        assert psi.A[psi.last].get_legs(axis=psi.right[0]).t == (nn,)
        assert psi.A[psi.first].get_legs(axis=psi.left[0]).t == ((0,) * len(nn),)
        bds = psi.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])


def test_generator_mpo():
    N = 10
    t = 1
    mu = 0.2
    #operators = yast.operators.SpinlessFermions(sym='Z2')
    #generate = yamps.Generator(N, operators)
    #parameters = {"t": t, "mu": mu}
    #H_str = "\sum_{j=0}^{"+str(N-1)+"} mu*cp_{j}.c_{j} + \sum_{j=0}^{"+str(N-2)+"} cp_{j}.c_{j+1} + \sum_{j=0}^{"+str(N-2)+"} t*cp_{j+1}.c_{j}"
    #H = generate.mpo(H_str, parameters)
    H_str = "- \sum_{j \in range1} mu cp_{j} c_{j} + \sum_{i \in range2} \sum_{j \in range3} ( t_{i,j} ) ( cp_{i} c_{j} + 2 cp_{j} c_{i} ) -  h_{0} cp_{0} c_{0}"
    basis_dict = {"cp": lambda j: "cp_"+str(j),
                  "c": lambda j: "c_"+str(j)}
    params_dict = { "mu": lambda j: 1.0,
                    "range1": range(5),
                    "range2": range(5),
                    "range3": range(5),
                    "h": lambda j: j+1,
                    "t": lambda i,j: 1+i*j}

    tmp = H_str
    while "  " in tmp:
        tmp = tmp.replace("  ", " ")
    lall = tmp.replace(" \in ", ".in.").replace("\sum_", ".sum_.").split(" ")

    sep = {}

    terminate, hodor = ['+','-'], False
    isep = 0
    id_sep = 0
    # split to separate sums
    sep[id_sep] = []
    for it in range(len(lall)):
        tmp = lall[it]
        if tmp == '(':
            hodor, terminate = True, [')']
        if hodor and tmp in terminate:
            hodor, terminate = False, ['+', '-']
        if not hodor and tmp in terminate and it > isep:
            isep = it+1
            id_sep += 1
            sep[id_sep] = []
        sep[id_sep].append(tmp)
    print(sep, '\n\n\n')
    # analyze terms, get info on sums
    for k in sep.keys():
        spart = sep[k]
        issum = ['sum' in ix for ix in spart]
        is_sum = list(compress(spart, issum))
        not_sum = list(compress(spart, [not ix for ix in issum]))
        print('is_sum: ', is_sum)
        print('not_sum: ', not_sum)
        print()
        # sum instruction
        is_sum = [re.match(r".sum_.{(\w+).in.(\w+)}", tmp).group(1, 2) for tmp in is_sum]
        is_index = [tmp[0] for tmp in is_sum]
        is_range = [params_dict[tmp[1]] for tmp in is_sum]
        print(is_sum, '\n')
        # iterate over is_index-es each over is_range-es range of numbers
        range_span = product(*is_range, repeat=1)
        for idx in range_span:
            read_not_sum = [None]*len(not_sum)
            sgn_math = ['+', '-', '(', ')']
            sgn_basis = list(basis_dict.keys())
            sgn_params = list(params_dict.keys())
            # ger order of operations
            iorder = 0
            for inum, ival in enumerate(not_sum[::-1]):
                read_not_sum[-inum-1] = iorder
                if ival in sgn_math:
                    iorder += 1
            # remove bracket for convenience
            remove_braket = [ix not in ['(', ')'] for ix in not_sum]
            not_sum = list(compress(not_sum, remove_braket))
            read_not_sum = list(compress(read_not_sum, remove_braket))
            print('not_sum: ', not_sum)
            print('read_not_sum: ', read_not_sum)
            for iorder in np.unique(read_not_sum):
                tmp = [ira == iorder for ira in read_not_sum]
                itake = list(compress(not_sum, tmp))
                print(itake)
                place_holder = []
                op_holder, ind_holder, amp_holder = [], [], 1
                iind = None
                for inum, ival in enumerate(itake[::-1]):
                    # extract using map if needed
                    print(inum, ival)
                    get_info = re.match(r"(?P<ival>.*)_{(?P<iind>.*)}", ival)
                    #get_info = re.match(r"(?<=\{)(?P<index>.*).in.(?P<range>.*)(?=\})", ival)
                    if get_info:
                        ival, iind = get_info.group("ival"), get_info.group("iind")
                        iind = iind.split(',')
                        if all([i.isnumeric() for i in iind]):
                            # convert to numerical index
                            iind = [int(i) for i in iind]
                        else:
                            # find index from sum indexing
                            for i_ind in range(len(iind)):
                                for i_index in range(len(is_index)):
                                    if iind[i_ind] == is_index[i_index]:
                                        iind[i_ind] = idx[i_index]
                    if ival in sgn_basis:
                        op_holder.append(basis_dict[ival](*iind))
                        ind_holder.append(*iind)
                    elif ival in sgn_params:
                        print(ival, iind)
                        amp_holder *= params_dict[ival](*iind)
                    else:
                        if ival == '+':
                            amp_holder *= +1.0
                        elif ival == '-':
                            amp_holder *= -1.0
                        else:
                            amp_holder *= float(ival)
                # generate mpo product using automatic
                place_holder.append([ind_holder, op_holder, amp_holder])
    print(place_holder, inum)
    exit()


if __name__ == "__main__":
    # test_generator_mps()
    test_generator_mpo()
