""" Environments for the <mps| mpo |mps> and <mps|mps>  contractions. """
from __future__ import annotations
from itertools import groupby
from typing import Sequence
from . import MpsMpoOBC
from ._env import Env


def vdot(*args) -> number:
    r"""
    Calculate the overlap :math:`\langle \textrm{bra}|\textrm{ket}\rangle`,
    or :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle` depending on the number of provided agruments.

    Parameters
    -----------
    *args: yastn.tn.mps.MpsMpoOBC
    """
    if len(args) == 2:
        return measure_overlap(*args)
    return measure_mpo(*args)


def measure_overlap(bra, ket) -> number:
    r"""
    Calculate overlap :math:`\langle \textrm{bra}|\textrm{ket} \rangle`.
    Conjugate of MPS :code:`bra` is computed internally.

    MPSs :code:`bra` and :code:`ket` must have matching length,
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    ket: yastn.tn.mps.MpsMpoOBC
    """
    env = Env(bra, ket)
    env.setup_(to='first')
    return env.measure(bd=(-1, 0))


def measure_mpo(bra, op: MpsMpoOBC | Sequence[tuple(MpsMpoOBC, number)], ket) -> number:
    r"""
    Calculate expectation value :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle`.

    Conjugate of MPS :code:`bra` is computed internally.
    MPSs :code:`bra`, :code:`ket`, and MPO :code:`op` must have matching length,
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    op: yastn.tn.mps.MpsMpoOBC or Sequence[tuple(MpsMpoOBC,number)]
        Operator written as (sums of) MPO.

    ket: yastn.tn.mps.MpsMpoOBC
    """
    env = Env(bra, [op, ket])
    env.setup_(to='first')
    return env.measure(bd=(-1, 0))


def measure_1site(bra, O, ket) -> dict[int, number]:
    r"""
    Calculate expectation values :math:`\langle \textrm{bra}|\textrm{O}_i|\textrm{ket} \rangle` for local operator :code:`O` at each lattice site `i`.

    Local operators can be provided as dictionary {site: operator}, limiting the calculation to provided sites.
    Conjugate of MPS :code:`bra` is computed internally.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    O: yastn.Tensor or dict
        An operator with signature (1, -1).
        It is possible to provide a dictionary {site: operator}

    ket: yastn.tn.mps.MpsMpoOBC
    """
    op = sorted(O.items()) if isinstance(O, dict) else [(n, O) for n in ket.sweep(to='last')]
    env = Env(bra, ket)
    env.setup_(to='first').setup_(to='last')
    results = {}
    for n, o in op:
        env.update_env_op_(n, o, to='first')
        results[n] = env.measure(bd=(n - 1, n))
    return results


def measure_2site(bra, O, P, ket, pairs=None) -> dict[tuple[int, int], number]:
    r"""
    Calculate expectation values :math:`\langle \textrm{bra}|\textrm{O}_i \textrm{P}_j|\textrm{ket} \rangle`
    of local operators :code:`O` and :code:`P` for each pair of lattice sites :math:`i < j`.

    Conjugate of MPS :code:`bra` is computed internally. Includes fermionic strings via swap_gate for fermionic operators.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    O, P: yastn.Tensor or dict
        Operators with signature (1, -1).
        It is possible to provide a dictionaries {site: operator}

    ket: yastn.tn.mps.MpsMpoOBC

    pairs: list[tuple[int, int]]
        It is possible to provide a list of pairs to limit the calculation.
        By default is None, when all pairs are calculated.
    """
    if pairs is not None:
        n1s = sorted(set(x[1] for x in pairs))
        pairs = sorted((-i, j) for i, j in pairs)
        pairs = [(-i, j) for i, j in pairs]
    else:
        n1s = range(ket.N)
        pairs = [(i, j) for i in range(ket.N - 1, -1, -1) for j in range(i + 1, ket.N)]

    env = Env(bra, ket)
    env.setup_(to='first').setup_(to='last')
    for n1 in n1s:
        env.update_env_op_(n1, P, to='first')

    results = {}
    for n0, n01s in groupby(pairs, key=lambda x: x[0]):
        env.update_env_op_(n0, O, to='last')
        _, n1 = next(n01s)
        for n in env.ket.sweep(to='last', df=n0 + 1):
            if n == n1:
                results[(n0, n1)] = env.measure(bd=(n - 1, n))
                try:
                    _, n1 = next(n01s)
                except StopIteration:
                    break
            env.update_env_(n, to='last')
    return dict(sorted(results.items()))
