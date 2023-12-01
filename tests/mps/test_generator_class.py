import numpy as np
import yastn
import yastn.tn.mps as mps
from yastn.tn.mps._latex2term import  splitt, interpret, string2list, latex2term, single_term
try:
    from .configs import config_dense as cfg
    from .test_build_mpo_manually import build_mpo_nn_hopping_manually
    from .test_generate_mpo import build_mpo_hopping_Hterm
except ImportError:
    from configs import config_dense as cfg
    from test_build_mpo_manually import build_mpo_nn_hopping_manually
    from test_generate_mpo import build_mpo_hopping_Hterm
# pytest modifies cfg to inject different backends and devices during tests


def mpo_nn_hopping_latex(N, t, mu, sym="U1", config=None):
    """
    Nearest-neighbor hopping Hamiltonian on N sites
    with hopping amplitude t and chemical potential mu.
    """
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)

    Hstr = "\sum_{j,k \in NN} t (cp_{j} c_{k}+cp_{k} c_{j})"
    Hstr += " + \sum_{i \in sites} mu cp_{i} c_{i}"
    parameters = {"t": t,
                  "mu": mu,
                  "sites": list(range(N)),
                  "NN": list((i, i+1) for i in range(N-1))}

    generate = mps.Generator(N, ops)
    H = generate.mpo_from_latex(Hstr, parameters=parameters)
    return H

def test_nn_hopping_latex_map(config=cfg, tol=1e-12):
    # uniform chain with nearest neighbor hopping
    # notation:
    # * in the sum there are all elements which are connected by multiplication, so \sum_{.} -1 ... should be \sum_{.} (-1) ...
    # * 1j is an imaginary number
    # * multiple sums are supported so you can write \sum_{.} \sum_{.} ...
    # * multiplication of the sum is allowed but '*' or bracket is needed.
    #   ---> this is an artifact of allowing space=' ' to be equivalent to multiplication
    #   E.g.1, 2 \sum... can be written as 2 (\sum...) or 2 * \sum... or (2) * \sum...
    #   E.g.2, \sum... \sum.. write as \sum... * \sum... or (\sum...) (\sum...)
    #   E.g.4, -\sum... is supported and equivalent to (-1) * \sum...
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    H_str = "\sum_{j,k \in NN} t_{j,k} (cp_{j} c_{k}+cp_{k} c_{j}) + \sum_{i \in sites} mu cp_{i} c_{i}"
    for sym in ['Z2', 'U1']:
        ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)
        for t in [0, 0.2, -0.3]:
            for mu in [0.2, -0.3]:
                for N in [3, 4]:
                    H2 = build_mpo_nn_hopping_manually(N=N, t=t, mu=mu, sym=sym, config=config)
                    H3 = mpo_nn_hopping_latex(N=N, t=t, mu=mu, sym=sym, config=config)
                    H2norm = H2.norm()
                    example_mapping = [{i: i for i in range(N)},
                                       {str(i): i for i in range(N)},
                                       {(str(i), 'A'): i for i in range(N)}]
                    example_parameters = [
                        {"t": t * np.ones((N,N)), "mu": mu,
                         "sites": list(range(N)),
                         "NN": list((i, i+1) for i in range(N - 1))},
                        {"t": t * np.ones((N,N)), "mu": mu,
                         "sites": [str(i) for i in range(N)],
                         "NN": list((str(i), str(i+1)) for i in range(N - 1))},
                        {"t": t * np.ones((N,N)), "mu": mu,
                         "sites": [(str(i),'A') for i in range(N)],
                         "NN": list(((str(i), 'A'), (str(i+1), 'A')) for i in range(N - 1))}]

                    for (emap, eparam) in zip(example_mapping, example_parameters):
                        generate = mps.Generator(N, ops, map=emap)
                        H1 = generate.mpo_from_latex(H_str, eparam)
                        assert (H1 - H2).norm() < tol * H2norm
                        assert (H1 - H3).norm() < tol * H2norm


def mpo_hopping_latex(J=np.array([[0.5, 1], [0, 0.2]]), sym="U1", config=None):
    """
    The upper triangular part of NxN matrix J defines hopping amplitudes,
    and the diagonal defines on-site chemical potentials of N-site Hamiltonian
    """
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)

    N = len(J)

    Hstr = "\sum_{j,k \in NN} J_{j,k} (cp_{j} c_{k}+cp_{k} c_{j})"
    Hstr += " + \sum_{i \in sites} J_{i,i} cp_{i} c_{i}"
    parameters = {"J": J,
                  "sites": list(range(N)),
                  "NN": list((i, j) for i in range(N-1)
                             for j in range(i + 1, N))}

    generate = mps.Generator(N, ops)
    H = generate.mpo_from_latex(Hstr, parameters=parameters)
    return H


def test_mpo_hopping_latex(config=cfg, tol=1e-12):
    opts_config = {} if config is None else \
                {'backend': config.backend,
                'default_device': config.default_device}

    # the model is random with handom hopping and on-site energies. sym is symmetry for tensors we will use
    sym, N = 'U1', 5

    # generate data for random Hamiltonian
    J = np.triu(np.random.rand(N, N))
    mu = np.diag(J)
    t = np.triu(J, 1)

    # initialize set of basic ops for the model we want to work with
    # and Generator instance
    ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)
    generate = mps.Generator(N, ops)

    # define parameters for automatic generator and Hamiltonian in a latex-like form
    eparam ={"t": t, "mu": mu, 'sites': list(range(N))}
    Hstr = "\sum_{j\in sites} \sum_{k\in sites} t_{j,k} (cp_{j} c_{k} + cp_{k} c_{j})"
    Hstr += " + \sum_{j\in sites} mu_{j} cp_{j} c_{j}"

    H1 = generate.mpo_from_latex(Hstr, eparam)
    H2 = build_mpo_hopping_Hterm(J, sym=sym, config=config)
    H3 = mpo_hopping_latex(J, sym=sym, config=config)

    assert abs(mps.vdot(H1, H2) - H1.norm() * H2.norm()) < tol
    assert abs(mps.vdot(H1, H3) - H1.norm() * H3.norm()) < tol


def test_latex2term_unit_tests():
    # Test splitting from latex-like instruction to list used as the input
    examples_A = ("   s_j   + d_j    +   1 + 2  + 3 ", \
                "   s_j   - d_j    +   1 - 2  + 3 ",\
                "s_j-d_j+1-2+3",\
                "\sum_{j\in range} (a_j + b_j)",\
                "\sum_{j \in   range}  (a_j + b_j)",\
                "\sum_{j\inrange}  (a_j + b_j)",\
                "\sum_{j\in range}\sum_{k\in range_L}(a_j+b_k)   ",\
                )
    test_examples_A1 = (["s_j","+","d_j","+","1","+","2","+","3"],\
                    ["s_j","+","minus",'*',"d_j","+","1","+","minus",'*',"2","+","3"],\
                    ["s_j","+","minus",'*',"d_j","+","1","+","minus",'*',"2","+","3"],\
                    ["sum","{j.in.range}","(","a_j","+","b_j", ")"],\
                    ["sum","{j.in.range}","(","a_j","+","b_j", ")"],\
                    ["sum","{j.in.range}","(","a_j","+","b_j", ")"],\
                    ["sum","{j.in.range}","sum","{k.in.range_L}","(","a_j","+","b_k", ")"],\
                    )
    for x, test_x in zip(examples_A, test_examples_A1):
        assert string2list(x) == test_x

    # Test interpretation of the string to single_term container
    examples_A = (  "   s_{j}   + d_{j}    +   1 + 2  + 3 ", \
                    "  s_{j} d_{j}   1 2  3 ", \
                    "  s_{j} d_{j} +  1 2  3 ", \
                    "sum {j.in.range} s_{j} d_{j} ", \
                    "sum {i,j.in.range} s_{j} d_{j} ", \
                    "sum {i.in.range} sum {j.in.range} s_{i} d_{j} ", \
                )
    test_examples_A2 = (\
                    ('+', [[single_term(op=(('s', 'j'),))], [single_term(op=(('d', 'j'),))], [single_term(op=(('1',),))], [single_term(op=(('2',),))], [single_term(op=(('3',),))]]),
                    [single_term(op=(('s', 'j'),('d', 'j'),('1',),('2',),('3',)))],
                    ('+', [[single_term(op=(('s', 'j'), ('d', 'j')))], [single_term(op=(('1',), ('2',), ('3',)))]]),
                    ('+', [(('sum', '{j.in.range}'), [single_term(op=(('s', 'j'), ('d', 'j')))])]),
                    ('+', [(('sum', '{i,j.in.range}'), [single_term(op=(('s', 'j'), ('d', 'j')))])]),
                    ('+', [(('sum', '{i.in.range}'), (('sum', '{j.in.range}'), [single_term(op=(('s', 'i'), ('d', 'j')))]))]),
                    )
    for x, test_x in zip(examples_A, test_examples_A2):
        assert splitt(string2list(x),0) == test_x

    examples_B = ("a*b*c*d+d*b",\
                "a*b*c*d*d*b",\
                "a b c*d d +b",\
                "a b c d d b",\
                )
    # Test the use of instrunction form split to make operations on single_term-s
    test_examples_B2 = (\
                        [single_term(op=(('a',), ('b',), ('c',), ('d',))), single_term(op=(('d',), ('b',)))],\
                        [single_term(op=(('a',), ('b',), ('c',), ('d',), ('d',), ('b',)))],\
                        [single_term(op=(('a',), ('b',), ('c',), ('d',), ('d',))), single_term(op=(('b',),))],\
                        [single_term(op=(('a',), ('b',), ('c',), ('d',), ('d',), ('b',)))],\
                        )
    for x, test_x in zip(examples_B, test_examples_B2):
        assert interpret(splitt(string2list(x),0), {}) == test_x

    # Test interpreter with sum substitution
    examples_C =["\sum_{j\in range} (a_{j}  b_{j})",
                 "\sum_{j\in range}\sum_{k\in range_L} a_{j} b_{j}",
                 "\sum_{j \in   range}  (a_{j} + b_{j})",
                 "\sum_{j\inrange}  (a_{j} + b_{j})",
                 "\sum_{j\in range}\sum_{k\in range_L}(a_{j}+b_{k})   ",
                 "\sum_{i,j\in NN} A_{i,j} ap_{i} a_{j}"]
    test_examples_C1 = [[single_term(op=(('a', 'i1'), ('b', 'i1'))), single_term(op=(('a', 'i2'), ('b', 'i2'))), single_term(op=(('a', 'i3'), ('b', 'i3')))],\
                        [single_term(op=(('a', 'i1'), ('b', 'i1'))), single_term(op=(('a', 'i1'), ('b', 'i1'))), single_term(op=(('a', 'i1'), ('b', 'i1'))), single_term(op=(('a', 'i2'), ('b', 'i2'))), single_term(op=(('a', 'i2'), ('b', 'i2'))), single_term(op=(('a', 'i2'), ('b', 'i2'))), single_term(op=(('a', 'i3'), ('b', 'i3'))), single_term(op=(('a', 'i3'), ('b', 'i3'))), single_term(op=(('a', 'i3'), ('b', 'i3')))],
                        [single_term(op=(('a', 'i1'),)), single_term(op=(('b', 'i1'),)), single_term(op=(('a', 'i2'),)), single_term(op=(('b', 'i2'),)), single_term(op=(('a', 'i3'),)), single_term(op=(('b', 'i3'),))],
                        [single_term(op=(('a', 'i1'),)), single_term(op=(('b', 'i1'),)), single_term(op=(('a', 'i2'),)), single_term(op=(('b', 'i2'),)), single_term(op=(('a', 'i3'),)), single_term(op=(('b', 'i3'),))],
                        [single_term(op=(('a', 'i1'),)), single_term(op=(('b', 'A'),)), single_term(op=(('a', 'i1'),)), single_term(op=(('b', 'B'),)), single_term(op=(('a', 'i1'),)), single_term(op=(('b', 'C'),)), single_term(op=(('a', 'i2'),)), single_term(op=(('b', 'A'),)), single_term(op=(('a', 'i2'),)), single_term(op=(('b', 'B'),)), single_term(op=(('a', 'i2'),)), single_term(op=(('b', 'C'),)), single_term(op=(('a', 'i3'),)), single_term(op=(('b', 'A'),)), single_term(op=(('a', 'i3'),)), single_term(op=(('b', 'B'),)), single_term(op=(('a', 'i3'),)), single_term(op=(('b', 'C'),))],
                        [single_term(op=(('A', 'i1', 'i2'), ('ap', 'i1'), ('a', 'i2'))), single_term(op=(('A', 'i2', 'i3'), ('ap', 'i2'), ('a', 'i3'))), single_term(op=(('A', 'i3', 'i4'), ('ap', 'i3'), ('a', 'i4')))]]
    param_dict = {}
    param_dict["minus"] = -1.0
    param_dict["1j"] = 1j
    param_dict["range"] = ("i1", "i2", "i3")
    param_dict["range_L"] = ("A", "B", "C")
    param_dict["NN"] = zip(["i1","i2","i3"], ["i2","i3","i4"])
    for x, test_x in zip(examples_C, test_examples_C1):
        assert latex2term(x, param_dict) == test_x


if __name__ == "__main__":
    test_nn_hopping_latex_map()
    test_mpo_hopping_latex()
    test_latex2term_unit_tests()
