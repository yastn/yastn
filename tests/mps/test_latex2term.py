import pytest
import sys
sys.path.insert(1, './../../yast/tn/mps')
from _latex2term import  splitt,\
                        interpret,\
                        string2list,\
                        latex2term,\
                        single_term

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
    examples_C =(\
                "\sum_{j\in range} (a_{j}  b_{j})",\
                "\sum_{j\in range}\sum_{k\in range_L} a_{j} b_{j}",\
                "\sum_{j \in   range}  (a_{j} + b_{j})",\
                "\sum_{j\inrange}  (a_{j} + b_{j})",\
                "\sum_{j\in range}\sum_{k\in range_L}(a_{j}+b_{k})   ",\
                "\sum_{i,j\in NN} A_{i,j} ap_{i} a_{j}",\
                )
    test_examples_C1 =  (\
                        [single_term(op=(('a', 'i1'), ('b', 'i1'))), single_term(op=(('a', 'i2'), ('b', 'i2'))), single_term(op=(('a', 'i3'), ('b', 'i3')))],\
                        [single_term(op=(('a', 'i1'), ('b', 'i1'))), single_term(op=(('a', 'i1'), ('b', 'i1'))), single_term(op=(('a', 'i1'), ('b', 'i1'))), single_term(op=(('a', 'i2'), ('b', 'i2'))), single_term(op=(('a', 'i2'), ('b', 'i2'))), single_term(op=(('a', 'i2'), ('b', 'i2'))), single_term(op=(('a', 'i3'), ('b', 'i3'))), single_term(op=(('a', 'i3'), ('b', 'i3'))), single_term(op=(('a', 'i3'), ('b', 'i3')))],\
                        [single_term(op=(('a', 'i1'),)), single_term(op=(('b', 'i1'),)), single_term(op=(('a', 'i2'),)), single_term(op=(('b', 'i2'),)), single_term(op=(('a', 'i3'),)), single_term(op=(('b', 'i3'),))],\
                        [single_term(op=(('a', 'i1'),)), single_term(op=(('b', 'i1'),)), single_term(op=(('a', 'i2'),)), single_term(op=(('b', 'i2'),)), single_term(op=(('a', 'i3'),)), single_term(op=(('b', 'i3'),))],\
                        [single_term(op=(('a', 'i1'),)), single_term(op=(('b', 'A'),)), single_term(op=(('a', 'i1'),)), single_term(op=(('b', 'B'),)), single_term(op=(('a', 'i1'),)), single_term(op=(('b', 'C'),)), single_term(op=(('a', 'i2'),)), single_term(op=(('b', 'A'),)), single_term(op=(('a', 'i2'),)), single_term(op=(('b', 'B'),)), single_term(op=(('a', 'i2'),)), single_term(op=(('b', 'C'),)), single_term(op=(('a', 'i3'),)), single_term(op=(('b', 'A'),)), single_term(op=(('a', 'i3'),)), single_term(op=(('b', 'B'),)), single_term(op=(('a', 'i3'),)), single_term(op=(('b', 'C'),))],\
                        [single_term(op=(('A', 'i1', 'i2'), ('ap', 'i1'), ('a', 'i2'))), single_term(op=(('A', 'i2', 'i3'), ('ap', 'i2'), ('a', 'i3'))), single_term(op=(('A', 'i3', 'i4'), ('ap', 'i3'), ('a', 'i4')))],
                        )
    param_dict = {}
    param_dict["minus"] = -1.0
    param_dict["1j"] = 1j
    param_dict["range"] = ("i1", "i2", "i3")
    param_dict["range_L"] = ("A", "B", "C")
    param_dict["NN"] = zip(["i1","i2","i3"], ["i2","i3","i4"])
    for x, test_x in zip(examples_C, test_examples_C1):
        assert latex2term(x, param_dict) == test_x


if __name__ == "__main__":
    test_latex2term_unit_tests()
