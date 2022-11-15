from typing import NamedTuple
from itertools import chain
import re

class gen_parameratorError(Exception):
    pass

basic_operation = ["+", "*"]

class single_term(NamedTuple):
    # Helper object to perform operations on elements
    # used to interpret order of operation into sum of products. 
    # doesn't distinguish what are elements in the product.
    op : tuple = ()

def get_variable(var):
    # Required format e.g., a_{i,j,k,<separated with commas, no spaces>}
    m = re.match(r'(.*)_{(.*)}', var)
    return (var,) if not m else (m.group(1), *m.group(2).split(","))

def issingle_term(d):
    if isinstance(d, single_term):
        return True
    elif isinstance(d, list):
        return all([issingle_term(x) for x in d])
    else:
        return False

def replace_index(single_term_el, old_name, new_name):
    old_name = [old_name] if not isinstance(old_name, (list, tuple)) else old_name
    new_name = [new_name] if not isinstance(new_name, (list, tuple)) else new_name
    if len(old_name) != len(new_name):
        raise gen_parameratorError("Number of iterators and indicies is inconsistent.")
    # returns single_term
    new_op = []
    for op in single_term_el.op:
        for jold, jnew in zip(old_name, new_name):
            op = tuple([jnew if ind == jold else ind for ind in op])
        new_op += (op,)
    return single_term(op=tuple(new_op))

def combine_lists(operation, list1, list2):
    if operation == "+":
        return list1 + list2
    elif operation == "*":
        new_list = []
        for n in list1:
            for m in list2:
                new_list.append(apply_operation(operation, [n, m]))
        return new_list

def apply_operation(operation, single_terms):
    if operation == "+":
        return single_terms
    elif operation == "*":
        new_op = ()
        for x in single_terms:
            new_op = new_op+x.op
        return single_term(op=new_op)
    elif operation[0] == "\sum":
        m = re.match(r'{(.*).\\in.(.*)}', operation[1])
        iterator, iterate_list = (m.group(1).split(",")), m.group(2)
        sum_elements = []
        # iterate over list read from gen_param parameters
        for new_iterator in gen_param[iterate_list]:
            for new_list in single_terms.copy():
                sum_elements.append(replace_index(new_list, iterator, new_iterator))
        return sum_elements


def interpret(c):
    """
    Interprets the outcome of splitt function. Simplify intruction to list of single_term-s
    """
    # Terminate if you have a list of single_term-s
    if all([isinstance(x, single_term) for x in c]):
        return c
    
    # Instruction is given by a tuple with operation as first element
    # and list of elements it connects as second
    # e.g., (operation, list of elements)
    operation, elements = c
    if isinstance(operation, tuple):
        # For iterating operation
        if issingle_term(elements):
            # Apply iteration
            return apply_operation(operation, elements)
        else:
            # Expand elements to simple list before iterating
            return (operation, interpret(elements))
    elif operation in basic_operation:
        # Use basic operations
        if all([isinstance(x, single_term) for x in elements]):
            # for simple list of single_terms combine_single_terms using single_term algebra
            return apply_operation(operation, elements)
        if all([issingle_term(x) for x in elements]):
            el_me = elements
            if operation == "+":
                return interpret(tuple([operation, list(chain(*el_me))]))
            elif operation == "*":
                initial = el_me[0]
                for n in range(1, len(el_me)):
                    initial = combine_lists(operation, initial, el_me[n])
                return initial
        else:
            # so elements are still embedded and there are other single_term interpreter-s needed
            return interpret(tuple([operation, list(map(lambda x: interpret(x), elements))]))

def splitt(b, eq_it):
    """
    Read list of string and interpret latex-like format using order of operation.
    """
    if not b or eq_it > len(basic_operation):
        raise gen_parameratorError("Not supported. You provided an empty list or there is no such basic operation.")
    # split against that operation
    operation = basic_operation[eq_it]

    # check brackets
    id_start = [i for i, x in enumerate(b) if x == "("]
    id_stop = [i for i, x in enumerate(b) if x == ")"]
    if len(id_start) != len(id_stop):
        raise gen_parameratorError("Inconsistent brackets!")
    # remove those we don't need
    if len(id_start) > 0:
        if id_start[0] == 0 and id_stop[-1] == len(b)-1:
            is_bad = False
            res = 0
            tmp = b.copy()
            b = b[1:-1]
            for n in range(len(b)):
                if b[n] == '(':
                    res += 1
                elif b[n] == ')':
                    res -= 1
                if res < 0:
                    is_bad = True
                    break
            if is_bad == True:
                b = tmp
            else:
                return splitt(b, 0)
    
    # check brackets
    id_startX = [i for i, x in enumerate(b) if x == "\sum"]
    id_stopX = [i for i, x in enumerate(b) if x == "\endsum"]
    if len(id_startX) and len(id_startX) == len(id_stopX):
        # remove those we don't need
        if id_startX[0] == 0 and id_stopX[-1] == len(b)-1:
            is_bad = False
            res = 0
            tmp = b[1:-1]
            for n in tmp:
                if n == '\sum':
                    res += 1
                elif n == '\endsum':
                    res -= 1
                if res < 0:
                    is_bad = True
                    break
            if not is_bad:
                in_sum = tmp[1:] if len(tmp) > 2 else tmp[1]
                return ("\sum", tmp[0]), splitt(in_sum, 0)
    
    # If a single string gen_paramerate single_term
    if not isinstance(b, list) or len(b) == 1:
        # Ends recursion.
        op = (get_variable(b),) if isinstance(b, str) else (get_variable(b[0]),)
        return [single_term(op=op)]
    
    # Check if there are any basic operations
    id_operation = [i for i, x in enumerate(b) if x is operation]
    id_basic_operation = [i for i, x in enumerate(b) if x in basic_operation]
    if not "\sum" in b and not id_basic_operation:
        # Report error if no relation defined
        raise gen_parameratorError("Not supported: Please define operation between elements: ", b)
    if not id_operation and id_basic_operation:
        # Check another operation if this one is not here
        return splitt(b, (eq_it + 1)%len(basic_operation))

    # Main splitter loop
    n, bracket_open, sum_open = 0, 0, 0
    # tunnel - list of elements connected by the operation
    # buffer - collects pieces of elements pushed into tunnel
    tunnel, buffer = [], []
    for n in range(len(b)):
        sig = b[n]

        # Brackets have higher rank than basic_operation
        if sig == '(':
            bracket_open += 1
        elif sig == ')':
            bracket_open -= 1
        
        # Itarative operations have higher rank than basic_operation and brackets
        if bracket_open == 0 and sig == "\sum":
            sum_open += 1
        if sig == "\endsum":
            sum_open -= 1
        if bracket_open == 0 and sum_open > 0 and sig == "+":
            # Addition outside all brackets ends sums
            [buffer.append("\endsum") for _ in range(sum_open)]
            sum_open = 0
        # Control buffer/tunnel flow
        tunnel_closed = (sig not in operation or bracket_open!=0) or sum_open > 0
        if tunnel_closed:
            buffer.append(sig)
        else:
            if len(buffer) == 1:
                tunnel.append(*buffer)
            else:    
                tunnel += [buffer]
            buffer = []
            tunnel_closed = False
    [buffer.append("\endsum") for _ in range(sum_open)]
    if len(buffer) == 1:
        tunnel.append(*buffer)
    else:
        tunnel += [buffer]
    out = tunnel
    # Apply interpreter to all elements separated by the operation
    return operation, list(map(lambda x: splitt(x, (eq_it + 1)%len(basic_operation)), out))


def latex2single_term(c0, param_dict):
    # Full interpreter from string to single_term-s
    # I make all parameters global but use only ranges from sum
    global gen_param
    gen_param = param_dict
    c0 = c0.split(" ") if c0 else []
    # TODO: write test
    c1 = splitt(c0, 0)
    # TODO: write test
    c2 = interpret(c1)
    # TODO: write test
    return c2
