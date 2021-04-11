import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './../..'))  # folder with yast
sys.path.append(os.path.join(os.path.dirname(__file__), './../configs'))  # folder with configs for tests
from test_swap_gate import test_swap_1
from test_syntax import test_syntax
from test_vdot import test_scalar_0, test_scalar_1R, test_scalar_1C, test_scalar_exceptions
from test_split import test_svd_0

if __name__ == '__main__':
    # test_swap_1()
    test_syntax()
    test_scalar_0()
    test_scalar_1R()
    test_scalar_1C()
    test_scalar_exceptions()
    test_svd_0()
