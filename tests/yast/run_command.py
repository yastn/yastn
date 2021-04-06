import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './../..'))  # folder with yast
sys.path.append(os.path.join(os.path.dirname(__file__), './../configs'))  # folder with configs for tests
from test_swap_gate import test_swap_1
from test_syntax import test_sytax

if __name__ == '__main__':
    # test_swap_1()
    test_syntax()
