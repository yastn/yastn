import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './../..'))  # folder with yast
sys.path.append(os.path.join(os.path.dirname(__file__), './../configs'))  # folder with configs for tests
from test_add import test_add_0
from test_to_number import test_to_number_1

if __name__ == '__main__':
    test_to_number_1()

