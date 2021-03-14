# import sys
# sys.path.append("D:/Programs/yamps/")

def pytest_addoption(parser):
    parser.addoption("--run", help='np or torch', default='np', action='store')
