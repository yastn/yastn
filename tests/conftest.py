import os
import sys
from fnmatch import fnmatch
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def pytest_addoption(parser):
    parser.addoption("--backend", help='np or torch', default='np', action='store')
    parser.addoption("--device", help='cpu or cuda', default='cpu', action='store')


def pytest_configure(config):
    if config.option.backend == 'torch':
        print('Using torch backend')
        import yast.backend.backend_torch as backend
    elif config.option.backend == 'np':
        print('Using numpy backend')
        import yast.backend.backend_np as backend

    for folder in ["tensor", "mps"]:
        confs = [name[:-3] for name in os.listdir(folder + "/configs") if fnmatch(name, 'config*.py')]
        for conf in confs:
            conf = importlib.import_module(folder + ".configs." + conf)
            conf.backend = backend
            conf.default_device = config.option.device
