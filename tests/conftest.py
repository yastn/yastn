import sys
import os
from fnmatch import fnmatch
import importlib

sys.path.append(os.path.join(os.path.dirname(__file__), './..'))  # folder with yast


def pytest_addoption(parser):
    parser.addoption("--backend", help='np or torch', default='np', action='store')


def pytest_configure(config):
    if config.option.backend == 'torch':
        print('Using torch backend')
        import yast.backend.backend_torch as backend
    elif config.option.backend == 'np':
        print('Using numpy backend')
        import yast.backend.backend_np as backend

    for folder in ["tensor", "mps"]:
        confs = [name[:-3] for name in os.listdir(folder) if fnmatch(name, 'config*.py')]
        for conf in confs:
            conf = importlib.import_module(folder + "." + conf)
            conf.backend = backend

