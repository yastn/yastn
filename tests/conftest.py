import os
import sys
from fnmatch import fnmatch
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def pytest_addoption(parser):
    parser.addoption("--backend", help='backend', default='np', choices=['np','torch','torch_cpp'],\
     action='store')
    parser.addoption("--device", help='cpu or cuda', default='cpu', action='store')
    parser.addoption("--bug_pytorch110", help='test complex conjugation bug in PyTorch 1.10',\
        action='store_true', dest="bug_pytorch110", default=False)
    parser.addoption("--quickstarts", help='execute quickstarts',\
        action='store_true', dest="quickstarts", default=False)


def pytest_configure(config):
    if config.option.backend == 'torch':
        print('Using torch backend')
        import yastn.backend.backend_torch as backend
    if config.option.backend == 'torch_cpp':
        print('Using torch_cpp backend')
        import yastn.backend.backend_torch_cpp as backend
    elif config.option.backend == 'np':
        print('Using numpy backend')
        import yastn.backend.backend_np as backend

    for folder in ["tensor", "mps", "operators", "peps"]:
        confs = [name[:-3] for name in os.listdir(folder + "/configs") if fnmatch(name, 'config*.py')]
        for conf in confs:
            conf = importlib.import_module(folder + ".configs." + conf)
            conf.backend = backend
            conf.default_device = config.option.device
