import pytest


def pytest_addoption(parser):
    parser.addoption("--backend", help='backend', default='np', choices=['np','torch','torch_cpp'], action='store')
    parser.addoption("--device", help='cpu or cuda', default='cpu', action='store')
    parser.addoption("--tensordot_policy", choices=['fuse_to_matrix', 'fuse_contracted', 'no_fusion'], default='fuse_to_matrix', action='store')
    parser.addoption("--default_fusion", choices=['hard', 'meta'], default='hard', action='store')
    parser.addoption("--quickstart", help='execute quickstarts', action='store_true', dest="quickstart", default=False)
    parser.addoption("--long_tests", help='run long duration tests', action='store_true', default=False)
    parser.addoption("--ray", help='tests using ray', action='store_true', default=False)


@pytest.fixture
def config_kwargs(request):
    return {'backend': request.config.getoption("--backend"),
            'default_device': request.config.getoption("--device"),
            'default_fusion': request.config.getoption("--default_fusion"),
            'tensordot_policy': request.config.getoption("--tensordot_policy")}
