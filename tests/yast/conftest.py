import config_dense_C
import config_dense_R
import config_U1_C
import config_U1_R
import config_Z2_C
import config_Z2_R
import config_Z2_U1_R


def pytest_addoption(parser):
    parser.addoption("--run", help='np or torch', default='np', action='store')


def pytest_configure(config):
    if config.option.run == 'torch':
        print('Using torch backend')
        import yast.backend.backend_torch as backend
        config_U1_R.backend = backend
    elif config.option.run == 'np':
        print('Using numpy backend')

