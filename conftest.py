import pytest


def pytest_addoption(parser):
    parser.addoption("--backend", help='backend', default='np', choices=['np','torch','torch_cutensor'], action='store')
    parser.addoption("--device", help='cpu or cuda', default='cpu', action='store')
    parser.addoption("--tensordot_policy", choices=['fuse_to_matrix', 'fuse_contracted', 'no_fusion'], default='fuse_to_matrix', action='store')
    parser.addoption("--lazy_threshold", type=float, default=0.5, action='store')
    parser.addoption("--default_fusion", choices=['hard', 'meta'], default='hard', action='store')
    parser.addoption("--quickstart", help='execute quickstarts', action='store_true', dest="quickstart", default=False)
    parser.addoption("--long_tests", help='run long duration tests', action='store_true', default=False)
    parser.addoption("--ray", help='tests using ray', action='store_true', default=False)
    parser.addoption("--devices", help='comma-separated device list for multi-device tests, e.g. cuda:0,cuda:1,cuda:2', default=None, action='store')


@pytest.fixture
def config_kwargs(request):
    return {'backend': request.config.getoption("--backend"),
            'default_device': request.config.getoption("--device"),
            'default_fusion': request.config.getoption("--default_fusion"),
            'tensordot_policy': request.config.getoption("--tensordot_policy"),
            'lazy_threshold': request.config.getoption("--lazy_threshold")}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "exclude_fusion_scatter_tiled: skip the 'tiled' fuse-scatter path variant for this test "
        "(e.g. gradcheck tests, which rerun forward O(numel) times so a small tile chunk is "
        "pathologically slow).")


@pytest.fixture(params=['scatter', 'tiled', 'loop'], ids=['scatter', 'tiled', 'loop'])
def fuse_scatter_path(request, monkeypatch):
    """Run each opted-in test under the GPU fuse/unfuse code paths, selected live via env
    ``YASTN_FUSE_SCATTER_CHUNK`` (read on every ``transpose_and_merge``/``unmerge`` call):

    * ``scatter`` -- unset   -> single-tile scatter/gather (GPU default),
    * ``tiled``   -- ``128`` -> tiled scatter/gather; a small chunk so the (small) test tensors
      actually cross tile boundaries. **Skipped for ``@pytest.mark.exclude_fusion_scatter_tiled``
      tests** (e.g. gradcheck), where forward reruns O(numel) times and a small chunk is pathological.
    * ``loop``    -- ``0``   -> forced per-block loop even on GPU.

    Only meaningful on cuda (CPU always uses the loop), so the extra variants are skipped off-cuda.
    Opt in per module with ``pytestmark = pytest.mark.usefixtures("fuse_scatter_path")``.
    """
    if request.config.getoption("--device") != 'cuda':
        if request.param != 'scatter':
            pytest.skip('YASTN_FUSE_SCATTER_CHUNK path split is a no-op off cuda')
        return
    if request.param == 'tiled' and request.node.get_closest_marker('exclude_fusion_scatter_tiled'):
        pytest.skip("'tiled' variant excluded for this test; boundaries covered by unit tests")
    if request.param == 'scatter':
        monkeypatch.delenv('YASTN_FUSE_SCATTER_CHUNK', raising=False)
    elif request.param == 'tiled':
        monkeypatch.setenv('YASTN_FUSE_SCATTER_CHUNK', '128')
    else:  # loop
        monkeypatch.setenv('YASTN_FUSE_SCATTER_CHUNK', '0')
