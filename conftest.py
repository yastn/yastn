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


@pytest.fixture(params=['scatter', 'tiled', 'hybrid', 'loop'], ids=['scatter', 'tiled', 'hybrid', 'loop'])
def fuse_scatter_path(request, monkeypatch):
    """Run each opted-in test under the GPU fuse/unfuse code paths, selected live via env
    ``YASTN_FUSE_SCATTER_CHUNK`` / ``YASTN_FUSE_SCATTER_THRESH`` (read on every
    ``transpose_and_merge``/``unmerge`` call):

    * ``scatter`` -- both unset -> pure single-tile scatter/gather (all small test blocks < default
      THRESH 2**16),
    * ``tiled``   -- CHUNK=128  -> tiled scatter/gather; a small chunk so the (small) test tensors
      cross tile boundaries. **Skipped for ``@pytest.mark.exclude_fusion_scatter_tiled`` tests**
      (e.g. gradcheck), where forward reruns O(numel) times and a small chunk is pathological.
    * ``hybrid``  -- THRESH=64  -> loop the >=64-element blocks, compact-scatter the smaller ones
      (exercises the large/small split on the small test tensors),
    * ``loop``    -- CHUNK=0    -> forced per-block loop even on GPU.

    Only meaningful on cuda (CPU always uses the loop), so the extra variants are skipped off-cuda.
    Opt in per module with ``pytestmark = pytest.mark.usefixtures("fuse_scatter_path")``.
    """
    if request.config.getoption("--device") != 'cuda':
        if request.param != 'scatter':
            pytest.skip('fuse-path split is a no-op off cuda')
        return
    if request.param == 'tiled' and request.node.get_closest_marker('exclude_fusion_scatter_tiled'):
        pytest.skip("'tiled' variant excluded for this test; boundaries covered by unit tests")
    monkeypatch.delenv('YASTN_FUSE_SCATTER_CHUNK', raising=False)
    monkeypatch.delenv('YASTN_FUSE_SCATTER_THRESH', raising=False)
    if request.param == 'tiled':
        monkeypatch.setenv('YASTN_FUSE_SCATTER_CHUNK', '128')
    elif request.param == 'hybrid':
        monkeypatch.setenv('YASTN_FUSE_SCATTER_THRESH', '64')
    elif request.param == 'loop':
        monkeypatch.setenv('YASTN_FUSE_SCATTER_CHUNK', '0')
    # 'scatter': both unset -> default THRESH 2**16 -> all (small) test blocks scatter
