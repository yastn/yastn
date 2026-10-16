import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--backend", help='backend', default='np', choices=['np','torch','torch_cutensor'], action='store')
    parser.addoption("--strict-host-conversion", action='store_true', default=False,
                     help="fail every implicit tensor->numpy conversion, as cuda does; catches tests "
                          "that only pass on cpu (see _install_strict_host_conversion, or set "
                          "YASTN_STRICT_HOST_CONVERSION=1)")
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


def _install_strict_host_conversion():
    """Make implicit tensor->numpy conversions fail on cpu exactly as they do on cuda.

    ``np.isclose(a.norm(), ...)`` and friends work for a cpu torch tensor but raise
    ``TypeError: can't convert cuda:0 device type tensor to numpy`` for a cuda one, so a
    test written that way breaks only in GPU runs -- which CI does not have.  Patching
    ``torch.Tensor.__array__``, the hook numpy uses for implicit conversion, reproduces
    the cuda behaviour on cpu and lets CI catch this class of mistakes.  Explicit
    conversion (``.to_numpy()``, ``.item()``, ``backend.to_numpy``) keeps working.
    """
    import traceback

    import torch

    def __array__(self, dtype=None):
        caller = ''
        for frame in reversed(traceback.extract_stack()[:-1]):
            if any(part in frame.filename for part in ('/numpy/', '/torch/', '/_pytest/')):
                continue
            caller = f' (implicitly converted at {frame.filename}:{frame.lineno})'
            break
        raise TypeError(
            "implicit conversion of a backend tensor to numpy; this raises on cuda. "
            "Use .to_numpy() on host tensors, or .item() for scalars." + caller)

    torch.Tensor.__array__ = __array__


def _ci_environment():
    """True inside a CI runner.  A plain ``CI``/``GITHUB_ACTIONS`` counts only when it is
    not explicitly false (the repo's workflow sets ``CI=false``)."""
    for key in ('GITHUB_ACTIONS', 'CI'):
        if os.environ.get(key, '').strip().lower() not in ('', 'false', '0'):
            return True
    return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "exclude_fusion_scatter_tiled: skip the 'tiled' fuse-scatter path variant for this test "
        "(e.g. gradcheck tests, which rerun forward O(numel) times so a small tile chunk is "
        "pathologically slow).")
    if config.getoption("--strict-host-conversion") or os.environ.get("YASTN_STRICT_HOST_CONVERSION"):
        # Local diagnostic only: it must never be armed by a CI run, so that CI keeps
        # testing the assigned backend and cannot start failing on this class for free.
        if _ci_environment():
            print('strict host conversion ignored: this is a local diagnostic, CI must not use it')
        else:
            _install_strict_host_conversion()


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
