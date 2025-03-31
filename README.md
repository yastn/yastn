# YASTN - Yet Another Symmetric Tensor Network

| **Documentation** | **Build status** | **Coverage** |
|:-----------------:|:----------------:|:------------:|
| [![][docs-img]][docs-url] | [![][CI-img]][ci-url] | [![][cov-img]][cov-url] |

[docs-img]: https://img.shields.io/badge/doc-master-blue.svg
[docs-url]: https://yastn.github.io/yastn/

[ci-img]: https://github.com/yastn/yastn/actions/workflows/main.yml/badge.svg?branch=master
[ci-url]: https://github.com/yastn/yastn/actions/workflows/main.yml

[cov-img]: https://codecov.io/gh/yastn/yastn/branch/master/graph/badge.svg?token=J548JFRTCZ
[cov-url]: https://codecov.io/gh/yastn/yastn


Python library for differentiable linear algebra with block-sparse tensors, supporting abelian symmetries.
It supports a range of Projected Entangled Pair States (PEPS) and Matrix Product States (MPS) algorithms employing the core symmetric Tensor class.

##### YASTN tensors can be defined with both discrete and continuous abelian groups

- Z<sub>2</sub> for parity conservation
- U(1) for particle number conservation
- Direct product of abelian groups such as Z<sub>3</sub>xU(1) or U(1)xU(1)xZ<sub>2</sub> and so on

##### YASTN can run with different backends, including

- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)

It allows automatic differentiation (autograd) on backends which support it.

##### To see YASTN in action, see

- [Quickstarts](https://yastn.github.io/yastn/yastn.quickstart.html) for a few end-to-end examples of PEPS calculations.
- [Benchmarks](https://github.com/yastn/yastn_benchmarks), including basic DMRG example.
- [Full documentation](https://yastn.github.io/yastn/index.html) for multiple further examples.
- Two-dimensional tensor networks library [peps-torch](https://github.com/jurajHasik/peps-torch). It includes variational optimization of iPEPS with abelian symmetries for spin models powered by YASTN tensor.


### Install using

```
git clone https://github.com/yastn/yastn.git
cd yastn
pip install .
```
or just clone and add YASTN root to your Python import path to successfully ``import yastn``
(see `pyproject.toml` for dependencies).

### Run the tests

Tests, which are also a good source of usage examples, can be found in the folder `tests`.
To verify that everything works, get `pytest`. To test specific elements of the repository, run

```
pytest -v ./tests/tensor
pytest -v ./tests/operators
pytest -v ./tests/mps
pytest -v ./tests/peps
```

To test YASTN on PyTorch backend and integration of autograd features (assuming PyTorch is installed)
```
pytest -v --backend torch
```

To execute the tests on a GPU

```
pytest -v --backend torch --device cuda
```

### Citing YASTN

If you find this package useful for a project published in an academic journal, you can consider citing the [release paper](https://scipost.org/SciPostPhysCodeb.52),
with the following BibTex entry
```
@Article{YASTN,
      title={{YASTN: Yet another symmetric tensor networks; A Python library for Abelian symmetric tensor network calculations}},
      author={Marek M. Rams and Gabriela Wójtowicz and Aritra Sinha and Juraj Hasik},
      journal={SciPost Phys. Codebases},
      pages={52},
      year={2025},
      publisher={SciPost},
      doi={10.21468/SciPostPhysCodeb.52},
      url={https://scipost.org/10.21468/SciPostPhysCodeb.52},
}
```
