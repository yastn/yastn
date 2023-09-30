# YASTN - Yet Another Symmetric Tensor Network
![Test status](https://github.com/yastn/yastn/actions/workflows/main.yml/badge.svg?branch=master)
![Docs status](https://github.com/yastn/yastn/actions/workflows/docs.yml/badge.svg?branch=master)

by Marek M. Rams, Gabriela Wójtowicz, Aritra Sinha, and Juraj Hasik

<br />
  
### Python library for differentiable linear algebra with block-sparse tensors, supporting abelian symmetries

##### YASTN tensors can be defined with both discrete and continuous abelian groups

- Z<sub>2</sub> for parity conservation 
- U(1) for particle number conservation
- direct product of abelian groups such as Z<sub>3</sub>xU(1) or U(1)xU(1)xZ<sub>2</sub> and so on

##### YASTN can run with different backends

- NumPy
- [PyTorch](https://pytorch.org/)

allowing for automatic differentiation (autograd) on backends which provide it.
<br />

##### To see YASTN in action, check

- Matrix product states (MPS) and algorithms powered by YASTN 

   Explore the entire [MPS module](https://yastn.github.io/yastn/yastn.mps.html) or 
   try out code examples running [DMRG](https://yastn.github.io/yastn/examples/mps/algorithms.html#dmrg)
   or [TDVP](https://yastn.github.io/yastn/examples/mps/algorithms.html#tdvp) optimizations

<br />

- Two-dimensional tensor networks library [peps-torch](https://github.com/jurajHasik/peps-torch) powered by YASTN

   Try variational optimization of iPEPS with abelian symmetries for spin models

<br />

### Jump to [full Documentation](https://yastn.github.io/yastn/index.html) or get started with [Examples](https://yastn.github.io/yastn/yastn.tensor.html#examples-basics-of-usage)

<br />

### Installation


Install using
```
git clone https://github.com/yastn/yastn.git && cd yastn
pip install .
```
or just clone and add YASTN root to your Python import path to sucessfully ``import yastn``.

##### YASTN depends on

- Python 3.9+
- NumPy

and optionally 

- PyTorch 1.11+ (for PyTorch backend)
- SciPy (for sparse linear algebra solvers)
- ...

### Run the tests

Tests, which are also a good source of examples of usage, can be found in the folder `tests`.
To verify that everything works, get `pytest`. See [Installing pytest](https://docs.pytest.org/en/6.2.x/getting-started.html) 
or `conda install -c conda-forge pytest`. Then you can test base yastn.Tensor and also yastn.mps modules

```
cd tests
pytest -v ./tensor
pytest -v ./mps
```

To test YASTN on PyTorch backend and integration of autograd features
```
cd tests
pytest --backend torch -v ./tensor
pytest --backend torch -v ./mps
```

### Building docs locally

You can build documentation using `sphinx`. The prerequisites are

   * sphinx
   * pydata-sphinx-theme

Get them with your favourite Python package manager. For example, using conda as `conda install -c conda-forge sphinx pydata-sphinx-theme`.
Then

```
cd docs && make html
```

The generated documentation can be found at `docs/build/html/index.html`
