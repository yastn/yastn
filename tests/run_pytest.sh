# conda activate pt

pytest --backend np --tensordot_policy fuse_to_matrix ./tests/tensor
pytest --backend np --tensordot_policy fuse_contracted ./tests/tensor
pytest --backend np --tensordot_policy no_fusion ./tests/tensor

pytest --backend np --cov=./../yastn --cov-report html ./
pytest --backend torch --cov=./../yastn --cov-append --cov-report html ./
pytest --backend torch --tensordot_policy no_fusion --cov=./../yastn --cov-append --cov-report html ./tests/tensor/test_tensordot.py
pytest --backend torch --tensordot_policy fuse_to_matrix --cov=./../yastn --cov-append --cov-report html ./tests/tensor/test_tensordot.py
pytest --backend torch --tensordot_policy fuse_contracted --cov=./../yastn --cov-append --cov-report html ./tests/tensor/test_tensordot.py
pytest --backend np --tensordot_policy no_fusion --cov=./../yastn --cov-append --cov-report html ./tests/tensor/test_tensordot.py
pytest --backend np --tensordot_policy fuse_to_matrix --cov=./../yastn --cov-append --cov-report html ./tests/tensor/test_tensordot.py
pytest --backend np --tensordot_policy fuse_contracted --cov=./../yastn --cov-append --cov-report html ./tests/tensor/test_tensordot.py
