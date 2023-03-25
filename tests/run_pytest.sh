conda activate pt
pytest --backend np --cov=./../yast --cov-report html ./tensor
pytest --backend torch --cov=./../yast --cov-append --cov-report html
