# conda activate pt
pytest --backend np --cov=./../yastn --cov-report html ./
pytest --backend torch --cov=./../yastn --cov-append --cov-report html ./
