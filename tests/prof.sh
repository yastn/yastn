#!/bin/bash
python -m cProfile -o ./cprof/${1}_cprof ${1}
pyprof2calltree -k -i ./cprof/${1}_cprof
