# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import contextlib
import glob
import os
from pathlib import Path
import re
import sys
import timeit

from benchmarks.bench_ctm.model_yastn_basic_single_layer import CtmBenchYastnBasicSL
from benchmarks.bench_ctm.model_yastn_basic_double_layer import CtmBenchYastnBasicDL
from model_yastn_fpeps import CtmBenchYastnfpeps

models = {"CtmBenchYastnBasicSL": CtmBenchYastnBasicSL,
          "CtmBenchYastnBasicDL": CtmBenchYastnBasicDL,
          "CtmBenchYastnfpeps": CtmBenchYastnfpeps}


def fname_output(model, fname, config):
    fpath = os.path.dirname(__file__)
    path = Path(f"{fpath}/results/{model}/{config['backend']}/{config['device']}")
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{fname.stem}.out"


def run_bench(model, fname, config, repeat, to_file):
    """
    Run a single benchmark and output results to file or to stdout
    """
    bench = models[model](fname, config)

    target = fname_output(model, fname, config) if to_file else None

    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(target, 'w')) if target else sys.stdout

        bench.print_header(file=f)

        for task in bench.bench_pipeline:
            times = timeit.repeat(stmt='bench.' + task + '()', repeat=repeat, number=1, globals=locals())
            print(task + "; times [seconds]", file=f)
            print(*(f"{t:.4f}" for t in times), file=f)

        bench.print_properties(file=f)
        bench.final_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-backend", type=str, default='np', choices=['np', 'torch'])
    parser.add_argument("-device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("-repeat", type=int, default=4)
    parser.add_argument("-to_file", type=bool, default=1)
    parser.add_argument("-fname", type=str, default='.', help="Use re.search to match names from /input_shapes")
    parser.add_argument("-model", type=str, default='.', help="Use re.search to match model class names")
    args = parser.parse_args()

    config = {"backend": args.backend,
              "device": args.device}

    use_models = [model for model in models if re.search(args.model, model)]
    fpath = os.path.join(os.path.dirname(__file__), "input_shapes/")
    fnames = [Path(fname) for fname in glob.glob(fpath + "*.json") if re.search(args.fname, Path(fname).stem)]

    for model in use_models:
        for fname in fnames:
            run_bench(model, fname, config, args.repeat, args.to_file)
