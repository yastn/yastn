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
""" examples for addition of the Mps-s """
import pytest
import numpy as np
import yastn
import yastn.tn.mps as mps



def test_sample_mps(config_kwargs):
    """ Test mps.sample() on an mps being uperposition of a few product states. """
    #
    # Create a state to sample from
    #
    ops = yastn.operators.Spin1(sym='dense', **config_kwargs)
    vecs = {v: ops.vec_z(val=v) for v in [-1, 0, 1]}
    states = np.array([[-1, 0,-1, 0, 1, 1,-1,-1],
                       [-1, 0, 1, 1,-1,-1, 0, 0],
                       [ 0, 1, 0, 0, 0, 1,-1,-1]])
    psis = [mps.product_mps([vecs[v] for v in st]) for st in states]

    amplitudes=[2, 2, 1]
    psi = mps.add(psis[0], psis[1], psis[2], amplitudes=amplitudes)

    number = 900
    samples, probabiliies = mps.sample(psi, projectors=vecs, number=number, return_probabilities=True)

    unique, inds, counts = np.unique(samples, axis=0, return_counts=True, return_inverse=True)

    probs = [a ** 2 for a in amplitudes]  # amplitudes to probabilities
    probs = [pr / sum(probs) for pr in probs]  # normalize
    for cnt, pr in zip(counts, probs):
        assert abs(cnt / number - pr) < 2 * number ** -0.5  # number of accurance wihin expected std.

    assert np.allclose(unique, states)
    assert all(abs(pr - probs[ind]) < 1e-10 for pr, ind in zip(probabiliies, inds))


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
