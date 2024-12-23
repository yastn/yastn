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
    # Create a state to sample from.
    #
    ops = yastn.operators.Spin1(sym='dense', **config_kwargs)
    ops.random_seed(seed=0)  # Fix seed for testing.
    #
    # Eigenvectors of local Sz spin-1 operator.
    vecs = {v: ops.vec_z(val=v) for v in [-1, 0, 1]}
    #
    # Set three classical product states
    states = np.array([[-1, 0,-1, 0, 1, 1,-1,-1],
                       [-1, 0, 1, 1,-1,-1, 0, 0],
                       [ 0, 1, 0, 0, 0, 1,-1,-1]])
    psis = [mps.product_mps([vecs[v] for v in st]) for st in states]
    amplitudes = [5, 4, 3]  # unnormalized amplitudes
    psi = mps.add(psis[0], psis[1], psis[2], amplitudes=amplitudes)
    # psi is a superposition of three product states with different weights.
    #
    # Sample local states in vecs from state psi.
    #
    number = 400  # number of samples
    samples, probabiliies = mps.sample(psi, projectors=vecs, number=number, return_probabilities=True)
    #
    # Check if the results match what is expected for sampling from state psi.
    #
    unique, inds, counts = np.unique(samples, axis=0, return_counts=True, return_inverse=True)
    probs = [a ** 2 for a in amplitudes]  # amplitudes to probabilities
    probs = [pr / sum(probs) for pr in probs]  # normalize
    for cnt, pr in zip(counts, probs):
        assert abs(cnt / number - pr) < 2 * number ** -0.5
        # number of occurances in samples is wihin the expected standard deviation.
    assert np.allclose(unique, states)
    assert all(abs(pr - probs[ind]) < 1e-10 for pr, ind in zip(probabiliies, inds))


def test_sample_mps_purification(config_kwargs):
    """ Test mps.sample() on purification and matrix projectors. """
    #
    # pure state + matrix projectors
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    ops.random_seed(seed=0)  # fix seed for testing
    #
    vecs = {v: ops.vec_n(val=v) for v in [0, 1]}
    #
    states = np.array([[0, 1, 0, 1, 0, 1],
                       [0, 1, 1, 0, 1, 1]])
    psis = [mps.product_mps([vecs[v] for v in st]) for st in states]
    amplitudes = [1, 2]  # unnormalized amplitudes
    psi = mps.add(psis[0], psis[1], amplitudes=amplitudes)
    # psi is a superposition of three product states with different weights
    #
    # perform sampling
    #
    number = 100  # number of samples
    projectors = [ops.I()-ops.n(), ops.n()]  # here use projector-operators.
    samples, probabiliies = mps.sample(psi, projectors=projectors, number=number, return_probabilities=True)
    #
    # check if the results match what is expected for sampling from state psi
    unique, inds, counts = np.unique(samples, axis=0, return_counts=True, return_inverse=True)
    probs = [a ** 2 for a in amplitudes]  # amplitudes to probabilities
    probs = [pr / sum(probs) for pr in probs]  # normalize
    for cnt, pr in zip(counts, probs):
        assert abs(cnt / number - pr) < number ** -0.5  # number of occurances in samples is wihin the expected standard deviation.
    assert np.allclose(unique, states)
    assert all(abs(pr - probs[ind]) < 1e-10 for pr, ind in zip(probabiliies, inds))
    #
    # purification + vector states to project on
    #
    vecs = [ops.c() @ ops.cp(), ops.cp() @ ops.c()]
    #
    # set three classical product states
    states = np.array([[0, 1, 1, 0, 1, 1],
                       [1, 0, 1, 0, 0, 1]])
    psis = [mps.product_mpo([vecs[v] for v in st]) for st in states]
    amplitudes = [1, 2]  # unnormalized amplitudes
    psi = mps.add(psis[0], psis[1], amplitudes=amplitudes)
    #
    # perform sampling
    #
    samples = mps.sample(psi, projectors=vecs, number=number, return_probabilities=False)
    #
    # check if the results match what is expected for sampling from state psi
    unique, inds, counts = np.unique(samples, axis=0, return_counts=True, return_inverse=True)
    probs = [a ** 2 for a in amplitudes]  # amplitudes to probabilities
    probs = [pr / sum(probs) for pr in probs]  # normalize
    for cnt, pr in zip(counts, probs):
        assert abs(cnt / number - pr) < number ** -0.5  # number of occurances in samples is wihin the expected standard deviation.
    assert np.allclose(unique, states)
    #
    # perform sampling
    #
    vecs = [ops.vec_n(0), ops.vec_n(1)]
    samples = mps.sample(psi, projectors=vecs, number=number, return_probabilities=False)
    #
    # check if the results match what is expected for sampling from state psi
    unique, inds, counts = np.unique(samples, axis=0, return_counts=True, return_inverse=True)
    probs = [a ** 2 for a in amplitudes]  # amplitudes to probabilities
    probs = [pr / sum(probs) for pr in probs]  # normalize
    for cnt, pr in zip(counts, probs):
        assert abs(cnt / number - pr) < number ** -0.5  # number of occurances in samples is wihin the expected standard deviation.
    assert np.allclose(unique, states)
    #
    # psi is not affected by mps.sample; copy of psi is made vefor canonization .
    assert not psi.is_canonical()


def test_sample_mps_raises(config_kwargs):
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    v0, v1 = ops.vec_n(0), ops.vec_n(1)
    v1101 = mps.product_mps([v1, v1, v0, v1])
    v0111 = mps.product_mps([v0, v1, v1, v1])
    psi = v1101 + v0111
    #
    with pytest.raises(yastn.YastnError,
                       match="Projectors not defined for some sites."):
        vecs = {v: ops.vec_n(val=v) for v in [0, 1]}
        mps.sample(psi, projectors={0: vecs, 2: vecs})
    #
    with pytest.raises(yastn.YastnError,
                       match="Use integer numbers for projector keys."):
        vecs = {'n0': ops.vec_n(val=0), 'n1': ops.vec_n(val=1)}
        mps.sample(psi, projectors=vecs)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
