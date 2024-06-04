// Copyright 2024 The YASTN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <omp.h>

std::tuple< std::vector<std::vector<int>>, std::vector<torch::Tensor> > mmib_forward(
	std::map< std::vector<int>, torch::Tensor > A,
	std::map< std::vector<int>, torch::Tensor > B,
	std::vector< std::vector< std::vector< int > > > meta_dot
	){
	std::vector<std::vector<int>> res_i;
	std::vector<torch::Tensor> res_t;

	for ( auto const &t : meta_dot ) {
		res_i.push_back(t[0]);
		res_t.push_back(A[t[1]].mm(B[t[2]]));
	}

	return std::make_tuple( res_i, res_t );
}

// std::vector< std::pair<std::vector<int64_t>, torch::Tensor> > mmib_forward(
// 	std::map< std::vector<int64_t>, torch::Tensor > A,
// 	std::map< std::vector<int64_t>, torch::Tensor > B,
// 	std::vector< std::vector< std::vector< int64_t > > > meta_dot
// 	){
// 	std::map< std::vector<int64_t>, torch::Tensor > res_map;

// 	#pragma omp parallel for
// 	for (auto it = meta_dot.begin(); it < meta_dot.end(); it++) {
// 		res_map[(*it)[0]]= A[(*it)[1]].mm(B[(*it)[2]]);
// 	}
// 	std::vector< std::pair<std::vector<int64_t>, torch::Tensor> > res_vec =
// 		std::vector< std::pair<std::vector<int64_t>, torch::Tensor> >(res_map.begin(), res_map.end());

// 	return res_vec;
// }

void mmib_backward(
	std::vector<torch::Tensor> A,
	std::vector<torch::Tensor> B
	){
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mmib_forward, "MMIB forward");
  m.def("backward", &mmib_backward, "MMIB backward");
}