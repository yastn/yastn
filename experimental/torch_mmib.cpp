// Copyright 2024 The YASTN Authors. All Rights Reserved.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
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