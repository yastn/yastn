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

std::vector< std::pair<std::vector<int64_t>, torch::Tensor> > mtm_forward(
	std::map< std::vector<int64_t>, torch::Tensor > A,
	std::vector<int64_t> order,
	std::vector< std::vector< std::vector<int64_t> > > meta_new,
	std::vector< std::tuple<
		std::vector<int64_t>,
		std::vector<int64_t>,
		std::vector<int64_t>,
		int64_t,
		std::vector<int64_t>,
		int64_t> > meta_mrg,
	std::string	device_str = "cpu"
	){

	std::map< std::vector<int64_t>, torch::Tensor > A_new;
	auto options= torch::TensorOptions()
	    .dtype(torch::kFloat64)
	    .layout(torch::kStrided)
		.device(device_str);
	for ( auto const &t : meta_new ) {
		A_new[t[0]]= torch::zeros( at::IntArrayRef(t[1]), options );
	}

	#pragma omp parallel for
	for ( auto const &t : meta_mrg ) {
		// tn, to, Dsl, Dl, Dsr, Dr <=> &t
		A_new[std::get<0>(t)].index_put_({
			torch::indexing::Slice(std::get<2>(t)[0],std::get<2>(t)[1]),
			torch::indexing::Slice(std::get<4>(t)[0],std::get<4>(t)[1])
			}, A[std::get<1>(t)].permute( at::IntArrayRef(order) ).reshape( at::IntArrayRef({ std::get<3>(t), std::get<5>(t) }) ) );
	}

	std::vector< std::pair<std::vector<int64_t>, torch::Tensor> > res =
		std::vector< std::pair<std::vector<int64_t>, torch::Tensor> >(A_new.begin(), A_new.end());
	return res;
}

void mtm_backward(
	std::vector<torch::Tensor> A
	){
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mtm_forward, "MTM forward");
  m.def("backward", &mtm_backward, "MTM backward");
}
