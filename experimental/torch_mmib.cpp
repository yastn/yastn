#include <torch/extension.h>
#include <iostream>
#include <vector>

std::tuple< std::vector<std::vector<int>>, std::vector<torch::Tensor> > mmib_forward(
	std::map< std::vector<int>, torch::Tensor > A,
	std::map< std::vector<int>, torch::Tensor > B,
	std::vector< std::vector< std::vector< int > > > meta_dot
	){
	std::vector<std::vector<int>> res_i;
	std::vector<torch::Tensor> res_t;
	for ( auto const &t : meta_dot ) { 
		// std::cout << t[0] << "|" << t[1] << "|" << t[2] << std::endl;
		res_i.push_back(t[0]);
		res_t.push_back(A[t[1]].mm(B[t[2]]));
	}
	return std::make_tuple( res_i, res_t );
}

void mmib_backward(
	std::vector<torch::Tensor> A,
	std::vector<torch::Tensor> B
	){
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mmib_forward, "MMIB forward");
  m.def("backward", &mmib_backward, "MMIB backward");
}