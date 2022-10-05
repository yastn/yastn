#include <torch/extension.h>
#include <iostream>
#include <vector>

// meta_new -> list of [(tn, Dn, sln), ...] where
//              tn -> effective charge for block in fused tensor
//              Dn -> effective shape of block tn in fused tensor
//              sln -> slice specifying the location of serialized tn block in 1d data of fused tensor  
//
// meta_mrg -> t1 is effective charge of source block after fusion. I.e. t1==tn, means, that 
//             this source block will belong to destination block tn
//          -> gr: tuple holding description of source data
//                  slo -> specifies the location of source block in 1d data
//                  Do  -> shape of the source block
//                  Dscl-> list of slice data which specifies the location of the "transformed"
//                         source block in the destination block tn
//                  Drsh-> the shape of the "transformed" source block in the destination block tn


torch::Tensor tm_forward_1d(
	torch::Tensor data,
	const std::vector<int64_t> & order,
	const std::vector< std::tuple<
		std::vector<int64_t> /* tn */, 
		std::vector<int64_t> /* Dn */, 
		std::vector<int64_t> /* Sln */, 
		std::vector<int64_t> /* t1 */,
		std::vector< 
			std::tuple <
				std::vector<int64_t> /* _ */,
				std::vector<int64_t> /* slo */,
				std::vector<int64_t> /* Do */,
				std::vector< std::vector<int64_t> >, /* Dscl */
				std::vector<int64_t> /* Drsh */
			>
		> 
	> > & jobs,
	int64_t Dsize
	){

	auto options= torch::TensorOptions()
	    .dtype(data.dtype())
	    .layout(data.layout())
		.device(data.device())
		.requires_grad(data.requires_grad());
	torch::Tensor newdata= torch::zeros( Dsize, options );

	for (auto const &job : jobs) {
		auto _tmp = newdata.index({torch::indexing::Slice(std::get<2>(job)[0], std::get<2>(job)[1])})
			.view(at::IntArrayRef(std::get<1>(job)));
		
		for (auto const &job_b : std::get<4>(job)) {

			std::vector<at::indexing::TensorIndex> slcs;
			slcs.reserve(std::get<3>(job_b).size());
			for (auto const &elem_Dslc : std::get<3>(job_b)) {
				slcs.emplace(slcs.end(), torch::indexing::Slice(elem_Dslc[0], elem_Dslc[1]));
			}
			auto _slcs = at::ArrayRef<at::indexing::TensorIndex>(slcs);

			_tmp.index_put_(_slcs,
				data.index({torch::indexing::Slice(std::get<1>(job_b)[0],std::get<1>(job_b)[1])})
					.reshape(at::IntArrayRef(std::get<2>(job_b)))
					.permute(at::IntArrayRef(order))
					.reshape(at::IntArrayRef(std::get<4>(job_b)))
				);
		}
	}
	return newdata;
}

torch::Tensor tm_forward_1d_omp(
	torch::Tensor data,
	const std::vector<int64_t> & order,
	const std::vector< std::tuple<
		std::vector<int64_t> /* tn */, 
		std::vector<int64_t> /* Dn */, 
		std::vector<int64_t> /* Sln */, 
		std::vector<int64_t> /* t1 */,
		std::vector< 
			std::tuple <
				std::vector<int64_t> /* _ */,
				std::vector<int64_t> /* slo */,
				std::vector<int64_t> /* Do */,
				std::vector< std::vector<int64_t> >, /* Dscl */
				std::vector<int64_t> /* Drsh */
			>
		> 
	> > & jobs,
	int64_t Dsize
	){

	auto options= torch::TensorOptions()
	  .dtype(data.dtype())
	  .layout(data.layout())
		.device(data.device())
		.requires_grad(data.requires_grad());
	torch::Tensor newdata= torch::zeros( Dsize, options );

	for (auto const &job : jobs) {
		auto _tmp = newdata.index({torch::indexing::Slice(std::get<2>(job)[0], std::get<2>(job)[1])})
			.view(at::IntArrayRef(std::get<1>(job)));
		
		#pragma omp parallel for
		for (auto const &job_b : std::get<4>(job)) {

			std::vector<at::indexing::TensorIndex> slcs;
			slcs.reserve(std::get<3>(job_b).size());
			for (auto const &elem_Dslc : std::get<3>(job_b)) {
				slcs.emplace(slcs.end(), torch::indexing::Slice(elem_Dslc[0], elem_Dslc[1]));
			}

			_tmp.index_put_(at::ArrayRef<at::indexing::TensorIndex>(slcs),
				data.index({torch::indexing::Slice(std::get<1>(job_b)[0],std::get<1>(job_b)[1])})
					.reshape(at::IntArrayRef(std::get<2>(job_b)))
					.permute(at::IntArrayRef(order))
					.reshape(at::IntArrayRef(std::get<4>(job_b)))
				);
		}
	}
	return newdata;
}

inline std::vector<int64_t> apply_perm(const std::vector<int64_t> & D, const std::vector<int64_t> & p){
	std::vector<int64_t> pD(D.size());
	for (int d=0; d<D.size(); d++) {
		pD[d]= D[p[d]];
	}
	return pD;
}

inline std::vector<int64_t> apply_inv_perm(const std::vector<int64_t> & D, const std::vector<int64_t> & p){
	std::vector<int64_t> pD(D.size());
	for (int d=0; d<D.size(); d++) {
		pD[p[d]]= D[d];
	}
	return pD;
}

// torch::Tensor map_source_to_dest_plain_omp(
// 	torch::Tensor data,
// 	const std::vector<int64_t> & order,
// 	const std::vector< std::tuple<
// 		std::vector<int64_t> /* tn */, 
// 		std::vector<int64_t> /* Dn */, 
// 		std::vector<int64_t> /* Sln */, 
// 		std::vector<int64_t> /* t1 */,
// 		std::vector< 
// 			std::tuple <
// 				std::vector<int64_t> /* _ */,
// 				std::vector<int64_t> /* slo */,
// 				std::vector<int64_t> /* Do */,
// 				std::vector< std::vector<int64_t> >, /* Dscl */
// 				std::vector<int64_t> /* Drsh */
// 			>
// 		>
// 	> > & jobs){

// 	std::vector<int64_t> _tmp_range(order.size());
// 	for (int i=0; i<order.size(); i++) { 
// 		_tmp_range[i]= i;
// 	}
// 	auto inv_order= apply_inv_perm(_tmp_range, order);

// 	auto options_int= torch::TensorOptions()
// 	    .dtype(torch::kInt64)
// 	    .layout(torch::kStrided)
// 		.device(torch::kCPU);
// 	torch::Tensor source_to_dest= torch::empty( data.numel(), options_int );

// 	for (auto const &job : jobs) {
// 		torch::Tensor tmp_b = torch::arange(std::get<2>(job)[0], std::get<2>(job)[1], options_int)
// 			.view(at::IntArrayRef(std::get<1>(job)));
		
// 		#pragma omp parallel for
// 		for (auto const &job_b : std::get<4>(job)) {
// 			// prelim)
//       // get strides of shape Do, strides of shape permute(Do; order)
// 			std::vector<at::indexing::TensorIndex> slcs;
// 			slcs.reserve(std::get<3>(job_b).size());
// 			for (auto const &elem_Dslc : std::get<3>(job_b)) {
// 				slcs.emplace(slcs.end(), torch::indexing::Slice(elem_Dslc[0], elem_Dslc[1]));
// 			}
// 			auto inv_Do= apply_perm(std::get<2>(job_b), order);
      
//       source_to_dest.index_put_({torch::indexing::Slice(std::get<1>(job_b)[0],std::get<1>(job_b)[1])},
//       	tmp_b.index(at::ArrayRef<at::indexing::TensorIndex>(slcs)).view(at::IntArrayRef(inv_Do))
//       		.permute(at::IntArrayRef(inv_order)).contiguous().view(-1)
//       	);
//     }
//   }
//   return source_to_dest;
// }

using row_meta_new = std::tuple<
		std::vector<int64_t> /* tn */, 
		std::vector<int64_t> /* Dn */, 
		std::vector<int64_t>  /* Sln */
		>;

using row_meta_mrg = std::tuple <
				std::vector<int64_t> /* t1 */,
				std::vector<int64_t> /* slo */,
				std::vector<int64_t> /* Do */,
				std::vector< std::vector<int64_t> >  /* Dscl */,
				std::vector<int64_t> /* Drsh or sl_index */
			>;

std::vector<torch::Tensor> map_source_to_dest_plain_omp_v3(
	const std::vector<int64_t> & order,
	const std::vector< row_meta_new > & meta_new,
	const std::vector< row_meta_mrg > & meta_mrg
	){
	// 0) permute inv_order
	std::vector<int64_t> _tmp_range(order.size());
	for (int i=0; i<order.size(); i++) { 
		_tmp_range[i]= i;
	}
	auto inv_order= apply_inv_perm(_tmp_range, order);

	// 1) build jobs
	std::map< std::vector<int64_t> /*tn*/, row_meta_new > jobs;
	std::map< std::vector<int64_t> /*tn == t1*/, std::vector< row_meta_mrg > > jobs_b;

	// 1.1 populate jobs and keys of jobs_b
	for (auto const &row : meta_new) {
		jobs[std::get<0>(row)]= row;
		jobs_b[std::get<0>(row)]= std::vector< row_meta_mrg >();
	}
	// 1.2 populate jobs_b
	int64_t n_elem=0;
	for (auto const &row : meta_mrg) {
		jobs_b[std::get<0>(row)].push_back(row);
		// compute location in source_inds, dest_inds
		int64_t D_inds= (std::get<1>(row)[1]-std::get<1>(row)[0]);
		std::get<4>(jobs_b[std::get<0>(row)].back())= {n_elem, n_elem+D_inds};
		n_elem+= D_inds;
	}

	auto options_int= torch::TensorOptions()
	    .dtype(torch::kInt64)
	    .layout(torch::kStrided)
		.device(torch::kCPU);
	torch::Tensor source_inds= torch::empty( n_elem, options_int );
	torch::Tensor dest_inds= torch::empty( n_elem, options_int );

	for (auto const &row : jobs) {
		auto job= row.second;
		torch::Tensor tmp_b = torch::arange(std::get<2>(job)[0], std::get<2>(job)[1], options_int)
			.view(at::IntArrayRef(std::get<1>(job)));
		
		#pragma omp parallel for
		for (auto const &job_b : jobs_b[std::get<0>(job)]) {
			// prelim)
      // get strides of shape Do, strides of shape permute(Do; order)
			std::vector<at::indexing::TensorIndex> slcs;
			slcs.reserve(std::get<3>(job_b).size());
			for (auto const &elem_Dslc : std::get<3>(job_b)) {
				slcs.emplace(slcs.end(), torch::indexing::Slice(elem_Dslc[0], elem_Dslc[1]));
			}
			auto inv_Do= apply_perm(std::get<2>(job_b), order);
      
      source_inds.index_put_({torch::indexing::Slice(std::get<4>(job_b)[0],std::get<4>(job_b)[1])},
      	torch::arange(std::get<1>(job_b)[0],std::get<1>(job_b)[1],options_int));
      dest_inds.index_put_({torch::indexing::Slice(std::get<4>(job_b)[0],std::get<4>(job_b)[1])},
      	tmp_b.index(at::ArrayRef<at::indexing::TensorIndex>(slcs)).view(at::IntArrayRef(inv_Do))
      		.permute(at::IntArrayRef(inv_order)).contiguous().view(-1)
      	);
    }
  }
  return {source_inds, dest_inds};
}


// torch::Tensor tm_forward_1d_p2p(
// 	torch::Tensor data,
// 	const std::vector<int64_t> & order,
// 	const std::vector< std::tuple<
// 		std::vector<int64_t> /* tn */, 
// 		std::vector<int64_t> /* Dn */, 
// 		std::vector<int64_t> /* Sln */, 
// 		std::vector<int64_t> /* t1 */,
// 		std::vector< 
// 			std::tuple <
// 				std::vector<int64_t> /* _ */,
// 				std::vector<int64_t> /* slo */,
// 				std::vector<int64_t> /* Do */,
// 				std::vector< std::vector<int64_t> >, /* Dscl */
// 				std::vector<int64_t> /* Drsh */
// 			>
// 		>
// 	> > & jobs,
// 	int64_t Dsize
// 	){

// 	auto options= torch::TensorOptions()
// 	  .dtype(data.dtype())
// 	  .layout(data.layout())
// 		.device(data.device())
// 		.requires_grad(data.requires_grad());
// 	torch::Tensor newdata= torch::zeros( Dsize, options );

// 	auto source_to_dest= map_source_to_dest_plain_omp(data,order,jobs);
// 	newdata.scatter_(0, source_to_dest, data);
// 	return newdata;
// }

torch::Tensor tm_forward_1d_p2p_v2(
	torch::Tensor data,
	torch::Tensor source_inds,
	torch::Tensor dest_inds,
	int64_t Dsize
	){

	auto options= torch::TensorOptions()
	  .dtype(data.dtype())
	  .layout(data.layout())
		.device(data.device())
		.requires_grad(data.requires_grad());
	torch::Tensor newdata= torch::zeros( Dsize, options );

	newdata.index_put_({dest_inds}, data.index({source_inds}));
	return newdata;
}

void tm_backward_1d(
	std::vector<torch::Tensor> A
	){
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tm_forward_1d, "TM forward");
  // m.def("forward_p2p", &tm_forward_1d_p2p, "TM forward p2p");
  m.def("forward_p2p_v2", &tm_forward_1d_p2p_v2, "TM forward p2p");
  m.def("forward_omp", &tm_forward_1d_omp, "TM forward omp");
  m.def("backward", &tm_backward_1d, "TM backward");
  m.def("map_source_to_dest_plain_omp", &map_source_to_dest_plain_omp, "source_to_dest map");
  m.def("map_source_to_dest_plain_omp_v3", &map_source_to_dest_plain_omp_v3, "source_to_dest map");
}