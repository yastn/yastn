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

torch::Tensor tm_forward_plain(
	torch::Tensor data,
	const std::vector<int64_t> & order,
	const std::vector< std::tuple<
		std::vector<int64_t> /* tn */, 
		std::vector<int64_t> /* Dn */, 
		std::vector<int64_t> /* Sln */, 
		std::vector<int64_t> /* t1 */,
		std::vector< row_meta_mrg > 
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

torch::Tensor tm_forward_plain_omp(
	torch::Tensor data,
	const std::vector<int64_t> & order,
	const std::vector< std::tuple<
		std::vector<int64_t> /* tn */, 
		std::vector<int64_t> /* Dn */, 
		std::vector<int64_t> /* Sln */, 
		std::vector<int64_t> /* t1 */,
		std::vector< row_meta_mrg > 
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
			std::cout<<"Parallel "<< at::in_parallel_region() 
				<<" #threads "<< at::get_num_threads() <<" tid "<< at::get_thread_num() << std::endl;

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

inline std::vector<int64_t> get_strides_v2(const std::vector<int64_t> & D){
	std::vector<int64_t> strides(D.size());
	strides[D.size()-1]=1;
	for (int d=D.size()-1;d>0;d--) {
		strides[d-1]=D[d]*strides[d];
	}
	return strides;
}

inline int64_t index_1d_v2(const std::vector<int64_t> & X, const std::vector<int64_t> & strides){
	int64_t i=0;
	for (int d=0;d<X.size();d++) {
		i+= X[d]*strides[d];
	}
	return i;
}

std::vector<torch::Tensor> map_source_to_dest_plain_omp_v2(
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
			#pragma omp critical
			{
			std::cout<<"Parallel "<< at::in_parallel_region() 
				<<" #threads "<< at::get_num_threads() <<" tid "<< omp_get_thread_num() << std::endl;
			}

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
  std::cout<<std::endl;
  return {source_inds, dest_inds};
}

std::vector<torch::Tensor> map_source_to_dest_v3(
	torch::Tensor data,
	const std::vector<int64_t> & order,
	const std::vector< row_meta_new > & meta_new,
	const std::vector< row_meta_mrg > & meta_mrg
	){

	// 1) build jobs (blocks in destination array), which are keys of map,
	//    holding array jobs_b of blocks in source array
	std::map< std::vector<int64_t> /*tn*/, row_meta_new > jobs;
	// std::map< std::vector<int64_t> /*tn == t1*/, std::vector< row_meta_mrg > > jobs_b;

	// 1.1 populate jobs and keys of jobs_b
	for (auto const &row : meta_new) {
		jobs[std::get<0>(row)]= row;
		// jobs_b[std::get<0>(row)]= std::vector< row_meta_mrg >();
	}

	std::vector< std::vector<int64_t> > sl_index;

	// 1.2 populate jobs_b
	int64_t n_elem=0;
	for (int64_t b=0; b<meta_mrg.size(); b++) {
	// for (auto const &row : meta_mrg) {
		// jobs_b[std::get<0>(row)].push_back(row);
		// compute location in source_inds, dest_inds
		const auto & row = meta_mrg[b];
		int64_t D_inds= (std::get<1>(row)[1]-std::get<1>(row)[0]);
		// std::get<4>(jobs_b[std::get<0>(row)].back())= {n_elem, n_elem+D_inds};
		sl_index.push_back({b, n_elem, n_elem+D_inds});
		n_elem+= D_inds;
	}

	// 1.3 sort in descending order by job size
	std::sort(sl_index.begin(), sl_index.end(), 
		[](const auto &l, const auto &r) {return l[2]-l[1] > r[2]-r[1];});

	// 2) prepare arrays to hold source and destination indices
	//    where element at source_inds[i] is mapped to dest_inds[i]
	auto options_int= torch::TensorOptions()
	    .dtype(torch::kInt64)
	    .layout(data.layout())
		.device(torch::kCPU);
	torch::Tensor source_inds= torch::empty( n_elem, options_int );
	torch::Tensor dest_inds= torch::empty( n_elem, options_int );
	auto a_source_inds = source_inds.accessor<int64_t,1>();
	auto a_dest_inds = dest_inds.accessor<int64_t,1>();

	// for (auto const &row : jobs) {
	// 	for (auto const &job_b : jobs_b[row.first]) {
	//    const auto & job= row.second;
	#pragma omp parallel for schedule(dynamic)
	for (int64_t b=0; b<meta_mrg.size(); b++) {
			const auto & job_b= meta_mrg[sl_index[b][0]];
			const auto & job= jobs[std::get<0>(job_b)];

			// prelim)
      // get strides of shape D_src, permuted D_src to D_src_perm
			const auto & D_src= std::get<2>(job_b);
			const int64_t r_src= D_src.size();
			const auto strides_src= get_strides_v2(D_src);
			const auto D_src_perm= apply_perm(D_src, order);

			const auto strides_dest= get_strides_v2(std::get<1>(job));

      
      const auto & D_offset_dest= std::get<3>(job_b);
      std::vector<int64_t> D_block_dest(D_offset_dest.size());
      const int64_t r_dest= D_offset_dest.size();
      for (int64_t i=0; i<r_dest; i++) {
				D_block_dest[i]= D_offset_dest[i][1]-D_offset_dest[i][0];
			}

			std::vector<int64_t> X_source_rp(r_src, 0);
			std::vector<int64_t> X_source_rpr(r_dest, 0);
			std::vector<int64_t> Y_dest_block(r_dest);
			for (int64_t i=0; i<r_dest; i++) {
				Y_dest_block[i]= D_offset_dest[i][0];
			}

			const int64_t index_default_src = r_src-1;
			const int64_t index_default_dest = r_dest-1;
			const int64_t stride_default_src = strides_src[order[index_default_src]];
			const int64_t stride_default_dest = strides_dest[index_default_dest];

			int64_t index=0;
			int64_t i_src= std::get<1>(job_b)[0];
			int64_t i_dest= index_1d_v2(Y_dest_block, strides_dest) + std::get<2>(job)[0];

			// 3) interpret as contiguous iteration over elements of reshape-permute-reshaped 
			//    source block, i.e., with shape D_src_perm (D_src_perm[i]= D_src[order[i]]) 
			//    OR D_block_dest (D_block_dest[i]= block_dest_slcs[i][1]-block_dest_slcs[i][0]) 		
			for (int64_t i=sl_index[b][1]; i<sl_index[b][2]; i++) {

				a_source_inds[i]= i_src;
				a_dest_inds[i]= i_dest;

				index = index_default_src;
				X_source_rp[index]++;
				i_src+= stride_default_src;
			  // carry on reshape-permuted source
			  while (X_source_rp[index] == D_src_perm[index]){
			    X_source_rp[index] = 0;
			 		i_src-= D_src_perm[index]*strides_src[order[index]];
			    index--;
			    if (index<0) {
			    	break;
			    }
			    X_source_rp[index]++;
			    i_src+= strides_src[order[index]];
			  }
			  
			  index = index_default_dest;
			  X_source_rpr[index]++;
			  i_dest += stride_default_dest;
			  // carry on block in dest
			  while (X_source_rpr[index] == D_block_dest[index]){
			    X_source_rpr[index] = 0;
			    i_dest-= D_block_dest[index]*strides_dest[index];
			    index--;
			    if (index<0) {
			    	break;
			    }
			    i_dest += strides_dest[index];
			    X_source_rpr[index]++;
			  }
			}

  //   }
  }
  return {source_inds, dest_inds};
}

torch::Tensor map_source_to_dest_unmerge(
	const int64_t dest_size,
	const std::vector< std::tuple <
				std::vector<int64_t> /* sln */,
				std::vector<int64_t> /* Dn */,
				std::vector<int64_t> /* slo */,
				std::vector<int64_t> /* Do */,
				std::vector< std::vector<int64_t> >  /* block_src_slcs */
			> > & meta
	){

	// 1) prepare arrays to hold source and destination indices
	//    where element at source_inds[i] is mapped to dest_inds[i]
	auto options_int= torch::TensorOptions()
	    .dtype(torch::kInt64)
	    .layout(torch::kStrided)
		.device(torch::kCPU);
	torch::Tensor source_inds= torch::empty( dest_size, options_int );
	auto a_source_inds = source_inds.accessor<int64_t,1>();

	#pragma omp parallel for
	for (int64_t b=0; b<meta.size(); b++) {
			const auto & job_b= meta[b];

			// prelim)
      // get strides of source shape D_src
			const auto & D_src= std::get<3>(job_b);
			const int64_t r_src= D_src.size();
			const auto strides_src= get_strides_v2(D_src);
			//
			// get location of sub-block in D_src
			const auto & D_offset_src= std::get<4>(job_b);
			// get D_src_b sub-block shape and strides
			std::vector<int64_t> D_src_b(D_offset_src.size());
			for (int64_t i=0; i<r_src; i++) {
				D_src_b[i]= D_offset_src[i][1]-D_offset_src[i][0];
			}
			
			std::vector<int64_t> X_src_block(r_src,0);
			// index of first element of sub-block src_b in source
			std::vector<int64_t> X_src_block_0(r_src);
			for (int64_t i=0; i<r_src; i++) {
				X_src_block_0[i]= D_offset_src[i][0];
			}

			const int64_t index_default_src = r_src-1;
			const int64_t stride_default_src = strides_src[index_default_src];

			int64_t index=0;
			int64_t i_src= index_1d_v2(X_src_block_0, strides_src) + std::get<2>(job_b)[0];

			// 3) interpret as contiguous iteration over elements of sub-block with shape D_src_b
			//    in source block with shape D_src OR destination of shape D_dest.
			//    The sub-block is located at D_offset_src, such that its shape is 
			//    D_src_b[i]= D_offset_src[i][1]-D_offset_src[i][0] 
			for (int64_t i=std::get<0>(job_b)[0]; i<std::get<0>(job_b)[1]; i++) {

				a_source_inds[i]= i_src;

				index = index_default_src;
				X_src_block[index]++;
				i_src+= stride_default_src;
			  // carry on reshape-permuted source
			  while (X_src_block[index] == D_src_b[index]){
			    X_src_block[index] = 0;
			 		i_src-= D_src_b[index]*strides_src[index];
			    index--;
			    if (index<0) {
			    	break;
			    }
			    X_src_block[index]++;
			    i_src+= strides_src[index];
			  }
			}

  //   }
  }
  return source_inds;
}

void tm_backward_1d(
	std::vector<torch::Tensor> A
	){
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tm_forward_plain", &tm_forward_plain, "TM forward");
  m.def("tm_forward_plain_omp", &tm_forward_plain_omp, "TM forward omp");
  m.def("forward_1d_p2p_v2", &tm_forward_1d_p2p_v2, "TM forward p2p");
  m.def("backward", &tm_backward_1d, "TM backward");
  // m.def("map_source_to_dest_plain_omp", &map_source_to_dest_plain_omp, "source_to_dest map");
  m.def("map_source_to_dest_plain_omp_v2", &map_source_to_dest_plain_omp_v2, "source_to_dest map");
  m.def("map_source_to_dest_v3", &map_source_to_dest_v3, "source_to_dest map merge");
  m.def("map_source_to_dest_unmerge", &map_source_to_dest_unmerge, "source_to_dest map unmerge");
}