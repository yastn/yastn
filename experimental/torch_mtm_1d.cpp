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


torch::Tensor mtm_forward_1d_plain(
	torch::Tensor data,
	std::vector<int64_t> order,
	std::vector< std::tuple<
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
	> > jobs,
	int64_t Dsize
	){

	auto options= torch::TensorOptions()
	    .dtype(data.dtype())
	    .layout(data.layout())
		.device(data.device());
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

torch::Tensor mtm_forward_1d_plain_omp(
	torch::Tensor data,
	std::vector<int64_t> order,
	std::vector< std::tuple<
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
	> > jobs,
	int64_t Dsize
	){

	auto options= torch::TensorOptions()
	    .dtype(data.dtype())
	    .layout(data.layout())
		.device(data.device());
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

inline std::vector<int64_t> get_strides(const std::vector<int64_t> & D){
	std::vector<int64_t> strides(D.size()+1);
	strides[D.size()]=1;
	for (int d=D.size()-1;d>=0;d--) {
		strides[d]=D[d]*strides[d+1];
	}
	return strides;
}

inline std::vector<int64_t> get_indices(int64_t i, const std::vector<int64_t> & strides){
	std::vector<int64_t> inds(strides.size()-1);
	for (int d=0;d<strides.size()-1;d++){
		inds[d]=(i%strides[d])/strides[d+1];
	}
	return inds;
}

inline int64_t index_1d(const std::vector<int64_t> & X, const std::vector<int64_t> & strides){
	int64_t i=0;
	for (int d=0;d<X.size();d++) {
		i+= X[d]*strides[d+1];
	}
	return i;
}

inline int64_t f_i_source_to_dest(
	int64_t i,
	int64_t dest_offset,
	const std::vector<int64_t> & strides_Do,
	const std::vector<int64_t> & order,
	const std::vector<int64_t> & strides_perm,
	const std::vector<int64_t> & strides_reshaped,
	const std::vector< std::vector<int64_t> > & Dslc,
	const std::vector<int64_t> & strides_Dn,
	int64_t source_offset
	){
  	//0) subtract offset
    i-=dest_offset;
  	//i) map from source 1d-array (specified by slice slo) to tuple X=(x_0,...,x_n)
  	//    specifying location inside Do shape
    auto X= get_indices(i, strides_Do);
  	// ii) permute
    //
    auto X_perm= apply_perm(X, order);
    auto i_perm= index_1d(X_perm, strides_perm);
    // iii) indices in (permute+reshape)d source
    auto X_reshaped= get_indices(i_perm,strides_reshaped);
    // iv) indices in reshaped destination
  	std::vector<int64_t> X_dest_block(X_reshaped.size());
    for (int d=0; d<X_reshaped.size(); d++) {
    	X_dest_block[d]= X_reshaped[d] + Dslc[d][0];
    }
    auto i_dest_block= index_1d(X_dest_block, strides_Dn);
    auto i_dest= i_dest_block + source_offset;
    return i_dest;
}

torch::Tensor map_source_to_dest_v1(
	torch::Tensor data,
	std::vector<int64_t> order,
	std::vector< std::tuple<
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
	> > jobs){

	auto options_int= torch::TensorOptions()
	    .dtype(torch::kInt64)
	    .layout(data.layout())
		.device(torch::kCPU);
	torch::Tensor source_to_dest= torch::arange( data.numel(), options_int );
	auto a_source_to_dest = source_to_dest.accessor<int64_t,1>();

	for (auto const &job : jobs) {
		auto strides_Dn= get_strides(std::get<1>(job));
		for (auto const &job_b : std::get<4>(job)) {
			// prelim)
      // get strides of shape Do, strides of shape permute(Do; order)
			auto strides_Do= get_strides(std::get<2>(job_b));
      auto strides_perm= get_strides(apply_perm(std::get<2>(job_b), order));
      auto strides_reshaped= get_strides(std::get<4>(job_b));

      auto i_source_to_dest = [&](int64_t i) -> int64_t {
      	//0) subtract offset
        i-=std::get<1>(job_b)[0];
      	//i) map from source 1d-array (specified by slice slo) to tuple X=(x_0,...,x_n)
      	//    specifying location inside Do shape
        auto X= get_indices(i, strides_Do);
      	// ii) permute
        //
        auto X_perm= apply_perm(X, order);
        auto i_perm= index_1d(X_perm, strides_perm);
        // iii) indices in (permute+reshape)d source
        auto X_reshaped= get_indices(i_perm,strides_reshaped);
        // iv) indices in reshaped destination
      	std::vector<int64_t> X_dest_block(X_reshaped.size());
        for (int d=0; d<X_reshaped.size(); d++) {
        	X_dest_block[d]= X_reshaped[d] + std::get<3>(job_b)[d][0];
        }
        auto i_dest_block= index_1d(X_dest_block, strides_Dn);
        auto i_dest= i_dest_block + std::get<2>(job)[0];
        return i_dest;
      };

      for (int64_t i=std::get<1>(job_b)[0]; i<std::get<1>(job_b)[1]; i++){
      	a_source_to_dest[i]= i_source_to_dest(a_source_to_dest[i]);
      }
    }
  }
  return source_to_dest;
}

torch::Tensor map_source_to_dest_v2(
	torch::Tensor data,
	const std::vector<int64_t> & order,
	std::vector< std::tuple<
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
	> > jobs){

	auto options_int= torch::TensorOptions()
	    .dtype(torch::kInt64)
	    .layout(data.layout())
		.device(torch::kCPU);
	torch::Tensor source_to_dest= torch::empty( data.numel(), options_int );
	auto a_source_to_dest = source_to_dest.accessor<int64_t,1>();

	for (auto const &job : jobs) {
		auto strides_Dn= get_strides(std::get<1>(job));
		#pragma omp parallel for
		for (auto const &job_b : std::get<4>(job)) {
			// prelim)
      // get strides of shape Do, strides of shape permute(Do; order)
			auto strides_Do= get_strides(std::get<2>(job_b));
      auto strides_perm= get_strides(apply_perm(std::get<2>(job_b), order));
      auto strides_reshaped= get_strides(std::get<4>(job_b));

      #pragma omp simd
      for (int64_t i=std::get<1>(job_b)[0]; i<std::get<1>(job_b)[1]; i++){
      	a_source_to_dest[i]= f_i_source_to_dest(i,
      		std::get<1>(job_b)[0], strides_Do, order, strides_perm, strides_reshaped,
      		std::get<3>(job_b), strides_Dn, std::get<2>(job)[0]);
      }
    }
  }
  return source_to_dest;
}

torch::Tensor map_source_to_dest_plain(
	torch::Tensor data,
	const std::vector<int64_t> & order,
	std::vector< std::tuple<
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
	> > jobs){

	std::vector<int64_t> _tmp_range(order.size());
	for (int i=0; i<order.size(); i++) {
		_tmp_range[i]= i;
	}
	auto inv_order= apply_inv_perm(_tmp_range, order);

	auto options_int= torch::TensorOptions()
	    .dtype(torch::kInt64)
	    .layout(data.layout())
		.device(torch::kCPU);
	torch::Tensor source_to_dest= torch::empty( data.numel(), options_int );

	for (auto const &job : jobs) {
		torch::Tensor tmp_b = torch::arange(std::get<2>(job)[0], std::get<2>(job)[1], options_int)
			.view(at::IntArrayRef(std::get<1>(job)));

		for (auto const &job_b : std::get<4>(job)) {
			// prelim)
      // get strides of shape Do, strides of shape permute(Do; order)
			std::vector<at::indexing::TensorIndex> slcs;
			slcs.reserve(std::get<3>(job_b).size());
			for (auto const &elem_Dslc : std::get<3>(job_b)) {
				slcs.emplace(slcs.end(), torch::indexing::Slice(elem_Dslc[0], elem_Dslc[1]));
			}
			auto inv_Do= apply_perm(std::get<2>(job_b), order);

      source_to_dest.index_put_({torch::indexing::Slice(std::get<1>(job_b)[0],std::get<1>(job_b)[1])},
      	tmp_b.index(at::ArrayRef<at::indexing::TensorIndex>(slcs)).view(at::IntArrayRef(inv_Do))
      		.permute(at::IntArrayRef(inv_order)).contiguous().view(-1)
      	);
    }
  }
  return source_to_dest;
}

torch::Tensor map_source_to_dest_plain_omp(
	torch::Tensor data,
	const std::vector<int64_t> & order,
	std::vector< std::tuple<
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
	> > jobs){

	std::vector<int64_t> _tmp_range(order.size());
	for (int i=0; i<order.size(); i++) {
		_tmp_range[i]= i;
	}
	auto inv_order= apply_inv_perm(_tmp_range, order);

	auto options_int= torch::TensorOptions()
	    .dtype(torch::kInt64)
	    .layout(data.layout())
		.device(torch::kCPU);
	torch::Tensor source_to_dest= torch::empty( data.numel(), options_int );

	for (auto const &job : jobs) {
		torch::Tensor tmp_b = torch::arange(std::get<2>(job)[0], std::get<2>(job)[1], options_int)
			.view(at::IntArrayRef(std::get<1>(job)));

		#pragma omp parallel for
		for (auto const &job_b : std::get<4>(job)) {
			// prelim)
      // get strides of shape Do, strides of shape permute(Do; order)
			std::vector<at::indexing::TensorIndex> slcs;
			slcs.reserve(std::get<3>(job_b).size());
			for (auto const &elem_Dslc : std::get<3>(job_b)) {
				slcs.emplace(slcs.end(), torch::indexing::Slice(elem_Dslc[0], elem_Dslc[1]));
			}
			auto inv_Do= apply_perm(std::get<2>(job_b), order);

      source_to_dest.index_put_({torch::indexing::Slice(std::get<1>(job_b)[0],std::get<1>(job_b)[1])},
      	tmp_b.index(at::ArrayRef<at::indexing::TensorIndex>(slcs)).view(at::IntArrayRef(inv_Do))
      		.permute(at::IntArrayRef(inv_order)).contiguous().view(-1)
      	);
    }
  }
  return source_to_dest;
}

torch::Tensor map_source_to_dest_plain_omp_v2(
	torch::Tensor data,
	const std::vector<int64_t> & order,
	const std::vector< std::tuple<
		std::vector<int64_t> /* tn */,
		std::vector<int64_t> /* Dn */,
		std::vector<int64_t>  /* Sln */
		> > & meta_new,
	const std::vector< std::tuple <
				std::vector<int64_t> /* t1 */,
				std::vector<int64_t> /* slo */,
				std::vector<int64_t> /* Do */,
				std::vector< std::vector<int64_t> >  /* Dscl */,
				std::vector<int64_t> /* Drsh */
			> > & meta_mrg
	){
	// 0) permute inv_order
	std::vector<int64_t> _tmp_range(order.size());
	for (int i=0; i<order.size(); i++) {
		_tmp_range[i]= i;
	}
	auto inv_order= apply_inv_perm(_tmp_range, order);

	// 1) build jobs
	std::map< std::vector<int64_t> /*tn*/, std::tuple<
		std::vector<int64_t>/*tn*/,
		std::vector<int64_t>/*Dn*/,
		std::vector<int64_t>/*Sln*/> > jobs;
	std::map< std::vector<int64_t> /*tn == t1*/, std::vector< std::tuple<
		std::vector<int64_t> /* t1 */,
		std::vector<int64_t> /* slo */,
		std::vector<int64_t> /* Do */,
		std::vector< std::vector<int64_t> >  /* Dscl */,
		std::vector<int64_t> /* Drsh */ > > > jobs_b;

	// 1.1 populate jobs and keys of jobs_b
	for (auto const &row : meta_new) {
		jobs[std::get<0>(row)]= row;
		jobs_b[std::get<0>(row)]= std::vector< std::tuple<
			std::vector<int64_t> /* t1 */,
			std::vector<int64_t> /* slo */,
			std::vector<int64_t> /* Do */,
			std::vector< std::vector<int64_t> >  /* Dscl */,
			std::vector<int64_t> /* sl_index */> >();
	}
	// 1.2 populate jobs_b
	int64_t n_elem=0;
	for (auto const &row : meta_mrg) {
		jobs_b[std::get<0>(row)].push_back(row);
	}

	auto options_int= torch::TensorOptions()
	    .dtype(torch::kInt64)
	    .layout(data.layout())
		.device(torch::kCPU);
	torch::Tensor source_to_dest= torch::empty( data.numel(), options_int );

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

      source_to_dest.index_put_({torch::indexing::Slice(std::get<1>(job_b)[0],std::get<1>(job_b)[1])},
      	tmp_b.index(at::ArrayRef<at::indexing::TensorIndex>(slcs)).view(at::IntArrayRef(inv_Do))
      		.permute(at::IntArrayRef(inv_order)).contiguous().view(-1)
      	);
    }
  }
  return source_to_dest;
}

std::vector<torch::Tensor> map_source_to_dest_plain_omp_v3(
	torch::Tensor data,
	const std::vector<int64_t> & order,
	const std::vector< std::tuple<
		std::vector<int64_t> /* tn */,
		std::vector<int64_t> /* Dn */,
		std::vector<int64_t>  /* Sln */
		> > & meta_new,
	const std::vector< std::tuple <
				std::vector<int64_t> /* t1 */,
				std::vector<int64_t> /* slo */,
				std::vector<int64_t> /* Do */,
				std::vector< std::vector<int64_t> >  /* Dscl */,
				std::vector<int64_t> /* Drsh */
			> > & meta_mrg
	){
	// 0) permute inv_order
	std::vector<int64_t> _tmp_range(order.size());
	for (int i=0; i<order.size(); i++) {
		_tmp_range[i]= i;
	}
	auto inv_order= apply_inv_perm(_tmp_range, order);

	// 1) build jobs
	std::map< std::vector<int64_t> /*tn*/, std::tuple<
		std::vector<int64_t>/*tn*/,
		std::vector<int64_t>/*Dn*/,
		std::vector<int64_t>/*Sln*/> > jobs;
	std::map< std::vector<int64_t> /*tn == t1*/, std::vector< std::tuple<
		std::vector<int64_t> /* t1 */,
		std::vector<int64_t> /* slo */,
		std::vector<int64_t> /* Do */,
		std::vector< std::vector<int64_t> >  /* Dscl */,
		std::vector<int64_t> /* sl_index */ > > > jobs_b;

	// 1.1 populate jobs and keys of jobs_b
	for (auto const &row : meta_new) {
		jobs[std::get<0>(row)]= row;
		jobs_b[std::get<0>(row)]= std::vector< std::tuple<
			std::vector<int64_t> /* t1 */,
			std::vector<int64_t> /* slo */,
			std::vector<int64_t> /* Do */,
			std::vector< std::vector<int64_t> >  /* Dscl */,
			std::vector<int64_t> /* sl_index */> >();
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
	    .layout(data.layout())
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
  }1
  return {source_inds, dest_inds};
}


torch::Tensor mtm_forward_1d_ptp(
	torch::Tensor data,
	torch::Tensor source_to_dest,
	int64_t Dsize
	){

	auto options= torch::TensorOptions()
	    .dtype(data.dtype())
	    .layout(data.layout())
		.device(data.device());
	torch::Tensor newdata= torch::zeros( Dsize, options );

	newdata.scatter_(0, source_to_dest, data);
	return newdata;
}

torch::Tensor mtm_forward_1d_ptp_v2(
	torch::Tensor data,
	torch::Tensor source_inds,
	torch::Tensor dest_inds,
	int64_t Dsize
	){

	auto options= torch::TensorOptions()
	    .dtype(data.dtype())
	    .layout(data.layout())
		.device(data.device());
	torch::Tensor newdata= torch::zeros( Dsize, options );

	newdata.index_put_({dest_inds}, data.index({source_inds}));
	return newdata;
}

void mtm_backward_1d(
	std::vector<torch::Tensor> A
	){
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_plain", &mtm_forward_1d_plain, "MTM forward");
  m.def("forward_plain_omp", &mtm_forward_1d_plain_omp, "MTM forward");
  m.def("forward_ptp", &mtm_forward_1d_ptp, "MTM forward");
  m.def("forward_ptp_v2", &mtm_forward_1d_ptp_v2, "MTM forward");
  m.def("backward", &mtm_backward_1d, "MTM backward");
  m.def("map_source_to_dest_v1", &map_source_to_dest_v1, "source_to_dest map");
  m.def("map_source_to_dest_v2", &map_source_to_dest_v2, "source_to_dest map");
  m.def("map_source_to_dest_plain", &map_source_to_dest_plain, "source_to_dest map");
  m.def("map_source_to_dest_plain_omp", &map_source_to_dest_plain_omp, "source_to_dest map");
  m.def("map_source_to_dest_plain_omp_v2", &map_source_to_dest_plain_omp_v2, "source_to_dest map");
  m.def("map_source_to_dest_plain_omp_v3", &map_source_to_dest_plain_omp_v3, "source_to_dest map");
}