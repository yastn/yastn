#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cutensor.h>

#include <random>
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <unordered_map>

// Handle cuTENSOR errors
#define HANDLE_ERROR(x)                                                           \
{                                                                                 \
    const cutensorStatus_t err = (x);                                             \
    if ( err != CUTENSOR_STATUS_SUCCESS )                                         \
    { throw std::runtime_error { std::string { cutensorGetErrorString(err) } }; } \
};

// Handle CUDA errors.
#define HANDLE_CUDA_ERROR(x)                                                  \
{                                                                             \
    const cudaError_t err = (x);                                              \
    if ( err != cudaSuccess )                                                 \
    { throw std::runtime_error { std::string { cudaGetErrorString(err) } }; } \
};

template <typename T>
using cuda_ptr = std::unique_ptr<T,decltype(&cudaFree)>;

template <typename T>
cuda_ptr<T> cuda_alloc( size_t count )
{
    void* result;
    cudaError_t err = cudaMalloc( &result, sizeof(T)*count );
    if ( err != cudaSuccess ) throw std::bad_alloc {};
    else return cuda_ptr<T> { reinterpret_cast<T*>(result), &cudaFree };
}

template <typename T>
struct Guard
{
    using destructor = cutensorStatus_t (*)( T );

    T p { nullptr };
    destructor destroy { nullptr };
    ~Guard() { if (p) destroy(p); }
};

namespace cublocksparse {

using ModeType   = int32_t;
using ExtentType = int64_t;
using StrideType = int64_t;

int tensordot_bs_cuda_impl(
    const double* a_ptr,
    const std::vector<ModeType> &modeA,                  // Modes of the tensor, which in turn also define the "rank/dim" of the tensor
    const std::vector<ModeType> nonZeroCoordinatesA,     // Coordinates of the non-zero blocks in the tensor, which are specified as a vector of indices
                                                         // with respect to sectionExtents already serialized into 1D, i.e.
                                                         // { { x0_0, x0_1, ..., x0_#modes-1 },
                                                         //   { x1_0,       ..., x1_#modes-1 },
                                                         //   ...                               } is given as 
                                                         // { x0_0, x0_1, ..., x0_#modes-1, x1_0, ..., x1_#modes-1, ... }
                                                         //    const std::vector<ModeType> modeB,
    const std::vector<StrideType> &offsetsA,
    const std::vector<StrideType> &stridesA,
    const double* b_ptr,
    const std::vector<ModeType> &modeB,
    const std::vector<ModeType> nonZeroCoordinatesB,
    const std::vector<StrideType> &offsetsB,
    const std::vector<StrideType> &stridesB,
    double* c_ptr,
    const std::vector<ModeType> &modeC,
    const std::vector<ModeType> nonZeroCoordinatesC,
    const std::vector<StrideType> &offsetsC,
    const std::vector<StrideType> &stridesC,
    const std::unordered_map<ModeType, std::vector<ExtentType>> sectionExtents
)
try
{
    const char* env = std::getenv("YASTN_LOG_LEVEL");
    int yastn_log_level = 0;
    if (env) {
        yastn_log_level = std::atoi(env);
        if ( yastn_log_level>1) { std::cout << "YASTN_LOG_LEVEL = " << yastn_log_level << std::endl; }
    } 

    auto bufA= a_ptr;
    auto bufB= b_ptr;
    auto bufC= c_ptr;

    // Initialise the library.
    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));
    Guard<cutensorHandle_t> guardHandle { handle, &cutensorDestroy };

    //////////////////////////////////////
    //                                  //
    // We compute C_{modeC} = A_{modeA}B_{modeB}   //
    //////////////////////////////////////
   
    auto printTensor = []
    (
      void * dev,
      int64_t size
    ) -> void
    {
        std::vector<double> temp(size, 0.0);
        HANDLE_CUDA_ERROR(cudaMemcpy(temp.data(), static_cast<double*>(dev),
                                     temp.size() * sizeof(double),
                                     cudaMemcpyDeviceToHost));
        for (int i = 0; i < size; ++i) {
            std::cout << temp[i] << ", ";
        }
        std::cout<<std::endl;
    };

    // Helper-λ to allocate and initialise block-sparse tensors with random
    // data. In this example we use 64-bit double precision numbers.
    cutensorDataType_t dataType = CUTENSOR_R_64F;
    auto initTensor = [&handle,&sectionExtents,&printTensor,&yastn_log_level,dataType]
    (
      const std::vector<ModeType>   &modes,              
      const std::vector<ModeType> &nonZeroCoordinates, 
      const std::vector<StrideType> &offsets,
      const std::vector<StrideType> &strides, 
      cutensorBlockSparseTensorDescriptor_t &desc, 
      Guard<cutensorBlockSparseTensorDescriptor_t> &guard,
      const double * buf, // Buffer to holding the non-zero blocks of the tensor
      std::vector<void*> &dev
    ) -> void
    {
        if (yastn_log_level > 3) {
            std::cout << "initTensor: modes: ";
            for (auto e : modes) { std::cout << e << " "; }
            std::cout << std::endl;
            std::cout << "initTensor: nonZeroCoordinates: ";
            for (auto e : nonZeroCoordinates) { std::cout << e << " "; }
            std::cout << std::endl;
        }

        uint32_t numModes         = modes.size();
        uint64_t numNonZeroBlocks = nonZeroCoordinates.size() / numModes;
        std::vector<uint32_t>     numSections;
        std::vector<ExtentType>   extents;
        for ( ModeType mode: modes ) // serialize sectionExtents into 1D for both number of section per mode (numSections)
                                     // and extents per each mode (extents) 
        {
            const std::vector<ExtentType> &modeExtents = sectionExtents.at(mode);

            numSections.push_back(modeExtents.size());
            extents.insert(extents.end(),modeExtents.begin(),modeExtents.end());
        }
        if (yastn_log_level > 3) {
            std::cout << "extents: ";
            for (auto e : extents) { std::cout << e << " "; }
            std::cout << std::endl << "strides: ";
            for (auto e : strides) { std::cout << e << " "; }
            std::cout << std::endl;
        }

        // We assume packed contiguous storage, column-major order.
        // This means that we may pass nullptr for the strides array later.
        // The offets are used to set the pointers in the dev vector.
        if (yastn_log_level > 3) {
            std::cout << "offsets: ";
            for (auto v : offsets) { std::cout << v << " "; }
            std::cout << std::endl;
        }
        dev.resize(numNonZeroBlocks);
        for ( uint64_t i = 0; i < numNonZeroBlocks; ++i ) {
            dev[i] = (void*)(buf + offsets[i]);
            if (yastn_log_level > 3) {
                ExtentType block_size = 1;
                for (size_t j = 0; j < numModes; ++j) {
                    block_size *= sectionExtents.at(modes[j])[nonZeroCoordinates[j + i * numModes]];
                }
                std::cout<< "block "<< i << " at " << offsets[i] << " size " << block_size << std::endl; 
                printTensor(dev[i], block_size);
            }
        }

        // Print contents of nonZeroCoordinates from its pointer
        if (yastn_log_level > 3) {
            std::cout << "nonZeroCoordinates (from pointer): ";
            const ModeType* ptr = nonZeroCoordinates.data();
            for (size_t i = 0; i < nonZeroCoordinates.size(); ++i) {
            std::cout << ptr[i] << " ";
            }
            std::cout << std::endl;
        }
        HANDLE_ERROR(cutensorCreateBlockSparseTensorDescriptor
        (
            handle, &desc,
            numModes, numNonZeroBlocks, numSections.data(), extents.data(),
            nonZeroCoordinates.data(), strides.data(), dataType
        ));
        guard.p = desc;
        guard.destroy = &cutensorDestroyBlockSparseTensorDescriptor;
    };
                                         

    //////////////
    // Tensor A //
    //////////////

    std::vector<void*> devA;
    cutensorBlockSparseTensorDescriptor_t descA;
    Guard<cutensorBlockSparseTensorDescriptor_t> guardDescA;
    initTensor(modeA,nonZeroCoordinatesA,offsetsA,stridesA,descA,guardDescA,bufA,devA);

    //////////////
    // Tensor B //
    //////////////

    std::vector<void*> devB;
    cutensorBlockSparseTensorDescriptor_t descB;
    Guard<cutensorBlockSparseTensorDescriptor_t> guardDescB;
    initTensor(modeB,nonZeroCoordinatesB,offsetsB,stridesB,descB,guardDescB,bufB,devB);

    //////////////
    // Tensor C //
    //////////////

    std::vector<void*> devC;
    cutensorBlockSparseTensorDescriptor_t descC;
    Guard<cutensorBlockSparseTensorDescriptor_t> guardDescC;
    initTensor(modeC,nonZeroCoordinatesC,offsetsC,stridesC,descC,guardDescC,bufC,devC);

    /*******************************
     * Block-sparse Contraction.   *
     *******************************/

    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateBlockSparseContraction(handle, &desc,
                descA, modeA.data(), CUTENSOR_OP_IDENTITY,
                descB, modeB.data(), CUTENSOR_OP_IDENTITY,
                descC, modeC.data(), CUTENSOR_OP_IDENTITY,
                descC, modeC.data(),
                CUTENSOR_COMPUTE_DESC_64F));
    Guard<cutensorOperationDescriptor_t> guardOpDesc { desc, &cutensorDestroyOperationDescriptor };

    // Plan preference
    cutensorPlanPreference_t planPref = nullptr;
    // const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
    // const cutensorJitMode_t jitMode = CUTENSOR_JIT_MODE_NONE;
    // HANDLE_ERROR(cutensorCreatePlanPreference(handle,&planPref,algo,jitMode));
    // Guard<cutensorPlanPreference_t> guardPlanPref { planPref, &cutensorDestroyPlanPreference };

    // Query workspace estimate
    uint64_t workspaceSize = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,desc,planPref,workspacePref,&workspaceSize));

    cuda_ptr<char> work = cuda_alloc<char>(workspaceSize);
    if ( uintptr_t(work.get()) % 128 != 0 ) throw std::bad_alloc {};

    // Create Contraction Plan
    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,&plan,desc,planPref,workspaceSize));
    Guard<cutensorPlan_t> guardPlan { plan, &cutensorDestroyPlan };

    // Execute
    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
    struct StreamGuard { cudaStream_t stream; ~StreamGuard() { cudaStreamDestroy(stream); } };
    StreamGuard guardStream { stream };

    double alpha = 1., beta = 0.;
    HANDLE_ERROR(cutensorBlockSparseContract(handle, plan,
                (void*) &alpha, (const void *const *) devA.data(), (const void *const *) devB.data(),
                (void*) &beta,  (const void *const *) devC.data(), (      void *const *) devC.data(), 
                (void*) work.get(), workspaceSize, stream));

    return EXIT_SUCCESS;
}
catch ( std::exception &ex )
{
    std::cerr << "Exception caught! Exiting." << std::endl;
    std::cerr << ex.what() << std::endl;
    return EXIT_FAILURE;
}
catch ( ... )
{
    std::cerr << "Unknown exception caught! Exiting." << std::endl;
    return EXIT_FAILURE;
}


at::Tensor tensordot_bs_cuda(
    const at::Tensor& a, 
    const at::Tensor& b,
    const std::vector<int64_t>& a_blocks, 
    const std::vector<int64_t>& a_offsets,
    const std::vector<int64_t>& a_strides,
    const at::Tensor& a_D_per_mode, 
    const std::vector<int64_t>& nout_a, 
    const std::vector<int64_t>& nin_a,
    const std::vector<int64_t>& b_blocks, 
    const std::vector<int64_t>& b_offsets,
    const std::vector<int64_t>& b_strides,
    const at::Tensor& b_D_per_mode, 
    const std::vector<int64_t>& nout_b, 
    const std::vector<int64_t>& nin_b,
    int64_t c_size,
    const std::vector<int64_t>& c_blocks, 
    const std::vector<int64_t>& c_offsets,
    const std::vector<int64_t>& c_strides,
    const at::Tensor& c_D_per_mode
) {
  const char* env = std::getenv("YASTN_LOG_LEVEL");
  int yastn_log_level = 0;
  if (env) {
    yastn_log_level = std::atoi(env);
    if (yastn_log_level>0) { std::cout << "YASTN_LOG_LEVEL = " << yastn_log_level << std::endl; }
  }
  TORCH_CHECK(a.dim() == 1, "Input 'a' must be 1D.");
  TORCH_CHECK(b.dim() == 1, "Input 'b' must be 1D.");
  TORCH_CHECK(a.dtype() == at::kDouble, "Input 'a' must be float64."); // At this stage, we support only float64.
  TORCH_CHECK(b.dtype() == at::kDouble, "Input 'b' must be float64.");
  TORCH_CHECK(a_D_per_mode.dtype() == at::kLong, "Input 'a_D_per_mode' must be int64 valued.");
  TORCH_CHECK(b_D_per_mode.dtype() == at::kLong, "Input 'b_D_per_mode' must be int64 valued.");
  TORCH_CHECK(c_D_per_mode.dtype() == at::kLong, "Input 'c_D_per_mode' must be int64 valued.");
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  //at::Tensor a_contig = a.contiguous(); // For 1D tensors, contiguous is not needed
  //at::Tensor b_contig = b.contiguous();

  const double* a_ptr = a.data_ptr<double>();
  const double* b_ptr = b.data_ptr<double>();

  at::Tensor result = torch::zeros(c_size, a.options());
  double* result_ptr = result.data_ptr<double>();

  // Prepare inputs for the CUDA implementation.
  //
  // 0) prepare mode labels (possibly use integers directly) and 1) prepare extents per mode
  TORCH_CHECK(a_D_per_mode.dim() == 2, "Input 'a_D_per_mode' must be a 2D tensor.");
  TORCH_CHECK(b_D_per_mode.dim() == 2, "Input 'b_D_per_mode' must be a 2D tensor.");
  TORCH_CHECK(c_D_per_mode.dim() == 2, "Input 'c_D_per_mode' must be a 2D tensor.");

  // Convert the 2D tensor to a vector of vectors.
  // This is needed for the CUDA implementation. 
  // Note: Otherwise a_D_per_mode[nout_a[i]][j] is an at::Tensor, so we need to call .item<int64_t>() to get the value.
  auto aAcc = a_D_per_mode.accessor<int64_t,2>();
  auto bAcc = b_D_per_mode.accessor<int64_t,2>();
  auto cAcc = c_D_per_mode.accessor<int64_t,2>();

  const std::string alphabet{"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"};
  std::unordered_map<ModeType, std::vector<ExtentType>> sectionExtents;
  std::vector<ModeType> a_modes, b_modes, c_modes;
  a_modes.resize(nout_a.size() + nin_a.size());
  b_modes.resize(nout_b.size() + nin_b.size());
  c_modes.resize(nout_a.size() + nout_b.size());
  for (size_t i = 0; i < nout_a.size(); ++i) { 
    a_modes[nout_a[i]]= alphabet[i]; 
    c_modes[i]= alphabet[i];
    for (size_t j = 0; j < a_D_per_mode.size(1); ++j) {
      if (aAcc[nout_a[i]][j] < 0) break; 
      sectionExtents[a_modes[nout_a[i]]].push_back(aAcc[nout_a[i]][j]);
    }
  } 
  for (size_t i = 0; i < nout_b.size(); ++i) { 
    b_modes[nout_b[i]]= alphabet[i + nout_a.size()];
    c_modes[i + nout_a.size()]= alphabet[i + nout_a.size()];
    for (size_t j = 0; j < b_D_per_mode.size(1); ++j) {
      if (bAcc[nout_b[i]][j] < 0) break; 
      sectionExtents[b_modes[nout_b[i]]].push_back(bAcc[nout_b[i]][j]);
    }
  }
  for (size_t i = 0; i < nin_a.size(); ++i) {
    a_modes[nin_a[i]]= b_modes[nin_b[i]]= alphabet[i + nout_a.size() + nout_b.size()];
    for (size_t j = 0; j < b_D_per_mode.size(1); ++j) {
      if (bAcc[nin_b[i]][j] < 0) break; 
      sectionExtents[b_modes[nin_b[i]]].push_back(bAcc[nin_b[i]][j]);
    }
  }

if (yastn_log_level > 3) {
    std::cout << "Computed a_modes: ";
    for (auto m : a_modes) {
        // std::cout << static_cast<char>(m) << ' ';
        std::cout << m << " ";
    }
    std::cout << std::endl << "sectionExtents a_modes: "<< std::endl;
    for (auto m : a_modes) {
        std::cout << m << ": ";
        for (auto e : sectionExtents[m]) {
            std::cout << e << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\nComputed b_modes: ";
    for (auto m : b_modes) {
        std::cout << m << " ";
    }
    std::cout << "sectionExtents b_modes: ";
    for (auto m : b_modes) {
        std::cout << m << ": ";
        for (auto e : sectionExtents[m]) {
            std::cout << e << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\nComputed c_modes: ";
    for (auto m : c_modes) {
        std::cout << m << ' ';
    }
    std::cout << "sectionExtents c_modes: ";
    for (auto m : c_modes) {
        std::cout << m << ": ";
        for (auto e : sectionExtents[m]) {
            std::cout << e << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  // non-zero block coord are expected to be int32_t instead of int64_t
  std::vector<int32_t> a_blocks32(a_blocks.size());
  std::vector<int32_t> b_blocks32(b_blocks.size());
  std::vector<int32_t> c_blocks32(c_blocks.size());
  std::transform(a_blocks.begin(), a_blocks.end(), a_blocks32.begin(),
               [](int64_t x) { return static_cast<int32_t>(x); });
  std::transform(b_blocks.begin(), b_blocks.end(), b_blocks32.begin(),
               [](int64_t x) { return static_cast<int32_t>(x); });
  std::transform(c_blocks.begin(), c_blocks.end(), c_blocks32.begin(),
               [](int64_t x) { return static_cast<int32_t>(x); });

  // Call the CUDA implementation.  
  tensordot_bs_cuda_impl(
    a_ptr,
    a_modes, 
    a_blocks32,
    a_offsets,
    a_strides,
    b_ptr,
    b_modes,
    b_blocks32,
    b_offsets,
    b_strides,
    result_ptr,
    c_modes, 
    c_blocks32,
    c_offsets,
    c_strides,
    sectionExtents
  );

  return result;
}

TORCH_LIBRARY_IMPL(cublocksparse, CUDA, m) {
  m.impl("tensordot_bs", &tensordot_bs_cuda);
}

} // namespace cublocksparse