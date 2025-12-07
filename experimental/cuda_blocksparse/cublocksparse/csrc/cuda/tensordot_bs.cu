#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cutensor.h>

// Optional NVTX3 support
#if __has_include(<nvtx3/nvtx3.hpp>)
    #include <nvtx3/nvtx3.hpp>
    #define CUBLOCKSPARSE_HAS_NVTX 1
#else
    #define CUBLOCKSPARSE_HAS_NVTX 0
#endif
// NVTX helper macro - no-op when NVTX is not available
#if CUBLOCKSPARSE_HAS_NVTX
    #define NVTX_MARK(msg) nvtx3::mark(msg)
#else
    #define NVTX_MARK(msg) ((void)0)
#endif

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


// ============================================================================
// Plan Cache Implementation
// ============================================================================

struct CachedPlan {
    cutensorHandle_t handle { nullptr };
    cutensorPlan_t plan { nullptr };
    cutensorOperationDescriptor_t opDesc { nullptr };
    cutensorBlockSparseTensorDescriptor_t descA { nullptr };
    cutensorBlockSparseTensorDescriptor_t descB { nullptr };
    cutensorBlockSparseTensorDescriptor_t descC { nullptr };
    uint64_t workspaceSize { 0 };
    size_t hitCount { 0 };
    
    ~CachedPlan() {
        if (plan) cutensorDestroyPlan(plan);
        if (opDesc) cutensorDestroyOperationDescriptor(opDesc);
        if (descA) cutensorDestroyBlockSparseTensorDescriptor(descA);
        if (descB) cutensorDestroyBlockSparseTensorDescriptor(descB);
        if (descC) cutensorDestroyBlockSparseTensorDescriptor(descC);
        if (handle) cutensorDestroy(handle);
    }
    
    // Non-copyable, movable
    CachedPlan() = default;
    CachedPlan(const CachedPlan&) = delete;
    CachedPlan& operator=(const CachedPlan&) = delete;
    CachedPlan(CachedPlan&& other) noexcept {
        handle = other.handle; other.handle = nullptr;
        plan = other.plan; other.plan = nullptr;
        opDesc = other.opDesc; other.opDesc = nullptr;
        descA = other.descA; other.descA = nullptr;
        descB = other.descB; other.descB = nullptr;
        descC = other.descC; other.descC = nullptr;
        workspaceSize = other.workspaceSize;
        hitCount = other.hitCount;
    }
    CachedPlan& operator=(CachedPlan&& other) noexcept {
        if (this != &other) {
            this->~CachedPlan();
            new (this) CachedPlan(std::move(other));
        }
        return *this;
    }
};

class PlanCache {
public:
    static PlanCache& instance() {
        static PlanCache cache;
        return cache;
    }
    
    // Generate a unique key from contraction parameters
    static std::string make_key(
        const std::vector<ModeType>& modeA,
        const std::vector<ModeType>& nonZeroCoordinatesA,
        const std::vector<StrideType>& stridesA,
        const std::vector<ModeType>& modeB,
        const std::vector<ModeType>& nonZeroCoordinatesB,
        const std::vector<StrideType>& stridesB,
        const std::vector<ModeType>& modeC,
        const std::vector<ModeType>& nonZeroCoordinatesC,
        const std::vector<StrideType>& stridesC,
        const std::unordered_map<ModeType, std::vector<ExtentType>>& sectionExtents,
        cudaDataType_t dataType
    ) {
        std::ostringstream oss;
        oss << "dtype=" << dataType << ";";
        
        auto appendVec = [&oss](const char* name, const auto& vec) {
            oss << name << "=[";
            for (size_t i = 0; i < vec.size(); ++i) {
                if (i > 0) oss << ",";
                oss << vec[i];
            }
            oss << "];";
        };
        
        appendVec("mA", modeA);
        appendVec("nzA", nonZeroCoordinatesA);
        appendVec("sA", stridesA);
        appendVec("mB", modeB);
        appendVec("nzB", nonZeroCoordinatesB);
        appendVec("sB", stridesB);
        appendVec("mC", modeC);
        appendVec("nzC", nonZeroCoordinatesC);
        appendVec("sC", stridesC);
        
        oss << "ext={";
        for (const auto& [mode, extents] : sectionExtents) {
            oss << mode << ":[";
            for (size_t i = 0; i < extents.size(); ++i) {
                if (i > 0) oss << ",";
                oss << extents[i];
            }
            oss << "],";
        }
        oss << "}";
        
        return oss.str();
    }
    
    CachedPlan* get(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            it->second->hitCount++;
            return it->second.get();
        }
        return nullptr;
    }
    
    void insert(const std::string& key, std::unique_ptr<CachedPlan> plan) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_[key] = std::move(plan);
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }
    
    std::vector<std::pair<std::string, size_t>> get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<std::string, size_t>> stats;
        stats.reserve(cache_.size());
        for (const auto& [key, plan] : cache_) {
            stats.emplace_back(key, plan->hitCount);
        }
        return stats;
    }
    
    std::vector<std::string> get_keys() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> keys;
        keys.reserve(cache_.size());
        for (const auto& [key, _] : cache_) {
            keys.push_back(key);
        }
        return keys;
    }

private:
    PlanCache() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<CachedPlan>> cache_;
};

// Global accessor functions for Python bindings
void clear_plan_cache() {
    PlanCache::instance().clear();
}

size_t plan_cache_size() {
    return PlanCache::instance().size();
}

std::vector<std::string> plan_cache_keys() {
    return PlanCache::instance().get_keys();
}

std::vector<std::pair<std::string, size_t>> plan_cache_stats() {
    return PlanCache::instance().get_stats();
}

// ============================================================================
// Tensor initialization helper (extracted for reuse)
// ============================================================================

template <typename scalar_t>
void initBlockSparseTensorDescriptor(
    cutensorHandle_t handle,
    const std::vector<ModeType>& modes,
    const std::vector<ModeType>& nonZeroCoordinates,
    const std::vector<StrideType>& strides,
    const std::unordered_map<ModeType, std::vector<ExtentType>>& sectionExtents,
    cudaDataType_t dataType,
    cutensorBlockSparseTensorDescriptor_t& desc
    // Guard<cutensorBlockSparseTensorDescriptor_t> &guard,
    // std::vector<void*> &dev
) {
    uint32_t numModes = modes.size();
    uint64_t numNonZeroBlocks = nonZeroCoordinates.size() / numModes;
    std::vector<uint32_t> numSections;
    std::vector<ExtentType> extents;
    
    for (ModeType mode : modes) {
        const std::vector<ExtentType>& modeExtents = sectionExtents.at(mode);
        numSections.push_back(modeExtents.size());
        extents.insert(extents.end(), modeExtents.begin(), modeExtents.end());
    }
    
    HANDLE_ERROR(cutensorCreateBlockSparseTensorDescriptor(
        handle, &desc,
        numModes, numNonZeroBlocks, numSections.data(), extents.data(),
        nonZeroCoordinates.data(), strides.data(), dataType
    ));
    // guard.p = desc;
    // guard.destroy = &cutensorDestroyBlockSparseTensorDescriptor;
}

template <typename scalar_t>
int tensordot_bs_cuda_impl(
    const scalar_t* a_ptr,
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
    const scalar_t* b_ptr,
    const std::vector<ModeType> &modeB,
    const std::vector<ModeType> nonZeroCoordinatesB,
    const std::vector<StrideType> &offsetsB,
    const std::vector<StrideType> &stridesB,
    scalar_t* c_ptr,
    const std::vector<ModeType> &modeC,
    const std::vector<ModeType> nonZeroCoordinatesC,
    const std::vector<StrideType> &offsetsC,
    const std::vector<StrideType> &stridesC,
    const std::unordered_map<ModeType, std::vector<ExtentType>> sectionExtents,
    const cudaDataType_t dataType
)
try
{
    const char* env = std::getenv("YASTN_LOG_LEVEL");
    int yastn_log_level = (env) ? std::atoi(env) : 0;
    env = std::getenv("YASTN_PROFILE");
    int yastn_profile = (env) ? std::atoi(env) : 0;

    // Generate cache key
    std::string cacheKey = PlanCache::make_key(
        modeA, nonZeroCoordinatesA, stridesA,
        modeB, nonZeroCoordinatesB, stridesB,
        modeC, nonZeroCoordinatesC, stridesC,
        sectionExtents, dataType
    );
    
    PlanCache& cache = PlanCache::instance();
    CachedPlan* cachedPlan = cache.get(cacheKey);
    
    // NVTX marker for plan cache hit/miss
    if (yastn_profile) NVTX_MARK( (cachedPlan) ? "cublocksparse::PlanCache HIT" : "cublocksparse::PlanCache MISS" );

    // Prepare device pointers for blocks
    uint32_t numModesA = modeA.size();
    uint32_t numModesB = modeB.size();
    uint32_t numModesC = modeC.size();
    uint64_t numBlocksA = nonZeroCoordinatesA.size() / numModesA;
    uint64_t numBlocksB = nonZeroCoordinatesB.size() / numModesB;
    uint64_t numBlocksC = nonZeroCoordinatesC.size() / numModesC;
    
    std::vector<void*> devA(numBlocksA);
    std::vector<void*> devB(numBlocksB);
    std::vector<void*> devC(numBlocksC);
    
    for (uint64_t i = 0; i < numBlocksA; ++i) {
        devA[i] = (void*)(a_ptr + offsetsA[i]);
    }
    for (uint64_t i = 0; i < numBlocksB; ++i) {
        devB[i] = (void*)(b_ptr + offsetsB[i]);
    }
    for (uint64_t i = 0; i < numBlocksC; ++i) {
        devC[i] = (void*)(c_ptr + offsetsC[i]);
    }

    auto bufA= a_ptr;
    auto bufB= b_ptr;
    auto bufC= c_ptr;

    cutensorHandle_t handle;
    cutensorPlan_t plan;
    uint64_t workspaceSize;

    // Initialise the library.
    // cutensorHandle_t handle;
    // HANDLE_ERROR(cutensorCreate(&handle));
    // Guard<cutensorHandle_t> guardHandle { handle, &cutensorDestroy };

    //////////////////////////////////////
    //                                  //
    // We compute C_{modeC} = A_{modeA}B_{modeB}   //
    //////////////////////////////////////
                          
    if (cachedPlan) {
        // Cache hit - reuse existing plan
        if (yastn_log_level > 5) {
            std::cout << "cublocksparse::PlanCache HIT (hits: " << cachedPlan->hitCount << ")" << std::endl;
        }
        handle = cachedPlan->handle;
        plan = cachedPlan->plan;
        workspaceSize = cachedPlan->workspaceSize;
    } else {
        // Cache miss - create new plan
        if (yastn_log_level > 5) {
            std::cout << "cublocksparse::PlanCache MISS - creating new plan" << std::endl;
        }
        
        auto newPlan = std::make_unique<CachedPlan>();
        
        HANDLE_ERROR(cutensorCreate(&newPlan->handle));
        handle = newPlan->handle;
        
        // Create tensor descriptors
        initBlockSparseTensorDescriptor<scalar_t>(
            handle, modeA, nonZeroCoordinatesA, stridesA,
            sectionExtents, dataType, newPlan->descA
        );
        initBlockSparseTensorDescriptor<scalar_t>(
            handle, modeB, nonZeroCoordinatesB, stridesB,
            sectionExtents, dataType, newPlan->descB
        );
        initBlockSparseTensorDescriptor<scalar_t>(
            handle, modeC, nonZeroCoordinatesC, stridesC,
            sectionExtents, dataType, newPlan->descC
        );
        
        // Create contraction descriptor
        HANDLE_ERROR(cutensorCreateBlockSparseContraction(
            handle, &newPlan->opDesc,
            newPlan->descA, modeA.data(), CUTENSOR_OP_IDENTITY,
            newPlan->descB, modeB.data(), CUTENSOR_OP_IDENTITY,
            newPlan->descC, modeC.data(), CUTENSOR_OP_IDENTITY,
            newPlan->descC, modeC.data(),
            CUTENSOR_COMPUTE_DESC_64F
        ));
        
        // Create plan preference (using default settings here)
        cutensorPlanPreference_t planPref = nullptr;
        // const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
        // const cutensorJitMode_t jitMode = CUTENSOR_JIT_MODE_NONE;
        // HANDLE_ERROR(cutensorCreatePlanPreference(handle,&planPref,algo,jitMode));
        // Guard<cutensorPlanPreference_t> guardPlanPref { planPref, &cutensorDestroyPlanPreference };
        
        // Query workspace
        const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
        HANDLE_ERROR(cutensorEstimateWorkspaceSize(
            handle, newPlan->opDesc, planPref, workspacePref, &newPlan->workspaceSize
        ));
        workspaceSize = newPlan->workspaceSize;

        // Create plan
        HANDLE_ERROR(cutensorCreatePlan(
            handle, &newPlan->plan, newPlan->opDesc, planPref, workspaceSize
        ));
        plan = newPlan->plan;
        
        cache.insert(cacheKey, std::move(newPlan));
    }

    /*******************************
     * Block-sparse Contraction.   *
     *******************************/

    cuda_ptr<char> work = cuda_alloc<char>(workspaceSize);
    if ( uintptr_t(work.get()) % 128 != 0 ) throw std::bad_alloc {};

    // Execute
    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
    struct StreamGuard { cudaStream_t stream; ~StreamGuard() { cudaStreamDestroy(stream); } };
    StreamGuard guardStream { stream };

    if (yastn_profile) NVTX_MARK( "cublocksparse::cutensorBlockSparseContract" );
    scalar_t alpha = 1., beta = 0.;
    HANDLE_ERROR(cutensorBlockSparseContract(handle, plan,
                (void*) &alpha, (const void *const *) devA.data(), (const void *const *) devB.data(),
                (void*) &beta,  (const void *const *) devC.data(), (      void *const *) devC.data(), 
                (void*) work.get(), workspaceSize, stream));

    // HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
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
  int yastn_log_level = (env) ? std::atoi(env) : 0;
  env = std::getenv("YASTN_PROFILE");
  int yastn_profile = (env) ? std::atoi(env) : 0;
//   if (yastn_log_level>0) { std::cout << "YASTN_LOG_LEVEL = " << yastn_log_level << std::endl; }

  TORCH_CHECK(a.dim() == 1, "Input 'a' must be 1D.");
  TORCH_CHECK(b.dim() == 1, "Input 'b' must be 1D.");
  // inputs must all have the same dtype
  TORCH_CHECK( a.dtype() == b.dtype(), "Inputs must have the same dtype" )
  cudaDataType_t dtype;
  if (a.dtype() == c10::ScalarType::Double) 
      dtype= CUDA_R_64F;
  else if (a.dtype() == c10::ScalarType::ComplexDouble)
      dtype= CUDA_C_64F;  
  else if (a.dtype() == c10::ScalarType::Float)
      dtype= CUDA_R_32F;
  else if (a.dtype() == c10::ScalarType::ComplexFloat)
      dtype= CUDA_C_32F;
  else
    throw std::runtime_error { "Unsupported dtype." };
  TORCH_CHECK(a_D_per_mode.dtype() == at::kLong, "Input 'a_D_per_mode' must be int64 valued.");
  TORCH_CHECK(b_D_per_mode.dtype() == at::kLong, "Input 'b_D_per_mode' must be int64 valued.");
  TORCH_CHECK(c_D_per_mode.dtype() == at::kLong, "Input 'c_D_per_mode' must be int64 valued.");
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

  at::Tensor result = torch::zeros(c_size, a.options());
  
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
    // modes of a: 0 1 2 3 ...
    //     nout_a  specifies outgoing indices of a AND in what order they should appear in c as
    // modes of c: nout_a[0], nout_a[1], ...
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

if (yastn_log_level > 5) {
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
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
    a.scalar_type(), "tensordot_bs_cuda_impl", [&] {
        const scalar_t* a_ptr = a.data_ptr<scalar_t>();
        const scalar_t* b_ptr = b.data_ptr<scalar_t>();
        scalar_t* result_ptr = result.data_ptr<scalar_t>();
    
        tensordot_bs_cuda_impl<scalar_t>(
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
            sectionExtents,
            dtype
    );
  });

  return result;
}

TORCH_LIBRARY_IMPL(cublocksparse, CUDA, m) {
  m.impl("tensordot_bs", &tensordot_bs_cuda);
}

} // namespace cublocksparse