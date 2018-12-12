#pragma once
#ifdef _MSC_VER
#pragma warning( push )  
#pragma warning( disable : 4244 )  
#endif
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <utility/identifier/uniform.h>
#include <utility/math.h>
#include <utility/unit_math.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h> 
#include <tbb/parallel_scan.h>
#include <thrust/system/cuda/execution_policy.h>
#ifdef _MSC_VER
#pragma warning( pop )  
#endif 
#include <utility/cuda.h>

namespace algorithm {
	template <typename T> using elem_t = const typename vec<T>::base_type &;
	template <typename T> using base_t = typename vec<T>::base_type;
}

#include <utility/algorithm/reduce.h>
#include <utility/algorithm/scan.h>
#include <utility/algorithm/sort.h>
#include <utility/algorithm/misc.h>
