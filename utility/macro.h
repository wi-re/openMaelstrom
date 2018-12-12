#pragma once
#include <vector_types.h>

extern thread_local int3 h_threadIdx;
extern thread_local int3 h_blockDim;
extern thread_local int3 h_blockIdx;
#include <utility/macro/cuda_defines.h>
#include <utility/macro/for_each.h>
#include <utility/macro/for_each_pair.h>
#include <utility/macro/helper_macros.h>