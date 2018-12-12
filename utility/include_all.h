#pragma once 
#include <utility/identifier.h>
#include <utility/cuda.h>
#include <vector_types.h>
#include <utility/macro.h>
#ifndef OLD_STYLE
#include <utility/mathv2/kernels.h>
#else
#include <utility/math/kernels.h>
#endif
#include <utility/cuda/error_handling.h>
#include <utility/cuda/cache.h>
#include <utility/launcher.h>
#include <utility/unit_math.h>
#include <utility/math.h>
#include <utility/atomic.h>
#include <utility/algorithm.h>
#include <utility/cuda.h>
#include <type_traits>
#include <utility/template/tuple_for_each.h>
#include <iostream>
#include <utility/helpers/timer.h>
#include <utility/helpers/log.h>
#include <utility/SPH.h>
#include <cuda_runtime.h>
#include <utility/SPH/boundaryFunctions.h>
#include <utility/SPH.h>
#include <utility/iterator.h>
