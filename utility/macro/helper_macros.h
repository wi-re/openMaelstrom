#pragma once
#include <utility/macro.h>
#include <utility/macro/cuda_defines.h>

#define RECURSIVE_CACHE_ARRAYS_FUNCTION(i, dummy, prior, x ) auto GET0 x = cache_array(arrays.GET1 x, GET0 prior .offset);
#define ROOT_CACHE_ARRAYS_FUNCTION(i, dummy, x ) auto GET0 x = cache_array(arrays.GET1 x);
#if defined(_MSC_VER) || !defined(__CUDA_ARCH__)
#define cache_arrays(...) EXPAND(FOR_EACH_SEQUENCE(ROOT_CACHE_ARRAYS_FUNCTION, RECURSIVE_CACHE_ARRAYS_FUNCTION,, __VA_ARGS__))
#else
#define cache_arrays(...) FOR_EACH_SEQUENCE(ROOT_CACHE_ARRAYS_FUNCTION, RECURSIVE_CACHE_ARRAYS_FUNCTION,, __VA_ARGS__)
#endif
#define RECURSIVE_ALIAS_ARRAYS_FUNCTION(i, dummy, prior, x ) auto GET0 x = arrays.GET1 x;
#define ROOT_ALIAS_ARRAYS_FUNCTION(i, dummy, x ) auto GET0 x = arrays.GET1 x;
#if defined(_MSC_VER) || !defined(__CUDA_ARCH__)
#define alias_arrays(...) EXPAND(FOR_EACH_SEQUENCE(ROOT_ALIAS_ARRAYS_FUNCTION, RECURSIVE_ALIAS_ARRAYS_FUNCTION,, __VA_ARGS__))
#else
#define alias_arrays(...) FOR_EACH_SEQUENCE(ROOT_ALIAS_ARRAYS_FUNCTION, RECURSIVE_ALIAS_ARRAYS_FUNCTION,, __VA_ARGS__)
#endif