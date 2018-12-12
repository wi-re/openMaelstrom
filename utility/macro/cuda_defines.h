#pragma once
#include <utility/macro.h>
#include <utility/macro/for_each.h>
#define EXPAND( x ) x
#define REMOVE_PARENS(...) __VA_ARGS__

#if defined(_MSC_VER)
#   define GET_ARG_COUNT(...)  INTERNAL_EXPAND_ARGS_PRIVATE(INTERNAL_ARGS_AUGMENTER(__VA_ARGS__))

#   define INTERNAL_ARGS_AUGMENTER(...) unused, __VA_ARGS__
#   define INTERNAL_EXPAND(x) x
#   define INTERNAL_EXPAND_ARGS_PRIVATE(...) INTERNAL_EXPAND(INTERNAL_GET_ARG_COUNT_PRIVATE(__VA_ARGS__, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#   define INTERNAL_GET_ARG_COUNT_PRIVATE(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, _10_, _11_, _12_, _13_, _14_, _15_, _16_, _17_, _18_, _19_, _20_, _21_, _22_, _23_, _24_, _25_, _26_, _27_, _28_, _29_, _30_, _31_, _32_, _33_, _34_, _35_, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, count, ...) count
#else 
#   define GET_ARG_COUNT(...) INTERNAL_GET_ARG_COUNT_PRIVATE(0, ## __VA_ARGS__, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#   define INTERNAL_GET_ARG_COUNT_PRIVATE(_0, _1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, _10_, _11_, _12_, _13_, _14_, _15_, _16_, _17_, _18_, _19_, _20_, _21_, _22_, _23_, _24_, _25_, _26_, _27_, _28_, _29_, _30_, _31_, _32_, _33_, _34_, _35_, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, count, ...) count
#endif

#define PPCAT_NX(A, B) A ## B
#define PPCAT(A, B) PPCAT_NX(A, B)

#if defined(_MSC_VER) || !defined(__CUDA_ARCH__)
#define FOR_EACH(action, data, ...) EXPAND(PPCAT(FE_, GET_ARG_COUNT(__VA_ARGS__))(action, data, __VA_ARGS__))
#else
#define FOR_EACH(action, data, ...) PPCAT(FE_, GET_ARG_COUNT(__VA_ARGS__))(action, data, __VA_ARGS__)
#endif

#if defined(_WIN32)
#define FOR_EACH_SEQUENCE_I(root_action, action, data, i, X, ...) EXPAND(root_action(i, data, X)) EXPAND(PPCAT(FAP_, GET_ARG_COUNT(__VA_ARGS__))(action, data,  X, __VA_ARGS__))
#define FOR_EACH_SEQUENCE(root_action, action, data, ...) EXPAND(FOR_EACH_SEQUENCE_I(root_action,action,data, GET_ARG_COUNT(__VA_ARGS__),__VA_ARGS__))
#elif defined(_MSC_VER) || !defined(__CUDA_ARCH__)
#define FOR_EACH_SEQUENCE_II(root_action, action, data, i, X, ...) EXPAND(root_action(i, data, X)) EXPAND(PPCAT(FAP_, GET_ARG_COUNT(__VA_ARGS__))(action, data,  X, __VA_ARGS__))
#define FOR_EACH_SEQUENCE_I(root_action, action, data, ...) EXPAND(FOR_EACH_SEQUENCE_II(root_action,action,data, GET_ARG_COUNT(__VA_ARGS__),__VA_ARGS__))
#define FOR_EACH_SEQUENCE_A(root_action, action, data, X) root_action(data, ,X)
#define FOR_EACH_SEQUENCE(root_action, action, data, ...) EXPAND(PPCAT(SPECIAL_, GET_ARG_COUNT(__VA_ARGS__))(FOR_EACH_SEQUENCE_A,FOR_EACH_SEQUENCE_I,root_action, action, data, __VA_ARGS__))

#else
#define FOR_EACH_SEQUENCE_II(root_action, action, data, i, X, ...) root_action(i, data, X) PPCAT(FAP_, GET_ARG_COUNT(__VA_ARGS__))(action, data,  X, __VA_ARGS__)
#define FOR_EACH_SEQUENCE_I(root_action, action, data, ...) FOR_EACH_SEQUENCE_II(root_action,action,data, GET_ARG_COUNT(__VA_ARGS__),__VA_ARGS__)
#define FOR_EACH_SEQUENCE_A(root_action, action, data, X) root_action(data, ,X)
#define FOR_EACH_SEQUENCE(root_action, action, data, ...) PPCAT(SPECIAL_, GET_ARG_COUNT(__VA_ARGS__))(FOR_EACH_SEQUENCE_A,FOR_EACH_SEQUENCE_I,root_action, action, data, __VA_ARGS__)
#endif

#define GET0(e0,...) e0
#define GET1(e0,e1,...) e1
#define GET2(e0,e1,e2,...) e2
#define GET3(e0,e1,e2,e3,...) e3
#define GET4(e0,e1,e2,e3,e4,...) e4
#define COMMA ,
#define STRING(x) #x
#define XSTRING(x) STRING(x)



#ifdef __CUDA_ARCH__
#define getThreadIdx_x() threadIdx.x
#define getThreadIdx() blockIdx.x *blockDim.x + threadIdx.x
#define checkedThreadIdx(x) int32_t x = getThreadIdx(); if(x >= threads) return;
#define checkedParticleIdx(x) int32_t x = getThreadIdx(); if(x >= arrays.num_ptcls) return;
#define hostDevice __host__ __device__
#define hostOnly __host__
#define deviceOnly __device__
#define hostDeviceInline __host__ __device__ __inline__
#define deviceInline __device__ __inline__
#define hostInline __host__ __inline__
#define constantDevice __constant__ __device__
#else
#define getThreadIdx_x() h_threadIdx.x
#define getThreadIdx() h_threadIdx.x
#define checkedThreadIdx(x) int32_t x = getThreadIdx();(void)(threads);
#define checkedParticleIdx(x) int32_t x = getThreadIdx();
#define hostDevice __host__ __device__ 
#define hostOnly __host__ 
#define deviceOnly __device__ 
#define hostDeviceInline __host__ __device__  inline
#define deviceInline __device__ inline
#define hostInline __host__ inline
#define constantDevice const
#endif
