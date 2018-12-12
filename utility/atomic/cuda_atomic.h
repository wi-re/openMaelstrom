#pragma once
#include <device_atomic_functions.h>
#include <utility/atomic.h>
namespace atomic {
#if defined(__CUDA_ARCH__)
template <typename T>
deviceOnly typename std::enable_if_t<sizeof(T) == sizeof(int32_t), T> CAS(T *ptr, T comp, T swap) {
  auto val = atomicCAS(reinterpret_cast<int32_t *>(ptr), *reinterpret_cast<int32_t *>(&comp),
                       *reinterpret_cast<int32_t *>(&swap));
  return *reinterpret_cast<T *>(&val);
}
template <typename T>
deviceOnly typename std::enable_if_t<sizeof(T) == sizeof(uint64_t), T> CAS(T *ptr, T comp, T swap) {
  auto val = atomicCAS(reinterpret_cast<uint64_t *>(ptr), *reinterpret_cast<uint64_t *>(&comp),
                       *reinterpret_cast<uint64_t *>(&swap));
  return *reinterpret_cast<T *>(&val);
}

template <typename T, typename Func> deviceOnly T apply(T *ptr, Func fn) {
  T old = *ptr, assumed;
  do {
    assumed = old;
    old = CAS(ptr, assumed, fn(assumed));
  } while (assumed != old);
  return old;
}

template <typename T>
deviceOnly
    typename std::enable_if_t<::is_any<T, int32_t, uint32_t, uint64_t, float, double>::value, T>
    add(T *ptr, T addend) {
  return ::atomicAdd(ptr, addend);
}
template <typename T>
hostDevice
    typename std::enable_if_t<!::is_any<T, int32_t, uint32_t, uint64_t, float, double>::value, T>
    add(volatile__ T *ptr, T addend) {
  return ::atomic::apply(ptr, [addend](T old) { return old + addend; });
}

template <typename T>
deviceOnly typename std::enable_if_t<::is_any<T, int32_t, uint32_t>::value, T> sub(T *ptr,
                                                                                   T addend) {
  return ::atomicSub(ptr, addend);
}
template <typename T>
deviceOnly typename std::enable_if_t<!::is_any<T, int32_t, uint32_t>::value, T> sub(T *ptr,
                                                                                    T addend) {
  return ::atomic::apply(ptr, [addend](T old) { return old - addend; });
}

template <typename T>
deviceOnly typename std::enable_if_t<::is_any<T, int32_t, uint32_t, uint64_t, float>::value, T>
exch(T *ptr, T swap) {
  return ::atomicExch(ptr, swap);
}
template <typename T>
deviceOnly typename std::enable_if_t<!::is_any<T, int32_t, uint32_t, uint64_t, float>::value, T>
exch(T *ptr, T swap) {
  return ::atomic::apply(ptr, [swap](T old) { return swap; });
}

template <typename T>
deviceOnly typename std::enable_if_t<::is_any<T, int32_t, uint32_t, uint64_t>::value, T>
min(T *ptr, T operand) {
  return ::atomicMin(ptr, operand);
}
template <typename T>
deviceOnly typename std::enable_if_t<!::is_any<T, int32_t, uint32_t, uint64_t>::value, T>
min(T *ptr, T operand) {
  return apply(ptr, [operand](T old) { return old < operand ? old : operand; });
}

template <typename T>
deviceOnly typename std::enable_if_t<::is_any<T, int32_t, uint32_t, uint64_t>::value, T>
max(T *ptr, T operand) {
  return ::atomicMax(ptr, operand);
}
template <typename T>
deviceOnly typename std::enable_if_t<!::is_any<T, int32_t, uint32_t, uint64_t>::value, T>
max(T *ptr, T operand) {
  return apply(ptr, [operand](T old) { return old > operand ? old : operand; });
}

template <typename T>
deviceOnly typename std::enable_if_t<::is_any<T, int32_t>::value, T> inc(T *ptr) {
  return add(ptr, T(1));
}
template <typename T>
deviceOnly typename std::enable_if_t<!::is_any<T, int32_t>::value, T> inc(T *ptr) {
  return ::atomic::apply(ptr, [](T old) { return ++old; });
}

template <typename T>
deviceOnly typename std::enable_if_t<::is_any<T, int32_t>::value, T> dec(T *ptr) {
  return sub(ptr, T(1));
}
template <typename T>
deviceOnly typename std::enable_if_t<!::is_any<T, int32_t>::value, T> dec(T *ptr) {
  return ::atomic::apply(ptr, [](T old) { return --old; });
}

template <typename T>
deviceOnly typename std::enable_if_t<::is_any<T, int32_t, uint32_t, uint64_t>::value, T>
And(T *ptr, T operand) {
  return atomicAnd(ptr, operand);
}
template <typename T>
deviceOnly typename std::enable_if_t<!::is_any<T, int32_t, uint32_t, uint64_t>::value, T>
And(T *ptr, T operand) {
  return ::atomic::apply(ptr, [operand](T old) { return old & addend; });
}

template <typename T>
deviceOnly typename std::enable_if_t<::is_any<T, int32_t, uint32_t, uint64_t>::value, T>
Or(T *ptr, T operand) {
  return atomicOr(ptr, operand);
}
template <typename T>
deviceOnly typename std::enable_if_t<!::is_any<T, int32_t, uint32_t, uint64_t>::value, T>
Or(T *ptr, T operand) {
  return ::atomic::apply(ptr, [operand](T old) { return old | addend; });
}

template <typename T>
deviceOnly typename std::enable_if_t<::is_any<T, int32_t, uint32_t, uint64_t>::value, T>
Xor(T *ptr, T operand) {
  return atomicXor(ptr, operand);
}
template <typename T>
deviceOnly typename std::enable_if_t<!::is_any<T, int32_t, uint32_t, uint64_t>::value, T>
Xor(T *ptr, T operand) {
  return ::atomic::apply(ptr, [operand](T old) { return old ^ addend; });
}
#endif
} // namespace atomic
