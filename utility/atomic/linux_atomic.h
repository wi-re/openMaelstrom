#pragma once
#ifndef _WIN32
#include <utility/atomic.h>
#include <utility/macro.h>

namespace atomic {
#if !defined(__CUDA_ARCH__)

	template <typename T> hostOnly T CAS(volatile__ T *ptr, T swap, T comp) {
		union i_t{ T t; int32_t i;};
		i_t sw, co;
		sw.t = swap;
		co.t = comp;
		return __sync_val_compare_and_swap(reinterpret_cast<int32_t*>(ptr), co.i, sw.i);
	}
	template <typename T, typename Func> hostOnly T apply(volatile__ T *ptr, Func fn) {
		T old = *ptr, assumed;
		do {
			assumed = old;
			old = CAS(ptr, assumed, fn(assumed));
		} while (assumed != old);
		return old;
	}

	template <typename T>
	hostDevice typename std::enable_if_t<::is_any<T, int32_t, uint32_t, int64_t>::value, T>
		add(volatile__ T *ptr, T addend) {
		return __sync_fetch_and_add(ptr, addend);
	}
	template <typename T>
	hostDevice std::enable_if_t<!::is_any<T, int32_t, uint32_t, int64_t>::value, T>
		add(volatile__ T *ptr, T addend) {
		return ::atomic::apply(ptr, [addend](T old) { return old + addend; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<::is_any<T, int32_t, uint32_t, int64_t>::value, T>
		sub(volatile__ T *ptr, T addend) {
		return __sync_fetch_and_sub(ptr, addend);
	}
	template <typename T>
	hostDevice std::enable_if_t<!::is_any<T, int32_t, uint32_t, int64_t>::value, T>
		sub(volatile__ T *ptr, T addend) {
		return ::atomic::apply(ptr, [addend](T old) { return old - addend; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<::is_any<T, int32_t, uint32_t, int64_t>::value, T>
		exch(volatile__ T *ptr, T swap) {
		return __atomic_exchange_n(ptr, swap, __ATOMIC_ACQ_REL);
	}
	template <typename T>
	hostDevice typename std::enable_if_t<!std::is_integral<T>::value, T> exch(volatile__ T *ptr,
		T addend) {
		return ::atomic::apply(ptr, [addend](T old) { return addend; });
	}

	template <typename T> hostDevice T min(volatile__ T *ptr, T operand) {
		return apply(ptr, [operand](T old) { return old < operand ? old : operand; });
	}

	template <typename T> hostDevice T max(volatile__ T *ptr, T operand) {
		return apply(ptr, [operand](T old) { return old > operand ? old : operand; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<std::is_integral<T>::value, T> inc(volatile__ T *ptr) {
		return __sync_fetch_and_add(ptr, 1);
	}
	template <typename T>
	hostDevice typename std::enable_if_t<!std::is_integral<T>::value, T> inc(volatile__ T *ptr) {
		return ::atomic::apply(ptr, [](T old) { return ++old; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<std::is_integral<T>::value, T> dec(volatile__ T *ptr) {
		return __sync_fetch_and_sub(ptr, 1);
	}
	template <typename T>
	hostDevice typename std::enable_if_t<!std::is_integral<T>::value, T> dec(volatile__ T *ptr) {
		return ::atomic::apply(ptr, [](T old) { return --old; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<std::is_integral<T>::value, T> And(volatile__ T *ptr,
		T operand) {
		return _sync_fetch_and_and(ptr, operand);
	}
	template <typename T>
	hostDevice typename std::enable_if_t<!std::is_integral<T>::value, T> And(volatile__ T *ptr,
		T addend) {
		return ::atomic::apply(ptr, [addend](T old) { return old & addend; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<std::is_integral<T>::value, T> Or(volatile__ T *ptr,
		T operand) {
		return __sync_fetch_and_or(ptr, operand);
	}
	template <typename T>
	hostDevice typename std::enable_if_t<!std::is_integral<T>::value, T> Or(volatile__ T *ptr,
		T addend) {
		return ::atomic::apply(ptr, [addend](T old) { return old | addend; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<std::is_integral<T>::value, T> Xor(volatile__ T *ptr,
		T operand) {
		return __sync_fetch_and_xor(ptr, operand);
	}
	template <typename T>
	hostDevice typename std::enable_if_t<!std::is_integral<T>::value, T> Xor(volatile__ T *ptr,
		T addend) {
		return ::atomic::apply(ptr, [addend](T old) { return old ^ addend; });
	}
#endif
}
#endif
