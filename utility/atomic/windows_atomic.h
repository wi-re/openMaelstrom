#pragma once
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <utility/atomic.h>
#include <Windows.h>
namespace atomic {
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#define WRAP_WINDOWS(name, fun)                                                                    \
  template <typename T, typename std::enable_if_t<sizeof(T) == sizeof(uint32_t)> * = nullptr,      \
            typename... Ts>                                                                        \
  hostOnly T name(volatile__ T *ptr, Ts... args) {                                                 \
    auto val = fun(reinterpret_cast<volatile__ long *>(ptr), *reinterpret_cast<long *>(&args)...); \
    return *reinterpret_cast<T *>(&val);                                                           \
  }                                                                                                \
  template <typename T, typename std::enable_if_t<sizeof(T) == sizeof(int64_t)> * = nullptr,       \
            typename... Ts>                                                                        \
  hostOnly T name(volatile__ T *ptr, Ts... args) {                                                 \
    auto val = fun##64(reinterpret_cast<volatile__ int64_t *>(ptr),                                \
                       *reinterpret_cast<int64_t *>(&args)...);                                    \
    return *reinterpret_cast<T *>(&val);                                                           \
  }

#define WRAP_WINDOWS16(name, fun)                                                                  \
  template <typename T, typename std::enable_if_t<sizeof(T) == sizeof(int16_t)> * = nullptr,       \
            typename... Ts>                                                                        \
  hostOnly T name(volatile__ T *ptr, Ts... args) {                                                 \
    auto val = fun##16(reinterpret_cast<volatile__ int16_t *>(ptr),                                \
                       *reinterpret_cast<int16_t *>(&args)...);                                    \
    return *reinterpret_cast<T *>(&val);                                                           \
  }                                                                                                \
  WRAP_WINDOWS(name, fun)

#define WRAP_WINDOWS8(name, fun)                                                                   \
  template <typename T, typename std::enable_if_t<sizeof(T) == sizeof(char)> * = nullptr,          \
            typename... Ts>                                                                        \
  hostOnly T name(volatile__ T *ptr, Ts... args) {                                                 \
    auto val = fun##8(reinterpret_cast<volatile__ char *>(ptr), *reinterpret_cast<char *>(&args)...); \
    return *reinterpret_cast<T *>(&val);                                                           \
  }                                                                                                \
  WRAP_WINDOWS16(name, fun)

	namespace detail {
		WRAP_WINDOWS16(CAS, InterlockedCompareExchange);
		WRAP_WINDOWS(Add, InterlockedAdd);
		WRAP_WINDOWS16(Inc, InterlockedIncrement);
		WRAP_WINDOWS16(Dec, InterlockedDecrement);
		WRAP_WINDOWS8(Exch, InterlockedExchange);
		WRAP_WINDOWS8(Or, InterlockedOr);
		WRAP_WINDOWS8(And, InterlockedAdd);
		WRAP_WINDOWS8(Xor, InterlockedXor);
	} // namespace detail

	template <typename T> hostOnly T CAS(volatile__ T *ptr, T swap, T comp) {
		return detail::CAS(ptr, comp, swap);
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
		return detail::Add(ptr, addend) - addend;
	}
	template <typename T>
	hostDevice std::enable_if_t<!::is_any<T, int32_t, uint32_t, int64_t>::value, T>
		add(volatile__ T *ptr, T addend) {
		return ::atomic::apply(ptr, [addend](T old) { return old + addend; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<::is_any<T, int32_t, uint32_t, int64_t>::value, T>
		sub(volatile__ T *ptr, T addend) {
		return detail::Add(ptr, -addend) + addend;
	}
	template <typename T>
	hostDevice std::enable_if_t<!::is_any<T, int32_t, uint32_t, int64_t>::value, T>
		sub(volatile__ T *ptr, T addend) {
		return ::atomic::apply(ptr, [addend](T old) { return old - addend; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<::is_any<T, int32_t, uint32_t, int64_t>::value, T>
		exch(volatile__ T *ptr, T swap) {
		return detail::Exch(ptr, swap);
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
		return detail::Inc(ptr) - 1;
	}
	template <typename T>
	hostDevice typename std::enable_if_t<!std::is_integral<T>::value, T> inc(volatile__ T *ptr) {
		return ::atomic::apply(ptr, [](T old) { return ++old; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<std::is_integral<T>::value, T> dec(volatile__ T *ptr) {
		return detail::Dec(ptr) + 1;
	}
	template <typename T>
	hostDevice typename std::enable_if_t<!std::is_integral<T>::value, T> dec(volatile__ T *ptr) {
		return ::atomic::apply(ptr, [](T old) { return --old; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<std::is_integral<T>::value, T> And(volatile__ T *ptr,
		T operand) {
		return detail::And(ptr);
	}
	template <typename T>
	hostDevice typename std::enable_if_t<!std::is_integral<T>::value, T> And(volatile__ T *ptr,
		T addend) {
		return ::atomic::apply(ptr, [addend](T old) { return old & addend; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<std::is_integral<T>::value, T> Or(volatile__ T *ptr,
		T operand) {
		return detail::Or(ptr);
	}
	template <typename T>
	hostDevice typename std::enable_if_t<!std::is_integral<T>::value, T> Or(volatile__ T *ptr,
		T addend) {
		return ::atomic::apply(ptr, [addend](T old) { return old | addend; });
	}

	template <typename T>
	hostDevice typename std::enable_if_t<std::is_integral<T>::value, T> Xor(volatile__ T *ptr,
		T operand) {
		return detail::Xor(ptr);
	}
	template <typename T>
	hostDevice typename std::enable_if_t<!std::is_integral<T>::value, T> Xor(volatile__ T *ptr,
		T addend) {
		return ::atomic::apply(ptr, [addend](T old) { return old ^ addend; });
	}
#endif
}
#endif
