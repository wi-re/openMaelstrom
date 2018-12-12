#pragma once
#include <utility/atomic.h>

template <typename T, typename Enable = void> struct cuda_atomic;

template <typename T>
struct cuda_atomic<
	T, std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>> {
private:
	volatile__ T *_data;

public:
	hostDevice cuda_atomic(volatile__ T *adr) : _data(adr) {}
	hostDevice cuda_atomic(T &val) : _data(&val) {}

	hostDevice T add(const T &arg) { return atomic::add(_data, arg); }
	hostDevice T sub(const T &arg) { return atomic::sub(_data, arg); }
	hostDevice T exch(const T &arg) { return atomic::exch(_data, arg); }
	hostDevice T min(const T &arg) { return atomic::exch(_data, arg); }
	hostDevice T max(const T &arg) { return atomic::exch(_data, arg); }
	hostDevice T inc() { return atomic::inc(_data); }
	hostDevice T dec() { return atomic::dec(_data); }
	hostDevice T CAS(const T &comp, const T &swap) { return atomic::CAS(_data, comp, swap); }
	hostDevice T And(const T &arg) { return atomic::And(_data); }
	hostDevice T Or(const T &arg) { return atomic::Or(_data); }
	hostDevice T Xor(const T &arg) { return atomic::Xor(_data); }

	hostDevice T mul(const T &arg) {
		return atomic::apply(_data, [arg](T old) { return old * arg; });
	}
	hostDevice T div(const T &arg) {
		return atomic::apply(_data, [arg](T old) { return old / arg; });
	}
	hostDevice T negate() {
		return atomic::apply(_data, [](T old) { return -old; });
	}

	hostDevice T val() const { return *_data; }
	explicit hostDevice operator T() const { return *_data; }

	hostDevice cuda_atomic &operator=(const T &val) {
		atomic::exch(_data, val);
		return *this;
	}

	hostDevice T operator++() {
		auto old = atomic::inc(_data);
		return old + 1;
	}
	hostDevice T operator++(int) { return atomic::inc(_data); }
	hostDevice T operator--() {
		auto old = atomic::dec(_data);
		return old - 1;
	}
	hostDevice T operator--(int) { return atomic::dec(_data); }

	hostDevice T operator+(const T &arg) { return val() + arg; }
	hostDevice T operator-(const T &arg) { return val() - arg; }
	hostDevice T operator*(const T &arg) { return val() * arg; }
	hostDevice T operator/(const T &arg) { return val() / arg; }
	hostDevice T operator-() { return -val(); }

	hostDevice T operator+=(const T &arg) { return add(arg) + arg; }
	hostDevice T operator-=(const T &arg) { return sub(arg) - arg; }
	hostDevice T operator*=(const T &arg) { return mul(arg) * arg; }
	hostDevice T operator/=(const T &arg) { return div(arg) / arg; }
};
template <typename T>
hostDevice typename std::enable_if_t<std::is_integral<T>::value, T>
operator|(const cuda_atomic<T> &lhs, const T &arg) {
	return lhs.val() | arg;
}

template <typename T>
hostDevice typename std::enable_if_t<std::is_integral<T>::value, T>
operator&(const cuda_atomic<T> &lhs, const T &arg) {
	return lhs.val() & arg;
}

template <typename T>
hostDevice typename std::enable_if_t<std::is_integral<T>::value, T>
operator^(const cuda_atomic<T> &lhs, const T &arg) {
	return lhs.val() ^ arg;
}

template <typename T>
hostDevice typename std::enable_if_t<std::is_integral<T>::value, T>
operator|=(const cuda_atomic<T> &lhs, const T &arg) {
	return lhs.Or(arg) | arg;
}

template <typename T>
hostDevice typename std::enable_if_t<std::is_integral<T>::value, T>
operator&=(const cuda_atomic<T> &lhs, const T &arg) {
	return lhs.And(arg)& arg;
}

template <typename T>
hostDevice typename std::enable_if_t<std::is_integral<T>::value, T>
operator^=(const cuda_atomic<T> &lhs, const T &arg) {
	return lhs.Xor(arg) ^ arg;
}

template <typename T>
struct cuda_atomic<T, typename std::enable_if_t<math::dimension<T>::value == 1>> {
	cuda_atomic<std::decay_t<decltype(std::declval<T>().x)>> x;

	hostDevice cuda_atomic(volatile__ T *adr) : x(adr->x) {}
	hostDevice cuda_atomic(T &val) : x(&val.x) {}
	explicit hostDevice operator T() const { return T{ x.val() }; }

	hostDevice cuda_atomic &operator=(const T &val) {
		x.exch(val.x);
		return *this;
	}
	hostDevice cuda_atomic &operator=(const std::decay_t<decltype(std::declval<T>().x)> &val) {
		x.exch(val);
		return *this;
	}
};
template <typename T>
struct cuda_atomic<T, typename std::enable_if_t<math::dimension<T>::value == 2>> {
	cuda_atomic<std::decay_t<decltype(std::declval<T>().x)>> x, y;

	hostDevice cuda_atomic(volatile__ T *adr) : x(adr->x), y(adr->y) {}
	hostDevice cuda_atomic(T &val) : x(&val.x), y(&val.y) {}
	explicit hostDevice operator T() const { return T{ x.val(), y.val() }; }

	hostDevice cuda_atomic &operator=(const T &val) {
		x.exch(val.x);
		y.exch(val.y);
		return *this;
	}
	hostDevice cuda_atomic &operator=(const std::decay_t<decltype(std::declval<T>().x)> &val) {
		x.exch(val);
		y.exch(val);
		return *this;
	}
};
template <typename T>
struct cuda_atomic<T, typename std::enable_if_t<math::dimension<T>::value == 3>> {
	cuda_atomic<std::decay_t<decltype(std::declval<T>().x)>> x, y, z;

	hostDevice cuda_atomic(volatile__ T *adr) : x(&adr->x), y(&adr->y), z(&adr->z) {}
	hostDevice cuda_atomic(T &val) : x(&val.x), y(&val.y), z(&val.z) {}
	explicit hostDevice operator T() const { return T{ x.val(), y.val(), z.val() }; }

	hostDevice cuda_atomic &operator=(const T &val) {
		x.exch(val.x);
		y.exch(val.y);
		z.exch(val.z);
		return *this;
	}
	hostDevice cuda_atomic &operator=(const std::decay_t<decltype(std::declval<T>().x)> &val) {
		x.exch(val);
		y.exch(val);
		z.exch(val);
		return *this;
	}
};
template <typename T>
struct cuda_atomic<T, typename std::enable_if_t<math::dimension<T>::value == 4>> {
	cuda_atomic<std::decay_t<decltype(std::declval<T>().x)>> x, y, z, w;

	hostDevice cuda_atomic(volatile__ T *adr) : x(adr->x), y(adr->y), z(adr->z), w(adr->w) {}
	hostDevice cuda_atomic(T &val) : x(&val.x), y(&val.y), z(&val.z), w(&val.w) {}
	explicit hostDevice operator T() const { return T{ x.val(), y.val(), z.val(), w.val() }; }

	hostDevice cuda_atomic &operator=(const T &val) {
		x.exch(val.x);
		y.exch(val.y);
		z.exch(val.z);
		w.exch(val.w);
		return *this;
	}
	hostDevice cuda_atomic &operator=(const std::decay_t<decltype(std::declval<T>().x)> &val) {
		x.exch(val);
		y.exch(val);
		z.exch(val);
		w.exch(val);
		return *this;
	}
};
