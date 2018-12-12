#pragma once 
#include <utility/unitmath/SI_Unit.h>
#include <utility/unitmath/operators.h>
namespace math {
	template<typename T, typename U>
	constexpr hostDeviceInline auto floorf(value_unit<T, U> arg) {
		return value_unit<T, U>{math::floorf(arg.val)};
	}
	template<typename T, typename U>
	hostDeviceInline auto ceilf(value_unit<T, U> arg) {
		return value_unit<T, U>{math::ceilf(arg.val)};
	}
	template<typename T, typename U>
	hostDeviceInline auto floor(value_unit<T, U> arg) {
		return value_unit<T, U>{math::floor(arg.val)};
	}
	template<typename T, typename U>
	hostDeviceInline auto ceil(value_unit<T, U> arg) {
		return value_unit<T, U>{math::ceil(arg.val)};
	}
	template<typename T, typename U>
	hostDeviceInline auto log2(value_unit<T, U> arg) {
		return value_unit<T, void>{math::log2(arg.val)};
	}
/* tanf wrapper*/
	template<typename T, typename U>
	hostDeviceInline auto atan(value_unit<T, U> arg) {
  return value_unit<T, void>{ math::atan(arg.val) };
}
/* cosf wrapper*/
	template<typename T, typename U>
	hostDeviceInline auto cosf(value_unit<T, U> arg) {
  return value_unit<T, void>{ math::cosf(arg.val) };
}
/* sinf wrapper*/
	template<typename T, typename U>
	hostDeviceInline auto sinf(value_unit<T, U> arg) {
  return value_unit<T, void>{ math::sinf(arg.val) };
}
	template<typename T, typename U>
	hostDeviceInline auto expf(value_unit<T, U> arg) {
  return value_unit<T, void>{ math::expf(arg.val) };
}
	template<typename T1, typename U1, typename T2, typename U2, typename = std::enable_if_t<SI::is_compatible<U1, U2>::value>>
	hostDeviceInline auto max(value_unit<T1, U1> lhs, value_unit<T2, U2> rhs) {
		return value_unit<T1, U1>{ math::max(lhs.val, rhs.val) };
	}
	template<typename T1, typename U1, typename T2>
	hostDeviceInline auto max(value_unit<T1, U1> lhs, T2 rhs) {
		return value_unit<T1, U1>{ math::max(lhs.val, rhs) };
	}
	template<typename T1, typename U1, typename T2>
	hostDeviceInline auto max(T2 rhs, value_unit<T1, U1> lhs) {
		return value_unit<T1, U1>{ math::max(lhs.val, rhs) };
	}
	template<typename T1, typename U1, typename T2, typename U2, typename = std::enable_if_t<SI::is_compatible<U1, U2>::value>>
	hostDeviceInline auto min(value_unit<T1, U1> lhs, value_unit<T2, U2> rhs) {
		return value_unit<T1, U1>{ math::min(lhs.val, rhs.val) };
	}
	template<typename T1, typename U1, typename T2>
	hostDeviceInline auto min(value_unit<T1, U1> lhs, T2 rhs) {
		return value_unit<T1, U1>{ math::min(lhs.val, rhs) };
	}
	template<typename T1, typename U1, typename T2>
	hostDeviceInline auto min(T2 rhs, value_unit<T1, U1> lhs) {
		return value_unit<T1, U1>{ math::min(lhs.val, rhs) };
	}
	template<typename T, typename U>
	hostDeviceInline auto sqrt(value_unit<T, U> lhs) {
		return value_unit< T, typename SI::multiply_ratio<U, ratio<1, 2>>>{math::sqrt(lhs.val)};
	}
	template<typename T, typename U>
	hostDeviceInline auto abs(value_unit<T, U> lhs) {
		return value_unit < T, U>{math::abs(lhs.val)};
	}
	template<typename T1, typename U1, typename T2, typename T3>
	hostDeviceInline auto lerp(value_unit<T1, U1> lhs, T2 rhs, T3 t) {
		return value_unit<T1, U1>{ math::lerp(lhs.val, rhs, t)};
	}
	template<typename T1, typename U1, typename T2, typename T3>
	hostDeviceInline auto clamp(value_unit<T1, U1> lhs, T2 rhs, T3 t) {
		return value_unit<T1, U1>{ math::clamp(lhs.val, rhs, t)};
	}
	template<typename T1, typename U1>
	hostDeviceInline auto clamp(value_unit<T1, U1> lhs, value_unit<T1, U1> rhs, value_unit<T1, U1> t) {
		return value_unit<T1, U1>{ math::clamp(lhs.val, rhs.val, t.val)};
	}
	template<typename T, typename U>
	hostDeviceInline auto sign(value_unit<T, U> lhs) {
		return value_unit<T, void>{math::sign(lhs.val)};
	}
	template<typename T, typename U>
	hostDeviceInline auto max_elem(value_unit<T, U> lhs) {
		return value_unit<T, U>{ math::max_elem(lhs.val) };
	}
	template<typename T, typename U>
	hostDeviceInline auto min_elem(value_unit<T, U> lhs) {
		return value_unit<T, U>{ math::min_elem(lhs.val) };
	}
	template<typename T1, typename U1, typename T2, typename U2>
	hostDeviceInline auto dot(value_unit<T1, U1> lhs, value_unit<T2, U2> rhs) {
		return value_unit<std::decay_t<decltype(math::dot(lhs.val, rhs.val))>, SI::add_unit<U1, U2>>{ math::dot(lhs.val, rhs.val) };
	}
	template<typename T1, typename U1, typename T2, typename U2>
	hostDeviceInline auto dot3(value_unit<T1, U1> lhs, value_unit<T2, U2> rhs) {
		return value_unit<std::decay_t<decltype(math::dot3(lhs.val, rhs.val))>, SI::add_unit<U1, U2>>{ math::dot3(lhs.val, rhs.val) };
	}
	template<typename T, typename U>
	hostDeviceInline auto length(value_unit<T, U> value) {
		return value_unit<std::decay_t<decltype(math::dot(value.val, value.val))>, U>{sqrtf(math::dot(value.val, value.val))};
	}
	template<typename T, typename U>
	hostDeviceInline auto length3(value_unit<T, U> value) {
		return sqrt(dot3(value, value));
	}	
	template<typename T, typename U>
		hostDeviceInline auto sqlength(value_unit<T, U> value) {
		return (math::dot(value.val, value.val));
	}
	template<typename T, typename U>
	hostDeviceInline auto sqlength3(value_unit<T, U> value) {
		return (dot3(value, value));
	}
	template<typename T, typename U>
	hostDeviceInline auto normalize(value_unit<T, U> value) {
		return value / length(value);
	}
	template<typename T, typename U>
	hostDeviceInline auto normalize3(value_unit<T, U> value) {
		return value / length3(value);
	}
	template<typename T1, typename U1, typename T2, typename U2>
	hostDeviceInline auto cross(value_unit<T1, U1> lhs, value_unit<T2, U2> rhs) {
		return value_unit<std::decay_t<decltype(math::cross(lhs.val, rhs.val))>, SI::add_unit<U1, U2>>{ math::cross(lhs.val, rhs.val) };
	}
	template<typename T1, typename U1, typename T2, typename U2>
	hostDeviceInline auto distance(value_unit<T1, U1> lhs, value_unit<T2, U2> rhs) {
		return length(rhs - lhs);
	}
	template<typename T1, typename U1, typename T2, typename U2>
	hostDeviceInline auto distance3(value_unit<T1, U1> lhs, value_unit<T2, U2> rhs) {
		return length3(rhs - lhs);
	}
	template<typename T1, typename U1, typename T2, typename U2>
	hostDeviceInline auto sqdistance(value_unit<T1, U1> lhs, value_unit<T2, U2> rhs) {
		return sqlength(rhs - lhs);
	}
	template<typename T1, typename U1, typename T2, typename U2>
	hostDeviceInline auto sqdistance3(value_unit<T1, U1> lhs, value_unit<T2, U2> rhs) {
		return sqlength3(rhs - lhs);
	}

	template<typename T>
	hostDeviceInline constexpr auto square(T a) {
		return a * a;
	}
	template<typename T>
	hostDeviceInline constexpr auto cubic(T a) {
		return a * a * a;
	}

	template<int i, typename T>
	hostDeviceInline constexpr auto power(T a) {
		constexpr float r = static_cast<float>(i);
		return  math::pow(a, r);
	}
	template<typename Ratio, typename T>
	hostDeviceInline constexpr auto power(T a) {
		constexpr float n = static_cast<float>(Ratio::num);
		constexpr float d = static_cast<float>(Ratio::den);
		constexpr float r = n / d;
		return  math::pow(a.val, r);
	}

	template<int i, typename T, typename U>
	hostDeviceInline constexpr auto power(value_unit<T, U> a) {
		constexpr float r = static_cast<float>(i);
		using ret_t = value_unit<T, typename SI::multiply_ratio<U, ratio<i, 1>> >;

		return ret_t{ math::pow(a.val, r) };
	}
	template<typename Ratio, typename T, typename U >
	hostDeviceInline constexpr auto power(value_unit<T, U> a) {
		constexpr float n = static_cast<float>(Ratio::num);
		constexpr float d = static_cast<float>(Ratio::den);
		constexpr float r = n / d;
		using ret_t = value_unit<T, typename SI::multiply_ratio<U, Ratio>>;
		return ret_t{ math::pow(a.val, r) };
	}


	template<typename T, typename U, typename Unit, typename = math::DimCheck<T, T, 4>, typename = void, typename = void, typename = void, typename = void, typename = void>
	constexpr hostDeviceInline auto castTo(value_unit<U, Unit> arg) {
		using type = decltype(std::declval<T>().x);
		return value_unit<T, Unit>{ T{ static_cast<type>(math::weak_get<1>(arg.val)),static_cast<type>(math::weak_get<2>(arg.val)),static_cast<type>(math::weak_get<3>(arg.val)),static_cast<type>(math::weak_get<4>(arg.val)) }};
	}
	template<typename T, typename U, typename Unit, typename = math::DimCheck<T, T, 3>, typename = void, typename = void, typename = void, typename = void>
	constexpr hostDeviceInline auto castTo(value_unit<U, Unit> arg) {
		using type = decltype(std::declval<T>().x);
		return value_unit<T, Unit>{ T{ static_cast<type>(math::weak_get<1>(arg.val)),static_cast<type>(math::weak_get<2>(arg.val)),static_cast<type>(math::weak_get<3>(arg.val)) }};
	}
	template<typename T, typename U, typename Unit, typename = math::DimCheck<T, T, 2>, typename = void, typename = void, typename = void>
	constexpr hostDeviceInline auto castTo(value_unit<U, Unit> arg) {
		using type = decltype(std::declval<T>().x);
		return value_unit<T, Unit>{ T{ static_cast<type>(math::weak_get<1>(arg.val)),static_cast<type>(math::weak_get<2>(arg.val)) }};
	}
	template<typename T, typename U, typename Unit, typename = math::DimCheck<T, T, 1>, typename = void>
	constexpr hostDeviceInline auto castTo(value_unit<U, Unit> arg) {
		using type = decltype(std::declval<T>().x);
		return value_unit<T, Unit>{ T{ static_cast<type>(math::weak_get<1>(arg.val)) }};
	}

	template<typename T, typename U>
	using unit_elem_t = value_unit <std::decay_t<decltype(math::weak_get<1>(std::declval<std::decay_t<T>>()))>, U>;

	template<typename T, typename U>
	using elem_t = std::decay_t<decltype(math::weak_get<1>(std::declval<std::decay_t<T>>()))>;

	template<uint32_t idx, typename T, typename U, typename std::enable_if<dimension<T>::value == 0, std::nullptr_t>::type* = nullptr>
	constexpr hostDeviceInline auto unit_get(value_unit<T, U> a) { return unit_elem_t<T, U>{a.val}; }
	template<uint32_t idx, typename T, typename U, greaterDimension<0, T>* = nullptr, std::enable_if_t<idx ==  1>* = nullptr>
	constexpr hostDeviceInline auto unit_get(value_unit<T, U> a) { return unit_elem_t<T, U>{a.val.x}; }
	template<uint32_t idx, typename T, typename U, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, std::enable_if_t<idx ==  2>* = nullptr>
	constexpr hostDeviceInline auto unit_get(value_unit<T, U> a) { return unit_elem_t<T, U>{a.val.y}; }
	template<uint32_t idx, typename T, typename U, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, greaterDimension<2, T>* = nullptr, std::enable_if_t<idx ==  3>* = nullptr>
	constexpr hostDeviceInline auto unit_get(value_unit<T, U> a) { return unit_elem_t<T, U>{a.val.z}; }
	template<uint32_t idx, typename T, typename U, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, greaterDimension<2, T>* = nullptr, greaterDimension<3, T>* = nullptr, std::enable_if_t<idx ==  4>* = nullptr>
	constexpr hostDeviceInline auto unit_get(value_unit<T, U> a) { return unit_elem_t<T, U>{a.val.w}; }

	template<uint32_t idx, typename T, typename std::enable_if<dimension<T>::value == 0, std::nullptr_t>::type* = nullptr>
	constexpr hostDeviceInline auto unit_get(T a) { return a; }
	template<uint32_t idx, typename T, greaterDimension<0, T>* = nullptr, std::enable_if_t<idx ==  1>* = nullptr>
	constexpr hostDeviceInline auto unit_get(T a) { return a.x; }
	template<uint32_t idx, typename T, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, std::enable_if_t<idx ==  2>* = nullptr>
	constexpr hostDeviceInline auto unit_get(T a) { return a.y; }
	template<uint32_t idx, typename T, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, greaterDimension<2, T>* = nullptr, std::enable_if_t<idx ==  3>* = nullptr>
	constexpr hostDeviceInline auto unit_get(T a) { return a.z; }
	template<uint32_t idx, typename T, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, greaterDimension<2, T>* = nullptr, greaterDimension<3, T>* = nullptr, std::enable_if_t<idx ==  4>* = nullptr>
	constexpr hostDeviceInline auto unit_get(T a) { return a.w; }

	template<uint32_t idx, typename T, typename V, typename U1, typename U2, typename  std::enable_if_t<SI::is_compatible<U1, U2>::value>* = nullptr, typename std::enable_if<dimension<T>::value == 0, std::nullptr_t>::type* = nullptr>
	constexpr hostDeviceInline auto unit_assign(value_unit<T, U1>& a, value_unit<V, U2> b) { a.val = b.val; }
	template<uint32_t idx, typename T, typename V, typename U1, typename U2, typename  std::enable_if_t<SI::is_compatible<U1, U2>::value>* = nullptr, greaterDimension<0, T>* = nullptr, std::enable_if_t<idx ==  1>* = nullptr>
	constexpr hostDeviceInline auto unit_assign(value_unit<T, U1>& a, value_unit<V, U2> b) { a.val.x = b.val; }
	template<uint32_t idx, typename T, typename V, typename U1, typename U2, typename  std::enable_if_t<SI::is_compatible<U1, U2>::value>* = nullptr, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, std::enable_if_t<idx ==  2>* = nullptr>
	constexpr hostDeviceInline auto unit_assign(value_unit<T, U1>& a, value_unit<V, U2> b) { a.val.y = b.val; }
	template<uint32_t idx, typename T, typename V, typename U1, typename U2, typename  std::enable_if_t<SI::is_compatible<U1, U2>::value>* = nullptr, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, greaterDimension<2, T>* = nullptr, std::enable_if_t<idx ==  3>* = nullptr>
	constexpr hostDeviceInline auto unit_assign(value_unit<T, U1>& a, value_unit<V, U2> b) { a.val.z = b.val; }
	template<uint32_t idx, typename T, typename V, typename U1, typename U2, typename  std::enable_if_t<SI::is_compatible<U1, U2>::value>* = nullptr, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, greaterDimension<2, T>* = nullptr, greaterDimension<3, T>* = nullptr, std::enable_if_t<idx ==  4>* = nullptr>
	constexpr hostDeviceInline auto unit_assign(value_unit<T, U1>& a, value_unit<V, U2> b) { a.val.w = b.val; }

	template<uint32_t idx, typename T, typename U1, typename std::enable_if<dimension<T>::value == 0, std::nullptr_t>::type* = nullptr>
	constexpr hostDeviceInline auto unit_assign(value_unit<T, U1>& a, elem_t<T, U1> b) { a.val = b.val; }
	template<uint32_t idx, typename T, typename U1, greaterDimension<0, T>* = nullptr, std::enable_if_t<idx ==  1>* = nullptr>
	constexpr hostDeviceInline auto unit_assign(value_unit<T, U1>& a, elem_t<T, U1> b) { a.val.x = b.val; }
	template<uint32_t idx, typename T, typename U1, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, std::enable_if_t<idx ==  2>* = nullptr>
	constexpr hostDeviceInline auto unit_assign(value_unit<T, U1>& a, elem_t<T, U1> b) { a.val.y = b.val; }
	template<uint32_t idx, typename T, typename U1, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, greaterDimension<2, T>* = nullptr, std::enable_if_t<idx ==  3>* = nullptr>
	constexpr hostDeviceInline auto unit_assign(value_unit<T, U1>& a, elem_t<T, U1> b) { a.val.z = b.val; }
	template<uint32_t idx, typename T, typename U1, greaterDimension<0, T>* = nullptr, greaterDimension<1, T>* = nullptr, greaterDimension<2, T>* = nullptr, greaterDimension<3, T>* = nullptr, std::enable_if_t<idx ==  4>* = nullptr>
	constexpr hostDeviceInline auto unit_assign(value_unit<T, U1>& a, elem_t<T, U1> b) { a.val.w = b.val; }

	template<typename T, typename U>
	hostDeviceInline constexpr auto planeDistance(T&& plane, U&&  point) {
		return math::dot3(plane, point) + (plane.val.w);
	}
	template<typename Func, typename T = float>
	hostDeviceInline auto brentsMethod(Func f, T lower, T upper, T tol, int32_t max_iter) {
		using R = decltype(f(lower));
		T a = lower;
		T b = upper;
		R fa = f(a);
		R fb = f(b);
		R fs = 0.f;

		if (!(fa * fb < 0.f)) {
			return T{ FLT_MAX };
		}

		if (math::abs(fa) < math::abs(fb)) {
			auto t1 = a; a = b; b = t1;
			auto t2 = fa; fa = fb; fb = t2;
		}

		T c = a;
		R fc = fa;
		bool mflag = true;
		T s{FLT_MAX};
		T d;

		for (int32_t iter = 1; iter < max_iter; ++iter) {

			if (math::abs(b - a) < tol) {
				return s;
			}

			if (fa != fc && fb != fc) {
				s = (a * fb * fc / ((fa - fb) * (fa - fc))) + (b * fa * fc / ((fb - fa) * (fb - fc))) +
					(c * fa * fb / ((fc - fa) * (fc - fb)));
			}
			else {
				s = b - fb * (b - a) / (fb - fa);
			}

			if (((s < (3.f * a + b) * 0.25f) || (s > b)) ||
				(mflag && (math::abs(s - b) >= (math::abs(b - c) * 0.5f))) ||
				(!mflag && (math::abs(s - b) >= (math::abs(c - d) * 0.5f))) ||
				(mflag && (math::abs(b - c) < tol)) || (!mflag && (math::abs(c - d) < tol))) {
				s = (a + b) * 0.5f;
				mflag = true;
			}
			else {
				mflag = false;
			}

			fs = f(s);
			d = c;
			c = b;
			fc = fb;

			if (fa * fs < 0) {
				b = s;
				fb = fs;
			}
			else {
				a = s;
				fa = fs;
			}

			if (math::abs(fa) < math::abs(fb)) {
				auto t1 = a; a = b; b = t1;
				auto t2 = fa; fa = fb; fb = t2;
			}

		}

		//std::cout << "The solution does not converge or iterations are not sufficient" << std::endl;
		return s;
	}
}

