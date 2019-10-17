#pragma once
#include <utility/include_all.h>
#define basicVolume (PI4O3 * math::power<3>(arrays.radius))
namespace math {
	hostDeviceInline float4_u<SI::m> point1Plane(float4_u<void_unit_ty> E, float4_u<SI::m> P) {
		auto d = math::planeDistance(E, P);
		return P - d * E;
	}
	hostDeviceInline float4_u<SI::m> point2Plane(float4_u<void_unit_ty> E0, float4_u<void_unit_ty> E1, float4_u<SI::m> P) {
		auto E2 = math::cross(E0, E1);
		auto det = math::sqlength3(E2);
		float4_u<SI::m> linePoint((((math::cross(E2, E1) * E0.val.w) + (math::cross(E0, E2) * E1.val.w)) / det).val);
		auto lineDirection = math::normalize3(E2);
		auto diff = P - linePoint;
		auto distance = math::dot3(diff, lineDirection);
		return linePoint + lineDirection * distance;
	}
	hostDeviceInline float4_u<SI::m> point3Plane(float4_u<void_unit_ty> E0, float4_u<void_unit_ty> E1, float4_u<void_unit_ty> E2, float4_u<SI::m>) {
		auto a = -E0.val.w * math::cross(E1, E2);
		auto b = -E1.val.w * math::cross(E2, E0);
		auto c = -E2.val.w * math::cross(E0, E1);
		auto d = math::dot3(E0, math::cross(E1, E2));
		return float4_u<SI::m>(((a + b + c) / d).val);
	}
	constexpr float kernel_scale = 0.509843426183f;
	constexpr float kernel_scale_i = 1.2517599686431f;
	hostDeviceInline auto scaleValue(float val, float targetVolume, float initialVolume) {
		auto hForInitial = radius_from_volume(initialVolume);
		auto hForTarget = radius_from_volume(targetVolume);
		return val * powf(hForInitial, 3.f) / powf(hForTarget, 1.f / 3.f);
	}
	hostDeviceInline auto scaleGradient(float val, float targetVolume, float initialVolume) {
		auto hForInitial = radius_from_volume(initialVolume);
		auto hForTarget = radius_from_volume(targetVolume);
		return val * powf(hForInitial, 4.f) / powf(hForTarget, 1.f / 4.f);
	}
	template<typename T>
	hostDeviceInline auto scaleValue(T val, float targetVolume) {
		float hForInitial = support_from_volume(1.f);// { 1.2517599686431f };
		float hForTarget = support_from_volume(targetVolume);
		return val * power<3>(hForInitial) / power<3>(hForTarget);
	}
	template<typename T>
	hostDeviceInline auto scaleGradient(T val, float targetVolume) {
		float hForInitial = support_from_volume(1.f);// { 1.2517599686431f };
		float hForTarget = support_from_volume(targetVolume);
		return val * power<4>(hForInitial) / power<4>(hForTarget);
	}
} // namespace math
namespace boundary {
//#define BOUNDARY_OFFSET 0.24509788f
//#define BOUNDARY_OFFSET 0.4002431620f
#define BOUNDARY_OFFSET 0.f
//#define BOUNDARY_OFFSET 0.41f

	hostDeviceInline float g(float4_u<SI::m> x, float_u<SI::m> h) {
		float d = x.val.w;
		//const float n = 1.f;
		const float gamma = 2.5f;
		const float offset = 1.f;
		//h = support_from_volume(1.f);
		auto H = h.val * kernelSize();

		return 1.f - d / H;

		float q = ( - d + BOUNDARY_OFFSET * H) / (1.f * H);
		//q = math::clamp(q, -FLT_MAX, 1.f);
		auto qq = q * gamma + offset;
		if (d >= BOUNDARY_OFFSET * H) return 1.f;
		return qq * 1.f;// powf(q * gamma + offset, n);
	}
	hostDeviceInline float dg(float4_u<SI::m> x, float_u<SI::m> h) {
		float d = x.val.w;
		//const float n = 1.f;
		const float gamma = 2.5f;
		//const float offset = 1.f;
		//h = support_from_volume(1.f);
		auto H = h.val * kernelSize();
		//float q = ( - d) / (2.f * H);
		//q = math::clamp(q, -FLT_MAX, 1.f);
		return -1.f / H;

		if (d >= BOUNDARY_OFFSET * H) return 0.f;
		//float g = q * gamma + offset;
		//float fac = 1.f;//support_from_volume(float_u<SI::volume>{1.f}).val / h.val;
		//auto qq = q * gamma + offset;
		return -gamma / (1.f * H) * 1.f; // *powf(q * gamma + offset, n - 1.f);
	}

	hostDeviceInline auto lookupOffset(float* offsetLUT, float val, int32_t lutSize) {
		float xRel = val * ((float)lutSize - 1.f);
		auto xL = math::floorf(xRel);
		auto xH = math::ceilf(xRel);
		auto xD = xRel - xL;
		int32_t xLi = math::clamp(static_cast<int32_t>(xL), 0, lutSize - 1);
		int32_t xHi = math::clamp(static_cast<int32_t>(xH), 0, lutSize - 1);
		auto lL = offsetLUT[xLi];
		auto lH = offsetLUT[xHi];
		auto v = lL * xD + (1.f - xD) * lH;
		return v;
	}

	template<typename T, typename U>
	hostDeviceInline auto lookupValue(const U& arrays, T* LUT, int32_t lutSize, float4_u<SI::m> distance, float_u<SI::volume> volume, float_u<> density, float_u<SI::m> h, float Hoffset = 0.f) {
		auto x = distance.val.w;
		auto H = h.val * kernelSize();
		auto xRel = math::clamp((x + H + H * (lookupOffset((float*)arrays.offsetLUT, density.val, lutSize) /*+ 0.24509788f*/)) / (2.f * H), 0.f, 1.f) * ((float)lutSize - 1.f);
		auto xL = math::floorf(xRel);
		auto xH = math::ceilf(xRel);
		auto xD = xRel - xL;
		int32_t xLi = math::clamp(static_cast<int32_t>(xL), 0, lutSize - 1);
		int32_t xHi = math::clamp(static_cast<int32_t>(xH), 0, lutSize - 1);
		auto lL = LUT[xLi];
		auto lH = LUT[xHi];
		auto val = lL * xD + (1.f - xD) * lH;
		return val;
		//return val * powf(2.28539074867f, 3.f) * powf(H, -3.f);
	}
	template<typename T, typename U>
	hostDeviceInline auto lookupGradient(const U& arrays, T* LUT, int32_t lutSize, float4_u<SI::m> distance, float_u<SI::volume> volume, float_u<> density, float_u<SI::m> h, float Hoffset = 0.f) {
		auto x = distance.val.w;
		auto H = h.val * kernelSize();
		auto xRel = math::clamp((x + H + H * (lookupOffset((float*)arrays.offsetLUT, density.val, lutSize) /*+ 0.24509788f*/)) / (2.f * H),0.f,1.f) * ((float)lutSize - 1.f);
		auto xL = math::floorf(xRel);
		auto xH = math::ceilf(xRel);
		auto xD = xRel - xL;
		int32_t xLi = math::clamp(static_cast<int32_t>(xL), 0, lutSize - 1);
		int32_t xHi = math::clamp(static_cast<int32_t>(xH), 0, lutSize - 1);
		auto lL = LUT[xLi];
		auto lH = LUT[xHi];
		auto val = lL * xD + (1.f - xD) * lH;
		//auto val = (lH - lL) / (2.f * float_u<SI::m>(H) / ((float)lutSize - 1.f));
		return val;
		//return val * powf(2.28539074867f, 4.f) * powf(H, -4.f);
	};
	

}
#define BOUNDARY_LIMIT (d.val.w < h.val * kernelSize() * 0.995f ? 1.f : 0.f)

namespace planeBoundary {
	namespace internal {
		template <typename T>
		hostDeviceInline float4_u<void_unit_ty> POSfunction(float4_u<SI::m> position, float_u<SI::volume> volume, float_u<SI::m> h, T &arrays) {
			auto H = h * kernelSize();
			auto H1 = support_from_volume(1.f) * kernelSize();
			if (H <= H1)
				H = H1;
			float4_u<void_unit_ty> e0, e1, e2;
			int32_t counter = 0;
			for (int32_t it = 0; it < arrays.boundaryCounter; ++it) {
				auto plane = arrays.boundaryPlanes[it];
				auto d = math::planeDistance(plane, position);
				if (d < 0.995f * H) {
					switch (counter) {
					case 0:
						e0 = plane;
						++counter;
						break;
					case 1:
						e1 = plane;
						++counter;
						break;
					case 2:
						e2 = plane;
						++counter;
						break;
					default:
						break;
					}
				}
			}
			float4_u<SI::m> c;
			float4_u<void_unit_ty> Hn;
			switch (counter) {
			case 1:
				c = math::point1Plane(e0, position);
				Hn = e0;
				break;
			case 2:
				c = math::point2Plane(e0, e1, position);
				Hn = e0 + e1;
				break;
			case 3:
				c = math::point3Plane(e0, e1, e2, position);
				Hn = e0 + e1 + e2;
				break;
			default:
				return float4_u<void_unit_ty>(0.f, 0.f, 0.f, 1e21f);
			}
			Hn = math::normalize3(Hn);
			float Hfactor = sqrtf((float)counter);
			float4_u<SI::m> Hp = c + Hfactor * Hn * H;
			float4_u<SI::m> diff = Hp - position;
			float_u<SI::m> diffL = math::length3(diff);
			float4_u<void_unit_ty> Hd = diff.val * (diffL.val < FLT_MIN ? 0.f : 1.f / diffL.val);
			float4_u<SI::m> pos = Hp - (H - 0.24509788f * h.val * kernelSize() * 0.f /* * 0.612f*/ ) * Hd;

			float4_u<void_unit_ty> plane = Hd;
			plane.val.w = -math::dot3(pos, plane).val;
			auto distance = math::planeDistance(plane, position);
			//if (distance < 0.f)
			//	plane.val = -plane.val;
			plane.val.w = distance.val;
			return plane;
		}
	}
	template<typename T>
	hostDeviceInline auto distance(float4_u<SI::m> p, float_u<SI::volume> v, T& arrays) {
		return float4_u<>(internal::POSfunction(p, v, support_from_volume(v), arrays).val);
	}
	template<typename T, typename U>
	hostDeviceInline auto value2(U* LUT, float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T& arrays) {
		using Ty = decltype(boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, position, volume, density, support_from_volume(volume)));
		auto d = distance(position, volume, arrays);
		auto h = support_from_volume(volume);
		//auto fac = d.val < 2.f * h.val * kernelSize() ? 1.f : 0.f;
		//if (!(d.val < 2.f * h.val * kernelSize())) return Ty{ 0.f };
		return boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, d, volume, h, arrays.LUTOffset) * BOUNDARY_LIMIT;
	}
	template<typename T, typename U>
	hostDeviceInline auto value(U* LUT, float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T& arrays) {
		using Ty = decltype(boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, position, volume, density, support_from_volume(volume)));
		auto d = distance(position, volume, arrays);
		auto h = support_from_volume(volume);
		auto h0 = support_from_volume(4.f / 3.f * CUDART_PI_F * math::cubic(arrays.radius));
		//auto fac = d.val < 2.f * h.val * kernelSize() ? 1.f : 0.f;
		//if (!(d.val < 2.f * h.val * kernelSize())) return Ty{ 0.f };
		return boundary::g(d,h) * boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, d, volume, density, h, arrays.LUTOffset) * BOUNDARY_LIMIT;
	}
	template<typename T, typename U, typename V>
	hostDeviceInline auto gradient(U* LUT, V* gradLUT, float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T& arrays) {
		auto d = distance(position, volume, arrays);
		using Ty = decltype(d * boundary::lookupGradient(arrays, gradLUT, arrays.boundaryLUTSize, position, volume, density, support_from_volume(volume)));
		auto h = support_from_volume(volume);
		//auto h0 = support_from_volume(4.f / 3.f * CUDART_PI_F * math::cubic(arrays.radius));
		//auto fac = d.val < 2.f * h.val * kernelSize() ? 1.f : 0.f;
		//if (!(d.val < 2.f * h.val * kernelSize())) return Ty{ 0.f, 0.f, 0.f, 0.f };
		return d  * BOUNDARY_LIMIT *
			(boundary::g(d, h) * boundary::lookupGradient(arrays, gradLUT, arrays.boundaryLUTSize, d, volume, density, h, arrays.LUTOffset) +
			 boundary::dg(d, h) * boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, d, volume, density, h, arrays.LUTOffset).val);
	}
} // namespace boundary

namespace volumeBoundary {
	namespace internal {

#if defined(__CUDA_ARCH__) || defined(__INTELLISENSE__)
		template <typename T> hostDeviceInline auto volumeDistanceFni(float4_u<SI::m> p, float_u<SI::volume> volume, float_u<SI::m> h, T &arrays, int32_t i) {
			float4 pt{ p.val.x, p.val.y, p.val.z, 1.f };
			float4_u<void_unit_ty> tp{ arrays.volumeBoundaryTransformMatrixInverse[i] * pt };
			//tp.val = pt;

			float4_u<SI::m> d_min = arrays.volumeBoundaryMin[i];
			float4_u<SI::m> d_max = arrays.volumeBoundaryMax[i];

			if ((d_min.val.x < tp.val.x) && (d_min.val.y < tp.val.y) && (d_min.val.z < tp.val.z) &&
				(d_max.val.x > tp.val.x) && (d_max.val.y > tp.val.y) && (d_max.val.z > tp.val.z)) {
				float4_u<void_unit_ty> d_p = (tp.val - d_min.val) / (d_max.val - d_min.val);
				#ifdef __clang__
				//asm("ld.global.cg.s32 %0, [%1];" : "=r"(*(reinterpret_cast<int32_t*>(&pSpan))) :
				float4_u<void_unit_ty> n =
					tex3D<float4>(arrays.volumeBoundaryVolumes[i], d_p.val.x, d_p.val.y, d_p.val.z);
				#else
				float4_u<void_unit_ty> n =
					tex3D<float4>(arrays.volumeBoundaryVolumes[i], d_p.val.x, d_p.val.y, d_p.val.z);
				#endif
				float d = n.val.w;
				n.val.w = 0.f;
				math::normalize3(n);

				if (d < 0.995f * h * kernelSize()) {
					n.val = arrays.volumeBoundaryTransformMatrixInverse[i].transpose() * n.val;
					//auto pbw = pt - d * nw;
					n = math::normalize3(n);
					n.val.w = d;
					return n;
				}
			}
			return float4_u<>{0.f, 0.f, 0.f, 1e21f};
		}
#else
		template <typename T> hostDeviceInline auto volumeDistanceFni(float4_u<SI::m>, float_u<SI::volume> volume, float_u<SI::m> h, T& arrays, int32_t idx) {
			return float4_u<>{0.f, 0.f, 0.f, 1e21f};
		}
#endif
	}
	template<typename T>
	hostDeviceInline auto distance_fn(float4_u<SI::m> p, float_u<SI::volume> v, T& arrays, int32_t idx) {
		return float4_u<>(internal::volumeDistanceFni(p, v, support_from_volume(v), arrays, idx).val);
	}
	template<typename T, typename U>
	hostDeviceInline auto value2(U* LUT, float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T& arrays, int32_t idx) {
		using Ty = decltype(boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, position, volume, density, support_from_volume(volume)));
		Ty sum{ 0.f };
		auto h = support_from_volume(volume);
		if (idx == -1) {
			for (int32_t i = 0; i < arrays.volumeBoundaryCounter; ++i) {
				auto d = distance_fn(position, volume, arrays, i);
				if (d.val.w < 0.9995f * h.val * kernelSize())
					sum += 1.f / boundary::g(d, h) *  1.f / boundary::g(d, h) * boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, d, volume, density, h, arrays.LUTOffset) * BOUNDARY_LIMIT;
			}
		}
		else {
			auto d = distance_fn(position, volume, arrays, idx);
			if (d.val.w < 0.9995f * h.val * kernelSize())
				return  1.f / boundary::g(d, h) *  1.f / boundary::g(d, h) * boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, d, volume, density, h, arrays.LUTOffset) * BOUNDARY_LIMIT;
		}
		return sum;
	}
	template<typename T, typename U>
	hostDeviceInline auto value(U* LUT, float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T& arrays, int32_t idx) {
		using Ty = decltype(boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, position,volume, density, support_from_volume(volume)));
		Ty sum{ 0.f };
		auto h = support_from_volume(volume);
		if (idx == -1) {
			for (int32_t i = 0; i < arrays.volumeBoundaryCounter; ++i) {
				auto d = distance_fn(position, volume, arrays, i);
				if(d.val.w < 0.9995f * h.val * kernelSize())
					sum += boundary::g(d, h) * boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, d, volume, density, h, arrays.LUTOffset) * BOUNDARY_LIMIT;
			}
		}
		else {
			auto d = distance_fn(position, volume, arrays, idx);
			if (d.val.w < 0.9995f * h.val * kernelSize())
				return boundary::g(d, h) * boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, d, volume, density, h, arrays.LUTOffset) * BOUNDARY_LIMIT;
		}
		return sum;
	}
	template<typename T, typename U, typename V>
	hostDeviceInline auto gradient(U* LUT, V* gradLUT, 
		float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T& arrays, int32_t idx = -1) {
		using Ty = decltype(boundary::lookupGradient(arrays, gradLUT, arrays.boundaryLUTSize, position, volume, density, support_from_volume(volume)));
		auto sum = float4_u<>{ 0.f, 0.f, 0.f, 0.f } *Ty{};
		auto h = support_from_volume(volume);
		if (idx == -1) {
			for (int32_t i = 0; i < arrays.volumeBoundaryCounter; ++i) {
				auto d = distance_fn(position, volume, arrays, i);
				if (d.val.w < 0.9995f * h.val * kernelSize())
					sum += d * (
						boundary::g(d, h) * boundary::lookupGradient(arrays, gradLUT, arrays.boundaryLUTSize, d, volume, density, h, arrays.LUTOffset) +
						boundary::dg(d, h) * boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, d, volume, density, h, arrays.LUTOffset).val
						)
					* BOUNDARY_LIMIT;
			}
		}
		else {
			auto d = distance_fn(position, volume, arrays, idx);
			if (d.val.w < 0.9995f * h.val * kernelSize())
				return d * (
					boundary::g(d, h) * boundary::lookupGradient(arrays, gradLUT, arrays.boundaryLUTSize, d, volume, density, h, arrays.LUTOffset) +
					boundary::dg(d, h) * boundary::lookupValue(arrays, LUT, arrays.boundaryLUTSize, d, volume, density, h, arrays.LUTOffset).val
					) * BOUNDARY_LIMIT;
		}
		return sum;
	}
} // namespace volumeBoundary
namespace boundary{
	enum struct kind {
		volume, plane, both
	};
	namespace internal{
		template<typename T, typename U> hostDeviceInline auto lookup2(T* LUT, float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, U &arrays, kind k = kind::both, int32_t idx = -1) {
			if (k == kind::plane)
				return planeBoundary::value2(LUT, position, volume, density, arrays);
			else if (k == kind::volume)
				return volumeBoundary::value2(LUT, position, volume, density, arrays, idx);
			else
				return planeBoundary::value2(LUT, position, volume, density, arrays) + volumeBoundary::value2(LUT, position, volume, density, arrays, idx);
		}
		template<typename T, typename U> hostDeviceInline auto lookup(T* LUT, float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, U &arrays, kind k = kind::both, int32_t idx = -1) {
			if (k == kind::plane)
				return planeBoundary::value(LUT, position, volume, density, arrays);
			else if (k == kind::volume)
				return volumeBoundary::value(LUT, position, volume, density, arrays, idx);
			else
				return planeBoundary::value(LUT, position, volume, density, arrays) + volumeBoundary::value(LUT, position, volume, density, arrays, idx);
		}
		template<typename T, typename U, typename V> hostDeviceInline auto lookupGradient(T* LUT, V* gradLUT, float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, U &arrays, kind k = kind::both, int32_t idx = -1) {
			if (k == kind::plane)
				return planeBoundary::gradient(LUT, gradLUT, position, volume, density, arrays);
			else if (k == kind::volume)
				return volumeBoundary::gradient(LUT, gradLUT, position, volume, density, arrays, idx);
			else
				return planeBoundary::gradient(LUT, gradLUT, position, volume, density, arrays) + volumeBoundary::gradient(LUT, gradLUT, position, volume, density, arrays, idx);
		}
		template<typename T, typename U> hostDeviceInline auto lookupScaled2Value(T* LUT, float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, U &arrays, kind k = kind::both, int32_t idx = -1) {
			return internal::lookup2(LUT, position, volume, density, arrays, k, idx);
		}
		template<typename T, typename U> hostDeviceInline auto lookupScaledValue(T* LUT, float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, U &arrays, kind k = kind::both, int32_t idx = -1) {
			auto val = internal::lookup(LUT, position, volume, density, arrays, k, idx);
			//auto vol = lookup(arrays.volumeLUT, position, volume, arrays, k, idx).val * volume;
			//if (vol > 0.f) {
				//auto kus = val;// / vol;
				return val;
				//return math::scaleValue(kus, volume.val);
			//}
			//else {
			//	return val * 0.f;// / float_u<SI::volume>{1.f} *0.f;
			//}
		}
		template<typename T, typename U, typename V> hostDeviceInline auto lookupScaledGradient(T* LUT, V* gradLUT, float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, U &arrays, kind k = kind::both, int32_t idx = -1) {
			auto val = internal::lookupGradient(LUT, gradLUT, position, volume, density, arrays, k, idx);
			//auto vol = lookup(arrays.volumeLUT, position, volume, arrays, k, idx).val * volume;
			//if (vol > 0.f) {
			//	auto kus = val;// / vol;
				return val * support_from_volume(float_u<SI::volume>{1.f}) / support_from_volume(volume);
				//return math::scaleGradient(kus, volume.val);
			//}
			//else {
			//	return val * 0.f; // / float_u<SI::volume>{1.f} *0.f;
			//}
		}
	}
	template<typename T> hostDeviceInline float_u<> unscaledVolume(float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T &arrays, kind k = kind::both, int32_t idx = -1) {
		return internal::lookup(arrays.volumeLUT, position, volume, density, arrays, k, idx) / float_u<SI::volume>{1.f /*(float)kernelNeighbors()*/};
	}
	template<typename T> hostDeviceInline float_u<SI::volume> volume(float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T &arrays, kind k = kind::both, int32_t idx = -1) {
		return volume * unscaledVolume(position, volume, density, arrays, k, idx);
	}
	template <typename T> hostDeviceInline float_u<> spline(float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T &arrays, kind k = kind::both, int32_t idx = -1) {
		return internal::lookupScaledValue(arrays.splineLUT, position, volume, density, arrays, k, idx);
	}
	template <typename T> hostDeviceInline float4_u<SI::m_1> splineGradient(float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T &arrays, kind k = kind::both, int32_t idx = -1) {
		return internal::lookupScaledGradient(arrays.splineLUT, arrays.splineGradientLUT, position, volume, density, arrays, k, idx);
	}
	template <typename T> hostDeviceInline float_u<SI::m_5> splineGradient2(float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T &arrays, kind k = kind::both, int32_t idx = -1) {
		return internal::lookupScaled2Value(arrays.spline2LUT, position, volume, density, arrays, k, idx) * math::power<5>(support_from_volume(float_u<SI::volume>{1.f})) / math::power<5>(support_from_volume(volume));
	}
	template <typename T> hostDeviceInline float_u<> spiky(float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T &arrays, kind k = kind::both, int32_t idx = -1) {
		return internal::lookupScaledValue(arrays.spikyLUT, position, volume, density, arrays, k, idx);
	}
	template <typename T> hostDeviceInline float4_u<SI::m_1> spikyGradient(float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T &arrays, kind k = kind::both, int32_t idx = -1) {
		return internal::lookupScaledGradient(arrays.spikyLUT, arrays.spikyGradientLUT, position, volume, density, arrays, k, idx);
	}
	template <typename T> hostDeviceInline auto cohesion(float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T &arrays, kind k = kind::both, int32_t idx = -1) {
		return internal::lookupScaledGradient(arrays.cohesionLUT, position, volume, density, arrays, k, idx);
	}
	template <typename T> hostDeviceInline auto adhesion(float4_u<SI::m> position, float_u<SI::volume> volume, float_u<> density, T &arrays, kind k = kind::both, int32_t idx = -1) {
		return internal::lookupScaledGradient(arrays.adhesionLUT, arrays.adhesionLUT, position, volume, density, arrays, k, idx);
	}
} // namespace boundary

#define pDistance planeBoundary::distance(arrays.position[i], arrays.volume[i], arrays)
#define volumeDistance(b) volumeBoundary::distance_fn(arrays.position[i], arrays.volume[i], arrays, b)

//#define V_b(k,j) boundary::volume(arrays.position[i], arrays.volume[i], arrays, k, j)
//#define pV_b V_b(boundary::kind::plane, -1)
//#define vV_b(j) V_b(boundary::kind::volume, j)
//#define bV_b V_b(boundary::kind::both, -1)

#define W_ib(k,j) boundary::spline(arrays.position[i], arrays.volume[i], arrays.density[i], arrays, k, j)
#define pW_ib W_ib(boundary::kind::plane, -1)
#define vW_ib(j) W_ib(boundary::kind::volume, j)
//#define bW_ib W_ib(boundary::kind::both, -1)

#define GW_ib(k,j) boundary::splineGradient(arrays.position[i], arrays.volume[i], arrays.density[i], arrays, k, j)
#define pGW_ib GW_ib(boundary::kind::plane, -1)
#define vGW_ib(j) GW_ib(boundary::kind::volume, j)
//#define bGW_ib GW_ib(boundary::kind::both, -1)

#define PW_ib(k,j) boundary::spiky(arrays.position[i], arrays.volume[i], arrays.density[i], arrays, k, j)
#define pPW_ib PW_ib(boundary::kind::plane, -1)
#define vPW_ib(j) PW_ib(boundary::kind::volume, j)
//#define bPW_ib PW_ib(boundary::kind::both, -1)

#define GPW_ib(k,j) boundary::spikyGradient(arrays.position[i], arrays.volume[i], arrays.density[i], arrays, k, j)
#define pGPW_ib GPW_ib(boundary::kind::plane, -1)
#define vGPW_ib(j) GPW_ib(boundary::kind::volume, j)
//#define bGPW_ib GPW_ib(boundary::kind::both, -1)
