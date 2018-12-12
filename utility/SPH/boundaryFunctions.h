#pragma once
#include <utility/include_all.h>
#define basicVolume (PI4O3 * math::power<3>(arrays.radius))


namespace math {
hostDeviceInline float4_u<SI::m> point1Plane(float4_u<void_unit_ty> E, float4_u<SI::m> P) {
  auto d = math::planeDistance(E, P);
  return P - d * E;
}
hostDeviceInline float4_u<SI::m> point2Plane(float4_u<void_unit_ty> E0, float4_u<void_unit_ty> E1,
                                             float4_u<SI::m> P) {
  auto E2 = math::cross(E0, E1);
  auto det = math::sqlength3(E2);
  float4_u<SI::m> linePoint((((math::cross(E2, E1) * E0.val.w) + (math::cross(E0, E2) * E1.val.w)) / det).val);
  auto lineDirection = math::normalize3(E2);
  auto diff = P - linePoint;
  auto distance = math::dot3(diff, lineDirection);
  return linePoint + lineDirection * distance;
}
hostDeviceInline float4_u<SI::m> point3Plane(float4_u<void_unit_ty> E0, float4_u<void_unit_ty> E1,
                                             float4_u<void_unit_ty> E2, float4_u<SI::m>) {
  auto a = -E0.val.w * math::cross(E1, E2);
  auto b = -E1.val.w * math::cross(E2, E0);
  auto c = -E2.val.w * math::cross(E0, E1);
  auto d = math::dot3(E0, math::cross(E1, E2));
  return float4_u<SI::m>(((a + b + c) / d).val);
}
} // namespace math


namespace boundary {
template <typename T>
hostDeviceInline float4_u<void_unit_ty> POSfunction(float4_u<SI::m> position, T &arrays) {
  auto volume = PI4O3 * math::power<3>(arrays.radius);
  auto h = support_from_volume(volume);
  auto H = h * kernelSize();

  float4_u<void_unit_ty> e0, e1, e2;
  int32_t counter = 0;
  for (int32_t it = 0; it < arrays.boundaryCounter; ++it) {
    auto plane = arrays.boundaryPlanes[it];
    if (math::planeDistance(plane, position) < H) {
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
  auto Hp = c + Hn * H;
  auto diff = Hp - position;
  //auto diffL = math::length3(diff);
  auto Hd = math::normalize3(diff);
  auto pos = Hp - H * Hd;

  auto plane = Hd;
  plane.val.w = -math::dot3(pos, plane).val;
  auto distance = math::planeDistance(plane, position);
  plane.val.w = distance.val;
  return plane;
}

template <typename T>
hostDeviceInline auto boundaryLookup(float_u<SI::m> distance, float_u<SI::m> H,
                                     float_u<SI::volume> vol, T &arrays) {
  int32_t idx =
      math::clamp(arrays.boundaryLUTSize -
                      static_cast<int32_t>(
                          ceilf((distance.val + H.val) / H.val * ((float)arrays.boundaryLUTSize + 1) * 0.5f)),
                  0, arrays.boundaryLUTSize - 1);
  if (distance >= H * 0.995f)
    return (decltype(arrays.boundaryLUT[idx] / vol))(0.f);
  else
    return arrays.boundaryLUT[idx] / vol;
}
template <typename T>
hostDeviceInline auto boundaryGradientLookup(float_u<SI::m> distance, float_u<SI::m> H,
                                             float_u<SI::volume> vol, T &arrays) {
  int32_t idx =
      math::clamp(arrays.boundaryLUTSize -
                      static_cast<int32_t>(
                          ceilf((distance.val + H.val) / H.val * ((float)arrays.boundaryLUTSize + 1) * 0.5f)),
                  0, arrays.boundaryLUTSize - 2);
  auto step = H / ((float)arrays.boundaryLUTSize + 1) * 2.f;
  auto a = arrays.boundaryLUT[idx + 1] / vol;
  auto b = arrays.boundaryLUT[idx] / vol;
  if (distance >= H * 0.995f)
    return (decltype((b - a) / step))(0.f);
  else
    return (b - a) / step;
}
template <typename T>
hostDeviceInline auto boundaryPressureLookup(float_u<SI::m> distance, float_u<SI::m> H,
                                             float_u<SI::volume> vol, T &arrays) {
  int32_t idx =
      math::clamp(arrays.boundaryLUTSize -
                      static_cast<int32_t>(
                          ceilf((distance.val + H.val) / H.val * ((float)arrays.boundaryLUTSize + 1) * 0.5f)),
                  0, arrays.boundaryLUTSize - 1);
  if (distance >= H * 0.995f)
    return (decltype(arrays.boundaryPressureLUT[idx] / vol))(0.f);
  else
    return arrays.boundaryPressureLUT[idx] / vol;
}
template <typename T>
hostDeviceInline auto boundaryPressureGradientLookup(float_u<SI::m> distance, float_u<SI::m> H,
                                                     float_u<SI::volume> vol, T &arrays) {
  int32_t size = arrays.boundaryLUTSize;
  auto volumeb = PI4O3 * math::power<3>(arrays.radius);
  auto hb = support_from_volume(volumeb);
  auto Hb = hb * kernelSize();

  auto step = Hb / ((float)size + 1) * 2.f;
  float_u<SI::m> h_0(1.2517599643209638f);
  auto h_c = H / kernelSize();
  auto ratio = h_0 / h_c;

  int32_t idx = math::clamp((int32_t)size - math::castTo<int32_t>(math::floorf((distance + Hb) / step)), 0,
                            (int32_t)size - 1);
  if (distance >= H * 0.995f)
    return (decltype(arrays.boundaryPressureLUT[idx] / vol))(0.f);
  else
    return arrays.boundaryPressureLUT[idx] / vol * ratio;
}

template <typename T>
hostDeviceInline auto xbarLookup(float_u<SI::m> distance, float_u<SI::m> H, float_u<SI::volume> vol,
                                 T &arrays) {
  int32_t size = arrays.boundaryLUTSize;

  auto step = H / ((float)size + 1) * 2.f;
  float_u<SI::m> h_0(1.2517599643209638f);
  auto h_c = H / kernelSize();
  auto hratio = h_0 / h_c;

  int32_t idx = math::clamp((int32_t)size - math::castTo<int32_t>(math::floorf((distance + H) / step)), 0,
                            (int32_t)size - 1);

  return pair<value_unit<float, SI::derived_unit<SI::recip_3<SI::m>, SI::recip_2<SI::m>> /*SI::SI_Unit<SI::Base::m, ratio<-5, 1>>*/>,
              float_u<SI::recip<SI::volume>>>{arrays.xbarLUT[idx] / vol / hratio,
                                              arrays.boundaryLUT[idx] / vol / hratio};
}
template <typename T>
hostDeviceInline auto ctrLookup(float_u<SI::m> distance, float_u<SI::m> H, float_u<SI::volume>,
                                T &arrays) {
  int32_t idx =
      math::clamp(arrays.boundaryLUTSize -
                      math::castTo<int32_t>(
                          math::ceilf((distance + H) / H * ((float)arrays.boundaryLUTSize + 1) * 0.5f)),
                  0, arrays.boundaryLUTSize - 1);
  return arrays.ctrLUT[idx];
}

template <typename T> hostDeviceInline auto spline4(float4_u<SI::m> position, T &arrays) {
  auto volumeb = basicVolume;
  auto hb = support_from_volume(volumeb);
  auto Hb = hb * kernelSize();
  auto POS = boundary::POSfunction(position, arrays);
  auto val = boundaryLookup(float_u<SI::m>(POS.val.w), Hb, volumeb, arrays) * volumeb;
  if (POS.val.w < 1e20f)
    return val;
  else 
    return val * 0.f;
}
template <typename T>
hostDeviceInline auto spikyGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
  auto H = float_u<SI::m>(position.val.w * Kernel<kernel_kind::spline4>::kernel_size());
  //auto volumeb = basicVolume;
  //auto hb = support_from_volume(volumeb);
  //auto Hb = hb * kernelSize();
  auto POS = boundary::POSfunction(position, arrays);
  float4_u<> n{POS.val.x, POS.val.y, POS.val.z, 0.f};
  auto val = n * boundaryPressureGradientLookup(float_u<SI::m>(POS.val.w), H, vol, arrays);
  if (POS.val.w < 1e20f)
    return val;
  else 
    return val * 0.f;
}
template <typename T>
hostDeviceInline auto splineGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
  auto H = float_u<SI::m>(position.val.w * Kernel<kernel_kind::spline4>::kernel_size());
  auto volumeb = basicVolume;
  auto hb = support_from_volume(volumeb);
  auto Hb = hb * kernelSize();
  auto POS = boundary::POSfunction(position, arrays);
  float4_u<> n{POS.val.x, POS.val.y, POS.val.z, 0.f};
  auto val = n * boundaryGradientLookup(float_u<SI::m>(POS.val.w), H, vol, arrays);
  if (POS.val.w < 1e20f)
    return val;
  else 
    return val * 0.f;
}
template <typename T> hostDeviceInline auto xBar(float4_u<SI::m> position, T &arrays) {
  auto volumeb = basicVolume;
  auto hb = support_from_volume(volumeb);
  auto Hb = hb * kernelSize();
  auto POS = boundary::POSfunction(position, arrays);
  auto values = xbarLookup(float_u<SI::m>(POS.val.w), Hb, volumeb, arrays);
  if (POS.val.w < 1e20f)
    return pair<float_u<SI::recip_2<SI::m>>, float_u<SI::recip<SI::volume>>>{values.first * volumeb,
                                                                             values.second};
  else
    return pair<float_u<SI::recip_2<SI::m>>, float_u<SI::recip<SI::volume>>>{
        float_u<SI::recip_2<SI::m>>{0.f}, float_u<SI::recip<SI::volume>>{0.f}};
}
template <typename T> hostDeviceInline auto count(float4_u<SI::m> position, T &arrays) {
  auto volumeb = basicVolume;
  auto hb = support_from_volume(volumeb);
  auto Hb = hb * kernelSize();
  auto POS = boundary::POSfunction(position, arrays);
  if (POS.val.w < 1e20f)
    return ctrLookup(float_u<SI::m>(POS.val.w), Hb, volumeb, arrays);
  else
    return int32_t(0);
}
} // namespace boundary

namespace volumeBoundary {
#ifdef __CUDA_ARCH__
template <typename T> hostDeviceInline auto volumeDistance(float4_u<SI::m> tp, T &arrays) {
  auto H = float_u<SI::m>(tp.val.w * Kernel<kernel_kind::spline4>::kernel_size());

  for (int32_t i = 0; i < arrays.volumeBoundaryCounter; ++i) {
    float4_u<SI::m> d_min = arrays.volumeBoundaryMin[i];
    float4_u<SI::m> d_max = arrays.volumeBoundaryMax[i];

	if ((d_min.val.x < tp.val.x) && (d_min.val.y < tp.val.y) && (d_min.val.z < tp.val.z) &&
		(d_max.val.x > tp.val.x) && (d_max.val.y > tp.val.y) && (d_max.val.z > tp.val.z)) {
      float4_u<void_unit_ty> d_p = (tp - d_min) / (d_max - d_min);
      float4_u<void_unit_ty> n =
          tex3D<float4>(arrays.volumeBoundaryVolumes[i], d_p.val.x, d_p.val.y, d_p.val.z);
      float d = n.val.w;
      n.val.w = 0.f;
      math::normalize3(n);
      n.val.w = d;

      if (d < H)
        return n;
    }
  }
  return float4_u<>{0.f, 0.f, 0.f, 1e21f};
}
#else
template <typename T> hostDeviceInline auto volumeDistance(float4_u<SI::m>, T&) {
  return float4_u<>{0.f, 0.f, 0.f, 1e21f};
}
#endif
template <typename T> hostDeviceInline auto spline4(float4_u<SI::m> position, T &arrays) {
  auto volumeb = basicVolume;
  auto hb = support_from_volume(volumeb);
  auto Hb = hb * kernelSize();
  auto POS = volumeDistance(position, arrays);
  auto val = boundary::boundaryLookup(float_u<SI::m>(POS.val.w), Hb, volumeb, arrays) * volumeb;
  if (POS.val.w < 1e20f)
    return val;
  else
    return val * 0.f;
}
template <typename T>
hostDeviceInline auto spikyGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
  auto H = float_u<SI::m>(position.val.w * Kernel<kernel_kind::spline4>::kernel_size());
  //auto volumeb = basicVolume;
  //auto hb = support_from_volume(volumeb);
  //auto Hb = hb * kernelSize();
  auto POS = volumeDistance(position, arrays);
  float4_u<> n{POS.val.x, POS.val.y, POS.val.z, 0.f};

  auto val = n * boundary::boundaryPressureGradientLookup(float_u<SI::m>(POS.val.w), H, vol, arrays);
  if (POS.val.w >= H)
    return val * 0.f;
  return val;
}
template <typename T>
hostDeviceInline auto splineGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
  auto H = float_u<SI::m>(position.val.w * Kernel<kernel_kind::spline4>::kernel_size());
  auto volumeb = basicVolume;
  auto hb = support_from_volume(volumeb);
  auto Hb = hb * kernelSize();
  auto POS = volumeDistance(position, arrays);
  float4_u<> n{POS.val.x, POS.val.y, POS.val.z, 0.f};
  auto val = n * boundary::boundaryGradientLookup(float_u<SI::m>(POS.val.w), H, vol, arrays);
  if (POS.val.w >= H)
    return val * 0.f;
  return val;
}
template <typename T> hostDeviceInline auto xBar(float4_u<SI::m> position, T &arrays) {
  auto volumeb = basicVolume;
  auto hb = support_from_volume(volumeb);
  auto Hb = hb * kernelSize();
  auto POS = volumeDistance(position, arrays);
  auto values = boundary::xbarLookup(float_u<SI::m>(POS.val.w), Hb, volumeb, arrays);
  if (POS.val.w < 1e20f)
    return pair<float_u<SI::recip_2<SI::m>>, float_u<SI::recip<SI::volume>>>{values.first * volumeb,
                                                                             values.second};
  else
    return pair<float_u<SI::recip_2<SI::m>>, float_u<SI::recip<SI::volume>>>{
        float_u<SI::recip_2<SI::m>>{0.f}, float_u<SI::recip<SI::volume>>{0.f}};
}
template <typename T> hostDeviceInline auto count(float4_u<SI::m> position, T &arrays) {
  auto volumeb = basicVolume;
  auto hb = support_from_volume(volumeb);
  auto Hb = hb * kernelSize();
  auto POS = volumeDistance(position, arrays);
  if (POS.val.w < 1e20f)
    return boundary::ctrLookup(float_u<SI::m>(POS.val.w), Hb, volumeb, arrays);
  else
    return int32_t(0);
}
} // namespace volumeBoundary

namespace SWH {
template <typename T> hostDeviceInline auto spline4(float4_u<SI::m> position, T &arrays) {
  return volumeBoundary::spline4(position, arrays) + 
          boundary::spline4(position, arrays);
}
template <typename T>
hostDeviceInline auto spikyGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
  return volumeBoundary::spikyGradient(position, vol, arrays) +
         boundary::spikyGradient(position, vol, arrays);
}
template <typename T>
hostDeviceInline auto splineGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
  return volumeBoundary::splineGradient(position, vol, arrays) +
         boundary::splineGradient(position, vol, arrays);
}
template <typename T> hostDeviceInline auto xBar(float4_u<SI::m> position, T &arrays) {
  auto x1 = volumeBoundary::xBar(position, arrays);
  auto x2 = boundary::xBar(position, arrays);
  x1.first += x2.first;
  x1.second += x2.second;
  return x1;
}
template <typename T> hostDeviceInline auto count(float4_u<SI::m> position, T &arrays) {
  return volumeBoundary::count(position, arrays) + boundary::count(position, arrays);
}

} // namespace SWH

enum struct SWHkind{volumes, planes, both};

template <typename T>
struct SWH2{
  T& arrays;
  float4_u<SI::m> position;
  float4_u<> POS;
  float4_u<> POS_vol;
  float_u<SI::m> distance;
  float_u<SI::m> H;
  float_u<SI::volume> vol;
private:
  hostDevice auto boundary_boundaryLookup(float_u<SI::m> distance, float_u<SI::m> H,
    float_u<SI::volume> vol, T &arrays) {
    int32_t idx =
        math::clamp(arrays.boundaryLUTSize -
                        static_cast<int32_t>(
                            ceilf((distance.val + H.val) / H.val * ((float)arrays.boundaryLUTSize + 1) * 0.5f)),
                    0, arrays.boundaryLUTSize - 1);
    if (distance >= H * 0.995f)
      return (decltype(arrays.boundaryLUT[idx] / vol))(0.f);
    else
      return arrays.boundaryLUT[idx] / vol;
  }
  hostDevice auto boundary_boundaryGradientLookup(float_u<SI::m> distance, float_u<SI::m> H,
    float_u<SI::volume> vol, T &arrays) {
    int32_t idx =
        math::clamp(arrays.boundaryLUTSize -
                        static_cast<int32_t>(
                            ceilf((distance.val + H.val) / H.val * ((float)arrays.boundaryLUTSize + 1) * 0.5f)),
                    0, arrays.boundaryLUTSize - 2);
    auto step = H / ((float)arrays.boundaryLUTSize + 1) * 2.f;
    auto a = arrays.boundaryLUT[idx + 1] / vol;
    auto b = arrays.boundaryLUT[idx] / vol;
    if (distance >= H * 0.995f)
      return (decltype((b - a) / step))(0.f);
    else
      return (b - a) / step;
  }
  hostDevice auto boundary_boundaryPressureLookup(float_u<SI::m> distance, float_u<SI::m> H,
    float_u<SI::volume> vol, T &arrays) {
    int32_t idx =
        math::clamp(arrays.boundaryLUTSize -
                        static_cast<int32_t>(
                            ceilf((distance.val + H.val) / H.val * ((float)arrays.boundaryLUTSize + 1) * 0.5f)),
                    0, arrays.boundaryLUTSize - 1);
    if (distance >= H * 0.995f)
      return (decltype(arrays.boundaryPressureLUT[idx] / vol))(0.f);
    else
      return arrays.boundaryPressureLUT[idx] / vol;
  }
  hostDevice auto boundary_boundaryPressureGradientLookup(float_u<SI::m> distance, float_u<SI::m> H,
    float_u<SI::volume> vol, T &arrays) {
    int32_t size = arrays.boundaryLUTSize;
    auto volumeb = PI4O3 * math::power<3>(arrays.radius);
    auto hb = support_from_volume(volumeb);
    auto Hb = hb * kernelSize();
  
    auto step = Hb / ((float)size + 1) * 2.f;
    float_u<SI::m> h_0(1.2517599643209638f);
    auto h_c = H / kernelSize();
    auto ratio = h_0 / h_c;
  
    int32_t idx = math::clamp((int32_t)size - math::castTo<int32_t>(math::floorf((distance + Hb) / step)), 0,
                              (int32_t)size - 1);
    if (distance >= H * 0.995f)
      return (decltype(arrays.boundaryPressureLUT[idx] / vol))(0.f);
    else
      return arrays.boundaryPressureLUT[idx] / vol * ratio;
  }
  hostDevice auto boundary_xbarLookup(float_u<SI::m> distance, float_u<SI::m> H, float_u<SI::volume> vol,
    T &arrays) {
    int32_t size = arrays.boundaryLUTSize;
  
    auto step = H / ((float)size + 1) * 2.f;
    float_u<SI::m> h_0(1.2517599643209638f);
    auto h_c = H / kernelSize();
    auto hratio = h_0 / h_c;
  
    int32_t idx = math::clamp((int32_t)size - math::castTo<int32_t>(math::floorf((distance + H) / step)), 0,
                              (int32_t)size - 1);
  
    return pair<value_unit<float, SI::derived_unit<SI::recip_3<SI::m>, SI::recip_2<SI::m>> /*SI::SI_Unit<SI::Base::m, ratio<-5, 1>>*/>,
                float_u<SI::recip<SI::volume>>>{arrays.xbarLUT[idx] / vol / hratio,
                                                arrays.boundaryLUT[idx] / vol / hratio};
  }
  hostDevice auto boundary_ctrLookup(float_u<SI::m> distance, float_u<SI::m> H, float_u<SI::volume>,
    T &arrays) {
    int32_t idx =
        math::clamp(arrays.boundaryLUTSize -
                        math::castTo<int32_t>(
                            math::ceilf((distance + H) / H * ((float)arrays.boundaryLUTSize + 1) * 0.5f)),
                    0, arrays.boundaryLUTSize - 1);
    return arrays.ctrLUT[idx];
  }
  hostDevice auto boundary_spline4() {
    auto volumeb = basicVolume;
    auto hb = support_from_volume(volumeb);
    auto Hb = hb * kernelSize();
    auto val = boundary_boundaryLookup(float_u<SI::m>(POS.val.w), Hb, volumeb, arrays) * volumeb;
    if (POS.val.w < 1e20f)
      return val;
    else 
      return val * 0.f;
  }
  hostDevice auto boundary_spikyGradient() {
    float4_u<> n{POS.val.x, POS.val.y, POS.val.z, 0.f};
    auto val = n * boundary_boundaryPressureGradientLookup(float_u<SI::m>(POS.val.w), H, vol, arrays);
    if (POS.val.w < 1e20f)
      return val;
    else 
      return val * 0.f;
  }
  hostDevice auto boundary_splineGradient() {
    auto volumeb = basicVolume;
    auto hb = support_from_volume(volumeb);
    auto Hb = hb * kernelSize();
    float4_u<> n{POS.val.x, POS.val.y, POS.val.z, 0.f};
    auto val = n * boundary_boundaryGradientLookup(float_u<SI::m>(POS.val.w), H, vol, arrays);
    if (POS.val.w < 1e20f)
      return val;
    else 
      return val * 0.f;
  }
  hostDevice auto boundary_xBar() {
    auto volumeb = basicVolume;
    auto hb = support_from_volume(volumeb);
    auto Hb = hb * kernelSize();
    auto values = boundary_xbarLookup(float_u<SI::m>(POS.val.w), Hb, volumeb, arrays);
    if (POS.val.w < 1e20f)
      return pair<float_u<SI::recip_2<SI::m>>, float_u<SI::recip<SI::volume>>>{values.first * volumeb,
                                                                               values.second};
    else
      return pair<float_u<SI::recip_2<SI::m>>, float_u<SI::recip<SI::volume>>>{
          float_u<SI::recip_2<SI::m>>{0.f}, float_u<SI::recip<SI::volume>>{0.f}};
  }
  hostDevice auto boundary_count() {
    auto volumeb = basicVolume;
    auto hb = support_from_volume(volumeb);
    auto Hb = hb * kernelSize();
    if (POS.val.w < 1e20f)
      return boundary_ctrLookup(float_u<SI::m>(POS.val.w), Hb, volumeb, arrays);
    else
      return int32_t(0);
  }
  #ifdef __CUDA_ARCH__
  hostDevice auto volumeDistance() {
    auto H = float_u<SI::m>(position.val.w * Kernel<kernel_kind::spline4>::kernel_size());
  
    for (int32_t i = 0; i < arrays.volumeBoundaryCounter; ++i) {
      float4_u<SI::m> d_min = arrays.volumeBoundaryMin[i];
      float4_u<SI::m> d_max = arrays.volumeBoundaryMax[i];
  
    if ((d_min.val.x < position.val.x) && (d_min.val.y < position.val.y) && (d_min.val.z < position.val.z) &&
      (d_max.val.x > position.val.x) && (d_max.val.y > position.val.y) && (d_max.val.z > position.val.z)) {
        float4_u<void_unit_ty> d_p = (position - d_min) / (d_max - d_min);
        float4_u<void_unit_ty> n =
            tex3D<float4>(arrays.volumeBoundaryVolumes[i], d_p.val.x, d_p.val.y, d_p.val.z);
        float d = n.val.w;
        n.val.w = 0.f;
        math::normalize3(n);
        n.val.w = d;
  
        if (d < H){
          POS_vol = n;
          return;
        }
      }
    }
    POS_vol = float4_u<>{0.f, 0.f, 0.f, 1e21f};
  }
  #else
  hostDevice auto volumeDistance() {
    POS_vol = float4_u<>{0.f, 0.f, 0.f, 1e21f};
  }
  #endif
  hostDevice auto volume_spline4() {
    auto volumeb = basicVolume;
    auto hb = support_from_volume(volumeb);
    auto Hb = hb * kernelSize();
    auto val = boundary_boundaryLookup(float_u<SI::m>(POS_vol.val.w), Hb, volumeb, arrays) * volumeb;
    if (POS_vol.val.w < 1e20f)
      return val;
    else
      return val * 0.f;
  }
  hostDevice auto volume_spikyGradient() {
    float4_u<> n{POS_vol.val.x, POS_vol.val.y, POS_vol.val.z, 0.f};
  
    auto val = n * boundary_boundaryPressureGradientLookup(float_u<SI::m>(POS.val.w), H, vol, arrays);
    if (POS.val.w >= H)
      return val * 0.f;
    return val;
  }
  hostDevice auto volume_splineGradient() {
    auto volumeb = basicVolume;
    auto hb = support_from_volume(volumeb);
    auto Hb = hb * kernelSize();
    float4_u<> n{POS_vol.val.x, POS_vol.val.y, POS_vol.val.z, 0.f};
    auto val = n * boundary_boundaryGradientLookup(float_u<SI::m>(POS_vol.val.w), H, vol, arrays);
    if (POS.val.w >= H)
      return val * 0.f;
    return val;
  }
  hostDevice auto volume_xBar() {
    auto volumeb = basicVolume;
    auto hb = support_from_volume(volumeb);
    auto Hb = hb * kernelSize();
    auto values = boundary_xbarLookup(float_u<SI::m>(POS_vol.val.w), Hb, volumeb, arrays);
    if (POS_vol.val.w < 1e20f)
      return pair<float_u<SI::recip_2<SI::m>>, float_u<SI::recip<SI::volume>>>{values.first * volumeb,
                                                                               values.second};
    else
      return pair<float_u<SI::recip_2<SI::m>>, float_u<SI::recip<SI::volume>>>{
          float_u<SI::recip_2<SI::m>>{0.f}, float_u<SI::recip<SI::volume>>{0.f}};
  }
  hostDevice auto volume_count() {
    auto volumeb = basicVolume;
    auto hb = support_from_volume(volumeb);
    auto Hb = hb * kernelSize();
    if (POS_vol.val.w < 1e20f)
      return boundary_ctrLookup(float_u<SI::m>(POS_vol.val.w), Hb, volumeb, arrays);
    else
      return int32_t(0);
  }


  hostDevice float4_u<SI::m> point1Plane(float4_u<void_unit_ty> E, float4_u<SI::m> P) {
    auto d = math::planeDistance(E, P);
    return P - d * E;
  }
  hostDevice float4_u<SI::m> point2Plane(float4_u<void_unit_ty> E0, float4_u<void_unit_ty> E1,
                                               float4_u<SI::m> P) {
    auto E2 = math::cross(E0, E1);
    auto det = math::sqlength3(E2);
    float4_u<SI::m> linePoint((((math::cross(E2, E1) * E0.val.w) + (math::cross(E0, E2) * E1.val.w)) / det).val);
    auto lineDirection = math::normalize3(E2);
    auto diff = P - linePoint;
    auto distance = math::dot3(diff, lineDirection);
    return linePoint + lineDirection * distance;
  }
  hostDevice float4_u<SI::m> point3Plane(float4_u<void_unit_ty> E0, float4_u<void_unit_ty> E1,
                                               float4_u<void_unit_ty> E2, float4_u<SI::m>) {
    auto a = -E0.val.w * math::cross(E1, E2);
    auto b = -E1.val.w * math::cross(E2, E0);
    auto c = -E2.val.w * math::cross(E0, E1);
    auto d = math::dot3(E0, math::cross(E1, E2));
    return float4_u<SI::m>(((a + b + c) / d).val);
  }
  hostDevice void POSfunction() {
    auto volume = PI4O3 * math::power<3>(arrays.radius);
    auto h = support_from_volume(volume);
    auto H = h * kernelSize();
  
    float4_u<void_unit_ty> e0, e1, e2;
    int32_t counter = 0;
    for (int32_t it = 0; it < arrays.boundaryCounter; ++it) {
      auto plane = arrays.boundaryPlanes[it];
      if (math::planeDistance(plane, position) < H) {
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
      c = point1Plane(e0, position);
      Hn = e0;
      break;
    case 2:
      c = point2Plane(e0, e1, position);
      Hn = e0 + e1;
      break;
    case 3:
      c = point3Plane(e0, e1, e2, position);
      Hn = e0 + e1 + e2;
      break;
    default:
    POS = float4_u<>(0.f, 0.f, 0.f, 1e21f);
      distance = 1e21f;
      return;
    }
    auto Hp = c + Hn * H;
    auto diff = Hp - position;
    //auto diffL = math::length3(diff);
    auto Hd = math::normalize3(diff);
    auto pos = Hp - H * Hd;
  
    auto plane = Hd;
    plane.val.w = -math::dot3(pos, plane).val;
    auto distance = math::planeDistance(plane, position);
    plane.val.w = distance.val;
    POS = plane;
    distance = math::getValue(math::get<4>(plane));
  }

public:
  hostDevice SWH2(T& arrays_a, float4_u<SI::m> p_a, float_u<SI::volume> vol_a):arrays(arrays_a), position(p_a), vol(vol_a){
    POSfunction();
    volumeDistance();
    H = float_u<SI::m>(position.val.w * Kernel<kernel_kind::spline4>::kernel_size());
  }
  hostDevice auto spline4(SWHkind kind = SWHkind::both) {
    if(kind == SWHkind::volumes)
      return volume_spline4();
    if(kind == SWHkind::planes)
      return boundary_spline4();
    return volume_spline4() + boundary_spline4();
  }
  hostDevice auto spikyGradient(SWHkind kind = SWHkind::both) {
    if(kind == SWHkind::volumes)
      return volume_spikyGradient();
    if(kind == SWHkind::planes)
      return boundary_spikyGradient();
    return volume_spikyGradient() + boundary_spikyGradient();
  }
  hostDevice auto splineGradient(SWHkind kind = SWHkind::both) {
    if(kind == SWHkind::volumes)
      return volume_splineGradient();
    if(kind == SWHkind::planes)
      return boundary_splineGradient();
    return volume_splineGradient() + boundary_splineGradient();
  }
  hostDevice auto xBar(SWHkind kind = SWHkind::both) {
    auto x1 = volume_xBar();
    auto x2 = boundary_xBar();
    if(kind == SWHkind::both){
      x1.first += x2.first;
      x1.second += x2.second;
      return x1;
    }
    if(kind == SWHkind::volumes)
      return x1;
    return x2;
  }
  hostDevice auto count(SWHkind kind = SWHkind::both) {
    if(kind == SWHkind::volumes)
      return volume_count();
    if(kind == SWHkind::planes)
      return boundary_count();
    return volume_count() + boundary_count();
  }
};


// namespace math {
// hostDeviceInline float4_u<SI::m> point1Plane(float4_u<void_unit_ty> E, float4_u<SI::m> P) {
//   auto d = math::planeDistance(E, P);
//   return P - d * E;
// }
// hostDeviceInline float4_u<SI::m> point2Plane(float4_u<void_unit_ty> E0, float4_u<void_unit_ty> E1,
//                                              float4_u<SI::m> P) {
//   auto E2 = math::cross(E0, E1);
//   auto det = math::sqlength3(E2);
//   float4_u<SI::m> linePoint((((math::cross(E2, E1) * E0.val.w) + (math::cross(E0, E2) * E1.val.w)) / det).val);
//   auto lineDirection = math::normalize3(E2);
//   auto diff = P - linePoint;
//   auto distance = math::dot3(diff, lineDirection);
//   return linePoint + lineDirection * distance;
// }
// hostDeviceInline float4_u<SI::m> point3Plane(float4_u<void_unit_ty> E0, float4_u<void_unit_ty> E1,
//                                              float4_u<void_unit_ty> E2, float4_u<SI::m>) {
//   auto a = -E0.val.w * math::cross(E1, E2);
//   auto b = -E1.val.w * math::cross(E2, E0);
//   auto c = -E2.val.w * math::cross(E0, E1);
//   auto d = math::dot3(E0, math::cross(E1, E2));
//   return float4_u<SI::m>(((a + b + c) / d).val);
// }
// } // namespace math


// namespace boundary {
// template <typename T> hostDeviceInline float4_u<void_unit_ty> POSfunction(float4_u<SI::m> position, T &arrays) {
//   SWH2<T> swh(arrays,position,basicVolume);
//   return swh.POS;
// }
// template <typename T> hostDeviceInline auto spline4(float4_u<SI::m> position, T &arrays) {
//   SWH2<T> swh(arrays,position,basicVolume);
//   return swh.spline4(SWHkind::planes);
// }
// template <typename T> hostDeviceInline auto spikyGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
//   SWH2<T> swh(arrays,position,vol);
//   return swh.spikyGradient(SWHkind::planes);
// }
// template <typename T> hostDeviceInline auto splineGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
//   SWH2<T> swh(arrays,position,vol);
//   return swh.splineGradient(SWHkind::planes);
// }
// template <typename T> hostDeviceInline auto xBar(float4_u<SI::m> position, T &arrays) {
//   SWH2<T> swh(arrays,position,basicVolume);
//   return swh.xBar(SWHkind::planes);
// }
// template <typename T> hostDeviceInline auto count(float4_u<SI::m> position, T &arrays) {
//   SWH2<T> swh(arrays,position,basicVolume);
//   return swh.count(SWHkind::planes);
// }
// } // namespace boundary

// namespace volumeBoundary {
// template <typename T> hostDeviceInline auto volumeDistance(float4_u<SI::m> tp, T &arrays) {
//   SWH2<T> swh(arrays,tp,basicVolume);
//   return swh.POS_vol;
// }
// template <typename T> hostDeviceInline auto spline4(float4_u<SI::m> position, T &arrays) {
//   SWH2<T> swh(arrays,position,basicVolume);
//   return swh.spline4(SWHkind::volumes);
// }
// template <typename T>
// hostDeviceInline auto spikyGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
//   SWH2<T> swh(arrays,position,vol);
//   return swh.spikyGradient(SWHkind::volumes);
// }
// template <typename T>
// hostDeviceInline auto splineGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
//   SWH2<T> swh(arrays,position,vol);
//   return swh.splineGradient(SWHkind::volumes);
// }
// template <typename T> hostDeviceInline auto xBar(float4_u<SI::m> position, T &arrays) {
//   SWH2<T> swh(arrays,position,basicVolume);
//   return swh.xBar(SWHkind::volumes);
// }
// template <typename T> hostDeviceInline auto count(float4_u<SI::m> position, T &arrays) {
//   SWH2<T> swh(arrays,position,basicVolume);
//   return swh.count(SWHkind::volumes);
// }
// } // namespace volumeBoundary

// namespace SWH {
// template <typename T> hostDeviceInline auto spline4(float4_u<SI::m> position, T &arrays) {
//   SWH2<T> swh(arrays,position,basicVolume);
//   return swh.spline4(SWHkind::both);
// }
// template <typename T>
// hostDeviceInline auto spikyGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
//   SWH2<T> swh(arrays,position,vol);
//   return swh.spikyGradient(SWHkind::both);
// }
// template <typename T>
// hostDeviceInline auto splineGradient(float4_u<SI::m> position, float_u<SI::volume> vol, T &arrays) {
//   SWH2<T> swh(arrays,position,vol);
//   return swh.splineGradient(SWHkind::both);
// }
// template <typename T> hostDeviceInline auto xBar(float4_u<SI::m> position, T &arrays) {
//   SWH2<T> swh(arrays,position,basicVolume);
//   return swh.xBar(SWHkind::both);
// }
// template <typename T> hostDeviceInline auto count(float4_u<SI::m> position, T &arrays) {
//   SWH2<T> swh(arrays,position,basicVolume);
//   return swh.count(SWHkind::both);
// }

// } // namespace SWH

#define BoundaryPressureGradient(x) (SWH::spikyGradient(x, arrays.volume[i], arrays))
#define BoundaryGradient(x) (SWH::splineGradient(x, arrays.volume[i], arrays))
#define BoundaryKernel(x) (SWH::spline4(x, arrays) / arrays.volume[i])
#define BoundaryParticleCount(x) (SWH::count(x, arrays))
#define BoundaryXBar(x) (SWH::xBar(x, arrays))

#define GPW_ib (SWH::spikyGradient(pos[i], arrays.volume[i], arrays))
#define GW_ib (SWH::splineGradient(pos[i], arrays.volume[i], arrays))
#define W_ib (SWH::spline4(pos[i], arrays) / arrays.volume[i])