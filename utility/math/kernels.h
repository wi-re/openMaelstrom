#pragma once
#include <utility/math.h>
#include <utility/unit_math.h>
enum kernel_kind { spline4 = 1, cohesion };

template <typename T, typename U> struct support_helper {
  hostDeviceInline static auto calculate_support(T a, U b) {
#ifdef NON_SYMMETRIC_KERNELS
    return a.w;
#else
    return (a.w + b.w) * 0.5f;
#endif
  }
};

template <typename T1, typename U1, typename T2>
struct support_helper<value_unit<T1, U1>, value_unit<T2, U1>> {
  hostDeviceInline static auto calculate_support(value_unit<T1, U1> a, value_unit<T2, U1> b) {
#ifdef NON_SYMMETRIC_KERNELS
    return value_unit<std::decay_t<decltype(a.val.w)>, U1>{(a.val.w)};
#else
    return value_unit<std::decay_t<decltype((a.val.w + b.val.w) * 0.5f)>, U1>{(a.val.w + b.val.w) *
                                                                              0.5f};
#endif
  }
};

template <typename T, typename U> constexpr hostDeviceInline auto support(T a, U b) {
  return support_helper<T, U>::calculate_support(a, b);
}

#define GENERIC_MEMBERS                                                                            \
  template <typename T, typename U> hostDeviceInline static auto gradient(T a, U b) {              \
    auto difference_vector = (a - b);                                                              \
    auto h = support_helper<T, U>::calculate_support(a, b);                                        \
    auto r = math::length3(difference_vector);                                                     \
    using res = decltype(difference_vector * derivative_impl(r, h) / r);                           \
    if (r < 1e-12f || r > kernel_size() * h)                                                       \
      return res{0.f, 0.f, 0.f, 0.f};                                                              \
    return difference_vector * derivative_impl(r, h) / r;                                          \
  }                                                                                                \
  template <typename T, typename U> hostDeviceInline static auto derivative(T a, U b) {            \
    auto difference_vector = (a - b);                                                              \
    auto h = support_helper<T, U>::calculate_support(a, b);                                        \
    auto r = math::length3(difference_vector);                                                     \
    using res = decltype(derivative_impl(r, h));                                                   \
    if (r < 1e-12f || r > kernel_size() * h)                                                       \
      return res{0.f};                                                                             \
    return derivative_impl(r, h);                                                                  \
  }                                                                                                \
                                                                                                   \
  template <typename T, typename U> hostDeviceInline static auto norm_derivative(T a, U b) {       \
    auto difference_vector = (a - b);                                                              \
    auto h = support_helper<T, U>::calculate_support(a, b);                                        \
    auto r = math::length3(difference_vector);                                                     \
    using res = decltype(difference_vector * derivative_impl(r, h) / r);                           \
    if (r < 1e-12f || r > kernel_size() * h)                                                       \
      return res{0.f};                                                                             \
    return derivative_impl(r, h) / r;                                                              \
  }

#define KERNEL(name, size, neighbors)                                                              \
                                                                                                   \
  template <> struct Kernel<name> {                                                                \
                                                                                                   \
  public:                                                                                          \
    static const int32_t neighbor_number = neighbors;                                              \
    template <typename T = float> hostDeviceInline static T kernel_size() { return float(size); }  \
    GENERIC_MEMBERS

template <kernel_kind K = kernel_kind::spline4> struct Kernel;

template <kernel_kind K = kernel_kind::spline4, typename T>
hostDeviceInline auto support_from_volume(T volume) {
  auto target_neighbors = Kernel<K>::neighbor_number;
  auto kernel_epsilon =
      (1.f) * powf((target_neighbors)*PI4O3_1, 1.f / 3.f) / Kernel<K>::kernel_size();
  auto h = kernel_epsilon * math::power<ratio<1, 3>>(volume);
  return h;
}

KERNEL(kernel_kind::spline4, 1.825742f, 50)
private:
template <typename T, typename U> hostDeviceInline static auto derivative_impl(T r, U half) {
  auto H = half * kernel_size();
  auto q = r / H;
  auto C = 16.f / CUDART_PI_F;
  auto kernel_scaling = C * math::power<-4>(H);
  auto kernel_value = 0.f;

  if (q <= 0.5f) {
    auto q1 = 1.f - q;
    auto q2 = 0.5f - q;
    kernel_value = -3.f * q1 * q1 + 12.f * q2 * q2;
  } else if ((q <= 1.0f) && (q > 0.5f)) {
    auto q1 = 1.f - q;
    kernel_value = -3.f * (q1 * q1);
  }
  return kernel_value * kernel_scaling;
}

public:
template <typename T, typename U> hostDeviceInline static auto value(T a, U b) {
  auto difference = a - b;
  auto r = math::length3(difference);
  auto half = support_helper<T, U>::calculate_support(a, b);

  auto H = half * kernel_size();
  auto q = r / H;
  auto C = 16.f / CUDART_PI_F;
  auto kernel_scaling = C / (H * H * H);
  auto kernel_value = 0.f;
  if (q <= 0.5f) {
    auto q1 = 1.f - q;
    auto q2 = 0.5f - q;
    kernel_value = (q1 * q1 * q1) - 4.f * (q2 * q2 * q2);
  } else if ((q <= 1.0f) && (q > 0.5f)) {
    auto q1 = 1.f - q;
    kernel_value = q1 * q1 * q1;
  }
  return kernel_value * kernel_scaling;
}
}
;

template <> struct Kernel<kernel_kind::cohesion> {

  template <kernel_kind reference_kernel = kernel_kind::spline4>
  hostDeviceInline static float kernel_size() {
    return Kernel<reference_kernel>::kernel_size();
  }

  template <kernel_kind reference_kernel = kernel_kind::spline4, typename T, typename U>
  hostDeviceInline static auto value(T a, U b) {
    auto difference = a - b;
    auto r = math::length3(difference);
    auto half = support(a, b);
    auto h = half * Kernel<reference_kernel>::kernel_size();

    const auto sphere_normalization = 32.f / CUDART_PI_F;
    auto h_1 = math::power<-9>(h);

    auto p = math::cubic(h - r) * math::cubic(r);
    decltype(p) spline{0.f};
    if (2.f * r > h && r <= h) {
      spline = p;
    } else if (r > 0.f && 2.f * r <= h) {
      spline = 2.f * p - math::cubic(h) * math::cubic(h) / 64.f;
    }

    auto result = spline * sphere_normalization * h_1 * difference;
    using res = decltype(result / r);
    if (r < 1e-12f || r > h)
      return res{0.f, 0.f, 0.f, 0.f};

    return result / r;
  }
};

template <kernel_kind kernel> struct PressureKernel;

#define LINK_TO_SPIKY(kernel)                                                                      \
  template <> struct PressureKernel<kernel> {                                                      \
    template <typename T, typename U> hostDeviceInline static auto gradient(T a, U b) {            \
      return SpikyKernel<kernel>::gradient(a, b);                                                  \
    }                                                                                              \
                                                                                                   \
    template <typename T, typename U> hostDeviceInline static auto value(T a, U b) {               \
      return SpikyKernel<kernel>::value(a, b);                                                     \
    }                                                                                              \
                                                                                                   \
    hostDeviceInline static auto kernel_size() { return Kernel<kernel>::kernel_size(); }           \
  };

#define NO_LINK_TO_SPIKY(kernel)                                                                   \
  template <> struct PressureKernel<kernel> {                                                      \
    template <typename T, typename U> hostDeviceInline static auto gradient(T a, U b) {            \
      return Kernel<kernel>::gradient(a, b);                                                       \
    }                                                                                              \
                                                                                                   \
    template <typename T, typename U> hostDeviceInline static auto value(T a, U b) {               \
      return Kernel<kernel>::value(a, b);                                                          \
    }                                                                                              \
                                                                                                   \
    hostDeviceInline static auto kernel_size() { return Kernel<kernel>::kernel_size(); }           \
  };

template <kernel_kind reference_kernel> struct SpikyKernel {
  static const int32_t neighbor_number = 15;

  template <typename T = float> hostDeviceInline static T kernel_size() { return float(2.f); }

  template <typename T, typename U> hostDeviceInline static auto gradient(T a, U b) {
    auto difference = ((a - b));
    auto h = support(a, b) * Kernel<reference_kernel>::kernel_size();
    decltype(h) r = math::length3(difference);
    // support constant
    auto factor = -45.f / (CUDART_PI_F);
    // distance based part
    decltype(h) p = h - r;

    auto c = factor * math::power<-6>(h);

    auto spline = p * p / r * c;

    auto result = difference * spline;

    using res = decltype(result);

    if (r < 1e-12f || r > h)
      return res{0.f, 0.f, 0.f, 0.f};
    return result;
  }

  template <typename T, typename U> hostDeviceInline static auto value(T a, U b) {
    auto difference = a - b;
    auto r = math::length3(difference);
    auto half = support(a, b);
    auto h = half * kernel_size();

    auto h2 = h * h;

    // support constant
    auto c = -45.f / (CUDART_PI_F * h2 * h2 * h2);

    // distance based part
    auto p = h - r;

    auto spline = c * p * p;

    if (r < 1e-12f)
      return decltype(spline){0.f};
    return spline;
  }
};

LINK_TO_SPIKY(kernel_kind::spline4);

template <typename T, typename U> hostDeviceInline auto spline4_kernel(T a, U b) {
  return Kernel<kernel_kind::spline4>::value(a, b);
}
template <typename T, typename U> hostDeviceInline auto spline4_gradient(T a, U b) {
  return Kernel<kernel_kind::spline4>::gradient(a, b);
}
template <typename T, typename U, kernel_kind K = kernel_kind::spline4>
hostDeviceInline auto kernel(T a, U b) {
  return Kernel<K>::value(a, b);
}
template <typename T, typename U, kernel_kind K = kernel_kind::spline4>
hostDeviceInline auto kernelDerivative(T a, U b) {
  return Kernel<K>::derivative(a, b);
}
template <typename T, typename U, kernel_kind K = kernel_kind::spline4>
hostDeviceInline auto gradient(T a, U b) {
  return Kernel<K>::gradient(a, b);
}
template <typename T, typename U, kernel_kind K = kernel_kind::spline4>
hostDeviceInline auto spikyGradient(T a, U b) {
  return PressureKernel<K>::gradient(a, b);
}
template <kernel_kind K = kernel_kind::spline4> hostDeviceInline auto kernelSize() {
  return Kernel<K>::kernel_size();
}
template <kernel_kind K = kernel_kind::spline4> hostDeviceInline auto kernelNeighbors() {
  return Kernel<K>::neighbor_number;
}

#define W_ij spline4_kernel(pos[i], pos[j])
#define W_ji spline4_kernel(pos[j], pos[i])
#define W_ij_UNCACHED spline4_kernel(arrays.position[i], arrays.position[j])
#define W_ji_UNCACHED spline4_kernel(arrays.position[j], arrays.position[i])

#define GW_ij spline4_gradient(pos[i], pos[j])
#define GW_ji spline4_gradient(pos[j], pos[i])
#define GW_ij_UNCACHED spline4_gradient(arrays.position[i], arrays.position[j])
#define GW_ji_UNCACHED spline4_gradient(arrays.position[j], arrays.position[i])

#define GPW_ij PressureKernel<kernel_kind::spline4>::gradient(pos[i], pos[j])
#define GPW_ji PressureKernel<kernel_kind::spline4>::gradient(pos[j], pos[i])
