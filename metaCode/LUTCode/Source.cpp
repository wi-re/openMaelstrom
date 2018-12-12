#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <utility/include_all.h>
#include <utility/iterator.h>
#include <utility/math.h>
#include <utility/unit_math.h>
#include <vector>
  
#ifndef TESTING
struct virtual_particles {
  int32_t requiredSlices_x, requiredSlices_y, requiredSlices_z;
  float r;
  value_unit<float4, SI::m> p;
  value_unit<float4, SI::m> c;
  bool valid = false;

  template <typename V, typename U> virtual_particles(U &&position, V &&volume, float spacing) {
    p = position;
    c = position; // float4_u<SI::m>(0., 0., 0., p.val.w);
    float radius = pow(volume.val / (4.f / 3.f * CUDART_PI_F), 1.f / 3.f);
    float h = p.val.w * Kernel<kernel_kind::spline4>::kernel_size();
    valid = true;

    r = spacing * radius;

    auto k = static_cast<int32_t>(floor(c.val.z / r * 3. / (2. * sqrt(6.))));
    auto j = static_cast<int32_t>(floor(c.val.y / r / sqrt(3.) - 1. / 3. * (k % 2)));
    auto i = static_cast<int32_t>(floor((c.val.x / r - (j + k) % 2) / 2.));
    c = value_unit<float4, SI::m>{2.f * i + ((j + k) % 2), sqrtf(3.f) * (j + 1.f / 3.f * (k % 2)),
                                  2.f * sqrtf(6.f) / 3.f * k, 0.f};
    c = value_unit<float4, SI::m>{c.val.x * r, c.val.y * r, c.val.z * r, 0.f};
    // std::cout << c.val.x << ", " << c.val.y << ", " << c.val.z << std::endl;
    c = value_unit<float4, SI::m>{0.f, 0.f, 0.f, 0.f};
    requiredSlices_x = 2 * static_cast<int32_t>(ceil(h / r));
    requiredSlices_y = 2 * static_cast<int32_t>(ceil(h / (sqrtf(3.f) * r)));
    requiredSlices_z = 2 * static_cast<int32_t>(ceil(h / r * 3.f / (sqrtf(6.f) * 2.f)));
  }
  struct neighbor_it {
    const int32_t requiredSlices_x, requiredSlices_y, requiredSlices_z;
    const float r;
    value_unit<float4, SI::m> p;
    value_unit<float4, SI::m> s;
    value_unit<float4, SI::m> c;
    int32_t i, j, k;

    neighbor_it(int32_t x_len, int32_t y_len, int32_t z_len, float r, value_unit<float4, SI::m> pos,
                value_unit<float4, SI::m> center, int32_t _x, int32_t _y, int32_t _z)
        : requiredSlices_x(x_len), requiredSlices_y(y_len), requiredSlices_z(z_len), r(r), p(pos), c(center), i(_x),
          j(_y), k(_z) {
      increment();
    }

    value_unit<float4, SI::m> operator*() { return s; }
    bool operator==(const neighbor_it &rawIterator) const { return (k == rawIterator.k); }
    bool operator!=(const neighbor_it &rawIterator) const { return (k != rawIterator.k); }

    void increment() {
      //using namespace math::ops;
      bool in_range = false;
      do {
        i++;
        if (i == requiredSlices_x + 1) {
          i = -requiredSlices_x;
          j++;
          if (j == requiredSlices_y + 1) {
            j = -requiredSlices_y;
            k++;
          }
        }
        float4 initial{2.f * i + ((j + k) % 2), sqrtf(3.f) * (j + 1.f / 3.f * (k % 2)), 2.f * sqrtf(6.f) / 3.f * k,
                       0.f};

        s = value_unit<float4, SI::m>{c.val.x + initial.x * r, c.val.y + initial.y * r, c.val.z + initial.z * r,
                                      p.val.w};
        //float dist = math::sqdistance3(p.val, s.val);
        //dist = sqrt(dist);
        in_range = true; // dist < (p.val.w *
                         // Kernel<kernel_kind::spline4>::kernel_size());
                         // s.val.w = 1. - d / (p.val.w *
        // Kernel<kernel_kind::spline4>::kernel_size());
      } while (k != requiredSlices_z + 1 && !(in_range));
    }

    neighbor_it &operator++() {
      increment();
      return (*this);
    }
    const neighbor_it operator++(int) {
      auto temp(*this);
      increment();
      return temp;
    }
  };
  neighbor_it begin() const {
    return neighbor_it{requiredSlices_x,
                       requiredSlices_y,
                       requiredSlices_z,
                       r,
                       p,
                       c,
                       -requiredSlices_x,
                       -requiredSlices_y,
                       !valid ? requiredSlices_z + 1 : -requiredSlices_z};
  }
  neighbor_it end() const {
    return neighbor_it{requiredSlices_x, requiredSlices_y, requiredSlices_z,    r, p, c,
                       requiredSlices_x, requiredSlices_y, requiredSlices_z + 1};
  }
  neighbor_it cbegin() const {
    return neighbor_it{requiredSlices_x,
                       requiredSlices_y,
                       requiredSlices_z,
                       r,
                       p,
                       c,
                       -requiredSlices_x,
                       -requiredSlices_y,
                       !valid ? requiredSlices_z + 1 : -requiredSlices_z};
  }
  neighbor_it cend() const {
    return neighbor_it{requiredSlices_x, requiredSlices_y, requiredSlices_z,    r, p, c,
                       requiredSlices_x, requiredSlices_y, requiredSlices_z + 1};
  }
};

template <typename T> auto print_val(const std::string& str, T val) {
  std::cout << std::setw(32) << str << " = " << std::setw(24)
            << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << val.val << " -> " << std::hexfloat
            << val.val << std::defaultfloat << std::endl;
};
template <> auto print_val<float>(const std::string& str, float val) {
  std::cout << std::setw(32) << str << " = " << std::setw(24)
            << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << val << " -> " << std::hexfloat
            << val << std::defaultfloat << std::endl;
};
template <> auto print_val<int32_t>(const std::string& str, int32_t val) {
  std::cout << std::setw(32) << str << " = " << val << std::endl;
};

#define PRINT(x) print_val(#x, x);

template <kernel_kind K = kernel_kind::spline4, typename T> hostDeviceInline auto supportFromVol(T volume) {
  auto target_neighbors = Kernel<K>::neighbor_number;
  auto kernel_epsilon = (1.f) * powf((target_neighbors)*PI4O3_1, 1.f / 3.f) / Kernel<K>::kernel_size();
  auto h = kernel_epsilon * math::power<ratio<1, 3>>(volume);
  return h;
}

template <typename C> auto createLUT(C func, int32_t steps, float spacing) {
  float_u<SI::volume> volume(1.f);
  auto h = supportFromVol(volume);
  auto H = h * Kernel<kernel_kind::spline4>::kernel_size<float>();
  auto step = H / (static_cast<float>(steps)) * 2.f;

  auto current_position = float4_u<SI::m>{0.f, 0.f, 0.f, h.val};
  auto center_position = float4_u<SI::m>{0.f, 0.f, 0.f, h.val};

  using res_t = decltype(math::unit_get<1>(func(current_position, current_position)) * volume);
  std::vector<res_t> LUT;

  for (int32_t i = 0; i < steps; ++i) {
    math::unit_assign<1>(current_position, H - step * static_cast<float>(i));
    res_t density{0.f};
    for (const auto &var : virtual_particles(center_position, volume, spacing)) {
      if (math::unit_get<1>(var) > 0._m){
        continue;
	  }
      density += volume * math::unit_get<1>(func(current_position, var));
    }
    LUT.push_back(density);
  }
  return LUT;
}

auto createCounterLUT(int32_t steps, float spacing) {
  float_u<SI::volume> volume(1.f);
  auto h = supportFromVol(volume);
  auto H = h * Kernel<kernel_kind::spline4>::kernel_size<float>();
  auto step = H / (static_cast<float>(steps)) * 2.f;

  auto current_position = float4_u<SI::m>{0.f, 0.f, 0.f, h.val};
  auto center_position = float4_u<SI::m>{0.f, 0.f, 0.f, h.val};

  std::vector<int32_t> LUT;

  for (int32_t i = 0; i < steps; ++i) {
    math::unit_assign<1>(current_position, H - step * static_cast<float>(i));
    int32_t counter = 0;
    for (const auto &var : virtual_particles(center_position, volume, spacing)) {
      if (math::unit_get<1>(var) > 0._m){
        continue;
	  }
      if (spline4_kernel(current_position, var) > 0.f){
        counter++;
	  }
    }
    LUT.push_back(counter);
  }
  return LUT;
}

template <typename C> auto create4DLUT(C func, int32_t steps, float spacing) {
  float_u<SI::volume> volume(1.f);
  auto h = supportFromVol(volume);
  auto H = h * Kernel<kernel_kind::spline4>::kernel_size<float>();
  auto step = H / (static_cast<float>(steps)) * 2.f;

  auto current_position = float4_u<SI::m>{0.f, 0.f, 0.f, h.val};
  auto center_position = float4_u<SI::m>{0.f, 0.f, 0.f, h.val};

  using res_t = decltype((func(current_position, current_position)) * volume);
  std::vector<res_t> LUT;

  for (int32_t i = 0; i < steps; ++i) {
    math::unit_assign<1>(current_position, H - step * static_cast<float>(i));
    res_t density{0.f};
    for (const auto &var : virtual_particles(center_position, volume, spacing)) {
      if (math::unit_get<1>(var) > 0._m){
        continue;
	  }
      density += volume * (func(current_position, var));
    }
    LUT.push_back(density);
  }
  return LUT;
}

template <typename T> auto printLUT(const std::string& name, const std::string& type, T LUT) {
  std::cout << "std::vector<" << type << "> " << name << "{ ";
  for (auto v : LUT) {
    std::cout << std::scientific << std::setprecision(std::numeric_limits<float>::digits10 + 1)
              << static_cast<float>(math::unit_get<1>(v).val) << "f, ";
  }
  std::cout << "};" << std::endl;
}

auto printLUT(const std::string& name, const std::vector<int32_t>& LUT) {
  std::cout << "std::vector<int32_t> " << name << "{ ";
  for (auto v : LUT) {
    std::cout << v << ", ";
  }
  std::cout << "};" << std::endl;
}
#include <config/config.h>
#include <fstream>
#ifdef _WIN32
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif

template <typename T> auto writeLUT(const std::string& name, const std::string& type, const std::vector<T>& LUT) {
  fs::path bin_dir(binaryDirectory);
  auto file = bin_dir / "config" / name;
  file.replace_extension("h");

  if (fs::exists(file)) {
    if (fs::exists(__FILE__)) {
      auto input_ts = fs::last_write_time(__FILE__);
      auto output_ts = fs::last_write_time(file);
      if (input_ts <= output_ts){
        return;
	  }
    }
  }
  std::cout << "Writing " << file.string() << std::endl;

  std::ofstream output(file.string());
  output << "std::vector<" << type << "> " << name << "{ ";
  int32_t ctr = 0;
  for (auto v : LUT) {
    output << std::scientific << std::setprecision(std::numeric_limits<float>::digits10 + 1)
           << static_cast<float>(math::unit_get<1>(v).val) << "f, " << (ctr++ % 5 == 0 ? "\n" : "");
  }
  output << "};" << std::endl;
  output.close();
}
auto writeLUT(const std::string& name, const std::vector<int32_t>& LUT) {
  fs::path bin_dir(binaryDirectory);
  auto file = bin_dir / "config" / name;
  file.replace_extension("h");
  if (fs::exists(file)) {
    if (fs::exists(__FILE__)) {
      auto input_ts = fs::last_write_time(__FILE__);
      auto output_ts = fs::last_write_time(file);
      if (input_ts <= output_ts){
        return;
	  }
    }
  }
  std::cout << "Writing " << file.string() << std::endl;
  std::ofstream output(file.string());
  int32_t ctr = 0;
  output << "std::vector<int32_t> " << name << "{ ";
  for (auto v : LUT) {
    output << v << ", " << (ctr++ % 10 == 0 ? "\n" : "");
  }
  output << "};" << std::endl;
  output.close();
}

int main() {
  std::cout << "Running LUT generation program" << std::endl;
  using pos_t = float4_u<SI::m>;
  bool test_spline4Kernel_LUT = false;
  bool test_spline4Gradient_LUT = false;
  bool test_estimateSpline4Gradient_LUT = false;
  bool test_testSpikyGradient_LUT = false;

  float_u<SI::volume> volume(1.f);
  auto h = supportFromVol(volume);
  auto H = h * Kernel<kernel_kind::spline4>::kernel_size<float>();

  auto position = float4_u<SI::m>{H.val / 2.f, 0.f, 0.f, h.val};

  float r = math::brentsMethod(
      [=](float_u<> radius) {
        float_u<> density = -1.f;
        for (const auto &var : virtual_particles(position, volume, math::getValue(radius))) {
          auto kernel = spline4_kernel(position, var);
          density += volume * kernel;
        }
        return density;
      },
      0.7f, 1.1f, 1e-7f, 10000);

  auto LUT = createLUT([](pos_t a, pos_t b) { return spline4_kernel(a, b); }, 2048, r);
  auto spline4gradientLUT = createLUT([](pos_t a, pos_t b) { return spline4_gradient(a, b); }, 2048, r);
  auto spikygradientLUT =
      createLUT([](pos_t a, pos_t b) { return PressureKernel<kernel_kind::spline4>::gradient(a, b); }, 2048, r);
  auto xbarLUT = createLUT([](pos_t a, pos_t b) { return b * spline4_kernel(a, b); }, 2048, r * 0.9f);
  auto ctrLUT = createCounterLUT(2048, r * 0.9f);

  // printLUT("boundaryLut" , "float", LUT);
  // printLUT("pressureLut", "float", spikygradientLUT);
  // printLUT("xbarLut", "float", xbarLUT);
  // printLUT("ctrLut", ctrLUT);

  writeLUT("boundaryLut", "float", LUT);
  writeLUT("pressureLut", "float", spline4gradientLUT);
  // writeLUT("pressureLut", "float", spikygradientLUT);
  writeLUT("xbarLut", "float", xbarLUT);
  writeLUT("ctrLut", ctrLUT);

  auto lookup = [&](float_u<SI::m> dist, float_u<SI::m> H, float_u<SI::volume> vol, auto LUT) {
    auto step = H / (static_cast<float>(LUT.size()) + 1) * 2.f;
    int32_t idx =
        math::clamp(static_cast<int32_t>(LUT.size()) - math::castTo<int32_t>(floor((dist.val + H.val) / step.val)), 0, static_cast<int32_t>(LUT.size()) - 1);
    return LUT[idx] / vol;
  };
  auto estimateGradient = [&](float_u<SI::m> dist, float_u<SI::m> H, float_u<SI::volume> vol, auto LUT) {
    auto step = H / (static_cast<float>(LUT.size()) + 1) * 2.f;
    int32_t idx =
        math::clamp(static_cast<int32_t>(LUT.size()) - math::castTo<int32_t>(floor((dist.val + H.val) / step.val)), 1, static_cast<int32_t>(LUT.size()) - 2);
    auto a = LUT[idx + 1] / vol;
    auto b = LUT[idx] / vol;
    auto ab = (b - a) / step;
    return ab;
  };
  auto lookupGradient = [&](float_u<SI::m> dist, float_u<SI::m> H, float_u<SI::volume> vol, auto LUT) {
    auto step = H / (static_cast<float>(LUT.size() + 1)) * 2.f;
    auto h_0 = support_from_volume(float_u<SI::volume>(1.f));
    auto h_c = support_from_volume(vol);

    auto ratio = h_0 / h_c;
    int32_t idx =
        math::clamp(static_cast<int32_t>(LUT.size()) - math::castTo<int32_t>(floor((dist.val + H.val) / step.val)), 0, static_cast<int32_t>(LUT.size()) - 1);
    return LUT[idx] / vol * ratio;
  };

  volume /= 10.f;
  h = support_from_volume(volume);
  H = h * kernelSize();
  position = float4_u<SI::m>{0.f, 2.f, 0.f, h.val};

  auto step = H / 32;
  // test kernel LUT
  if (test_spline4Kernel_LUT) {
    std::cout << std::endl
              << "Testing LUT for "
              << "spline4 kernel" << std::endl;
    for (float_u<SI::m> it = H; it >= -H - step; it -= step) {
      float_u<SI::recip_3<SI::m>> kernelSum{0.f};
      math::unit_assign<1>(position, it);
      for (const auto &var : virtual_particles(position, volume, r)) {
        if (math::unit_get<1>(var) > 0.0_m){
          continue;
		}
        kernelSum += kernel(position, var);
      }

      std::cout << std::setw(7) << std::fixed << std::setprecision(4) << it / H;
      std::cout << " -> " << std::fixed << std::setprecision(4) << std::setw(18)
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << math::unit_get<1>(kernelSum).val
                << " : " << std::setw(18) << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << lookup(it, H, volume, LUT).val << " : " << std::setw(18)
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << lookup(it, H, volume, LUT) / math::unit_get<1>(kernelSum) << std::endl;
    }
  }
  if (test_spline4Gradient_LUT) {
    std::cout << std::endl
              << "Testing LUT for "
              << "spline4 gradient" << std::endl;
    for (float_u<SI::m> it = H; it >= -H - step; it -= step) {
      float4_u<SI::multiply_ratios<SI::m, ratio<-4, 1>>> gradientSum{0.f, 0.f, 0.f, 0.f};
      math::unit_assign<1>(position, it);
      for (const auto &var : virtual_particles(position, volume, r)) {
        if (math::unit_get<1>(var) > 0._m){
          continue;
		}
        gradientSum += spline4_gradient(position, var);
      }

      std::cout << std::setw(7) << std::fixed << std::setprecision(4) << it / H;
      std::cout << " -> " << std::fixed << std::setprecision(4) << std::setw(18)
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << math::unit_get<1>(gradientSum).val << " : " << std::setw(18)
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << lookupGradient(it, H, volume, spline4gradientLUT).val << " : " << std::setw(18)
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << lookupGradient(it, H, volume, spline4gradientLUT) / math::unit_get<1>(gradientSum) << std::endl;
    }
  }
  if (test_estimateSpline4Gradient_LUT) {
    std::cout << std::endl
              << "Testing LUT for "
              << "estimating spline4 gradient" << std::endl;
    for (float_u<SI::m> it = H; it >= -H - step; it -= step) {
      float4_u<SI::SI_Unit<SI::m, ratio<-4, 1>>> gradientSum{0.f, 0.f, 0.f, 0.f};
      math::unit_assign<1>(position, it);
      for (const auto &var : virtual_particles(position, volume, r)) {
        if (math::unit_get<1>(var) > 0._m){
          continue;
		}
        gradientSum += spline4_gradient(position, var);
      }

      std::cout << std::setw(7) << std::fixed << std::setprecision(4) << it / H;
      std::cout << " -> " << std::fixed << std::setprecision(4) << std::setw(18)
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << math::unit_get<1>(gradientSum).val << " : " << std::setw(18)
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << estimateGradient(it, H, volume, LUT).val << " : " << std::setw(18)
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << estimateGradient(it, H, volume, LUT) / math::unit_get<1>(gradientSum) << std::endl;
    }
  }
  if (test_testSpikyGradient_LUT) {
    std::cout << std::endl
              << "Testing LUT for "
              << "spiky gradient" << std::endl;
    for (float_u<SI::m> it = H; it >= -H - step; it -= step) {
      float4_u<SI::SI_Unit<SI::m, ratio<-4, 1>>> gradientSum{0.f, 0.f, 0.f, 0.f};
      math::unit_assign<1>(position, it);
      for (const auto &var : virtual_particles(position, volume, r)) {
        if (math::unit_get<1>(var) > 0._m){
          continue;
		}
        gradientSum += PressureKernel<kernel_kind::spline4>::gradient(position, var);
      }

      std::cout << std::setw(7) << std::fixed << std::setprecision(4) << it / H;
      std::cout << " -> " << std::fixed << std::setprecision(4) << std::setw(18)
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << math::unit_get<1>(gradientSum).val << " : " << std::setw(18)
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << lookupGradient(it, H, volume, spikygradientLUT).val << " : " << std::setw(18)
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << lookupGradient(it, H, volume, spikygradientLUT) / math::unit_get<1>(gradientSum) << std::endl;
    }
  }
  // getchar();
}
#else
struct mem {
  int *x, *y, *z, *w;
};

int main() {
  mem arrays;
  cache_arrays((m_01, x));
  cache_arrays((m_02, x), (m_03, y));

  float4_u<SI::m> test_x;
  float4_u<SI::velocity> test_v;
  float_u<SI::s> test_s;
  test_x + test_v *test_s;
  test_x = test_v * test_s;
  std::cout << SI::is_same_unit<decltype(test_x)::unit, decltype(test_v * test_s)::unit>::value << std::endl;

  using Tuple1 = decltype(test_x)::unit;
  using Tuple2 = decltype(test_v * test_s)::unit;

  std::cout << typeid(Tuple1).name() << std::endl;
  std::cout << typeid(Tuple2).name() << std::endl;
  std::cout << typeid(SI::flatten2<Tuple2>::type).name() << std::endl;

  // constexpr auto s1 = std::tuple_size<Tuple1>::value;
  // constexpr auto s2 = std::tuple_size<Tuple2>::value;

  return 0;
}
#endif