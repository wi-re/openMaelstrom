#if defined(__MSC_VER__) && defined(__clang__)
#include <host_defines.h>
#undef __builtin_align__
#define __builtin_align__(x)
#endif
#include <IO/config/config.h> 
#include <IO/config/initialize.h>
#include <IO/config/snapshot.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility/cuda.h>
#include <utility/helpers/log.h>
#include <utility/template/metafunctions.h>
#include <utility/template/tuple_for_each.h>
#include <utility/mathv2.h>

bool IsPrime(int32_t number){
	if (number == 2 || number == 3)
		return true;
	if (number % 2 == 0 || number % 3 == 0)
		return false;
	int divisor = 6;
	while (divisor * divisor - 2 * divisor + 1 <= number){
		if (number % (divisor - 1) == 0)
			return false;
		if (number % (divisor + 1) == 0)
			return false;
		divisor += 6;
	}
	return true;
}

int32_t NextPrime(int32_t a){
	while (!IsPrime(++a)){}
	return a;
}

void IO::config::initParameters() {
  if (get<parameters::sorting>() == "mlm")
    *parameters::sorting::ptr = "MLM";
  if (get<parameters::launch_cfg>() == "cpu") {
    get<parameters::target>() = launch_config::host;
  } else if (get<parameters::launch_cfg>() == "debug") {
    get<parameters::target>() = launch_config::debug;
  } else if (get<parameters::launch_cfg>() == "pure") {
    get<parameters::target>() = launch_config::pure_host;
  } else
    get<parameters::target>() = launch_config::device;

  if (get<parameters::sorting>() == "linear_cell") {
    get<parameters::resort_algorithm>() = 0;
    get<parameters::cell_order>() = cell_ordering::linear_order;
    get<parameters::cell_structure>() = cell_structuring::complete;
    get<parameters::hash_size>() = hash_length::bit_32;
  } else if (get<parameters::sorting>() == "hashed_cell" &&
             get<parameters::hash_width>() == "64bit") {
    get<parameters::resort_algorithm>() = 1;
    get<parameters::cell_order>() = cell_ordering::z_order;
    get<parameters::cell_structure>() = cell_structuring::hashed;
    get<parameters::hash_size>() = hash_length::bit_64;
  } else if (get<parameters::sorting>() == "hashed_cell" &&
             get<parameters::hash_width>() == "32bit") {
    get<parameters::resort_algorithm>() = 2;
    get<parameters::cell_order>() = cell_ordering::z_order;
    get<parameters::cell_structure>() = cell_structuring::hashed;
    get<parameters::hash_size>() = hash_length::bit_32;
  } else if (get<parameters::sorting>() == "MLM" && get<parameters::hash_width>() == "64bit") {
    get<parameters::resort_algorithm>() = 3;
    get<parameters::cell_order>() = cell_ordering::z_order;
    get<parameters::cell_structure>() = cell_structuring::MLM;
    get<parameters::hash_size>() = hash_length::bit_64;
  }
  else if (get<parameters::sorting>() == "MLM" && get<parameters::hash_width>() == "32bit") {
	  get<parameters::resort_algorithm>() = 3;
	  get<parameters::cell_order>() = cell_ordering::z_order;
	  get<parameters::cell_structure>() = cell_structuring::MLM;
	  get<parameters::hash_size>() = hash_length::bit_32;
  }

  if (get<parameters::sorting>() == "MLM" || get<parameters::sorting>() == "hashed_cell") {
    if (parameters::hash_entries{} == UINT_MAX)
      *parameters::hash_entries::ptr = NextPrime(parameters::max_numptcls{});
    if (parameters::mlm_schemes{} == UINT_MAX) {
      if (parameters::adaptive{} == true && get<parameters::sorting>() == "MLM")
        *parameters::mlm_schemes::ptr =
            static_cast<uint32_t>(ceilf(log2f(pow(parameters::adaptive_resolution{}, 1.f / 3.f)))) + 1u;
      else
        *parameters::mlm_schemes::ptr = 1;
    }
  }

  if (get<parameters::neighborhood>() == "constrained") {
    get<parameters::neighborhood_kind>() = neighbor_list::constrained;
  } else if (get<parameters::neighborhood>() == "cell_based") {
    get<parameters::neighborhood_kind>() = neighbor_list::cell_based;
  } else if (get<parameters::neighborhood>() == "basic") {
    get<parameters::neighborhood_kind>() = neighbor_list::basic;
  }
  else if (get<parameters::neighborhood>() == "compactCell") {
	  get<parameters::neighborhood_kind>() = neighbor_list::compactCell;	  
  }

  // timestep
  auto &dt = get<parameters::timestep>();
  if (dt == 0.f)
    dt = (get<parameters::timestep_max>() - get<parameters::timestep_min>()) / 2.f;
  // Init parameters for constrained neighbor list
  uint32_t preferred_count = Kernel<kernel_kind::spline4>::neighbor_number;
  uint32_t count = get<parameters::neighborlimit>() - 1;
  uint32_t leeway = count - preferred_count;
  get<parameters::support_leeway>() = leeway;
  get<parameters::target_neighbors>() = preferred_count;
  get<parameters::overhead_size>() = get<parameters::max_numptcls>() / 10;
  if (get<parameters::neighborhood_kind>() == neighbor_list::constrained ||
      get<parameters::neighborhood_kind>() == neighbor_list::basic) {
    info<arrays::neighborListSwap>().allocate(sizeof(uint32_t));
  }

  
  std::string host = get<parameters::hostRegex>();
  std::string device = get<parameters::deviceRegex>();
  std::string debug = get<parameters::debugRegex>();
  if (host != "" || device != "" || debug != "")
	  get<parameters::regex_cfg>() = true;
}

void IO::config::initBoundary() {
	if (get<parameters::moving_planes>().size() > 0)
		get<parameters::movingBoundaries>() = true;
	if (get<parameters::inlet_volumes>().size() > 0)
		get<parameters::volumeInlets>() = true;
	// implicit boundary parameters
	std::vector<float4> planes;
	float3 max = get<parameters::max_domain>();
	float3 min = get<parameters::min_domain>();
	float r = -get<parameters::radius>();
	{
		char prev = 'x';
		for (auto c : get<parameters::domainWalls>()) {
			if (c == 'x' || c == 'y' || c == 'z')
				prev = c;
			float3 dim;
			float sign = -1.f;
			if (c == '+')
				dim = max - r;
			if (c == '-') {
				dim = -(min + r);
				sign = 1.f;
			}
			if (c == '+' || c == '-') {
				if (prev == 'x')
					planes.push_back(float4{ sign, 0.f, 0.f, dim.x });
				if (prev == 'y')
					planes.push_back(float4{ 0.f, sign, 0.f, dim.y });
				if (prev == 'z')
					planes.push_back(float4{ 0.f, 0.f, sign, dim.z });
			}
		}
	}
  for(auto& plane : get<parameters::moving_planes>()){
    //auto t = plane.duration.value;
    auto f = plane.frequency.value;
    auto m = plane.magnitude.value;
    auto p = plane.plane_position.value;
    auto n = plane.plane_normal.value;
    auto dir = plane.plane_direction.value;
    plane.index.value = (int32_t) planes.size();
    p += dir * m * sinf(2.f * CUDART_PI_F * f * get<parameters::simulationTime>());
    auto nn     = math::normalize(n);
    auto d      = math::dot3(p, nn);
    planes.push_back(float4{nn.x, nn.y, nn.z, d});
  }
	get<parameters::boundaryCounter>() = (int32_t) planes.size();
	arrays::boundaryPlanes::ptr = (float4*)cuda::malloc(sizeof(float4) * planes.size());
	//info<arrays::boundaryPlanes>().allocate(sizeof(float4) * planes.size());
	cuda::memcpy(arrays::boundaryPlanes::ptr, planes.data(), planes.size() * sizeof(float4), cudaMemcpyHostToDevice);
	//for (uint32_t i = 0; i < planes.size(); ++i)
	//	arrays::boundaryPlanes::ptr[i] = planes[i];

#include <config/boundaryLut.h>
#include <config/pressureLut.h>
#include <config/xbarLut.h>
#include <config/ctrLut.h>

        *parameters::boundaryLUTSize::ptr = (int32_t)boundaryLut.size();
        arrays::xbarLUT::ptr = (float*)cuda::malloc(sizeof(float) * boundaryLut.size());
	arrays::boundaryLUT::ptr = (float*)cuda::malloc(sizeof(float) * boundaryLut.size());
	arrays::boundaryPressureLUT::ptr = (float*)cuda::malloc(sizeof(float) * pressureLut.size());
	arrays::ctrLUT::ptr = (int32_t*)cuda::malloc(sizeof(int32_t) * ctrLut.size());
	cuda::memcpy(arrays::xbarLUT::ptr, xbarLut.data(), xbarLut.size() * sizeof(float), cudaMemcpyHostToDevice);
	cuda::memcpy(arrays::boundaryLUT::ptr, boundaryLut.data(), boundaryLut.size() * sizeof(float), cudaMemcpyHostToDevice);
	cuda::memcpy(arrays::boundaryPressureLUT::ptr, pressureLut.data(), pressureLut.size() * sizeof(float), cudaMemcpyHostToDevice);
	cuda::memcpy(arrays::ctrLUT::ptr, ctrLut.data(), ctrLut.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

}

void IO::config::initKernel() {
  float_u<SI::volume> volume(PI4O3 * math::power<3>(get_u<parameters::radius>()));
  auto h = support_from_volume(volume);
  auto H = h * ::kernelSize();

  auto gen_position = [&](float_u<SI::m> r, int32_t i, int32_t j, int32_t k) {
    float4_u<> initial{2.f * i + ((j + k) % 2), sqrtf(3.f) * (j + 1.f / 3.f * (k % 2)),
                       2.f * sqrtf(6.f) / 3.f * k, h / r};
    return initial * r;
  };

  auto radius = get_u<parameters::radius>();

  auto r = math::brentsMethod(
      [=](float_u<SI::m> r) { 
        float_u<> density = -1.0f;
        int32_t requiredSlices_x = (int32_t) math::ceilf(h / r).val;
        int32_t requiredSlices_y = (int32_t) math::ceilf(h / (sqrtf(3.f) * r)).val;
        int32_t requiredSlices_z = (int32_t) math::ceilf(h / r * 3.f / (sqrtf(6.f) * 2.f)).val;
        density = -1.f;
        float4_u<SI::m> center_position{0.f, 0.f, 0.f, h.val};
        for (int32_t x_it = -requiredSlices_x; x_it <= requiredSlices_x; x_it++)
          for (int32_t y_it = -requiredSlices_y; y_it <= requiredSlices_y; y_it++)
            for (int32_t z_it = -requiredSlices_z; z_it <= requiredSlices_z; z_it++)
              density +=
                  volume * spline4_kernel(center_position, gen_position(r, x_it, y_it, z_it));
        return density;
      },
      radius*0.1f, radius*4.f, 1e-6_m, 10000);
  get_u<parameters::ptcl_support>() = h;
  get_u<parameters::cell_size>() = float3_u<SI::m>{H.val, H.val, H.val};
  get_u<parameters::ptcl_spacing>() = r;
}

void IO::config::initDomain() {
  if (get<parameters::boundaryObject>() != "") {
    std::string boundaryFile = get<parameters::config_folder>() + get<parameters::boundaryObject>();

    auto &min = get<parameters::min_domain>();
    auto &max = get<parameters::max_domain>();
    min = vector_t<float, 3>::max();
    max = vector_t<float, 3>::min();

    std::string line;
    std::ifstream file(boundaryFile);
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      char id;
      iss >> id;
      if (id == 'v') {
        float3 vtx;
        iss >> vtx.x >> vtx.y >> vtx.z;
        min = math::min(min, vtx);
        max = math::max(max, vtx);
      }
    }
    auto distFlowCoord = max - min;
    auto grid_size = math::ceilf(distFlowCoord / get<parameters::cell_size>());

    get<parameters::grid_size>() =
        int3{(int32_t)grid_size.y, (int32_t)grid_size.y, (int32_t)grid_size.y};
  } else {
		throw std::runtime_error("No boundary object provided.");
  }
  get<parameters::min_coord>() = get<parameters::min_domain>();
  get<parameters::max_coord>() = get<parameters::max_domain>();

}

void IO::config::defaultAllocate() {
  for_each(allocations_list, [](auto x) {
	  if (!decltype(x)::valid()) return;
      decltype(x)::leanAllocate();
	  //decltype(x)::defaultAllocate();
    logger(log_level::debug) << "Array: " << decltype(x)::variableName
                             << " valid: " << decltype(x)::valid()
                             << " size: " << IO::config::bytesToString(decltype(x)::alloc_size)
                             << std::endl;
  });
}

void IO::config::initSnapshot() {
  for_each(uniforms_list, [](auto x) { snaps.push_back(new UniformSnap<decltype(x)>()); });
  for_each(sorting_list, [](auto x) { snaps.push_back(new ArraySnap<decltype(x)>()); });
}
