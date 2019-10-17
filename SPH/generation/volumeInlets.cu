#define NEW_STYLE
#include <SPH/generation/volumeInlets.cuh>
#include <utility/include_all.h>

// Variadic template base case
hostDeviceInline void emit_particle(uint32_t) {}

// This function sets all parameters of a create particle to a default constructed value, if
// something else is desired it should be done in the particleInlet function.
template <typename T, typename... Ts> hostDeviceInline void emit_particle(uint32_t trgIdx, std::pair<T *, T *> arg, Ts... ref) {
  if (arg.second != nullptr && arg.first != nullptr)
    arg.first[trgIdx] = T{};
  emit_particle(trgIdx, ref...);
}

// This function emits all particles from a single emitter based on the seed positions (
// float4_u<SI::m> *positions ) with the given parameters. This function checks if there is a
// particle in close proximity which for hash based methods requires checking all hash collisions of
// a cell as this method takes places after the position integration which means that the particle
// positions of particles might not be equal to the morton code of the cell they occupy. Similarly
// this interpolation is done at the basic resolution for MLM.
cellFunctionType particleInlet(SPH::streamInlet::Memory arrays, 
	int32_t threads, int32_t ptclsEmitted, int32_t *ptclCounter,
	float4_u<SI::m> *positions, float4_u<SI::velocity> velocity, float_u<SI::volume> particleVolume,  float_u<SI::s> delay, float_u<SI::m> distance, float* distanceBuffer, int32_t* type, int32_t* indicator,
	Ts... tup) {
  checkedThreadIdx(i);

  float4_u<SI::m> position = positions[ptclsEmitted + i];
  bool flag = false;

  float_u<SI::m> a = math::power<ratio<1, 3>>(particleVolume) * 1.f;
  iterateAllCells(position, j) {
	  flag = flag || math::distance3(arrays.position[j], position) < a;
	  if (math::distance3(arrays.position[j], position) < a)
		  arrays.velocity[j] = velocity;
  }

  auto p = position;
  auto r = 0.35f * arrays.radius;
  auto POS = planeBoundary::distance(position, particleVolume, arrays);
  if (POS.val.w <= r) {
	  flag = true;
  }
  for (int32_t b = 0; b < arrays.volumeBoundaryCounter; ++b) {
	  auto VOS = volumeBoundary::distance_fn(position, particleVolume, arrays, b);
	  if (VOS.val.w <= r) {
		  //flag = true;
	  }
  }
  if (flag)
    return;
  cuda_atomic<int32_t> counter(ptclCounter);
  int32_t new_idx = counter.add(1);
  emit_particle(new_idx, tup...);
  arrays.lifetime[new_idx] = -delay;
  position.val.w = support_from_volume(particleVolume).val;
  arrays.position[new_idx] = position;

  arrays.velocity[new_idx] = velocity;
  arrays.volume[new_idx] = particleVolume;
  
  if(distanceBuffer != nullptr) distanceBuffer[new_idx] = distance.val;
  if (type != nullptr) type[new_idx] = 0;
  if (indicator != nullptr) indicator[new_idx] = 0;
}

cellFunction(Inlet, particleInlet, "Particle Inlet")

template <typename... Ts, typename... Vs>
auto callInlet(std::tuple<Ts...>, SPH::streamInlet::Memory mem, int32_t num_ptcls, Vs... args) {
  launch<Inlet>(num_ptcls, mem, num_ptcls, args..., std::make_pair((typename Ts::unit_type *)Ts::ptr, (typename Ts::unit_type *)Ts::rear_ptr)...);
}

void SPH::streamInlet::emit(Memory mem) {
  int32_t num_ptcl_sum = 0;
  for (auto &fluidVolume : get<parameters::inletVolumes>()) {
    //float_u<SI::m> radius = fluidVolume.inlet_radius.value * 1.f / powf(get<parameters::adaptive::resolution>() / 8.f, 1.f / 3.f);
	  float_u<SI::m> radius = fluidVolume.inlet_radius.value;
    float_u<SI::volume> particleVolume = PI4O3 * math::power<3>(radius);

    auto num_ptcls = fluidVolume.particles_emitted.value;
    auto v = fluidVolume.emitter_velocity.value;
    float4_u<SI::velocity> vel{v.x, v.y, v.z, v.w};

	auto factor = sinf(get<parameters::simulationTime>() * (CUDART_PI_F * 1.f)) * float4_u<SI::velocity>(0.f,5.f,0.f,0.f);
	//vel += factor;

    float_u<SI::s> dur{fluidVolume.duration.value};
    float_u<SI::s> del{fluidVolume.delay.value};

    if (dur < get<parameters::simulationTime>() && dur > 0.0_s || del > get<parameters::simulationTime>()) {
      num_ptcl_sum += num_ptcls;
      continue;
    }

    int32_t old_count = parameters::num_ptcls{};

    if (old_count + num_ptcls >= parameters::max_numptcls{})
      continue;
	//std::cout << "-----------------------------------------------------" << std::endl;
	//std::cout << *parameters::num_ptcls::ptr << std::endl;
	//std::cout << *parameters::num_ptcls::ptr << std::endl;
	//std::cout << old_count << std::endl;
	//std::cout << "######################################################" << std::endl;
	cuda::memcpy(arrays::inletCounter::ptr, parameters::num_ptcls::ptr, sizeof(int32_t), cudaMemcpyHostToDevice);
    callInlet(sorting_list, mem, num_ptcls, num_ptcl_sum, arrays::inletCounter::ptr, (float4_u<SI::m> *)arrays::inletPositions::ptr, vel,
              particleVolume, *parameters::adaptive::delay::ptr, *parameters::surfaceDistance::surface_levelLimit::ptr, arrays::surfaceArrays::distanceBuffer::ptr, arrays::particle_type::ptr, arrays::splitIndicator::ptr );

	//std::cout << *parameters::num_ptcls::ptr << std::endl;
	//std::cout << *parameters::num_ptcls::ptr << std::endl;
	//std::cout << old_count << std::endl;
	cuda::sync("Testing");
	cuda::memcpy(parameters::num_ptcls::ptr, arrays::inletCounter::ptr, sizeof(int32_t), cudaMemcpyDeviceToHost);
    num_ptcl_sum += num_ptcls;
  }
}
