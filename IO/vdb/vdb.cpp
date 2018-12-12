//#define NO_OPERATORS
#define BOOST_USE_WINDOWS_H
#include <IO/vdb/vdb.h>
#include <iostream>
#include <utility/identifier.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

#include <utility/generation.h>



void IO::vdb::emitParticleVolumes() {
  for (auto fluidVolume : get<parameters::particle_volumes>()) {
    auto r = get<parameters::radius>();
    auto volume = PI4O3 * math::power<3>(r);
    auto out_points = generation::generateParticles(fluidVolume.fileName.value, r, genTechnique::hex_grid);
    auto inserted_particles = (int32_t) out_points.size();

    int32_t old_ptcls = get<parameters::num_ptcls>();
    if (old_ptcls + inserted_particles > get<parameters::max_numptcls>()) {
      std::cerr << "Not enough memory to insert particles." << std::endl;
      continue;
    }

    get<parameters::num_ptcls>() += inserted_particles;

#ifdef UNIFIED_MEMORY
    for (int32_t i = old_ptcls; i < old_ptcls + inserted_particles; ++i) {
      openvdb::Vec4f ptcl_position = out_points[i - old_ptcls];
      get<arrays::position>()[i] =
          float4{ptcl_position.x(), ptcl_position.y(), ptcl_position.z(), ptcl_position.w()};
      get<arrays::velocity>()[i] = float4{0.f, 0.f, 0.f, 0.f};

      get<arrays::volume>()[i] = volume;
    }
#else
	std::vector<float4> positions;
	std::vector<float4> velocities;
	std::vector<float> volumes;

	for (int32_t i = old_ptcls; i < old_ptcls + inserted_particles; ++i) {
		openvdb::Vec4f ptcl_position = out_points[i - old_ptcls];
		positions.push_back(float4{ ptcl_position.x(), ptcl_position.y(), ptcl_position.z(), ptcl_position.w() });
		velocities.push_back(float4{ 0.f, 0.f, 0.f, 0.f });
		volumes.push_back(volume);
	}
	cudaMemcpy(arrays::position::ptr + old_ptcls, positions.data(), inserted_particles * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::velocity::ptr + old_ptcls, velocities.data(), inserted_particles * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volume::ptr + old_ptcls, volumes.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
#endif

  }
}
