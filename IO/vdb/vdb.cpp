//#define NO_OPERATORS
#define BOOST_USE_WINDOWS_H
#include <IO/vdb/vdb.h>
#include <iostream>
#include <utility/identifier.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

#include <utility/generation.h>
#include <utility/bullet/DynamicsWorld.h>

void IO::vdb::emitParticleVolumes() {
  //emit fluid volumes
   for (auto fluidVolume : get<parameters::particleVolumes>()) {
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
	std::vector<int> particle_type;
	std::vector<int> particle_type_x;

  bool first = false;
	for (int32_t i = old_ptcls; i < old_ptcls + inserted_particles; ++i) {
		openvdb::Vec4f ptcl_position = out_points[i - old_ptcls];
    
		positions.push_back(float4{ ptcl_position.x(), ptcl_position.y(), ptcl_position.z()+(first ? 20 : 0), ptcl_position.w() });
		velocities.push_back(float4{ 0.f, 0.f, 0.f, 0.f });
		volumes.push_back(volume);
    particle_type.push_back(FLUID_PARTICLE);
    // particle_type_x.push_back(FLUID_PARTICLE);
    particle_type_x.push_back(first ? 1 : FLUID_PARTICLE);
    first = false;
	}
	cudaMemcpy(arrays::position::ptr + old_ptcls, positions.data(), inserted_particles * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::velocity::ptr + old_ptcls, velocities.data(), inserted_particles * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volume::ptr + old_ptcls, volumes.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::particle_type::ptr + old_ptcls, particle_type.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(arrays::particle_type_x::ptr + old_ptcls, particle_type_x.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
#endif
  }
  

   std::vector<float> densities;
   std::vector<float4> rigid_velocities;
   std::vector<float3> rigid_avelocities;
   std::vector<float> rigid_volumes;
   std::vector<float3> rigid_origins;
   std::vector<float4> rigid_quaternions;
   std::vector<std::string> rigid_files;
   if (get<parameters::volumeBoundaryCounter>() == 0 && get<parameters::rigidVolumes>().size() > 0) {
	   int rigid_nums = 1;
	   auto wrld = DynamicsWorld::getInstance();
	   wrld->createWorld();
	   for (auto fluidVolume : get<parameters::rigidVolumes>()) {
		   densities.push_back(fluidVolume.density.value);
		   rigid_velocities.push_back({ 0.f, 0.f, 0.f, 0.f });
		   rigid_avelocities.push_back({ 0.f, 0.f, 0.f });
		   rigid_quaternions.push_back({ 0.f, 0.f, 0.f, 1.f });
		   rigid_files.push_back(fluidVolume.fileName.value);

		   auto maxmin = wrld->addInfoBody(fluidVolume.fileName.value, fluidVolume.density.value, fluidVolume.shift.value);
		   auto sht = fluidVolume.shift.value;
		   rigid_origins.push_back({ (maxmin["max"].x + maxmin["min"].x) / 2 + sht.x, (maxmin["max"].y + maxmin["min"].y) / 2 + sht.y,
			 (maxmin["max"].z + maxmin["min"].z) / 2 + sht.z });
		   // std::cout << "saved origin: " << (maxmin["max"].x+maxmin["min"].x)/2 << " " << 
		   //   (maxmin["max"].y+maxmin["min"].y)/2 << " " << (maxmin["max"].z+maxmin["min"].z)/2 << std::endl;
		   //TODO: combine above and below functions and make a review on the body of function below

		   auto r = get<parameters::radius>();
		   auto volume = PI4O3 * math::power<3>(r);
		   auto out_points = generation::generateParticlesRigid(fluidVolume.fileName.value, r, genTechnique::square_grid, false, maxmin);
		   auto inserted_particles = (int32_t)out_points.size();

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
				   float4{ ptcl_position.x(), ptcl_position.y(), ptcl_position.z(), ptcl_position.w() };
			   get<arrays::velocity>()[i] = float4{ 0.f, 0.f, 0.f, 0.f };

			   get<arrays::volume>()[i] = volume;
		   }
#else
		   std::vector<float4> positions;
		   std::vector<float4> velocities;
		   std::vector<float> volumes;
		   std::vector<int> particle_type;
		   std::vector<int> particle_type_x;
		   // float4 center = {0, 0, 0, 0};
		   // for (int32_t i = 0; i < inserted_particles; ++i) {
		   // 	openvdb::Vec4f ptcl_position = out_points[i];
		   //   center += float4{ ptcl_position.x(), ptcl_position.y(), ptcl_position.z(), ptcl_position.w()};

		   // }

		   // center = center / inserted_particles;

		   auto shift = fluidVolume.shift.value;
		   for (int32_t i = old_ptcls; i < old_ptcls + inserted_particles; ++i) {
			   openvdb::Vec4f ptcl_position = out_points[i - old_ptcls];
			   positions.push_back(float4{ ptcl_position.x() + shift.x, ptcl_position.y() + shift.y, ptcl_position.z() + shift.z, ptcl_position.w() });
			   velocities.push_back(float4{ 0.f, 0.f, 0.f, 0.f });
			   particle_type.push_back(rigid_nums);
			   particle_type_x.push_back(1);
		   }
		   for (auto& p_i : positions) {
			   float v = 0.f;
			   for (auto& p_j : positions) {
				   v += kernel(p_i, p_j);
			   }
			   volumes.push_back(get<parameters::gamma>() / v);
			   // volumes.push_back(volume);

		   }

		   cudaMemcpy(arrays::position::ptr + old_ptcls, positions.data(), inserted_particles * sizeof(float4), cudaMemcpyHostToDevice);
		   cudaMemcpy(arrays::velocity::ptr + old_ptcls, velocities.data(), inserted_particles * sizeof(float4), cudaMemcpyHostToDevice);
		   cudaMemcpy(arrays::volume::ptr + old_ptcls, volumes.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
		   cudaMemcpy(arrays::particle_type::ptr + old_ptcls, particle_type.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
		   //cudaMemcpy(arrays::particle_type_x::ptr + old_ptcls, particle_type_x.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
#endif


  // DynamicsWorld::getInstance()->addBody(fluidVolume.fileName.value);
		   float total_vol = 0;
		   for (auto i = 0; i < inserted_particles; i++)
		   {
			   total_vol += volumes[i];
		   }

		   rigid_volumes.push_back(total_vol);
		   int32_t index = rigid_nums - 1;
		   wrld->addBody(total_vol, index);
		   rigid_nums++;
	   }
	   if (get<parameters::rigidVolumes>().size()) {

		   int32_t rigidCount = get<parameters::rigidVolumes>().size();
		   DynamicsWorld::getInstance()->createBoundingBox();

		   arrays::rigidOrigins::allocate(sizeof(float3) * rigidCount);
		   cudaMemcpy(arrays::rigidOrigins::ptr, rigid_origins.data(), sizeof(float3) * rigidCount, cudaMemcpyHostToDevice);

		   //arrays::rigidFiles::allocate(sizeof(std::string) * rigidCount);
		   //cudaMemcpy(arrays::rigidFiles::ptr, rigid_files.data(), sizeof(std::string) * rigidCount, cudaMemcpyHostToDevice);

		   arrays::rigidQuaternions::allocate(sizeof(float4) * rigidCount);
		   cudaMemcpy(arrays::rigidQuaternions::ptr, rigid_quaternions.data(), sizeof(float4) * rigidCount, cudaMemcpyHostToDevice);

		   arrays::rigidVolumes::allocate(sizeof(float) * rigidCount);
		   cudaMemcpy(arrays::rigidVolumes::ptr, rigid_volumes.data(), sizeof(float) * rigidCount, cudaMemcpyHostToDevice);

		   arrays::rigidDensities::allocate(sizeof(float) * rigidCount);
		   cudaMemcpy(arrays::rigidDensities::ptr, densities.data(), sizeof(float) * rigidCount, cudaMemcpyHostToDevice);

		   arrays::rigidLinearVelocities::allocate(sizeof(float4) * rigidCount);
		   cudaMemcpy(arrays::rigidLinearVelocities::ptr, rigid_velocities.data(), sizeof(float4) * rigidCount, cudaMemcpyHostToDevice);

		   arrays::rigidAVelocities::allocate(sizeof(float3) * rigidCount);
		   cudaMemcpy(arrays::rigidAVelocities::ptr, rigid_avelocities.data(), sizeof(float3) * rigidCount, cudaMemcpyHostToDevice);

	   }
	   for (auto boundaryVolume : get<parameters::boundaryVolumes>()) {
		   wrld->addBoundary(boundaryVolume.fileName.value);
	   }
   }
	   else {
		   rigid_origins.push_back({ 0.f, 0.f, 0.f });
		   arrays::rigidOrigins::allocate(sizeof(float3));
		   cudaMemcpy(arrays::rigidOrigins::ptr, rigid_origins.data(), sizeof(float3), cudaMemcpyHostToDevice);

		   rigid_files.push_back("");
		   //arrays::rigidFiles::allocate(sizeof(std::string));
		   //cudaMemcpy(arrays::rigidFiles::ptr, rigid_files.data(), sizeof(std::string), cudaMemcpyHostToDevice);

		   rigid_quaternions.push_back({ 0.f, 0.f, 0.f, 0.f });
		   arrays::rigidQuaternions::allocate(sizeof(float4));
		   cudaMemcpy(arrays::rigidQuaternions::ptr, rigid_quaternions.data(), sizeof(float4), cudaMemcpyHostToDevice);

		   rigid_volumes.push_back(0.f);
		   arrays::rigidVolumes::allocate(sizeof(float));
		   cudaMemcpy(arrays::rigidVolumes::ptr, rigid_volumes.data(), sizeof(float), cudaMemcpyHostToDevice);

		   densities.push_back(0.f);
		   arrays::rigidDensities::allocate(sizeof(float));
		   cudaMemcpy(arrays::rigidDensities::ptr, densities.data(), sizeof(float), cudaMemcpyHostToDevice);

		   rigid_velocities.push_back({ 0.0f, 0.0f, 0.0f, 0.0f });
		   arrays::rigidLinearVelocities::allocate(sizeof(float4));
		   cudaMemcpy(arrays::rigidLinearVelocities::ptr, rigid_velocities.data(), sizeof(float4), cudaMemcpyHostToDevice);

		   rigid_avelocities.push_back({ 0.0f, 0.0f, 0.0f });
		   arrays::rigidAVelocities::allocate(sizeof(float3));
		   cudaMemcpy(arrays::rigidAVelocities::ptr, rigid_avelocities.data(), sizeof(float3), cudaMemcpyHostToDevice);

	   }

}

void IO::vdb::recreateRigids() {
  int rigid_nums = 1;
  auto wrld = DynamicsWorld::getInstance();
  wrld->createWorld();
  for (auto fluidVolume : get<parameters::rigidVolumes>()) {
    
    // DynamicsWorld::getInstance()->addBody(fluidVolume.fileName.value);
    int32_t index = rigid_nums - 1;
    auto dens = fluidVolume.density.value;
    auto vol = arrays::rigidVolumes::ptr[index];
    auto rfile = fluidVolume.fileName.value;
    wrld->addInfoBody(rfile, dens, {0, 0, 0});
    wrld->addBody(vol, index);
    rigid_nums++;
  }

  if (get<parameters::rigidVolumes>().size()){
    
    DynamicsWorld::getInstance()->createBoundingBox();
  
  }
  else{}

  for (auto boundaryVolume : get<parameters::boundaryVolumes>()) {
    wrld->addBoundary(boundaryVolume.fileName.value);
  }
}