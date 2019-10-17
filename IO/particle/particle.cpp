#define BOOST_USE_WINDOWS_H
#include <IO/particle/particle.h>
#include <utility/cuda.h>
#include <iostream>
#include <utility/identifier/arrays.h>
#include <utility/identifier/uniform.h>
#include <utility/math.h>
#ifdef _WIN32
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif
#include <fstream>
#include <utility/helpers/log.h>
#include <utility/cuda.h>
#include <simulation/particleSystem.h>
#include <utility/helpers/pathfinder.h>

namespace IO {
void particle::loadParticles() {
	for (auto sets : get<parameters::particleSets::particleSets>()) {
		auto& numPtcls = get<parameters::num_ptcls>();
		auto file = resolveFile(sets, { get<parameters::config_folder>() });
		auto ptcls = 0;
		std::ifstream pSet(file.string(), std::ios::binary);
		pSet.read(reinterpret_cast<char*>(&ptcls), sizeof(int32_t));
		pSet.read(reinterpret_cast<char*>(arrays::position::ptr + numPtcls), sizeof(float4) * ptcls);
		pSet.read(reinterpret_cast<char*>(arrays::velocity::ptr + numPtcls), sizeof(float4) * ptcls);
		pSet.read(reinterpret_cast<char*>(arrays::volume::ptr + numPtcls), sizeof(float) * ptcls);
		pSet.read(reinterpret_cast<char*>(arrays::lifetime::ptr + numPtcls), sizeof(float) * ptcls);
		numPtcls += ptcls;
		std::cout << "Read set file " << sets << " with "<< ptcls << " particles. " << std::endl;
		pSet.close();
	}
} 
void particle::saveParticles() {
	std::lock_guard<std::mutex> guard(cuda_particleSystem::instance().simulation_lock);
	std::ofstream pSet("particles.set", std::ios::binary);
	auto ptcls = get<parameters::num_ptcls>();
	pSet.write(reinterpret_cast<char*>(&ptcls), sizeof(int32_t));
	pSet.write(reinterpret_cast<char*>(arrays::position::ptr), sizeof(float4) * ptcls);
	pSet.write(reinterpret_cast<char*>(arrays::velocity::ptr), sizeof(float4) * ptcls);
	pSet.write(reinterpret_cast<char*>(arrays::volume::ptr), sizeof(float) * ptcls);
	pSet.write(reinterpret_cast<char*>(arrays::lifetime::ptr), sizeof(float) * ptcls);
	pSet.close();
	std::cout << "Done writing particle set file" << std::endl;
}
} // namespace IO
