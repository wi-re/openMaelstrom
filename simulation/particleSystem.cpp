#include <IO/alembic/alembic.h>
#include <IO/config/config.h>
#include <IO/vdb/vdb.h>
#include <simulation/functions.h>
#include <simulation/particleSystem.h>
#include <utility/MemoryManager.h>
#include <utility/helpers/arguments.h>

auto setup_simulation() {
	simulation<> sim;
	return sim
		//.then(&SPH::moving_planes::update_boundaries, "Movingboundary: position", Color::firebrick3)
		.then(&SPH::Outlet::remove, "Outlet: remove", Color::broadwaypink)

		// resorting algorithms
		.then(&SPH::Resort::resortParticles, "Linear Sort: resort", Color::aquaman)
		.then(&SPH::resort_mlm::resortParticles, "MLM sort: resort", Color::aquarium)

		.then(&SPH::cluster::clusterParticles, "Clustering", Color::aquarium)

		.then(&SPH::enforceSymmetry::constrain_support, "Constraining Support", Color::braeburn_apple)

		// create neighborhoods
		.then(&SPH::BasicNeighborList::calculate_neighborlist, "Basic Neighborlist creation", Color::burgundy)
		.then(&SPH::spanNeighborList::calculate_neighborlist, "Cell Neighborlist creation", Color::burgundy)
		.then(&SPH::ConstrainedNeighborList::calculate_neighborlist, "Constrained Neighborlist creation", Color::burgundy)

		// calculate density
		.then(&SPH::Density::estimate_density, "Density estimate", Color::orange4)
		.then(&SPH::MLSDensity::estimate_density, "Density estimate", Color::orange4)
		.then(&SPH::shepardDensity::estimate_density, "Density estimate", Color::orange4)
		.then(&SPH::adaptive::blendDensity, "Adaptive: blend density", Color::ruby_red)

		.then(array_clear<arrays::acceleration>())
		.then(&SPH::DFSPH::divergence_solve, "DFSPH: divergence solver", Color::greencopper)

		// non pressure forces
		.then(&SPH::BenderVorticity::vorticity, "Force: Micropolar", Color::masters_jacket)
		.then(&SPH::GisslerDrag::drag, "Force: Drag", Color::ivoryblack)

		.then(&SPH::AkinciTension::tension, "Force: Akinci Tension", Color::crema)
		.then(&SPH::Viscosity::viscosity, "Force: Viscosity Monaghan", Color::jack_pine)
		.then(&SPH::External::gravity_force, "Force: External", Color::amethyst)
		.then(&SPH::distance::distance, "Surface distance", Color::presidential_blue)

		// pressure forces / enforce incomressibility
		.then(&SPH::DFSPH::density_solve, "DFSPH: incompressibility", Color::greencopper)
		.then(&SPH::IISPH::pressure_solve, "IISPH: incompressibility", Color::greencopper)
		.then(&SPH::IISPH17::pressure_solve, "IISPH17: incompressibility", Color::greencopper)


		.then(&SPH::XSPH::viscosity, "Force: Viscosity XSPH", Color::wasabi_sauce)
		.then(&SPH::adaptive::blendVelocity, "Adaptivity: blend velocity", Color::green_scrubs)
		.then(&SPH::shepardDensity::update_density, "Density: predicting new densities", Color::black)
		// update velocity
		.then(&SPH::Integration::update_velocities, "Integrate: velocities", Color::alizarin_crimson)
		//.then(&SPH::moving_planes::correct_velocity, "Movingboundary: velocities", Color::firebrick)
		// update positions
		
		.then(&SPH::Integration::update_positions, "Integrate: position", Color::crimson)
		.then(&SPH::streamInlet::emit, "Inlet: emit", Color::broadwaypink)
		//.then(&SPH::moving_planes::correct_position, "Movingboundary: position", Color::firebrick3)
		
		.then(&SPH::adaptive::adapt, "Adaptivity: adapt", Color::braeburn_apple)
		.then(&SPH::Visualizer::visualizeParticles, "Visualize", Color::neonpink)
		;
}

std::decay_t<decltype(setup_simulation().functions)> simulation_functions;


void* cuda_particleSystem::retainArray(std::string arrayName){
	std::lock_guard<std::mutex> guard(simulation_lock);
	bool found = false;
	void* ptr = nullptr;
for_each_r(arrays_list,[&](auto x){
	using Ty = decltype(x);
	if(Ty::variableName == arrayName){
		if(!Ty::valid())
			throw std::runtime_error("Trying to retain array that cannot be valid under given configuration");
		found = true;
		if(Ty::ptr == nullptr)
			MemoryManager::instance().allocate(x);
		ptr = Ty::ptr;
		MemoryManager::instance().persistentArrays.push_back(arrayName);		
	}
});
if(!found)
	throw std::runtime_error("Cannot find array for given name " + arrayName);
return ptr;
}

void cuda_particleSystem::init_simulation() {
  IO::config::load_config(get<parameters::config_file>());
  IO::config::show_config();
  if(!arguments::cmd::instance().snapped)
  IO::vdb::emitParticleVolumes();
  SPH::volume::init_volumes();
  SPH::streamInlet::init();
  SPH::Outlet::init();
  simulation_functions = setup_simulation().functions;
  init = true;
}
#include <utility/helpers/exporter.h>
void cuda_particleSystem::step() {
  if (running) {
	  GRAPH_CODE(Lock, Color::black, std::lock_guard<std::mutex> guard(simulation_lock));
	  TIME_CODE(Frame, Color::neonpink,
		  get<parameters::simulationTime>() += get<parameters::timestep>();
	  for_each(simulation_functions, [](auto &x) { x(); });
	  if (get<parameters::alembic_export>()) {
		  TIME_CODE(Export, Color::neonblue, IO::alembic::save_particles(););
	  }
	  MemoryManager::instance().reclaimMemory(); 
	  get<parameters::frame>() += 1;
	  );
  }
  if(get<parameters::dumpNextframe>()){
	  get<parameters::dumpNextframe>() = 0;
	  IO::dumpAll(get<parameters::dumpFile>());
  }
}

cuda_particleSystem &cuda_particleSystem::instance() {
  static cuda_particleSystem cps;
  return cps;
}
