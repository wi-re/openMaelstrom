#include <IO/alembic/alembic.h>
#include <IO/particle/particle.h>
#include <IO/config/config.h>
#include <IO/vdb/vdb.h>
#include <simulation/functions.h>
#include <simulation/particleSystem.h>
#include <utility/MemoryManager.h>
#include <utility/helpers/arguments.h>
#include <utility/template/for_struct.h>

template<typename Ty> struct arrayReset : public simulationStep {
	Ty arr;
	arrayReset(Ty ar) : arr(ar), simulationStep(Ty::qualifiedName, Color::flatpink) {	};

	virtual void call() {
		if (!isValid())
			return;
		if (!logger::silent)
			std::cout << "Resetting array " << description << std::endl;
		MemoryManager::instance().allocate(Ty{});
		cuda::Memset(Ty::ptr, 0, Ty::alloc_size);
	}
	virtual bool isValid() {
		return Ty::valid();
	}
	virtual std::vector<array_enum> getSwapArrays() { return{}; }
	virtual std::vector<array_enum> getInputsArrays() { return{}; }
	virtual std::vector<array_enum> getOutputArrays() { return{ Ty::identifier }; }
	virtual std::vector<array_enum> getTemporaryArrays() { return{}; }
	virtual std::vector<array_enum> getCellArrays() { return{}; }
	virtual std::vector<array_enum> getVirtualArrays() { return{}; }
	virtual std::vector<array_enum> getBoundaryArrays() { return{}; }
	virtual std::vector<array_enum> getNeighborArrays() { return{}; }
};

template<typename... Ts>
std::vector<array_enum> tupToVec(std::tuple<Ts...> tup) {
	return std::vector{ Ts::identifier... };
}
std::vector<array_enum> tupToVec(std::tuple<> tup) {
	return std::vector<array_enum>{};
}


template<typename Ty> struct moduleCall : public simulationStep{
	std::function<void(Ty)> fn;
	/*constexpr std::array inputs = */

	moduleCall(std::function<void(Ty)> fun, 
		const char* name = "", 
		Color c = Color::flatpink) : fn(fun), simulationStep(name, c) {	};
	moduleCall(void(*fun)(Ty), 
		const char* name = "", Color
		c = Color::flatpink) :fn(fun), simulationStep(name, c) {};

	virtual void call() {
		if (!isValid())
			return;
		 if (!logger::silent)
			 std::cout << "Calling function " << description << std::endl;

		if (timer == nullptr && has_timer)
			timer = TimerManager::createGPUTimer(description, color, graph);

		if (timer)
			timer->start();

		auto memory = prepareMemory2<Ty>();
		fn(memory);
		clearMemory2(memory);

		if (this->timer)
		this->timer->stop();
		if (get<parameters::error_checking>()) {
			cuda::sync(description);
		}
	}
	virtual bool isValid() {
		Ty valid_test_mem{};
		if (!valid(valid_test_mem))
			return false;
		return true;
	}

	virtual std::vector<array_enum> getSwapArrays() { return tupToVec(typename Ty::swap_arrays{}); }
	virtual std::vector<array_enum> getInputsArrays() { return tupToVec(typename Ty::input_arrays{}); }
	virtual std::vector<array_enum> getOutputArrays() { return tupToVec(typename Ty::output_arrays{}); }
	virtual std::vector<array_enum> getTemporaryArrays() { return tupToVec(typename Ty::temporary_arrays{}); }
	virtual std::vector<array_enum> getCellArrays() { return tupToVec(typename Ty::cell_info_arrays{}); }
	virtual std::vector<array_enum> getVirtualArrays() { return tupToVec(typename Ty::virtual_info_arrays{}); }
	virtual std::vector<array_enum> getBoundaryArrays() { return tupToVec(typename Ty::boundaryInfo_arrays{}); }
	virtual std::vector<array_enum> getNeighborArrays() { return tupToVec(typename Ty::neighbor_info_arrays{}); }
};

void cuda_particleSystem::setup_simulation() {
	functions.push_back(new moduleCall(&SPH::moving_planes::update_boundaries, "Movingboundary: position", Color::firebrick3));
	functions.push_back(new moduleCall(&SPH::Outlet::remove, "Outlet: remove", Color::broadwaypink));

	// resorting algorithms
	functions.push_back(new moduleCall( &SPH::Resort::resortParticles, "Linear Sort: resort", Color::aquaman));
	functions.push_back(new moduleCall( &SPH::resort_mlm::resortParticles, "MLM sort: resort", Color::aquarium));
	functions.push_back(new moduleCall( &SPH::compactMLM::resortParticles, "compactMLM sort: resort", Color::aquarium));


	//functions.push_back(new moduleCall( &SPH::cluster::clusterParticles, "Clustering", Color::aquarium));

	functions.push_back(new moduleCall( &SPH::enforceSymmetry::constrain_support, "Constraining Support", Color::braeburn_apple));

	// create neighborhoods
	functions.push_back(new moduleCall( &SPH::BasicNeighborList::calculate_neighborlist, "Basic Neighborlist creation", Color::burgundy));
	functions.push_back(new moduleCall( &SPH::spanNeighborList::calculate_neighborlist, "Cell Neighborlist creation", Color::burgundy));
	functions.push_back(new moduleCall( &SPH::compactCellList::calculate_neighborlist, "Compact Cell Neighborlist creation", Color::burgundy));
	functions.push_back(new moduleCall( &SPH::ConstrainedNeighborList::calculate_neighborlist, "Constrained Neighborlist creation", Color::burgundy));
	functions.push_back(new moduleCall( &SPH::compactNeighborMask::calculate_neighborlist, "Compact Masklist creation", Color::burgundy));
	functions.push_back(new moduleCall(&SPH::sortedNeighborList::sort, "Sorting neighbor list", Color::pink3));
	// calculate density
	functions.push_back(new moduleCall( &SPH::Density::estimate_density, "Density estimate", Color::orange4));
	functions.push_back(new moduleCall( &SPH::MLSDensity::estimate_density, "Density estimate", Color::orange4));
	functions.push_back(new moduleCall( &SPH::shepardDensity::estimate_density, "Density estimate", Color::orange4));
	functions.push_back(new moduleCall( &SPH::adaptive::blendDensity, "Adaptive: blend density", Color::ruby_red));

	functions.push_back(new arrayReset(arrays::acceleration{}));
	functions.push_back(new moduleCall(&SPH::DFSPH::divergence_solve, "DFSPH: divergence solver", Color::greencopper));

	//// non pressure forces
	functions.push_back(new moduleCall( &SPH::BenderVorticity::vorticity, "Force: Micropolar", Color::masters_jacket));
	functions.push_back(new moduleCall( &SPH::GisslerDrag::drag, "Force: Drag", Color::ivoryblack));

	functions.push_back(new moduleCall( &SPH::AkinciTension::tension, "Force: Akinci Tension", Color::crema));
	functions.push_back(new moduleCall( &SPH::External::gravity_force, "Force: External", Color::amethyst));
	//functions.push_back(new moduleCall( &SPH::Viscosity::drag, "Force: Viscosity Monaghan", Color::jack_pine));

	// pressure forces / enforce incomressibility
	functions.push_back(new moduleCall( &SPH::DFSPH::density_solve, "DFSPH: incompressibility", Color::greencopper));
	functions.push_back(new moduleCall( &SPH::IISPH::pressure_solve, "IISPH: incompressibility", Color::greencopper));
	functions.push_back(new moduleCall( &SPH::IISPH17::pressure_solve, "IISPH17: incompressibility", Color::greencopper));
	functions.push_back(new moduleCall( &SPH::IISPH17_BAND_RB::pressure_solve, "IISPH17 Stefan Band rigid bodies: incompressibility", Color::greencopper));



	//functions.push_back(new moduleCall(&SPH::Viscosity::viscosity, "Force: Viscosity Monaghan", Color::jack_pine));

	functions.push_back(new moduleCall(&SPH::Integration::update_velocities, "Integrate: velocities", Color::alizarin_crimson));

	functions.push_back(new moduleCall(&SPH::XSPH::viscosity, "Force: Viscosity XSPH", Color::wasabi_sauce));

	functions.push_back(new moduleCall( &SPH::adaptive::blendVelocity, "Adaptivity: blend velocity", Color::green_scrubs));
	functions.push_back(new moduleCall( &SPH::shepardDensity::update_density, "Density: predicting new densities", Color::black));

	functions.push_back(new moduleCall( &SPH::Integration::update_positions, "Integrate: position", Color::crimson));




	if (get<parameters::modules::adaptive>() == true || get<parameters::inletVolumes>().size() > 0) {
		//// calculate density
		functions.push_back(new moduleCall(&SPH::Resort::resortParticles, "Linear Sort: resort", Color::aquaman));
		functions.push_back(new moduleCall(&SPH::resort_mlm::resortParticles, "MLM sort: resort", Color::aquarium));
		functions.push_back(new moduleCall(&SPH::compactMLM::resortParticles, "compactMLM sort: resort", Color::aquarium));


		functions.push_back(new moduleCall(&SPH::cluster::clusterParticles, "Clustering", Color::aquarium));

		functions.push_back(new moduleCall(&SPH::enforceSymmetry::constrain_support, "Constraining Support", Color::braeburn_apple));

		// create neighborhoods
		functions.push_back(new moduleCall(&SPH::BasicNeighborList::calculate_neighborlist, "Basic Neighborlist creation", Color::burgundy));
		functions.push_back(new moduleCall(&SPH::spanNeighborList::calculate_neighborlist, "Cell Neighborlist creation", Color::burgundy));
		functions.push_back(new moduleCall(&SPH::compactCellList::calculate_neighborlist, "Compact Cell Neighborlist creation", Color::burgundy));
		functions.push_back(new moduleCall(&SPH::ConstrainedNeighborList::calculate_neighborlist, "Constrained Neighborlist creation", Color::burgundy));
		functions.push_back(new moduleCall(&SPH::compactNeighborMask::calculate_neighborlist, "Compact Masklist creation", Color::burgundy));
		functions.push_back(new moduleCall(&SPH::sortedNeighborList::sort, "Sorting neighbor list", Color::pink3));

		functions.push_back(new moduleCall(&SPH::Density::estimate_density, "Density estimate", Color::orange4));
	}
	functions.push_back(new moduleCall(&SPH::distance::distance, "Surface distance", Color::presidential_blue));
	functions.push_back(new moduleCall(&SPH::detection::distance, "Surface detection", Color::presidential_blue));
	functions.push_back(new moduleCall(&SPH::adaptive::adapt, "Adaptivity: adapt", Color::braeburn_apple));
	functions.push_back(new moduleCall(&SPH::Visualizer::visualizeParticles, "Visualize", Color::neonpink));
	functions.push_back(new moduleCall(&SPH::streamInlet::emit, "Inlet: emit", Color::broadwaypink));
	if (get<parameters::rayTracing>() == true) {
		functions.push_back(new moduleCall(&SPH::Resort::resortParticles, "Linear Sort: resort", Color::aquaman));
		functions.push_back(new moduleCall(&SPH::resort_mlm::resortParticles, "MLM sort: resort", Color::aquarium));
		functions.push_back(new moduleCall(&SPH::compactMLM::resortParticles, "compactMLM sort: resort", Color::aquarium));
		functions.push_back(new moduleCall(&SPH::enforceSymmetry::constrain_support, "Constraining Support", Color::braeburn_apple));
		functions.push_back(new moduleCall(&SPH::BasicNeighborList::calculate_neighborlist, "Basic Neighborlist creation", Color::burgundy));
		functions.push_back(new moduleCall(&SPH::spanNeighborList::calculate_neighborlist, "Cell Neighborlist creation", Color::burgundy));
		functions.push_back(new moduleCall(&SPH::compactCellList::calculate_neighborlist, "Compact Cell Neighborlist creation", Color::burgundy));
		functions.push_back(new moduleCall(&SPH::ConstrainedNeighborList::calculate_neighborlist, "Constrained Neighborlist creation", Color::burgundy));
		functions.push_back(new moduleCall(&SPH::compactNeighborMask::calculate_neighborlist, "Compact Masklist creation", Color::burgundy));
		functions.push_back(new moduleCall(&SPH::Density::estimate_density, "Density estimate", Color::orange4));
		functions.push_back(new moduleCall(&SPH::auxilliaryMLM::generateAuxilliaryGrid, "compactMLM sort: resort", Color::aquarium));
		functions.push_back(new moduleCall(&SPH::anisotropy::generateAnisotropicMatrices, "Anisotropy", Color::aquarium));
	}
	

}

void* cuda_particleSystem::retainArray(std::string arrayName){
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
#include <iomanip>

void cuda_particleSystem::storeDebugArrays() {
	if (get<parameters::debug>() == false) return;

	for_struct(m_debugArrays, [](auto& arr) {
		using Ta = std::decay_t<decltype(arr)>;
		using Ty = typename Ta::value_type;
		if (arr.valid) {
			cudaMemcpy(arr.ptr, Ty::ptr, Ty::alloc_size, cudaMemcpyDeviceToHost);
		}
		});
}
debugArrays cuda_particleSystem::getDebugArrays() { return m_debugArrays; }

void cuda_particleSystem::init_simulation() {
  IO::config::load_config(get<parameters::config_file>());
  IO::config::show_config();
  if (arguments::cmd::instance().snapped)
	  get<parameters::gl_record>() = false;
  SPH::volume::init_volumes();
  if(!arguments::cmd::instance().snapped)
  IO::vdb::emitParticleVolumes();
  else if(get<parameters::rigidVolumes>().size() > 0) IO::vdb::recreateRigids();
  SPH::streamInlet::init();
  SPH::Outlet::init();
  if (!arguments::cmd::instance().snapped)
  IO::particle::loadParticles();
  setup_simulation();


  if (!arguments::cmd::instance().snapped) {
	  auto memory = prepareMemory2<SPH::cleanup::Memory>();
	  SPH::cleanup::cleanup_particles(memory);
	  clearMemory2(memory);
  }

  if (get<parameters::debug>() == true) {
	  for_struct(m_debugArrays, [&](auto& arr) {
		  using Ta = std::decay_t<decltype(arr)>;
		  using Ty = typename Ta::value_type;
		  arr.valid = Ty::valid();
		  if (arr.valid) {
			  arr.ptr = (typename Ty::type*) malloc(Ty::alloc_size);
			  retainArray(Ty::variableName);
		  }
		  });
	  storeDebugArrays();
  }

  init = true;
  if (!logger::silent) {
	  size_t longestName = 0;
	  size_t longestSize = 0;
	  std::vector<array_enum> arraysInUse;
	  for (auto fn : functions)
		  if (fn->isValid())
			  for (auto a : fn->getArrays()) {
				  auto qn = getArrayQualifiedName(a);
				  auto s = getArrayAllocationSize(a);
				  auto sz = IO::config::bytesToString(s);
				  if (s != 0) {
					  arraysInUse.push_back(a);
					  longestName = std::max(longestName, std::string(qn).length());
					  longestSize = std::max(longestSize, std::string(sz).length());
				  }
			  }
	  std::sort(arraysInUse.begin(), arraysInUse.end());
	  arraysInUse.erase(std::unique(arraysInUse.begin(), arraysInUse.end()), arraysInUse.end());

	  std::cout << "Arrays in use : " << std::endl;
	  for (auto a : arraysInUse) {
		  std::cout << getArrayQualifiedName(a) << " -> " << IO::config::bytesToString(getArrayAllocationSize(a)) << std::endl;
	  }

	  auto contains = [](const auto& v, const auto& e) {
		  return std::find(v.begin(), v.end(), e) != v.end();
	  };

	  for (auto a : arraysInUse) {
		  std::cout << std::setw(longestName + 1) << getArrayQualifiedName(a) << " -> " << std::setw(longestSize + 1) << IO::config::bytesToString(getArrayAllocationSize(a)) << "\t";

		  for (auto fn : functions)
			  if (fn->isValid()) {
				  if (contains(fn->getSwapArrays(), a)) std::cout << "S";
				  else if (contains(fn->getInputsArrays(), a)) std::cout << "I";
				  else if (contains(fn->getOutputArrays(), a)) std::cout << "O";
				  else if (contains(fn->getTemporaryArrays(), a)) std::cout << "T";
				  else if (contains(fn->getCellArrays(), a)) std::cout << "C";
				  else if (contains(fn->getVirtualArrays(), a)) std::cout << "V";
				  else if (contains(fn->getBoundaryArrays(), a)) std::cout << "B";
				  else if (contains(fn->getNeighborArrays(), a)) std::cout << "N";
				  else std::cout << " ";
			  }
		  std::cout << std::endl;
	  }
  }
}
#include <utility/helpers/exporter.h>
void cuda_particleSystem::step() {
	static cudaEvent_t start, stop;
	static bool once = true;
	if (once) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		once = false;
	}
	if (renderFlag && (running || single)) {
		{
			GRAPH_CODE(Lock, Color::black, std::lock_guard<std::mutex> guard(simulation_lock));
			renderFlag = false;
			cudaEventRecord(start, 0);
			TIME_CODE(Frame, Color::neonpink,
				get<parameters::simulationTime>() += get<parameters::timestep>();
			for (auto& fun : functions)
				fun->call();
			if (get<parameters::alembic_export>()) {
				TIME_CODE(Export, Color::neonblue, IO::alembic::save_particles(););
			}
			for (auto& cb : callbacks)
				cb();
			arguments::cmd::instance().parameter_stats();
			MemoryManager::instance().reclaimMemory();
			get<parameters::frame>() += 1;
			);
			cudaEventRecord(stop, 0);
			storeDebugArrays();
			if (single) {
				running = false;
				single = false;
			}
		}
		//std::this_thread::sleep_for(std::chrono::milliseconds(100));
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

void cuda_particleSystem::addCallback(std::function<void()> fn){
	callbacks.push_back(fn);
}