#define NEW_STYLE
#include <SPH/density/shepardDensity.cuh>
#include <utility/include_all.h>
#include <utility/SPH/boundaryFunctions.h>

/** This function calculates a density estimate using SPH for all particles and additionally uses a
 * lookup table to improve the boundary density contribution of flat boundary planes (no
 * source/novel). Only calculates unit density not actual density  **/
 neighFunctionType init_density(SPH::shepardDensity::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume));	 

	auto unit_density = SWH::spline4(pos[i], arrays);
	auto shepFilter = SWH::spline4(pos[i], arrays);
	iterateNeighbors(j){
	  unit_density += vol[j] * W_ij;
	  shepFilter += arrays.volume[j] / arrays.density.first[j] * W_ij;
	}
	
	arrays.density.second[i] = unit_density / shepFilter;
}

neighFunctionType estimate_density(SPH::shepardDensity::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume));	 

	auto unit_density = SWH::spline4(pos[i], arrays);
	iterateNeighbors(j){
	  unit_density += vol[j] * W_ij;
	}
	
	arrays.density.first[i] = unit_density;
}

neighFunctionType update_density(SPH::shepardDensity::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume));	 

	auto V_i = basicVolume;
	auto boundaryKernel = SWH::spikyGradient(pos[i], V_i, arrays);
	auto rho_star = arrays.density.first[i] + arrays.timestep * V_i * math::dot3(arrays.velocity[i], boundaryKernel);

	iterateNeighbors(j) {
		auto spikyGradient = GW_ij;
		rho_star = rho_star + arrays.timestep * vol[j] * math::dot3(arrays.velocity[i] - arrays.velocity[j], spikyGradient);
	}
	arrays.density.second[i] = rho_star;
}


neighFunction(estimateDensity, estimate_density, "Estimate Density", caches<float4, float>{});
neighFunction(initDensity, init_density, "Estimate Density", caches<float4, float>{});
neighFunction(updateDensity, update_density, "Estimate Density", caches<float4, float>{});

void SPH::shepardDensity::estimate_density(Memory mem) {
	
	std::cout << get<parameters::frame>() << std::endl;
	if(get<parameters::frame>() < 10)
		launch<estimateDensity>(mem.num_ptcls, mem);
	else{
		if(get<parameters::frame>() % get<parameters::densitySteps>() == 0)
			launch<estimateDensity>(mem.num_ptcls, mem);
		launch<initDensity>(mem.num_ptcls, mem);
		arrays::density::swap();
	}
}

void SPH::shepardDensity::update_density(Memory mem) {
	launch<updateDensity>(mem.num_ptcls, mem);
	arrays::density::swap();
}
