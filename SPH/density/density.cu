#define NEW_STYLE
#include <SPH/density/density.cuh>
#include <utility/include_all.h>
#include <utility/SPH/boundaryFunctions.h>

/** This function calculates a density estimate using SPH for all particles and additionally uses a
 * lookup table to improve the boundary density contribution of flat boundary planes (no
 * source/novel). Only calculates unit density not actual density  **/
 neighFunctionType estimate_density(SPH::Density::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume));	 
	auto unit_density = SWH::spline4(pos[i], arrays);
	iterateNeighbors(j)
	  unit_density += vol[j] * W_ij;
	
	arrays.density[i] = unit_density;
}

neighFunction(estimateDensity, estimate_density, "Estimate Density", caches<float4, float>{});

void SPH::Density::estimate_density(Memory mem) {
	launch<estimateDensity>(mem.num_ptcls, mem); 
}
 