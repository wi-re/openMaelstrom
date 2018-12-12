#include <SPH/pressure/dfsph.cuh>
#include <utility/include_all.h>

basicFunctionType boundaryCorrect(SPH::DFSPH::Memory arrays) {
  checkedParticleIdx(i);
  auto p = arrays.position[i];
  auto v = arrays.velocity.second[i];
  auto volume = PI4O3 * math::power<3>(arrays.radius);
  auto H = support_from_volume(volume) * kernelSize();
  auto r = H * 0.4f;

  auto POS = boundary::POSfunction(p + v * arrays.timestep, arrays);
  if (POS.val.w <= r)
    v -= math::dot3(POS, v) * POS;

  auto VOS = volumeBoundary::volumeDistance(p + v * arrays.timestep, arrays);
  if (VOS.val.w <= r)
    v -= math::dot3(VOS, v) * VOS;
  v.val.w = 0.f;
  arrays.velocity.second[i] = v;
}

basicFunctionType initialize(SPH::DFSPH::Memory arrays, arrays::dfsphKappa::unit_type* kappa) {
  checkedParticleIdx(i);

  arrays.velocity.second[i] = arrays.velocity.first[i] + arrays.timestep * arrays.acceleration[i];
  kappa[i] = math::max(kappa[i], -0.5f);
}

neighFunctionType computeAlpha(SPH::DFSPH::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume), (rho, density));

  auto baseSum = vol[i] * arrays.rest_density * GPW_ib;
  auto squareSum = math::dot3(baseSum, baseSum);

  iterateNeighbors(j) {
    auto v = vol[j] * arrays.rest_density * GPW_ij;
    baseSum += v;
    squareSum += math::dot3(v, v);
  }

  arrays.dfsphAlpha[i] = rho[i] * arrays.rest_density / math::max(math::dot3(baseSum, baseSum) + squareSum, 1e-6f);
}

neighFunctionType kappa(SPH::DFSPH::Memory arrays, bool density) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (velocity, velocity.second), (vol, volume));

  auto dpdt = vol[i] * arrays.rest_density * math::dot3(velocity[i], GPW_ib);

  iterateNeighbors(j)
	  dpdt += vol[j] * arrays.rest_density * math::dot3(velocity[i] - velocity[j] , GPW_ij);

  dpdt = math::max(dpdt, 0.f);
  auto rho_s = math::max(arrays.rest_density, arrays.rest_density + arrays.timestep * dpdt);
  
  arrays.dfsphRhoStar[i] = rho_s.val;
  arrays.dfsphDpDt[i] = dpdt;

  if (density) {
	  arrays.dfsphKappa.first[i] = (rho_s - arrays.rest_density) * arrays.dfsphAlpha[i] / (arrays.timestep * arrays.timestep);
	  arrays.dfsphKappa.second[i] += arrays.dfsphKappa.first[i];
  }
  else {
	  arrays.dfsphKappaDivergence.first[i] = 1.f / arrays.timestep * dpdt * arrays.dfsphAlpha[i];
	  arrays.dfsphKappaDivergence.second[i] += arrays.dfsphKappaDivergence.first[i];
  }
}

neighFunctionType solve(SPH::DFSPH::Memory arrays, arrays::dfsphKappa::unit_type* kappa) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (velocity, velocity.second), (rho, density));
  auto k = cache_array(kappa, rho.offset);

  auto sum = arrays.volume[i] * arrays.rest_density * k[i] / (rho[i] * arrays.rest_density) * GPW_ib;

  iterateNeighbors(j) 
	  if (math::abs(k[i] + k[j]) > 1e-6f)
		  sum += arrays.volume[j] * arrays.rest_density * (k[i] / (rho[i] * arrays.rest_density) + k[j] / (rho[j] * arrays.rest_density)) * GPW_ij;

  arrays.velocity.second[i] -= - arrays.timestep * sum;
}

basicFunctionType finalize(SPH::DFSPH::Memory arrays, decltype(arrays.dfsphKappa) kappa) {
  checkedParticleIdx(i);
  arrays.acceleration[i] = (arrays.velocity.second[i] - arrays.velocity.first[i]) / arrays.timestep;
  kappa.first[i] = kappa.second[i];
}

basicFunction(Boundary, boundaryCorrect, "DFSPH: Boundary correction");
neighFunction(Alpha, computeAlpha, "DFSPH: compute Alpha", caches<float4, float, float>{});

basicFunction(Init, initialize, "DFSPH: Init");
neighFunction(Kappa, kappa, "DFSPH: compute Kappa", caches<float4, float4, float, float>{});
neighFunction(Solve, solve, "DFSPH: solve", caches<float4, float4, float, float>{});
basicFunction(Finalize, finalize, "DFSPH: finalize");


void SPH::DFSPH::density_solve(Memory arrays) {
  int32_t it = 0;
  int32_t numPtcls = arrays.num_ptcls;
  auto limit = get<parameters::dfsph_density_eta>() * 0.001f * 1.f;
  decltype(limit) error{0.f};

  launch<Alpha>(numPtcls, arrays);
  launch<Init>(numPtcls, arrays, arrays.dfsphKappa.first);
  launch<Solve>(numPtcls, arrays, arrays.dfsphKappa.first);
  launch<Boundary>(numPtcls, arrays);
  cuda::Memset(arrays::dfsphKappa::rear_ptr, 0, arrays::dfsphKappa::alloc_size);

  do {
    launch<Kappa>(numPtcls, arrays, true);
    launch<Solve>(numPtcls, arrays, arrays.dfsphKappa.first);
    launch<Boundary>(numPtcls, arrays);
    error =  math::getValue(algorithm::reduce_sum(arrays.dfsphRhoStar, numPtcls) / ((float)numPtcls) - 1.f);

    //std::cout << "Density: " << it << " -> " << error.val << " : " << limit.val << std::endl;
  } while (++it < 2 || (error > limit));

  launch<Finalize>(numPtcls, arrays, arrays.dfsphKappa);
}
void SPH::DFSPH::divergence_solve(Memory arrays) {
  int32_t it = 0;
  int32_t numPtcls = arrays.num_ptcls;
  auto limit = get<parameters::dfsph_divergence_eta>() * 0.01f * arrays.rest_density / 1.0_s;
  decltype(limit) error{0.f};

  launch<Alpha>(numPtcls, arrays);
  launch<Init>(numPtcls, arrays, arrays.dfsphKappaDivergence.first);
  launch<Solve>(numPtcls, arrays, arrays.dfsphKappaDivergence.first);
  launch<Boundary>(numPtcls, arrays);
  cuda::Memset(arrays::dfsphKappaDivergence::rear_ptr, 0, arrays::dfsphKappaDivergence::alloc_size);

  do {
	  launch<Kappa>(numPtcls, arrays, false);
	  launch<Solve>(numPtcls, arrays, arrays.dfsphKappaDivergence.first);
    launch<Boundary>(numPtcls, arrays);
    error = algorithm::reduce_sum(arrays.dfsphDpDt, numPtcls) / ((float)numPtcls);

    //std::cout << "Divergence: " << it << " -> " << error.val << " : " << limit.val << std::endl;
  } while (++it < 2 || (error > limit));

  launch<Finalize>(numPtcls, arrays, arrays.dfsphKappaDivergence);
}
