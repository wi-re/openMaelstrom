#include <SPH/pressure/iisph17.cuh>
#include <utility/include_all.h>

basicFunctionType IISPH_predictAdvection2(SPH::IISPH17::Memory arrays) {
	checkedParticleIdx(i);
  arrays.velocity[i] = arrays.velocity[i] + arrays.timestep * arrays.acceleration[i];
  arrays.iisphVolume[i] = arrays.volume[i] / arrays.density[i];
}

neighFunctionType IISPH_prepareSolver2(SPH::IISPH17::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (v_adv, velocity), (i_vol, iisphVolume), (vol, volume),
               (omega, iisphOmega));

  auto s_i = 1.f / arrays.density[i] - 1.f;

  auto V_i = basicVolume;
  auto m_i = V_i * arrays.rest_density;
  auto boundaryKernel = SWH::spikyGradient(pos[i], V_i, arrays);
  auto boundaryKernelSpline = SWH::spikyGradient(pos[i], V_i, arrays);

  auto a_ii = V_i * V_i / m_i * math::dot3(boundaryKernelSpline, boundaryKernelSpline) ;
  auto kernelSum = V_i * boundaryKernel ;
  s_i = s_i - arrays.timestep * V_i * math::dot3(v_adv[i], boundaryKernel);

  iterateNeighbors(j) {
    auto spikyGradient = GPW_ij;
    auto m_j = vol[j] * arrays.rest_density;

    kernelSum += i_vol[j] * spikyGradient / omega[j];
    a_ii += i_vol[j] * i_vol[j] / m_j * math::dot3(spikyGradient, spikyGradient) / omega[j];
    s_i = s_i - arrays.timestep * i_vol[j] * math::dot3(v_adv[i] - v_adv[j], spikyGradient);
  }

  a_ii += math::dot3(kernelSum, kernelSum) / (arrays.iisphVolume[i] * arrays.rest_density);

  arrays.iisphSource[i] = s_i;
  arrays.iisphAii[i] = -math::square(arrays.timestep) * i_vol[i] * a_ii;
  arrays.pressure.second[i] = 0.75f * arrays.pressure.first[i];
}

neighFunctionType IISPH_jacobiFirst2(const SPH::IISPH17::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (i_vol, iisphVolume), (omega, iisphOmega),
               (pressure, pressure.second));

  const auto factor = -1.f * i_vol[i] / (arrays.volume[i] * arrays.rest_density);

  auto V_i = basicVolume; 
  auto boundaryKernel = SWH::spikyGradient(pos[i], V_i, arrays);

  auto kernelSum = factor * V_i * (pressure[i] ) / 2.f * boundaryKernel;

  iterateNeighbors(j) 
	  kernelSum += factor * i_vol[j] * (pressure[i] / omega[i] + pressure[j] / omega[j]) / 2.f * GPW_ij;

  arrays.iisphAcceleration[i] = kernelSum;
  arrays.pressure.first[i] = arrays.pressure.second[i];
}

neighFunctionType IISPH_jacobiSecond2(const SPH::IISPH17::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (acc, iisphAcceleration));
  alias_arrays((vol, volume), (i_vol, iisphVolume));

  auto V_i = basicVolume;
  auto boundaryKernel = SWH::spikyGradient(pos[i], V_i, arrays);

  auto kernelSum = math::square(arrays.timestep) * V_i * math::dot3(acc[i], boundaryKernel);

  iterateNeighbors(j) 
    kernelSum += math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[i] - acc[j], GPW_ij);;
  
  float omega = math::max(0.5f, 0.5f * i_vol[i] / vol[i]);

  auto pressure = math::max(arrays.pressure.first[i] + omega * (arrays.iisphSource[i] - kernelSum) / arrays.iisphAii[i], 0.f);
  auto residual = kernelSum - arrays.iisphSource[i];

  if (math::abs(arrays.iisphAii[i]) < 1e-30f) {
    pressure = 0._Pa;
    residual = decltype(residual)(0.f);
  }

  arrays.pressure.second[i] = pressure;
  arrays.iisphVolumeError[i] = math::max(residual, -arrays.eta / 100.f).val;
}

basicFunctionType updateAccelerations2(SPH::IISPH17::Memory arrays) {
  checkedParticleIdx(i);
  arrays.velocity[i] = arrays.velocity[i] - arrays.timestep * arrays.acceleration[i];
  arrays.acceleration[i] += arrays.iisphAcceleration[i];
}

neighFunctionType iisphOmega(const SPH::IISPH17::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume));

	auto x_i = pos[i];
	auto H_i = support_H(x_i);
	auto rho_i = arrays.density[i];

  float omega = 1.f;

  iterateNeighbors(j) {
	  auto x_j = pos[j];
	  auto H_j = support_H(x_j);
	  auto dWdr = kernelDerivative(x_i, x_j);
	  auto W = kernel(x_i, x_j);
	  //auto x_ij = math::distance3(x_i, x_j);
	  auto f = -/*x_ij / H_i **/ dWdr - 3.f / (H_i + H_j) * 2.f * W;
	  omega += H_i * f * vol[j] / (3.f * rho_i);
  }
  if (omega != omega)
	  omega = 1.f;
  else
	  omega = math::clamp(omega, 0.125f, 2.5f);
  arrays.iisphOmega[i] = omega;
  arrays.iisphOmega[i] = 1.f;
}

neighFunction(Omega, iisphOmega, "IISPH17: compute omega", caches<float4, float>{});
basicFunction(Predict, IISPH_predictAdvection2, "IISPH17: predict Advection");
neighFunction(Prepare, IISPH_prepareSolver2, "IISPH17: prepare Solver", caches<float4, float4, float, float, float>{});
neighFunction(Jacobi1, IISPH_jacobiFirst2, "IISPH17: jacobi First", caches<float4, float, float, float>{});
neighFunction(Jacobi2, IISPH_jacobiSecond2, "IISPH17: jacobi Second", caches<float4, float4, float>{});
basicFunction(Update, updateAccelerations2, "IISPH17: updaate Acceleration");

void SPH::IISPH17::pressure_solve(Memory mem) {
  size_t iteration = 0;
  float limit = get<parameters::eta>() / 100.f;
  bool test = true;
  launch<Omega>(mem.num_ptcls, mem);
  launch<Predict>(mem.num_ptcls, mem);
  launch<Prepare>(mem.num_ptcls, mem);
  do {
    iteration++;
	launch<Jacobi1>(mem.num_ptcls, mem);
	launch<Jacobi2>(mem.num_ptcls, mem);
    if (iteration > 1) {     
       get_u<parameters::density_error>() = math::getValue(algorithm::reduce_sum(mem.iisphVolumeError, mem.num_ptcls) / ((float) mem.num_ptcls));
    }
    test = iteration < 2;
    test = test || (get<parameters::density_error>() > limit);
    test = test && (iteration < 256);
  } while (test);
  launch<Jacobi1>(mem.num_ptcls, mem);
  launch<Update>(mem.num_ptcls, mem);
  get<parameters::iterations>() = (int32_t) iteration;
}
