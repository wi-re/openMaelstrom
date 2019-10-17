#include <SPH/pressure/iisph17_band_rb.cuh>
#include <utility/include_all.h>

#define DEBUG_PRINT4(val) printf("[%d] @ %s : %d -> %s = %f %f %f %f\n", i, __FILE__, __LINE__, #val ,(val).x, (val).y, (val).z, (val).w);
#define DEBUG_PRINT(val) printf("[%d] @ %s : %d ->  %s = %f\n", i, __FILE__, __LINE__, #val ,(val));

neighFunctionType IISPH_predictAdvection2(SPH::IISPH17_BAND_RB::Memory arrays) {
	checkedParticleIdx(i);
  cache_arrays((pos, position));
  //TODO: why volume for fluids is different
  arrays.velocity[i] = arrays.velocity[i] + arrays.timestep * arrays.acceleration[i];
  // if (arrays.particle_type[i] == FLUID_PARTICLE && arrays.particle_type_x[i] == 0)
  if (arrays.particle_type[i] == FLUID_PARTICLE)
    arrays._apparentVolume[i] = arrays.volume[i] / arrays.density[i];
  else
  {
    float fluids = 0;
    iterateNeighbors(j){
      if (arrays.particle_type[j] == FLUID_PARTICLE)
        fluids += arrays.volume[j] * W_ij;
    }
    auto rad3 = arrays.radius * arrays.radius * arrays.radius * 8;
    arrays._apparentVolume[i] = arrays.volume[i] / (fluids + arrays.gamma + arrays.beta * rad3);
  }
    
}

neighFunctionType IISPH_prepareSolver2(SPH::IISPH17_BAND_RB::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (v_adv, velocity), (i_vol, _apparentVolume), (vol, volume),
               (omega, _omega));
  //TODO: source is not correct for rigid body
  auto s_i = 1.f / arrays.density[i] - 1.f;
  // auto s_i = 1.f - vol[i] / i_vol[i];

  auto V_i = basicVolume;
  V_i = 1.f;  
  // auto m_i = V_i * (arrays.particle_type[i] == FLUID_PARTICLE
  //   ? arrays.rest_density : arrays.rigidDensities[arrays.particle_type[i]-1]);
  
  auto m_i = V_i * arrays.rest_density;
    
  auto boundaryKernel =  pGPW_ib;// SWH::spikyGradient(pos[i], arrays);
  auto boundaryKernelSpline = pGPW_ib / V_i;//SWH::spikyGradient(pos[i], arrays)/ V_i;

  auto a_ii = V_i * V_i / m_i * math::dot3(boundaryKernelSpline, boundaryKernelSpline) ;
  // a_ii = 0;
  auto kernelSum = boundaryKernel ;
  if (arrays.particle_type[i] != FLUID_PARTICLE)
    kernelSum = {0, 0, 0, 0};
  s_i = s_i - arrays.timestep * math::dot3(v_adv[i], boundaryKernel);
  if (arrays.particle_type[i] != FLUID_PARTICLE)
    s_i = 0.f;
  iterateNeighbors(j) {
    auto spikyGradient = GPW_ij;
    auto m_j = vol[j] * (arrays.particle_type[j] == FLUID_PARTICLE
      ? arrays.rest_density : arrays.rigidDensities[arrays.particle_type[j]-1]);

    // auto m_j = vol[j] * arrays.rest_density;
  
    kernelSum += i_vol[j] * spikyGradient / omega[j];
    a_ii += i_vol[j] * i_vol[j] / m_j * math::dot3(spikyGradient, spikyGradient) / omega[j];
    
    if (arrays.particle_type[i] == FLUID_PARTICLE || (arrays.particle_type[i] != FLUID_PARTICLE && arrays.particle_type[j] == FLUID_PARTICLE))  
      s_i = s_i - arrays.timestep * i_vol[j] * math::dot3(v_adv[i] - v_adv[j], spikyGradient);
  }

  if (arrays.particle_type[i] == FLUID_PARTICLE)
    a_ii += math::dot3(kernelSum, kernelSum) / (arrays._apparentVolume[i] * (arrays.particle_type[i] == FLUID_PARTICLE
      ? arrays.rest_density : arrays.rigidDensities[arrays.particle_type[i]-1]));
  
  // a_ii += math::dot3(kernelSum, kernelSum) / (arrays._apparentVolume[i] * arrays.rest_density);
  
  // if (s_i < 0)
      // printf("s_i: %f, dens: %f, adv: %f, %f, %f\n", s_i.val, arrays.density[i].val, v_adv[i].val.x, v_adv[i].val.y, v_adv[i].val.z);

  arrays._sourceTerm[i] = s_i;
  arrays._Aii[i] = -math::square(arrays.timestep) * i_vol[i] * a_ii;
  arrays.pressure.second[i] = 0.75f * arrays.pressure.first[i];
  if (arrays.pressure.second[i] != arrays.pressure.second[i] || arrays.pressure.second[i] < 0.f)
		arrays.pressure.second[i] = 0.f;
}

neighFunctionType IISPH_jacobiFirst2(const SPH::IISPH17_BAND_RB::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (i_vol, _apparentVolume), (omega, _omega),
               (pressure, pressure.second));

  const auto factor = -1.f * i_vol[i] / (arrays.volume[i] * (arrays.particle_type[i] == FLUID_PARTICLE
    ? arrays.rest_density : arrays.rigidDensities[arrays.particle_type[i]-1]));

  // const auto factor = -1.f * i_vol[i] / (arrays.volume[i] * arrays.rest_density);
  // printf("factor:  %f\n",factor.val);
  auto V_i = basicVolume;
  V_i = 1.f;
  auto boundaryKernel = pGPW_ib;//SWH::spikyGradient(pos[i], arrays);

  auto kernelSum = factor * (pressure[i]) / 2.f * boundaryKernel;
  if (arrays.particle_type[i] != FLUID_PARTICLE)
    kernelSum = {0.f, 0.f, 0.f};
    // printf("kernel: %f\n", kernelSum.val);
  iterateNeighbors(j)
  {
    // if (kernelSum.val.x != kernelSum.val.x) DEBUG_PRINT4(kernelSum.val);
	  kernelSum += factor * i_vol[j] * (pressure[i] / omega[i] + pressure[j] / omega[j]) / 2.f * GPW_ij;
  }

  //if (arrays._predictedAcceleration[i].val.x != arrays._predictedAcceleration[i].val.x)
  //{
  //  DEBUG_PRINT4(boundaryKernel.val);
  //  DEBUG_PRINT4(kernelSum.val);
  //  DEBUG_PRINT(omega[i].val);
  //  DEBUG_PRINT(pressure[i].val);
  //  DEBUG_PRINT(i_vol[i].val);
  //  DEBUG_PRINT(factor.val);

  //  DEBUG_PRINT4(arrays._predictedAcceleration[i].val);
  //  DEBUG_PRINT4(arrays.acceleration[i].val);

  //}
  arrays._predictedAcceleration[i] = kernelSum;
  arrays.pressure.first[i] = arrays.pressure.second[i];
}

neighFunctionType IISPH_jacobiSecond2(const SPH::IISPH17_BAND_RB::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (acc, _predictedAcceleration));
  alias_arrays((vol, volume), (i_vol, _apparentVolume));

  auto V_i = basicVolume;
  V_i = 1.f;
  auto boundaryKernel = pGPW_ib;//SWH::spikyGradient(pos[i], arrays);

  auto kernelSum = math::square(arrays.timestep) * math::dot3(acc[i], boundaryKernel);
  if (arrays.particle_type[i] != FLUID_PARTICLE)
    kernelSum = 0.f;
  // auto kernelSum2 = math::square(arrays.timestep) * V_i * math::dot3(acc[i], boundaryKernel);

  //int32_t ncnt_f = 0;
  //int32_t ncnt_r = 0;
  // iterateNeighbors(j) {
  //     kernelSum += math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[i] - acc[j], GPW_ij);
  //     if (arrays.particle_type[j] == FLUID_PARTICLE) ncnt_f++;
  //     else ncnt_r++;
  // }

  
  if (arrays.particle_type[i] == FLUID_PARTICLE)
  {
    iterateNeighbors(j) 
      // kernelSum += math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[i] - acc[j], GPW_ij);;
      kernelSum += math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[i] - (arrays.particle_type[j] == FLUID_PARTICLE ? acc[j] : 0), GPW_ij);;
  }
  else
  {
    //kernelSum = 0;
    iterateNeighbors(j)
      if (arrays.particle_type[j] == FLUID_PARTICLE)
        kernelSum += -math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[j], GPW_ij);;
    // kernelSum += math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[i] - (arrays.particle_type[j] != FLUID_PARTICLE ? 0 : acc[j]), GPW_ij);;

  }

  float omega = math::max(0.5f, 0.5f * i_vol[i] / vol[i]);
  if (arrays.particle_type[i] != FLUID_PARTICLE) omega = 0.5f * vol[i].val / (arrays.radius.val*arrays.radius.val*arrays.radius.val);

  auto pressure = math::max(arrays.pressure.first[i] + omega * (arrays._sourceTerm[i] - kernelSum) / arrays._Aii[i], 0.f);
  
  // if (pressure > 100)
  // if (arrays.pressure.first[i].val == 0 /* && pressure > 100 */)
  //   printf("pressure: new %f, old %f, omega %f, src %f, kernel %f, diff %f, aii %.10f, \n", pressure.val, arrays.pressure.first[i].val, omega,
  //     arrays._sourceTerm[i].val, kernelSum.val, (arrays._sourceTerm[i] - kernelSum).val, arrays._Aii[i].val);

  auto residual = kernelSum - arrays._sourceTerm[i];

  if (math::abs(arrays._Aii[i]) < 1e-20f || pressure != pressure || pressure > 1e16f) {
		pressure = 0._Pa;
		residual = decltype(residual)(0.f);
	}

  arrays.pressure.second[i] = pressure;
  arrays._volumeError[i] = math::max(residual, 0.f).val;
}

basicFunctionType updateAccelerations2(SPH::IISPH17_BAND_RB::Memory arrays) {
  checkedParticleIdx(i);
  arrays.velocity[i] = arrays.velocity[i] - arrays.timestep * arrays.acceleration[i];
  //if (arrays.particle_type[i] == FLUID_PARTICLE)
  
  arrays.acceleration[i] += arrays._predictedAcceleration[i];
  // if (abs(arrays._predictedAcceleration[i].val.x) > 2000)
  //   printf("acc: %f, %f, %f\n", arrays._predictedAcceleration[i].val.x, arrays._predictedAcceleration[i].val.y, arrays._predictedAcceleration[i].val.z);
  
  
}

neighFunctionType _omega(const SPH::IISPH17_BAND_RB::Memory arrays) {
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
  arrays._omega[i] = omega;
  arrays._omega[i] = 1.f;

  // printf("particle: %i, iishpomega: %f, omega: %f\n", i, arrays._omega[i].val, omega);
}

neighFunction(Omega, _omega, "IISPH17_BAND_RB: compute omega", caches<float4, float>{});
neighFunction(Predict, IISPH_predictAdvection2, "IISPH17_BAND_RB: predict Advection", caches<float4>{});
neighFunction(Prepare, IISPH_prepareSolver2, "IISPH17_BAND_RB: prepare Solver", caches<float4, float4, float, float, float>{});
neighFunction(Jacobi1, IISPH_jacobiFirst2, "IISPH17_BAND_RB: jacobi First", caches<float4, float, float, float>{});
neighFunction(Jacobi2, IISPH_jacobiSecond2, "IISPH17_BAND_RB: jacobi Second", caches<float4, float4, float>{});
basicFunction(Update, updateAccelerations2, "IISPH17_BAND_RB: updaate Acceleration");

void SPH::IISPH17_BAND_RB::pressure_solve(Memory mem) {
  size_t iteration = 0;
  float limit = get<parameters::eta>() / 100.f;
  bool test = true;

  //currently omega alwasy 1.0f, forced to be 1.0f (useless) | for adaptivity
  launch<Omega>(mem.num_ptcls, mem);
  //
  launch<Predict>(mem.num_ptcls, mem);
  launch<Prepare>(mem.num_ptcls, mem);
  do {
    iteration++;
    launch<Jacobi1>(mem.num_ptcls, mem);
    launch<Jacobi2>(mem.num_ptcls, mem);
    if (iteration > 1) {     
      get_u<parameters::density_error>() = math::getValue(algorithm::reduce_sum(mem._volumeError, mem.num_ptcls) / algorithm::reduce_sum(arrays::volume::ptr, mem.num_ptcls));
    }
    test = iteration < 2;
    test = test || (get<parameters::density_error>() > limit);
    test = test && (iteration < 256);
    // std::cout << iteration << " -> " << get<parameters::density_error>() << std::endl;
  } while (test);
  launch<Jacobi1>(mem.num_ptcls, mem);
  launch<Update>(mem.num_ptcls, mem);
  get<parameters::iterations>() = (int32_t) iteration;
}
