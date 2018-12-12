#include <SPH/pressure/iisph.cuh>
#include <utility/include_all.h>

basicFunctionType updateAccelerations(SPH::IISPH::Memory arrays) {
  checkedParticleIdx(i);
  arrays.acceleration[i] += (arrays.iisphDii[i] * arrays.pressure.first[i] + arrays.iisphSum[i]) /
                            (arrays.timestep * arrays.timestep);
  arrays.iisphDensityAdvection[i] =
      arrays.iisphDensityIteration[i] * arrays.rest_density + arrays.rest_density;
}

neighFunctionType IISPH_predictAdvection(const SPH::IISPH::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume));

  auto factor = math::square(arrays.timestep) /
                math::square(arrays.density[i] * arrays.rest_density) * arrays.rest_density;
  //auto rho_i = 1.f / math::square(arrays.density[i] * arrays.rest_density);

  std::decay_t<decltype(arrays.iisphDii[i])> d_ii{0.f, 0.f, 0.f, 0.f};

  iterateNeighbors(j) { d_ii += -vol[j] * factor * GPW_ij; }

  arrays.velocity.second[i] = arrays.velocity.first[i] + arrays.timestep * arrays.acceleration[i];
  arrays.iisphDii[i] = d_ii;
}

neighFunctionType IISPH_prepareSolver(const SPH::IISPH::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (v_adv, velocity.second));

  auto m_i = arrays.volume[i] * arrays.rest_density;
  auto rho_i = arrays.density[i] * arrays.rest_density;
  auto d_ii = arrays.iisphDii[i];

  decltype(rho_i) rho_adv{0.f};
  std::decay_t<decltype(arrays.iisphAiiOld[i])> a_ii{0.f};
  auto factor = -math::square(arrays.timestep) * m_i / math::square(rho_i);

  iterateNeighbors(j) {
    auto kernel = GPW_ij;
    auto m_j = arrays.volume[j] * arrays.rest_density;
    auto d_ji = -factor * kernel;

    rho_adv += arrays.timestep * m_j * math::dot3(v_adv[i] - v_adv[i], kernel);
    a_ii += m_j * math::dot3(d_ii - d_ji, kernel);
  }

  arrays.pressure.second[i] = 0.5f * arrays.pressure.first[i];
  arrays.iisphDensityAdvection[i] = rho_i + rho_adv;
  arrays.iisphAiiOld[i] = a_ii;
}

neighFunctionType IISPH_jacobiFirst(const SPH::IISPH::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume), (rho, density), (pressure, pressure.second));

  std::decay_t<decltype(arrays.iisphSum[i] / math::square(arrays.timestep))> kernel_sum{0.f, 0.f,
                                                                                        0.f, 0.f};

  iterateNeighbors(j) kernel_sum += -vol[j] * arrays.rest_density /
                                    math::square(rho[j] * arrays.rest_density) * pressure[j] *
                                    GPW_ij;

  arrays.iisphSum[i] = math::square(arrays.timestep) * kernel_sum;
  arrays.pressure.first[i] = arrays.pressure.second[i];
}

neighFunctionType IISPH_jacobiSecond(const SPH::IISPH::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (d_ij, iisphSum), (d_ii, iisphDii), (vol, volume),
               (pressure, pressure.first));

  auto rho_i = arrays.density[i] * arrays.rest_density;
  auto a_ii = arrays.iisphAiiOld[i];

  decltype(arrays.rest_density) sum{0.f};
  auto factor = -math::square(arrays.timestep) * vol[i] * arrays.rest_density / math::square(rho_i);

  iterateNeighbors(j) {
    auto kernel = GPW_ij;
    auto m_j = vol[j] * arrays.rest_density;

    auto d_ji = -factor * kernel;

    auto first = d_ij[i] - d_ii[i] * pressure[j];
    auto second = d_ij[i] - d_ji * pressure[i];

    sum += m_j * math::dot3(first - second, kernel);
  }

  sum = arrays.rest_density - arrays.iisphDensityAdvection[i] - sum;

  auto new_pressure = (1.f - 0.5f) * arrays.pressure.first[i];
  new_pressure += 0.5f / a_ii * sum;

  std::decay_t<decltype(arrays.iisphDensityIteration[i])> compression{0.f};
  if (new_pressure < 0.f || math::abs(a_ii) < 1e-30f)
    new_pressure = 0._Pa;

  if (new_pressure > 0._Pa)
    compression = (a_ii * new_pressure - sum + arrays.rest_density) / arrays.rest_density - 1.0f;
  else
    compression = arrays.iisphDensityAdvection[i] / arrays.rest_density - 1.f;
  compression.val = math::clamp(compression.val, 0.f, compression.val);

  arrays.iisphDensityIteration[i] = (compression);
  arrays.pressure.second[i] = new_pressure;
}

neighFunction(Predict, IISPH_predictAdvection, "IISPH: predict Advection", caches<float4, float>{});
neighFunction(Prepare, IISPH_prepareSolver, "IISPH: prepare Solver", caches<float4, float4>{});
neighFunction(Jacobi1, IISPH_jacobiFirst, "IISPH: jacobi First", caches<float4, float, float, float>{});
neighFunction(Jacobi2, IISPH_jacobiSecond, "IISPH: jacobi Second", caches<float4, float4, float4, float, float>{});
basicFunction(Update, updateAccelerations, "IISPH: update", caches<>{});

void SPH::IISPH::pressure_solve(Memory mem) {
  size_t iteration = 0;
  float limit = get<parameters::eta>() / 100.f;
  bool test = true;

  cuda::Memset(mem.iisphDensityAdvection, 0, sizeof(float) * mem.num_ptcls);
  cuda::Memset(mem.kernelBuffer, 0x0, sizeof(float4) * mem.num_ptcls);
  launch<Predict>(mem.num_ptcls, mem);
  launch<Prepare>(mem.num_ptcls, mem);
  do {
    launch<Jacobi1>(mem.num_ptcls, mem);
    launch<Jacobi2>(mem.num_ptcls, mem);
    get_u<parameters::density_error>() =
        math::getValue(algorithm::reduce_sum(mem.iisphDensityIteration, mem.num_ptcls) / ((float)mem.num_ptcls));

    iteration++;
    test = iteration < 2;
    test = test || (get<parameters::density_error>() > limit);
    test = test && (iteration < 256);
  } while (test);
  launch<Jacobi1>(mem.num_ptcls, mem);
  launch<Update>(mem.num_ptcls, mem);

  get<parameters::iterations>() = (int32_t) iteration;
}
