
#include <SPH/integration/simple.cuh>
#include <utility/include_all.h>
#include "dummy.h" 
// This function does a lot of things.
/*
        - It corrects the particle position against all boundary planes
				- It updates the position using the new velocity and corrects the velocity against all planes
				- It calculates the new position and corrects it against all boundary planes
				- It calculates a shifting algorithm for position and velocity to improve particle sampling
				- It updates the lifetime of the particle by the timestep
*/
// These are a lot of steps but due to most of them being fairly quick to calculate the only
// expensive part is the shifting which involves a single SPH interpolation.
hostDeviceInline auto correct_position(float4_u<SI::m> p, float_u<SI::m> r, SPH::Integration::Memory& arrays) {
  auto POS = boundary::POSfunction(p, arrays);
  if (POS.val.w <= r) {
    float4_u<> n{POS.val.x, POS.val.y, POS.val.z, 0.f};
    float_u<SI::m> impulseStrength = float_u<SI::m>(POS.val.w) - r;
    float4_u<> boundImpulse = -n;
    p += boundImpulse * impulseStrength;
  }
  auto VOS = volumeBoundary::volumeDistance(p, arrays);
  if (VOS.val.w <= r) {
    float4_u<> n{VOS.val.x, VOS.val.y, VOS.val.z, 0.f};
    float_u<SI::m> impulseStrength = float_u<SI::m>(VOS.val.w) - r;
    float4_u<> boundImpulse = -n;
    p += boundImpulse * impulseStrength;
  }
  return p;
}

hostDeviceInline auto correct_velocity(float4_u<SI::m> p, float4_u<SI::velocity> v, float_u<SI::m> r, SPH::Integration::Memory& arrays) {
  auto POS = boundary::POSfunction(p + v * arrays.timestep, arrays);
  if (POS.val.w <= r)
    v -= math::dot3(POS, v) * POS;  

  auto VOS = volumeBoundary::volumeDistance(p + v * arrays.timestep, arrays);
  if (VOS.val.w <= r)
    v -= math::dot3(VOS, v) * VOS;
  v.val.w = 0.f;
  return v;
}
basicFunctionType correct_particles(SPH::Integration::Memory arrays) {
  checkedParticleIdx(i);
  auto volume = PI4O3 * math::power<3>(arrays.radius);
  auto H = support_from_volume(volume) * kernelSize();
  auto r = H * 0.25f;
  //r = H / kernelSize();
  auto dt = arrays.timestep;

  auto p = correct_position(arrays.position.first[i], r, arrays);
  auto v = correct_velocity(p, arrays.velocity.first[i], r, arrays);

  arrays.velocity.first[i] = v;
  arrays.position.first[i] = correct_position(p + dt * v, r, arrays);
}
neighFunctionType position_update(SPH::Integration::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position.first), (vol, volume), (vel, velocity.first), (den, density));

  auto v = arrays.velocity.first[i];
  auto p = arrays.position.first[i];

  int32_t N_i = 0;
  auto r_i = 0.0_m;
  float4_u<SI::recip_2<SI::m>> X_i;
  float_u<SI::velocity> v_local_max;
  float_u<SI::Hz> vel_gradient{ 0.f };

  iterateNeighbors(j) {
    if (W_ij == 0.f)
      continue;
	vel_gradient += vol[j] / den[j] * math::dot3((vel[j] - vel[i]), GW_ij);
    v_local_max = math::max(v_local_max, math::length3(vel[j]));
    if (i == j)
      continue;
	auto x_ij = math::distance3(pos[i], pos[j]);
    r_i += x_ij;
    N_i++;
    X_i += vol[j] / vol[i] * (pos[i] - pos[j]) / (x_ij * x_ij * x_ij);
  }
  r_i /= (float)N_i;
  auto X_irr = X_i * r_i * r_i;
  X_irr.val.w = 0.f;

  auto C = 0.004f;
  auto a = arrays.max_velocity * arrays.timestep * C;
  auto b = v_local_max * arrays.timestep * C;
  auto d = r_i * 0.001_s / arrays.timestep;
  auto f = math::max(b, d);
  auto Cdash = math::min(a, f);
  auto delta_ii = Cdash * X_irr;
  auto delta_ii_l = math::length3(delta_ii);
  math::unit_assign<4>(delta_ii, 0.0_m);
  if (N_i < 10)
    delta_ii = float4_u<SI::m>{0.f, 0.f, 0.f, 0.f};
  if (delta_ii_l < 0.75f * math::unit_get<4>(p))
    arrays.position.second[i] = p + delta_ii;
  else
    arrays.position.second[i] = p;
  if (delta_ii_l < 0.75f * math::unit_get<4>(p))
    arrays.velocity.first[i] = v + delta_ii * vel_gradient;
  else
    arrays.velocity.first[i] = v;

  arrays.lifetime[i] = arrays.lifetime[i] + arrays.timestep;
}

// This function updates the velocity of all particles based on their current acceleration and
// stores the length of the particle velocity in an auxillary array to calculate the shifting
// process.
basicFunctionType velocity_update(SPH::Integration::Memory arrays) {
  checkedParticleIdx(i);
  auto v = arrays.velocity.first[i];
  auto a = arrays.acceleration[i];
  auto dt = arrays.timestep;

  v += dt * a;
  v.val.w = 0.f;

  arrays.velocity.second[i] = v;
  arrays.maxVelocity[i] = math::length(v);
}

// This function calculates the ideal timestep for the simulation based on a CFL condition. This CFL
// condition only considers the term based on the particle support radius and velocity and not any
// shock effects. These could be included but usually don't improve results enough to be worthwhile
// the cost of longer simulation.
basicFunctionType cflCalculate(SPH::Integration::Memory arrays) {
  checkedParticleIdx(i);
  auto v = arrays.velocity.first[i];
  auto p = arrays.position.first[i];

  arrays.cflValue[i] = 0.4f * support_H(p) / math::length3(v);
}

basicFunction(updatePosition,correct_particles, "Integrate: Boundary");
neighFunction(particleShifting, position_update, "Integrate: Positions", caches<float4, float4, float, float>{});
basicFunction(calculateCFL, cflCalculate, "Integrate: CFL");
basicFunction(updateVelocity, velocity_update, "Integrate: Velocity");

// This function updates the positions of all particles and additionally updates the timestep
// according to a CFL condition.
void SPH::Integration::update_positions(Memory mem) {
	//std::cout << arrays::previousPosition::ptr << " - " << arrays::position::ptr << std::endl;
	if (get<parameters::rayTracing>() == true)
		cuda::memcpy(arrays::previousPosition::ptr, arrays::position::ptr, sizeof(float4) * mem.num_ptcls, cudaMemcpyDeviceToDevice);
  launch<updatePosition>(mem.num_ptcls, mem);
  launch<particleShifting>(mem.num_ptcls, mem);
  info<arrays::position>().swap();
  launch<calculateCFL>(mem.num_ptcls, mem);

  float_u<SI::s> dt = algorithm::reduce_min(mem.cflValue, mem.num_ptcls);

  dt = math::clamp(dt, get<parameters::timestep_min>(), get<parameters::timestep_max>());
  dt = math::clamp(dt, get<parameters::timestep>() * 0.9f, get<parameters::timestep>() * 1.1f);

  get_u<parameters::timestep>() = dt;
}

// This function updates the velocity of all particles based on the current velocity and prepares
// the max velocity parameter for the shifting algorithm.
void SPH::Integration::update_velocities(Memory mem) {
  launch<updateVelocity>(mem.num_ptcls, mem);

  info<arrays::velocity>().swap();
  get_u<parameters::max_velocity>() = algorithm::reduce_max(mem.maxVelocity, mem.num_ptcls);
}
