
#include "dummy.h"
#include <SPH/integration/simple.cuh>
#include <utility/bullet/DynamicsWorld.h>
#include <utility/include_all.h>
// This function does a lot of things.
/*
        - It corrects the particle position against all boundary planes
                                - It updates the position using the new velocity and corrects the velocity against all
   planes
                                - It calculates the new position and corrects it against all boundary planes
                                - It calculates a shifting algorithm for position and velocity to improve particle
   sampling
                                - It updates the lifetime of the particle by the timestep
*/

#define hDistance 0.f

// These are a lot of steps but due to most of them being fairly quick to calculate the only
// expensive part is the shifting which involves a single SPH interpolation.
hostDeviceInline auto correct_position(float4_u<SI::m> p, float_u<SI::volume> volume, SPH::Integration::Memory &arrays, float4_u<SI::velocity> vel) {
	for (int32_t b = 0; b < arrays.boundaryCounter; ++b) {
		auto VOS = volumeBoundary::distance_fn(p, volume, arrays, b);
		if (VOS.val.w <= 0.f) {
			float4_u<> n{ VOS.val.x, VOS.val.y, VOS.val.z, 0.f };
			float_u<SI::m> impulseStrength = float_u<SI::m>(VOS.val.w);
			float4_u<> boundImpulse = -math::normalize3(n);
			p += boundImpulse * impulseStrength;
			///p -= vel * arrays.timestep;
		}
	}
	//return p;
  auto POS = planeBoundary::distance(p, volume, arrays);
  if (POS.val.w <= -0.f*hDistance) {
    float4_u<> n{POS.val.x, POS.val.y, POS.val.z, 0.f};
	float_u<SI::m> impulseStrength = float_u<SI::m>(POS.val.w) + 0.f*hDistance;
    float4_u<> boundImpulse = -math::normalize3(n);
    p += boundImpulse * impulseStrength;
  }
  return p;
}

neighFunctionHostType walkRigidNeigh(SPH::Integration::Memory arrays, float4_u<SI::m> *pos,
                                     float4_u<SI::acceleration> *acc, float_u<SI::volume> *vol,
                                     int32_t *particle_type) {
  checkedParticleIdx(i);
  if (particle_type[i] == FLUID_PARTICLE)
    return;
  auto dens = DynamicsWorld::getInstance()->getDensity(particle_type[i]);

  float4_u<SI::N> frc = dens * vol[i] * acc[i];
  DynamicsWorld::getInstance()->applyForce(frc.val, particle_type[i], pos[i].val);
}

neighFunctionHost(walkRigidNeighE, walkRigidNeigh, "Integrate: update rigids");

void updateRigids(SPH::Integration::Memory mem) {

  // auto pos = mem.position.first;
  // static int32_t* particle_type = (int32_t*) malloc(sizeof(int32_t) * parameters::max_numptcls{});
  // cudaMemcpy(particle_type, mem.particle_type, sizeof(int32_t) * parameters::max_numptcls{}, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  static auto pos = std::make_unique<float4_u<SI::m>[]>(mem.max_numptcls);
  static auto accels = std::make_unique<float4_u<SI::acceleration>[]>(mem.max_numptcls);
  static auto volumes = std::make_unique<float_u<SI::volume>[]>(mem.max_numptcls);
  static auto types = std::make_unique<int32_t[]>(mem.max_numptcls);
  cudaMemcpy(pos.get(), mem.position.first, sizeof(float4) * mem.num_ptcls, cudaMemcpyDeviceToHost);
  cudaMemcpy(accels.get(), mem.acceleration, sizeof(float4) * mem.num_ptcls, cudaMemcpyDeviceToHost);
  cudaMemcpy(volumes.get(), mem.volume, sizeof(float) * mem.num_ptcls, cudaMemcpyDeviceToHost);
  cudaMemcpy(types.get(), mem.particle_type, sizeof(int32_t) * mem.num_ptcls, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  launchHost<walkRigidNeighE>(mem.num_ptcls, mem, pos.get(), accels.get(), volumes.get(), types.get());
  cudaDeviceSynchronize();

  // auto body = DynamicsWorld::getInstance()->rigid_bodies[0];
  // auto frc = body->getTotalForce();

  DynamicsWorld::getInstance()->stepSimulation(get<parameters::timestep>());

  // auto cntr = body->getCenterOfMassPosition();
  // std::cout << "center mass: " << cntr.getX() << " " << cntr.getY() << " " << cntr.getZ() << std::endl;
  // std::cout << "timestep: " << get<parameters::timestep>() << std::endl;
  // auto body
  // DynamicsWorld::getInstance()->writeRotation();

  for (int j = 0; j < DynamicsWorld::getInstance()->getRigidBodiesCount(); j++) {
    mem.rigidLinearVelocities[j].val = DynamicsWorld::getInstance()->getLinearVelocity(j);

    for (int32_t i = 0; i < mem.num_ptcls; ++i) {
      // auto position = mem.positions.first[i];
      if (mem.particle_type[i] == j + 1) {
        auto tmp = mem.position.first[i].val;
        // auto tmp1 = DynamicsWorld::getInstance()->moveRigidParticleFake(tmp, line);
        auto tmp1 = DynamicsWorld::getInstance()->moveRigidParticle(tmp, j);
        mem.position.first[i].val = tmp1;
        // std::cout << "rigid z: " << tmp1.z << std::endl;
      }
    }

    // std::cout << "XXXX: " << mem.position.first[0].val.z << std::endl;
    DynamicsWorld::getInstance()->updateCenter(j);
    DynamicsWorld::getInstance()->clearForces(j);
  }

  cudaDeviceSynchronize();
}

hostDeviceInline auto correct_velocity(float4_u<SI::m> p, float4_u<SI::velocity> v, float_u<SI::volume> volume, float_u<SI::s> dt, 
                                       SPH::Integration::Memory &arrays) {
for (int32_t b = 0; b < arrays.boundaryCounter; ++b) {
	auto VOS = volumeBoundary::distance_fn(p + v * dt, volume, arrays, b);
	if (VOS.val.w <= -0.f*hDistance) {
//		// auto d = math::dot3(VOS, v);
//		// if (d < 0.f)
//		// v += -d * VOS;
//		// else
//		// v += - d * VOS;
//		// if (fabs(POS.val.x) > fabs(POS.val.y) && fabs(POS.val.x) > fabs(POS.val.z))
//		// v.val.x = 0.f;
//		// if (fabs(POS.val.y) > fabs(POS.val.x) && fabs(POS.val.x) > fabs(POS.val.z))
//		// v.val.y = 0.f;
//		// if (fabs(POS.val.z) > fabs(POS.val.y) && fabs(POS.val.z) > fabs(POS.val.x))
//		// v.val.z = 0.f;
//		// v = -v;
//		// v = 0.f;
		VOS = math::normalize3(VOS);
		v -= math::dot3(VOS, v) * VOS;// *arrays.boundaryDampening;
	}
}
//return v;
  auto POS = planeBoundary::distance(p + v * dt,volume, arrays);
  if (POS.val.w <= - 0.f * hDistance) {
    // auto d = math::dot3(POS, v);
    // if (d < 0.f)
    // v += - d * POS;
    // else
    // v += - d * POS;
    // if (fabs(POS.val.x) > fabs(POS.val.y) && fabs(POS.val.x) > fabs(POS.val.z))
    // v.val.x = 0.f;
    // if (fabs(POS.val.y) > fabs(POS.val.x) && fabs(POS.val.x) > fabs(POS.val.z))
    // v.val.y = 0.f;
    // if (fabs(POS.val.z) > fabs(POS.val.y) && fabs(POS.val.z) > fabs(POS.val.x))
    // v.val.z = 0.f;
    // v = -v;
    // v = 0.f;
    POS = math::normalize3(POS);
	v -= math::dot3(POS, v) * POS;// *arrays.boundaryDampening;
  }

  // v -= math::dot3(VOS, v) * VOS;
  v.val.w = 0.f;
  return v;
}
basicFunctionType correct_particles(SPH::Integration::Memory arrays, float_u<SI::s> dt) {
  checkedParticleIdx(i);
  if (arrays.particle_type[i] != FLUID_PARTICLE /*|| arrays.particle_type_x[i] == 1*/) {
    // printf("vel of type_x: %f, %f, %f\n", arrays.velocity.first[i].val.x, arrays.velocity.first[i].val.y,
    // arrays.velocity.first[i].val.z);
    return;
  }

#ifdef DEBUG_INVALID_PARITLCES
  auto old_p = arrays.position.first[i].val;
  auto old_v = arrays.velocity.first[i].val;
#endif
  auto p = arrays.position.first[i];
  //p = correct_position(p, arrays.volume[i], arrays, arrays.velocity.first[i]);
  auto v = arrays.velocity.first[i];
  //v = correct_velocity(p, v, arrays.volume[i], dt, arrays);  
  p = p + dt * v;
 // p = correct_position(p, arrays.volume[i], arrays, arrays.velocity.first[i]);
  arrays.velocity.first[i] = v;
  arrays.position.first[i] = p;

#ifdef DEBUG_INVALID_PARITLCES
  auto new_p = p.val;
  auto new_v = v.val;

  if (new_p.x != new_p.x)
    printf("%s: Invalid particle %d: [%+.8e %+.8e %+.8e %+.8e] x [%+.8e %+.8e %+.8e %+.8e] -> [%+.8e %+.8e %+.8e "
           "%+.8e] x [%+.8e %+.8e %+.8e %+.8e]\n",
           __FUNCTION__, i, old_p.x, old_p.y, old_p.z, old_p.w, old_v.x, old_v.y, old_v.z, old_v.w, new_p.x, new_p.y,
           new_p.z, new_p.w, new_v.x, new_v.y, new_v.z, new_v.w);
#endif
}
neighFunctionType position_update(SPH::Integration::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position.first), (vol, volume), (vel, velocity.first), (den, density));

  auto v = arrays.velocity.first[i];
  auto p = arrays.position.first[i];

  // if (arrays.particle_type[i] != FLUID_PARTICLE){
  //  arrays.position.second[i] = p;
  //  return;
  //}

  // if (arrays.particle_type_x[i] == 1){
  //  arrays.position.second[i] = p;
  //  return;
  //}
  
  int32_t N_i = 0;
  auto r_i = 0.0_m;
  float4_u<SI::recip_2<SI::m>> X_i;
  float_u<SI::velocity> v_local_max;
  float_u<SI::Hz> vel_gradient{0.f};

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
  if (planeBoundary::distance(p, arrays.volume[i], arrays).val.w < support_from_volume(arrays.volume[i]).val * kernelSize()
	  || volumeBoundary::distance_fn(p, arrays.volume[i], arrays, -1).val.w < support_from_volume(arrays.volume[i]).val * kernelSize()
	  ) {
	  arrays.position.second[i] = p;
	  return;
  }
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
  

#ifdef DEBUG_INVALID_PARITLCES
  auto p_new = arrays.position.second[i].val;
  auto p_old = p.val;
  auto vel_new = v.val;
  if (p_new.x != p_new.x)
    printf(
        "%s: Invalid particle %d: [%+.8e %+.8e %+.8e %+.8e] x [%+.8e %+.8e %+.8e %+.8e] -> [%+.8e %+.8e %+.8e %+.8e]\n",
        __FUNCTION__, i, p_old.x, p_old.y, p_old.z, p_old.w, vel_new.x, vel_new.y, vel_new.z, vel_new.w, p_new.x,
        p_new.y, p_new.z, p_old.w);
#endif
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
#ifdef DEBUG_INVALID_PARITLCES
  auto new_vel = v.val;
  auto old_vel = arrays.velocity.first[i].val;
  auto acc = a.val;
  auto timestep = dt.val;
  if (new_vel.x != new_vel.x)
    printf("%s: Invalid particle %d: [%+.8e %+.8e %+.8e %+.8e] + [%+.8e %+.8e %+.8e %+.8e] * %+.8e -> [%+.8e %+.8e "
           "%+.8e %+.8e]\n",
           __FUNCTION__, i, new_vel.x, new_vel.y, new_vel.z, new_vel.w, acc.x, acc.y, acc.z, acc.w, timestep, new_vel.x,
           new_vel.y, new_vel.z, new_vel.w);
#endif
}

basicFunctionType velocity_update_rb(SPH::Integration::Memory arrays) {
  checkedParticleIdx(i);
  auto v = arrays.velocity.first[i];
  auto a = arrays.acceleration[i];
  auto dt = arrays.timestep;

  v += dt * a;
  v.val.w = 0.f;

  if (arrays.particle_type[i] != FLUID_PARTICLE) {
    auto tmp_vel = arrays.rigidLinearVelocities[arrays.particle_type[i] - 1].val;
    // printf("vel x: %f, %f, %f", tmp_vel.x, tmp_vel.y, tmp_vel.z);
    arrays.velocity.second[i].val = tmp_vel;
    // arrays.velocity.second[i].val = {0, 0, 0, 0};
  } else
    arrays.velocity.second[i] = v;

  arrays.maxVelocity[i] = math::length(v);
#ifdef DEBUG_INVALID_PARITLCES
  auto new_vel = v.val;
  auto old_vel = arrays.velocity.first[i].val;
  auto acc = a.val;
  auto timestep = dt.val;
  if (new_vel.x != new_vel.x)
    printf("%s: Invalid particle %d: [%+.8e %+.8e %+.8e %+.8e] + [%+.8e %+.8e %+.8e %+.8e] * %+.8e -> [%+.8e %+.8e "
           "%+.8e %+.8e]\n",
           __FUNCTION__, i, new_vel.x, new_vel.y, new_vel.z, new_vel.w, acc.x, acc.y, acc.z, acc.w, timestep, new_vel.x,
           new_vel.y, new_vel.z, new_vel.w);
#endif
}

// This function calculates the ideal timestep for the simulation based on a CFL condition. This CFL
// condition only considers the term based on the particle support radius and velocity and not any
// shock effects. These could be included but usually don't improve results enough to be worthwhile
// the cost of longer simulation.
basicFunctionType cflCalculate(SPH::Integration::Memory arrays) {
  checkedParticleIdx(i);
  auto v = arrays.velocity.first[i];
  auto p = arrays.position.first[i];

  arrays.lifetime[i] = arrays.lifetime[i] + arrays.timestep;
  auto cfl_velocity = 0.4f * support_H(p) / math::length3(v);
  auto cfl_acceleration = 0.25f * math::sqrt(support_H(p) / math::length3(arrays.acceleration[i] * arrays.volume[i] * arrays.rest_density));
  arrays.cflValue[i] = cfl_velocity;
  //arrays.cflValue[i] = math::min(cfl_velocity.val, cfl_acceleration.val);
}

basicFunction(updatePosition, correct_particles, "Integrate: Boundary");
neighFunction(particleShifting, position_update, "Integrate: Positions", caches<float4, float4, float, float>{});
basicFunction(calculateCFL, cflCalculate, "Integrate: CFL");
basicFunction(updateVelocity, velocity_update, "Integrate: Velocity");
basicFunction(updateVelocityRB, velocity_update_rb, "Integrate: Velocity + rigig body");

#include <utility/volumeBullet.h>
//#include <utility/generation/util.h>
void updateVolumeBoundary(float dt) {
	if (get<parameters::volumeBoundaryCounter>() == 0) {
		return;
	}
	int32_t n = get<parameters::volumeBoundaryCounter>();
	std::vector<float4> positions(n), quaternions(n);
	std::vector<float4> velocities(n), angularVelocities(n);
	std::vector<float4> forces(n), torques(n);
	std::vector<float> volumes(n), densities(n);
	std::vector<int32_t> kinds(n);
	std::vector<Matrix4x4> Txs(n), Txs_1(n);
	std::vector<Matrix4x4> Ixs(n), Ixs_1(n);

	cudaMemcpy(positions.data(), arrays::volumeBoundaryPosition::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(quaternions.data(), arrays::volumeBoundaryQuaternion::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(velocities.data(), arrays::volumeBoundaryVelocity::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(angularVelocities.data(), arrays::volumeBoundaryAngularVelocity::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(forces.data(), arrays::volumeBoundaryAcceleration::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(torques.data(), arrays::volumeBoundaryAngularAcceleration::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(volumes.data(), arrays::volumeBoundaryVolume::ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(densities.data(), arrays::volumeBoundaryDensity::ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Txs.data(), arrays::volumeBoundaryTransformMatrix::ptr, n * sizeof(Matrix4x4), cudaMemcpyDeviceToHost);
	cudaMemcpy(Txs_1.data(), arrays::volumeBoundaryTransformMatrixInverse::ptr, n * sizeof(Matrix4x4), cudaMemcpyDeviceToHost);
	cudaMemcpy(Ixs.data(), arrays::volumeBoundaryInertiaMatrix::ptr, n * sizeof(Matrix4x4), cudaMemcpyDeviceToHost);
	cudaMemcpy(Ixs_1.data(), arrays::volumeBoundaryInertiaMatrixInverse::ptr, n * sizeof(Matrix4x4), cudaMemcpyDeviceToHost);


	auto vols = get<parameters::boundaryVolumes>();
	//std::cout << "##################################################################" << std::endl;
	for (int32_t b = 0; b < get<parameters::volumeBoundaryCounter>(); ++b) {
		//float3 totalR{ 0.f, 0.f, 1.f * CUDART_PI_F };
		//totalR = totalR / 2.f;
		float3 totalR = math::castTo<float3>(vols[b].angularVelocity.value);
		float3 totalV = vols[b].velocity.value * dt;
		auto dq = eul2quat(totalR * dt);
		btQuaternion dbq(dq.x, dq.y, dq.z, dq.w);
		// auto vol = vols[b];
		// auto dq = eul2quat(float3{ CUDART_PI_F / 360.f, 0.f, 0.f });
		//std::cout << quaternions[b] << " x " << dq << " -> " << quatMul(quaternions[b], dq) << " => " << toAngularVelocity(quaternions[b], quatMul(quaternions[b], dq), 0.008f) << std::endl;

		auto V = volumes[b];
		auto rho = densities[b];
		auto m = V * rho;

		auto qStart = quaternions[b];
		auto pStart = positions[b];

		auto linearForce = forces[b] * volumes[b] * densities[b];
		auto angularAccel = torques[b];
		angularAccel.w = 0.f;
		auto angularForce = Txs[b] * (Ixs[b] * (Txs_1[b] * angularAccel));
#ifndef __CUDA_ARCH__
		//std::cout << "Inertia: " << std::endl;
		//Ixs[b].print();
		//std::cout << "Inertia Inverse: " << std::endl;
		//Ixs_1[b].print();
		//std::cout << "rigid body " << b << " => ";
		//std::cout << "F = " << linearForce << " -> " << forces[b] << ", ";
		//std::cout << "T = " << angularForce << " -> " << angularAccel << std::endl;
#endif
		auto c_obj = bt::World::instance().dynamicsWorld->getCollisionObjectArray()[bt::World::instance().boundaryMap[b]];
		auto body = btRigidBody::upcast(c_obj);
		//std::cout << f4frombt3(body->getTotalForce()) << std::endl;
		body->applyCentralForce(bt::bt3fromf4(linearForce));
		//std::cout << f4frombt3(body->getTotalForce() )<< std::endl;
		body->applyTorque(bt::bt3fromf4(angularForce));
		//body->applyCentralForce(btVector3(-1e8f, 0.0, 0.0));
		//std::cout << f4frombt3(body->getTotalForce()) << std::endl;

		if (get<parameters::boundaryVolumes>()[b].kind.value == 1) {
			//std::cout << bt::f4frombtq(body->getWorldTransform().getRotation()) << " * " << bt::f4frombtq(dbq) << "" << bt::f4frombtq(body->getWorldTransform().getRotation() * dbq) << std::endl;
			//btTransform newTrans;
			//body->getMotionState()->getWorldTransform(newTrans);
			//newTrans.getRotation() *= dbq;
			//body->getMotionState()->setWorldTransform(newTrans);

			body->getWorldTransform().setRotation(body->getWorldTransform().getRotation() * dbq);
			body->getWorldTransform().setOrigin(bt::bt3fromf3(totalV) + body->getWorldTransform().getOrigin());
			//std::cout << bt::f4frombtq(body->getWorldTransform().getRotation()) << std::endl;
		}
	}

	bt::World::instance().step(dt);

	for (int32_t b = 0; b < get<parameters::volumeBoundaryCounter>(); ++b) {

		float3 totalR = math::castTo<float3>(vols[b].angularVelocity.value);
		float3 totalV = vols[b].velocity.value * dt;
		auto dq = eul2quat(totalR * dt);
		btQuaternion dbq(dq.x, dq.y, dq.z, dq.w);

		auto V = volumes[b];
		auto rho = densities[b];
		auto m = V * rho;

		auto qStart = quaternions[b];
		auto pStart = positions[b];

		auto linearForce = forces[b];
		auto angularForce = torques[b];

		auto c_obj = bt::World::instance().dynamicsWorld->getCollisionObjectArray()[bt::World::instance().boundaryMap[b]];
		auto body = btRigidBody::upcast(c_obj);

		auto vel = bt::f4frombt3(body->getLinearVelocity());
		auto angularVel = bt::f4frombt3(body->getAngularVelocity());
		//std::cout << bt::f4frombtq(body->getWorldTransform().getRotation()) << std::endl;
		auto pEnd = bt::f4frombt3(body->getWorldTransform().getOrigin());
		auto qEnd = bt::f4frombtq(body->getWorldTransform().getRotation());

#ifndef __CUDA_ARCH__
		//std::cout << angularVel << std::endl;
#endif

		if (get<parameters::boundaryVolumes>()[b].kind.value == 1) {
			angularVel = bt::toAngularVelocity(float4{ 0.f,0.f,0.f,0.f }, dq, dt);
			vel = math::castTo<float4>(totalV / dt);
		}
		//angularVel = float4{ 1000.f,1000.f,1000.f,0.f };
#ifndef __CUDA_ARCH__
		/*std::cout << bt::toAngularVelocity(qStart, qEnd, dt) << std::endl;
		std::cout << vel << std::endl;
		std::cout << pStart << " -> " << pEnd << " : " << qStart << " -> " << qEnd << std::endl;
		std::cout << angularVel << std::endl;*/
#endif
		Matrix4x4 Q = Matrix4x4::fromQuaternion(qEnd);
		Matrix4x4 T = Matrix4x4::fromTranspose(pEnd);
		auto M =  T * Q;
		auto invM = M.inverse();
		//Q.print();
		//T.print();
		//M.print();
		//invM.print();

		quaternions[b] = qEnd;
		angularVelocities[b] = angularVel;
		torques[b] = float4{ 0.f,0.f,0.f,0.f };
		velocities[b] = vel;
		positions[b] = pEnd;
		forces[b] = float4{ 0.f,0.f,0.f,0.f };
		Txs[b] = M;
		Txs_1[b] = invM;
	}

	cudaMemcpy(arrays::volumeBoundaryPosition::ptr, positions.data(), n * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volumeBoundaryQuaternion::ptr, quaternions.data(), n * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volumeBoundaryVelocity::ptr, velocities.data(), n * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volumeBoundaryAngularVelocity::ptr, angularVelocities.data(), n * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volumeBoundaryAcceleration::ptr, forces.data(), n * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volumeBoundaryAngularAcceleration::ptr, torques.data(), n * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volumeBoundaryVolume::ptr, volumes.data(), n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volumeBoundaryDensity::ptr, densities.data(), n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volumeBoundaryTransformMatrix::ptr, Txs.data(), n * sizeof(Matrix4x4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volumeBoundaryTransformMatrixInverse::ptr, Txs_1.data(), n * sizeof(Matrix4x4), cudaMemcpyHostToDevice);

}

// This function updates the positions of all particles and additionally updates the timestep
// according to a CFL condition.
void SPH::Integration::update_positions(Memory mem) {
	if (mem.num_ptcls == 0) return;
	// std::cout << arrays::previousPosition::ptr << " - " << arrays::position::ptr << std::endl;
	// if (get<parameters::rayTracing>() == true)
	//	cuda::memcpy(arrays::previousPosition::ptr, arrays::position::ptr, sizeof(float4) * mem.num_ptcls,
	// cudaMemcpyDeviceToDevice);
	int32_t substeps = 1;
	float factor = 1.f / ((float)substeps);
	for (int32_t s = 0; s < substeps; ++s) {
		updateVolumeBoundary(mem.timestep.val * factor);
		launch<updatePosition>(mem.num_ptcls, mem, mem.timestep * factor);
	}
  if (get<parameters::volumeBoundaryCounter>() == 0)
    if (get<parameters::rigidVolumes>().size())
      updateRigids(mem);

  //launch<particleShifting>(mem.num_ptcls, mem);
  //info<arrays::position>().swap();
  launch<calculateCFL>(mem.num_ptcls, mem);

  float_u<SI::s> dt = algorithm::reduce_min(mem.cflValue, mem.num_ptcls);
  // dt = math::min(dt, get<parameters::timestep_max>());
  int32_t n = get<parameters::volumeBoundaryCounter>();
  if (n > 0) {
	  std::vector<float4> velocities(n), angularVelocities(n);
	  std::vector<float4> minAABBs(n), maxAABBs(n);
	  cudaMemcpy(minAABBs.data(), arrays::volumeBoundaryMin::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
	  cudaMemcpy(maxAABBs.data(), arrays::volumeBoundaryMax::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
	  cudaMemcpy(velocities.data(), arrays::volumeBoundaryVelocity::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
	  cudaMemcpy(angularVelocities.data(), arrays::volumeBoundaryAngularVelocity::ptr, n * sizeof(float4), cudaMemcpyDeviceToHost);
	  auto minVol = algorithm::reduce_min(arrays::volume::ptr, mem.num_ptcls);
	  auto Hmin = support_from_volume(minVol) * kernelSize();
	  auto vols = get<parameters::boundaryVolumes>();
	  for (int32_t b = 0; b < n; ++b) {
		  auto c_obj = bt::World::instance().dynamicsWorld->getCollisionObjectArray()[bt::World::instance().boundaryMap[b]];
		  auto body = btRigidBody::upcast(c_obj);
		  auto maxAbs = [](auto lhs, auto rhs) { return math::max(math::abs(lhs), math::abs(rhs)); };
		  auto min = minAABBs[b];
		  auto max = maxAABBs[b];
		  auto extent = float4{ maxAbs(min.x, max.x),maxAbs(min.y, max.y), maxAbs(min.z, max.z), 0.f };
		  auto angularVel = angularVelocities[b];

		  auto vel = math::length3(velocities[b] + math::cross(extent, angularVel));
#ifndef __CUDA_ARCH__
		 // std::cout << b << std::endl;
		  //std::cout << "Velocity of rigid body: " << vel << " - " << velocities[b] << " + " << angularVel << " x " << extent << std::endl;
		  //std::cout << "H " << Hmin << " - " << 0.4f * Hmin / vel << " | " << dt.val << std::endl;
#endif
		  auto cfl_dt = 0.2f * Hmin / vel;
		  dt = math::min(dt.val, cfl_dt);
	  }
  }

  dt = math::clamp(dt, get<parameters::timestep_min>(), get<parameters::timestep_max>());
  // dt = math::clamp(dt, get<parameters::timestep>() * 0.9f, get<parameters::timestep>() * 1.1f);

  get_u<parameters::timestep>() = dt;
}

// This function updates the velocity of all particles based on the current velocity and prepares
// the max velocity parameter for the shifting algorithm.
void SPH::Integration::update_velocities(Memory mem) {
	if (mem.num_ptcls == 0) return;
  if (get<parameters::volumeBoundaryCounter>() == 0) {
    if (get<parameters::modules::pressure>() == "IISPH17_BAND_RB")
      launch<updateVelocityRB>(mem.num_ptcls, mem);
    else
      launch<updateVelocity>(mem.num_ptcls, mem);
  } else
    launch<updateVelocity>(mem.num_ptcls, mem);

  info<arrays::velocity>().swap();
  get_u<parameters::max_velocity>() = algorithm::reduce_max(mem.maxVelocity, mem.num_ptcls);
}
