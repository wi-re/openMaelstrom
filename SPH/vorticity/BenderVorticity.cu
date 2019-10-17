#include <SPH/vorticity/BenderVorticity.cuh>
#include <utility/include_all.h>

#define p(i) arrays.position[i]
#define omega(i) arrays.angularVelocity.first[i]
#define omega_new(i) arrays.angularVelocity.second[i]
#define a(i) arrays.acceleration[i]
#define m(x) (arrays.volume[x] * arrays.rest_density)
#define rho(x) (arrays.density[x] * arrays.rest_density)
#define rho2(x) (rho(x) * rho(x))
#define v(x) (arrays.velocity[x])
#define dt arrays.timestep
#define invDt 1.f / dt

#define rho0 arrays.rest_density
#define H(x) support_H(arrays.position[x])

#define zeta arrays.viscosityOmega
#define nu_t float_u<SI::velocity>(arrays.vorticityCoeff)
#define inertiaInverse arrays.intertiaInverse

#define W(i,j) kernel(arrays.position[i], arrays.position[j])
#define GradW(i,j) gradient(arrays.position[i], arrays.position[j])


// This function implements the Micropolar SPH Model to improve the vorticity of the simulation at
// no significant cost.
neighFunctionType computeVorticity(SPH::BenderVorticity::Memory arrays) {
	checkedParticleIdx(i);
	float4_u<SI::acceleration> angular{ 0.f,0.f,0.f,0.f }, linear{ 0.f,0.f,0.f,0.f };
	//constexpr auto d = float_u<SI::add_units<SI::area, SI::Hz>>(10.0);
	auto h2 = H(i) * H(i);
	//float ctr = 0.f;
	angular -= invDt * inertiaInverse * zeta * omega(i) * pW_ib;
	auto mb = uFloat<SI::volume>{ 1.f } *arrays.rest_density;
	linear -= nu_t * rho(i) *arrays.rest_density * math::cross(omega(i) / rho2(i), pGW_ib);
	angular -= nu_t * rho(i) * inertiaInverse *arrays.rest_density * math::cross(v(i) / rho2(i), pGW_ib);

	iterateNeighbors(j) {
		auto xij = p(i) - p(j);
		auto omegaij = omega(i) - omega(j);
		auto gradW = GradW(i, j);
		// XSPH for angular velocity field
		angular -= invDt * inertiaInverse * zeta *(m(j) / rho(j)) * omegaij * W(i, j);

		// Viscosity
		//angular += d * inertiaInverse * zeta * (m(j) / rho(i)) * math::dot3(omegaij, xij) / (math::sqlength3(xij) + 0.01f * h2) * gradW;
		
		// difference curl 
		//linear += nu_t * 1.f / rho(i) * m(j) * math::cross(omegaij, gradW);
		//angular += nu_t * 1.f / rho(i) * inertiaInverse * m(j) * math::cross(v(i) - v(j), gradW);

		// symmetric curl 
		linear -= nu_t * rho(i) * m(j) * math::cross(omega(i) / rho2(i) + omega(j) / rho2(j), gradW);
		angular -= nu_t * rho(i) * inertiaInverse * m(j) * math::cross(v(i) / rho2(i) + v(j) / rho2(j), gradW);
		linear.val.w = 0.f;
		angular.val.w = 0.f;
#ifdef DEBUG_INVALID_PARITLCES
		if (angular.val.x != angular.val.x || angular.val.y != angular.val.y || angular.val.z != angular.val.z ||
			linear.val.x != linear.val.x || linear.val.y != linear.val.y || linear.val.z != linear.val.z) {

			printf("%s: Invalid particle %dx%d: " _VECSTR _VECSTR _VECSTR _VECSTR _VECSTR " %f %f %f %f %f\n",
				__FUNCTION__, i, j, _VEC(v(i).val), _VEC(v(j).val), _VEC(linear.val), _VEC(angular.val), _VEC(omega(i).val), h2.val, m(i).val, rho(i).val, m(j).val, rho(j).val);
		}
#endif
	}
	angular -= float_u<SI::recip<SI::m>>(2.0) * inertiaInverse * nu_t * omega(i);
	omega_new(i) = omega(i) + dt * angular;

#ifdef DEBUG_INVALID_PARITLCES
	if (angular.val.x != angular.val.x || angular.val.y != angular.val.y || angular.val.z != angular.val.z ||
		linear.val.x != linear.val.x || linear.val.y != linear.val.y || linear.val.z != linear.val.z) {

		printf("%s: Invalid particle %d: " _VECSTR _VECSTR _VECSTR _VECSTR " %f %f %f\n",
			__FUNCTION__, i, _VEC(v(i).val), _VEC(linear.val), _VEC(angular.val), _VEC(omega(i).val), h2.val, m(i).val, rho(i).val);
	}
#endif


	a(i) += linear;
}

neighFunction(Vorticity, computeVorticity, "Bender Vorticity", caches<float4, float>{});

void SPH::BenderVorticity::vorticity(Memory mem) {
	launch<Vorticity>(mem.num_ptcls, mem);
	cuda::memcpy(arrays::angularVelocity::ptr, 
		arrays::angularVelocity::rear_ptr, 
		sizeof(arrays::angularVelocity::type) * mem.num_ptcls, 
		cudaMemcpyDeviceToDevice);
}
