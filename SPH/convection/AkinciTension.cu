#define NEW_STYLE
#include <SPH/convection/AkinciTension.cuh>
#include <utility/include_all.h>

hostDeviceInline static auto cohesionKernel(float4_u<SI::m> a, float4_u<SI::m> b) {
	auto difference = a - b;
	auto r = math::length3(difference);
	auto half = support(a, b);
	auto h = half * kernelSize();
	const auto surfaceTensionConstant = 32.f / (CUDART_PI_F * math::power<9>(h));
	const auto surfaceTensionOffset = - math::power<6>(h) / 64.f;

	float_u<SI::multiply_ratios<SI::m, SI::ratio<6, 1>>> spline = 0.f;
	if (2.f * r > h && r <= h) {
		spline = math::cubic(h - r) * math::cubic(r);
	}
	else if (r > 0.f && 2.f * r <= h) {
		spline = 2.f * math::cubic(h - r) * math::cubic(r) + surfaceTensionOffset;
	}

	auto result = spline * surfaceTensionConstant * difference;
	using res = decltype(result / r);
	return result / r;
}

#define m(x) (arrays.volume[x] * arrays.rest_density)
#define rho(x) (arrays.density[x] * arrays.rest_density)
#define n(x) arrays.particleNormal[x]
#define rho0 arrays.rest_density
#define H(x) support_H(arrays.position[x])
#define k arrays.tension_akinci
#define CW(i,j) cohesionKernel(arrays.position[i], arrays.position[j])
#define GradW(i,j) gradient(arrays.position[i], arrays.position[j])

// This function estimates the normal of a particle according to the methods described in the source
// paper. This needs to be done in an initial step.
neighFunctionType colorField(SPH::AkinciTension::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (normal, particleNormal), (vol, volume),
		(dens, density));

	uFloat<> color{ 0.f };
	uFloat4<> colorGrad{ 0.f, 0.f, 0.f, 0.f };
	iterateNeighbors(j) {
		colorGrad += (m(j) / rho(j) * GW_ij).val;
		color += m(j) / rho(j) * W_ij;
	}
	colorGrad.val /= color.val;
	colorGrad.val.w = math::sqlength3(colorGrad).val;
	n(i) = colorGrad.val;
}
neighFunctionType colorFieldGradient(SPH::AkinciTension::Memory arrays) {
}
neighFunctionType heTension(SPH::AkinciTension::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (normal, particleNormal), (vol, volume),
		(dens, density));

	float4_u<SI::N> heForce{ 0.f,0.f,0.f,0.f };
	float4_u<SI::N> atmForce{ 0.f,0.f,0.f,0.f };
	auto kh = 0.25f * k / rho(i);
	iterateNeighbors(j) {
		heForce += (kh * m(i) * m(j) / rho(j) * (n(i).w + n(j).w)* GW_ij).val;
		atmForce += (m(i) / rho(i) * m(j) / rho(j) * GW_ij).val;
	}
	auto new_accel = (heForce) / m(i) + 0.f * atmForce * 101325.f / m(i);
	//n(i) = new_accel.val;
	//if (i % 128 == 0)
	//	printf("Surface tension on %d = [%f %f %f %f] @ [%f %f %f]/%f\n", i, 
	//		new_accel.val.x, new_accel.val.y, new_accel.val.z, new_accel.val.w,
	//		n(i).x, n(i).y, n(i).z, n(i).w);
	arrays.acceleration[i] += new_accel;
}

neighFunction(heFirst, colorField, "Akinci Tension: normals", caches<float4, float4, float, float>{});
neighFunction(heSecond, colorFieldGradient, "Akinci Tension: normals", caches<float4, float4, float, float>{});
neighFunction(heThird, heTension, "Akinci Tension: force", caches<float4, float4, float, float>{});

// This function estimates the normal of a particle according to the methods described in the source
// paper. This needs to be done in an initial step.
neighFunctionType akinciFirst(SPH::AkinciTension::Memory arrays) {
  checkedParticleIdx(i);
  float4_u<SI::recip<SI::m>> kernelSum{ 0.f,0.f,0.f,0.f };
  iterateNeighbors(j) {
	  kernelSum += m(j) / rho(j) * GradW(i,j);
  }
  kernelSum += boundary::splineGradient(arrays.position[i], arrays.volume[i], float_u<>{0.5f}, arrays, boundary::kind::plane).val;
  kernelSum += boundary::splineGradient(arrays.position[i], arrays.volume[i], float_u<>{0.5f}, arrays, boundary::kind::volume, -1).val;
  auto normal = kernelSum * H(i);
  n(i) = normal.val;
}

// This function calculates the actual surface tension, however it does not exactly replicate the
// source paper as the source has some issues with the units of the scaling parameter that need to
// manually fixed.
neighFunctionType akinciSecond(SPH::AkinciTension::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (normal, particleNormal), (vol, volume),
               (dens, density));

  float4_u<SI::N> cohesionForce{ 0.f,0.f,0.f,0.f }, curvatureForce{ 0.f,0.f,0.f,0.f };
//  iterateBoundaryPlanes(p)
//	cohesionForce += -k * 2.f * rho0 / rho(i) * m(i) * (n(i) - p);

  float4_u<SI::N> heForce{ 0.f,0.f,0.f,0.f };
  iterateNeighbors(j) {
	  if (math::distance3(arrays.position[i], arrays.position[j]) < 1e-12f) continue;
	  auto h = support(arrays.position[i], arrays.position[j]) * kernelSize();
	  if(math::distance3(arrays.position[i], arrays.position[j]) > h) continue;
	  auto scaling = 2.f * rho0 / (rho(i) + rho(j));
	  auto k_ij = k;
	  curvatureForce += (-k_ij * scaling * m(i) * m(j) * CW(i, j)).val;
	  cohesionForce += -(k_ij * scaling * m(i) * (n(i) - n(j))).val;
  }
  auto new_accel = (curvatureForce + cohesionForce) / m(i);
  auto V_p = boundary::volume(pos[i], arrays.volume[i], arrays.density[i], arrays, boundary::kind::plane);
  new_accel += -(arrays.boundaryAdhesion_akinci * arrays.rest_density * float_u<SI::volume>{1.f} *boundary::adhesion(pos[i], vol[i], arrays.density[i], arrays, boundary::kind::plane)).val;
  auto V_v = boundary::volume(pos[i], arrays.volume[i], arrays.density[i], arrays, boundary::kind::volume);
  new_accel += -(arrays.rigidAdhesion_akinci * arrays.rest_density * float_u<SI::volume>{1.f} * boundary::adhesion(pos[i], vol[i], arrays.density[i], arrays, boundary::kind::volume)).val;
#ifdef DEBUG_INVALID_PARITLCES
  auto a_new = arrays.acceleration[i].val;
  auto a_old = new_accel.val;
  auto f_curv = curvatureForce.val;
  auto f_cohesion = cohesionForce.val;
  auto mass = m(i).val;
  if (a_new.x != a_new.x)
	  printf("%s: Invalid particle %d: " _VECSTR " + " _VECSTR "(" _VECSTR " + " _VECSTR ") / %+.8e\n",
		  __FUNCTION__, i, 
		  _VEC(a_old), _VEC(a_new), _VEC(f_curv), _VEC(f_cohesion), mass);
#endif
  arrays.acceleration[i] += new_accel;
}

neighFunction(calculateNormals, akinciFirst, "Akinci Tension: normals", caches<float4, float, float>{});
neighFunction(calculateSurfaceTension, akinciSecond, "Akinci Tension: force", caches<float4, float4, float, float>{});

void SPH::AkinciTension::tension(SPH::AkinciTension::Memory mem) {
	static bool once = true;
	if (once) {
		std::cout << "Surface tension module built " << __TIMESTAMP__ << std::endl;
		once = false;
	}
	//launch<heFirst>(mem.num_ptcls, mem);
	//launch<heSecond>(mem.num_ptcls, mem);
	//launch<heThird>(mem.num_ptcls, mem);
	//return;


  launch<calculateNormals>(mem.num_ptcls, mem);
  launch<calculateSurfaceTension>(mem.num_ptcls, mem);
}
