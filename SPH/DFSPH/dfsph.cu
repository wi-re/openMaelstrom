#include <SPH/DFSPH/dfsph.cuh>
#include <utility/include_all.h>
#include <utility/SVD.h> 

using alpha1 = SI::combine<SI::m_4, SI::kg>;
using alpha2 = SI::square<alpha1>;

hostDeviceInline auto rhob(const SPH::DFSPH::Memory& arrays, uFloat4<SI::m> position, uFloat<SI::volume> vol, uFloat<SI::density> rho, boundary::kind k = boundary::kind::plane, int32_t idx = -1) {
	return boundary::spline(position, vol, rho, arrays, k, idx);
}
hostDeviceInline auto drhob_dt(const SPH::DFSPH::Memory& arrays, uFloat4<SI::m> position, uFloat4<SI::velocity> v, uFloat<SI::volume> vol,uFloat<SI::density> rho ) {
	return (rhob(arrays, position + v * arrays.timestep, vol, rho) - rhob(arrays, position, vol, rho)) / arrays.timestep;
}
hostDeviceInline auto pressureMLS(const SPH::DFSPH::Memory& arrays, uFloat4<SI::m> pb, uFloat<SI::volume> vol) {
	constexpr auto hash_width = hash_length::bit_64;
	constexpr auto order = cell_ordering::z_order;
	constexpr auto structure = cell_structuring::compactMLM;
	float4 vecSum{ 0.f,0.f,0.f,0.f };
	SVD::Mat3x3 M{ 0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f };
	float sumA = 0.f;
	float sumB = 0.f;
	pb.val.w = support_from_volume(vol).val;
	auto x_b = pb.val;
	float4 d_bar{ 0.f,0.f,0.f,0.f };
	float d_sum = 0.f;
	iterateCells(pb, j) {
		auto pj = arrays.position[j];
		auto fac = arrays.volume[j].val * kernel(pj, pb).val;
		d_bar += pj.val * fac;
		d_sum += fac;
	}
	d_bar /= d_sum;
	d_bar.w = 0.f;
	x_b -= d_bar;
	iterateCells(pb, j) {
		auto xx_j = arrays.position[j];
		auto x_j = arrays.position[j].val - d_bar;
		auto V_j = arrays.volume[j].val;
		auto p_j = arrays.pressure.second[j].val;

		auto Wbbf = kernel(xx_j, pb).val;
		M = M + SVD::Mat3x3{
			x_j.x * x_j.x, x_j.x * x_j.y, x_j.x * x_j.z,
			x_j.x * x_j.y, x_j.y * x_j.y, x_j.y * x_j.z ,
			x_j.z * x_j.x, x_j.z * x_j.y, x_j.z * x_j.z
		} *V_j * Wbbf;
		vecSum += x_j * p_j * V_j * Wbbf;
		sumA += p_j * V_j * Wbbf;
		sumB += V_j * Wbbf;
	}
	auto svd = SVD::svd(M);
	auto U = svd.U;
	auto S = svd.S;
	auto V = svd.V;
	S.m_00 = (fabsf(S.m_00) > 1e-2f ? 1.f / S.m_00 : 0.f);
	S.m_11 = (fabsf(S.m_11) > 1e-2f ? 1.f / S.m_11 : 0.f);
	S.m_22 = (fabsf(S.m_22) > 1e-2f ? 1.f / S.m_22 : 0.f);
	S.m_01 = S.m_02 = S.m_10 = S.m_12 = S.m_20 = S.m_21 = 0.f;
	auto Mp = V * S * U.transpose();
	auto MpM = Mp * M;
	auto alpha = sumA / sumB;
	auto beta = Mp.m_00 * vecSum.x + Mp.m_01 * vecSum.y + Mp.m_02 * vecSum.z;
	auto gamma = Mp.m_10 * vecSum.x + Mp.m_11 * vecSum.y + Mp.m_12 * vecSum.z;
	auto delta = Mp.m_20 * vecSum.x + Mp.m_21 * vecSum.y + Mp.m_22 * vecSum.z;


	auto det = M.det();
	if (det != det
		//|| math::abs(det) < 1e-6f || fabsf(svd.S.m_00) < 1e-8f || fabsf(svd.S.m_11) < 1e-8f || fabsf(svd.S.m_22) < 1e-8f
		) {
		beta = gamma = delta = 0.f;
	}
	auto pressure = alpha * 1.f + beta * x_b.x + gamma * x_b.y + delta * x_b.z;

	if (pressure != pressure)
		pressure = 0.f;
	//if(density)
	pressure = math::max(pressure, 0.f);

	return uFloat<SI::Pa>(pressure);
}
hostDeviceInline auto gradRho(const SPH::DFSPH::Memory& arrays, uFloat4<SI::m> position, uFloat<SI::volume> vol, uFloat<SI::density> rho, boundary::kind k = boundary::kind::plane, int32_t idx = -1) {
	return boundary::splineGradient(position, vol, rho, arrays, k, idx);
}
hostDeviceInline auto grad2Rho(const SPH::DFSPH::Memory& arrays, uFloat4<SI::m> position, uFloat<SI::volume> vol, uFloat<SI::density> rho, boundary::kind k = boundary::kind::plane, int32_t idx = -1) {
	auto fac = (support_from_volume(1.f) / support_from_volume(vol)).val;
	return boundary::splineGradient2(position, vol, rho, arrays, k, idx) * math::power<5>(fac);
}

#define LOOKUP_DENSITY arrays.fluidDensity[i]
#define GRADIENT_BOUNDARY boundary::splineGradient(arrays.position[i], arrays.volume[i], LOOKUP_DENSITY, arrays, boundary::kind::plane, -1)
#define GRADIENT2_BOUNDARY boundary::splineGradient2(arrays.position[i], arrays.volume[i], LOOKUP_DENSITY, arrays, boundary::kind::plane, -1)
//#define BOUNDARY_DENSITY arrays.density[i]
#define BOUNDARY_DENSITY 1.f

neighFunctionType computeAlphaKernel(SPH::DFSPH::Memory& arrays, bool density, bool boundary) {
	cache_arrays((pos, position), (vol, volume));
	checkedParticleIdx(i);

	uFloat4<SI::m_1> kernelSum1{ 0.f, 0.f, 0.f, 0.f };
	//kernelSum1 += GRADIENT_BOUNDARY / BOUNDARY_DENSITY;

	uFloat<SI::exp<SI::Pa,-1>, SI::s_2, SI::m_3> kernelSum2{ 0.f };
	if (density && boundary)
	kernelSum1 += gradRho(arrays, pos[i], vol[i], arrays.fluidDensity[i]);
	if(density)
	for (int32_t v = 0; v < arrays.volumeBoundaryCounter; ++v) {
		if (volumeBoundary::distance_fn(arrays.position[i], arrays.volume[i], arrays, v).val.w >= 1e20f)
			continue;
		kernelSum1 += gradRho(arrays, pos[i], vol[i], arrays.fluidDensity[i], boundary::kind::volume, v);
		if(arrays.volumeBoundaryKind[v] == 0)
			kernelSum2 += grad2Rho(arrays, pos[i], vol[i], arrays.fluidDensity[i], boundary::kind::volume, v) / arrays.volumeBoundaryDensity[v];
	}
	
	iterateNeighbors(j) {
		auto kernel = GPW_ij;
		kernelSum1 += arrays.apparentVolume[j] * kernel;
		kernelSum2 += arrays.apparentVolume[j] * arrays.apparentVolume[j] /( arrays.volume[j] * arrays.rest_density ) * math::dot3(kernel, kernel);
	}
	uFloat<SI::exp<SI::Pa, -1>> alpha = 
		- arrays.timestep * arrays.timestep * arrays.apparentVolume[i] / (arrays.volume[i] * arrays.rest_density) * math::dot3(kernelSum1, kernelSum1) -
		arrays.timestep * arrays.timestep * arrays.apparentVolume[i] * kernelSum2;
	arrays.dfsphAlpha[i] = alpha;
}
hostDeviceInline auto boundaryPosition(const SPH::DFSPH::Memory& arrays, uFloat4<SI::m> pb, uFloat<SI::volume> v, boundary::kind k = boundary::kind::plane, int32_t idx = -1) {
	if (k == boundary::kind::volume) {
		auto tp = pb * 0.f;
		tp.val.w = 1e21f;
		auto volumeDistance = volumeBoundary::distance_fn(pb, v, arrays, idx).val;
		if (volumeDistance.w < 1e20f) {
			pb = pb - volumeDistance.w * volumeDistance;
			pb.val.w = 1.f;
			tp = float4_u<SI::m>{ arrays.volumeBoundaryTransformMatrix[idx] * float4{0.f,0.f,0.f,1.f} };
			tp.val.w = volumeDistance.w;
		}
		return std::make_pair(pb, tp);
	}
	else {
		auto planeDistance = planeBoundary::distance(pb, v, arrays);
		pb = pb - planeDistance.val.w * planeDistance.val;
		return std::make_pair(pb, uFloat4<SI::m>{0.f});
	}
}
hostDeviceInline auto boundaryVelocity(const SPH::DFSPH::Memory& arrays, uFloat4<SI::m> pb, uFloat<SI::volume> v, boundary::kind k = boundary::kind::plane, int32_t idx = -1) {
	if (k == boundary::kind::plane)
		return uFloat4<SI::velocity>{0.f, 0.f, 0.f, 0.f};
	auto rp = boundaryPosition(arrays, pb, v, k, idx);
	return arrays.volumeBoundaryVelocity[idx] + math::cross(arrays.volumeBoundaryAngularVelocity[idx].val, rp.first.val - rp.second.val);
}
hostDeviceInline auto boundaryAcceleration(const SPH::DFSPH::Memory& arrays, uFloat4<SI::m> pb, uFloat<SI::volume> v, boundary::kind k = boundary::kind::plane, int32_t idx = -1) {
	if (k == boundary::kind::plane)
		return uFloat4<SI::acceleration>{0.f, 0.f, 0.f, 0.f};
	auto rp = boundaryPosition(arrays, pb, v, k, idx);
	return (arrays.volumeBoundaryKind[idx] != 0 ? 0.f : 1.f) * (arrays.volumeBoundaryAcceleration[idx] +
		math::cross(arrays.volumeBoundaryAngularAcceleration[idx].val, rp.first.val - rp.second.val));
}

neighFunctionType computeSourceTermKernel(const SPH::DFSPH::Memory& arrays, bool density, bool boundary) {
	checkedParticleIdx(i);
	cache_arrays((pos, position));
	uFloat<> sourceTerm = density ? 1.f - arrays.density[i] : 0.f;

	//sourceTerm += -arrays.timestep * drhob_dt(arrays, arrays.position[i], arrays.velocity.first[i], arrays.volume[i]);
	if(density && boundary)
	sourceTerm = sourceTerm - arrays.timestep * math::dot3(arrays.velocity.first[i], GRADIENT_BOUNDARY) / (BOUNDARY_DENSITY);

	if (density)
	for (int32_t v = 0; v < arrays.volumeBoundaryCounter; ++v) {
		if (volumeBoundary::distance_fn(arrays.position[i], arrays.volume[i], arrays, v).val.w >= 1e20f)
			continue;
		auto vel = boundaryVelocity(arrays, pos[i], arrays.volume[i], boundary::kind::volume, v);
		//vel.val = float4{ 0.f,0.f,0.f,0.f };
		sourceTerm = sourceTerm - arrays.timestep * math::dot3(arrays.velocity.first[i] - vel, gradRho(arrays, pos[i], arrays.volume[i], arrays.fluidDensity[i], boundary::kind::volume, v));
	}


	//arrays.debugArray[i] = float4{
	//	(-arrays.timestep * math::dot3(arrays.velocity.first[i], GRADIENT_BOUNDARY)).val,
	//	rhob(arrays,arrays.position[i], arrays.volume[i]).val,
	//	rhob(arrays,arrays.position[i] + arrays.timestep * arrays.velocity.first[i], arrays.volume[i]).val,
	//	(-arrays.timestep * drhob_dt(arrays, arrays.position[i], arrays.velocity.first[i], arrays.volume[i])).val
	//};

	iterateNeighbors(j)
		sourceTerm = sourceTerm - arrays.timestep * arrays.apparentVolume[j] * math::dot3(arrays.velocity.first[i] - arrays.velocity.first[j], GPW_ij);
	arrays.dfsphSource[i] = density ? sourceTerm : (arrays.neighborListLength[i] < 20 ? 0.f :  math::max(sourceTerm, 0.f));
	arrays.dfsphSource[i] = sourceTerm;
	arrays.pressure.second[i] = 0.f;
}

neighFunctionType computeAccelerationKernel(const SPH::DFSPH::Memory arrays, bool density, bool boundary) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (i_vol, apparentVolume), (pressure, pressure.second));

	float4_u<SI::acceleration> kernelSum{ 0.f,0.f,0.f,0.f };
	iterateNeighbors(j)
		kernelSum += -arrays.volume[j] * arrays.rest_density *
		(pressure[i] / math::square(arrays.density[i] * arrays.rest_density)
			+ pressure[j] / math::square(arrays.density[j] * arrays.rest_density)) *  GPW_ij;

	arrays.predictedAcceleration[i] += kernelSum;
	arrays.pressure.first[i] = arrays.pressure.second[i];
}

neighFunctionType updatePressureKernel(const SPH::DFSPH::Memory arrays, bool density, bool boundary) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (acc, predictedAcceleration));
	alias_arrays((vol, volume), (i_vol, apparentVolume));

	uFloat<> kernelSum{ 0.f };
	auto dt = arrays.timestep;
	auto v_i = arrays.velocity.first[i];
	auto a_i = arrays.predictedAcceleration[i];
	auto x_i = pos[i];
	
	//kernelSum += arrays.timestep * drhob_dt(arrays, x_i + v_i * dt, a_i * dt, arrays.volume[i]);
	//kernelSum += arrays.timestep * arrays.timestep * math::dot3(acc[i], gradRho(arrays, pos[i], vol[i]));
	if (density && boundary)
	kernelSum += arrays.timestep * arrays.timestep * math::dot3(acc[i], GRADIENT_BOUNDARY) / (BOUNDARY_DENSITY);

	if (density)
	for (int32_t v = 0; v < arrays.volumeBoundaryCounter; ++v) {
		if (volumeBoundary::distance_fn(arrays.position[i], arrays.volume[i], arrays, v).val.w >= 1e20f)
			continue;
		auto bacc = boundaryAcceleration(arrays, pos[i], arrays.volume[i], boundary::kind::volume, v);
		kernelSum += arrays.timestep * arrays.timestep * math::dot3(acc[i] - bacc, 
			gradRho(arrays, pos[i], arrays.volume[i], arrays.fluidDensity[i], boundary::kind::volume, v)
		);
	}


	//arrays.debugArray[i] = float4{
	//	(arrays.timestep * arrays.timestep * math::dot3(acc[i], GRADIENT_BOUNDARY)).val,
	//	(arrays.timestep * drhob_dt(arrays, x_i, a_i * dt, arrays.volume[i])).val,
	//	(arrays.timestep * drhob_dt(arrays, x_i + v_i * dt, a_i * dt, arrays.volume[i])).val,
	//	(arrays.timestep * drhob_dt(arrays, x_i, v_i + a_i * dt, arrays.volume[i])).val
	//};

	iterateNeighbors(j) {
		kernelSum += arrays.timestep * arrays.timestep * i_vol[j] * math::dot3(acc[i] - acc[j], GPW_ij);
	}

	float omega = 0.5f;
	//if (density) omega = math::clamp(omega / arrays.density[i].val,0.25f, 0.5f);
	//omega = 0.5f;
	uFloat<SI::Pa> pressure = arrays.pressure.first[i] +
		omega / arrays.dfsphAlpha[i] * (arrays.dfsphSource[i] - kernelSum);
	if (density)
		pressure = math::max(uFloat<SI::Pa>{0.f}, pressure);

	uFloat<> residual = kernelSum - arrays.dfsphSource[i];

	if (math::abs(arrays.dfsphAlpha[i]) < 1e-12f || pressure != pressure || pressure > 1e15f) {
		pressure = 0._Pa;
		residual = decltype(residual)(0.f);
	}

	arrays.pressure.second[i] = pressure;
	arrays.dfsphDpDt[i] = math::max(residual.val, -0.001f) * arrays.volume[i].val;
	//arrays.dfsphRhoStar[i] = math::max(residual, 0.f) * arrays.volume[i].val;
	//arrays.dfsphRhoStar[i] = residual * arrays.volume[i].val ;
	arrays.dfsphRhoStar[i] = math::max(residual, -0.001f) * arrays.volume[i].val; /// arrays.rest_density.val;
}

template <hash_length hash_width, cell_ordering order, cell_structuring structure> hostDeviceInline auto
calculateBoundaryPressure(int32_t i, float4_u<SI::m> pb, const SPH::DFSPH::Memory& arrays, bool density, int32_t b, boundary::kind k) {
	float4 vecSum{ 0.f,0.f,0.f,0.f };
	SVD::Mat3x3 M{ 0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f };
	float sumA = 0.f;
	float sumB = 0.f;
	pb.val.w = arrays.position[i].val.w;
	auto x_b = pb.val;
	float4 d_bar{ 0.f,0.f,0.f,0.f };
	float d_sum = 0.f;
	iterateCells(pb, j) {
		auto pj = arrays.position[j];
		auto fac = arrays.volume[j].val * kernel(pj, pb).val;
		d_bar += pj.val * fac;
		d_sum += fac;
	}
	d_bar /= d_sum;
	d_bar.w = 0.f;
	x_b -= d_bar;
	iterateCells(pb, j) {
		auto xx_j = arrays.position[j];
		auto x_j = arrays.position[j].val - d_bar;
		auto V_j = arrays.volume[j].val;
		auto p_j = arrays.pressure.second[j].val;

		auto Wbbf = kernel(xx_j, pb).val;
		M = M + SVD::Mat3x3{
			x_j.x * x_j.x, x_j.x * x_j.y, x_j.x * x_j.z,
			x_j.x * x_j.y, x_j.y * x_j.y, x_j.y * x_j.z ,
			x_j.z * x_j.x, x_j.z * x_j.y, x_j.z * x_j.z
		} * V_j * Wbbf;
		vecSum += x_j * p_j * V_j * Wbbf;
		sumA += p_j * V_j * Wbbf;
		sumB += V_j * Wbbf;
	}
	auto svd = SVD::svd(M);
	auto U = svd.U;
	auto S = svd.S;
	auto V = svd.V;
	S.m_00 = (fabsf(S.m_00) > 1e-2f ? 1.f / S.m_00 : 0.f);
	S.m_11 = (fabsf(S.m_11) > 1e-2f ? 1.f / S.m_11 : 0.f);	
	S.m_22 = (fabsf(S.m_22) > 1e-2f ? 1.f / S.m_22 : 0.f);
	S.m_01 = S.m_02 = S.m_10 = S.m_12 = S.m_20 = S.m_21 = 0.f;
	auto Mp = V * S * U.transpose();
	auto MpM = Mp * M;
	auto alpha = sumA / sumB;
	auto beta = Mp.m_00 * vecSum.x + Mp.m_01 * vecSum.y + Mp.m_02 * vecSum.z;
	auto gamma = Mp.m_10 * vecSum.x + Mp.m_11 * vecSum.y + Mp.m_12 * vecSum.z;
	auto delta = Mp.m_20 * vecSum.x + Mp.m_21 * vecSum.y + Mp.m_22 * vecSum.z;


	auto det = M.det();
	if (det != det 
		//|| math::abs(det) < 1e-6f || fabsf(svd.S.m_00) < 1e-8f || fabsf(svd.S.m_11) < 1e-8f || fabsf(svd.S.m_22) < 1e-8f
		) {
		beta = gamma = delta = 0.f;
	}
	auto pressure = alpha * 1.f + beta * x_b.x + gamma * x_b.y + delta * x_b.z;

	if (pressure != pressure)
		pressure = 0.f;
	if(density)
		pressure = math::max(pressure, 0.f);

	return pressure;
}
cellFunctionType boundaryMLSPressureKernel(const SPH::DFSPH::Memory arrays, bool density, bool boundary, int32_t iteration) {
	//if (!density) return;
	checkedParticleIdx(i);
	arrays.predictedAcceleration[i] = float4_u <SI::acceleration>{ 0.f,0.f,0.f,0.f };
	float4_u <SI::acceleration> accel{ 0.f,0.f,0.f,0.f };
	float_u<SI::Pa> boundaryP{ 0.f };
	float pressure = 0.f;

	auto x_i = arrays.position[i];
	auto V_i = arrays.volume[i];
	auto rho_i = arrays.density[i];
	auto p_i = arrays.pressure.second[i];

	//arrays.lifetime[i].val = -arrays.timestep.val;
	auto boundaryDistance = pDistance.val;
	if (density && boundary)
	if (boundaryDistance.w < 1e20f) {
		//arrays.lifetime[i].val = boundaryDistance.w - arrays.timestep.val;
		auto pb = arrays.position[i] - boundaryDistance.w * math::normalize3(boundaryDistance);
		//auto boundaryPressure = pressureMLS(arrays, pb, arrays.volume[i]);
		float_u<SI::Pa> boundaryPressure{ calculateBoundaryPressure<hash_width, order, structure>(
			i, pb, arrays, density, -1, boundary::kind::plane) };
		auto fluid = density ? math::max(0.0_Pa, arrays.pressure.second[i]) : arrays.pressure.second[i];
		auto boundary = density ? math::max(0.0_Pa, boundaryPressure) : boundaryPressure;
		//boundaryPressure = arrays.pressure.second[i];
		//boundaryPressure = 0.f;
		auto boundaryKernel = gradRho(arrays, x_i, V_i, arrays.fluidDensity[i]);
		//boundaryKernel = boundary::splineGradient(arrays.position[i] + arrays.velocity.first[i] * arrays.timestep, arrays.volume[i], LOOKUP_DENSITY, arrays, boundary::kind::plane, -1);
		//if (iteration > 0)
		accel +=- 1.f * arrays.rest_density *
			(fluid / math::square(arrays.density[i] * arrays.rest_density) +
			boundary / math::square(BOUNDARY_DENSITY * arrays.rest_density)) *
				(boundaryKernel);// *(density ? 1.f : 0.f);

			//accel += - arrays.density[i] / (arrays.rest_density) * (arrays.pressure.second[i] + boundaryPressure) * pGW_ib;
		boundaryP += boundaryPressure;


		//auto fluid = fluidPressure / math::square(arrays.density[i] * arrays.rest_density);
		//auto boundary = boundaryPressure / math::square(arrays.rest_density);

		//accel += -arrays.rest_density * (fluid + boundary) * gradRho(arrays, arrays.position[i], arrays.volume[i]);

		//boundaryP += boundaryPressure;
	}

	if (density)
	for (int32_t v = 0; v < arrays.volumeBoundaryCounter; ++v) {
		auto pbp = boundaryPosition(arrays, x_i, V_i, boundary::kind::volume, v);
		auto pb = pbp.first;
		pb.val.w = x_i.val.w;
		auto com = pbp.second;
		auto d = pbp.second.val.w;
		com.val.w = 1.f;
		if (d < 1e20f) {
			float_u<SI::Pa> boundaryPressure{ calculateBoundaryPressure<hash_width, order, structure>(
				i, pb, arrays, density, v, boundary::kind::volume) };
			auto fluid =  math::max(0.0_Pa, arrays.pressure.second[i]);
			auto boundary =  math::max(0.0_Pa, boundaryPressure);
			auto boundaryKernel = gradRho(arrays, x_i, V_i, arrays.fluidDensity[i], boundary::kind::volume, v);

			auto a = -1.f * arrays.rest_density *
				(fluid / math::square(arrays.density[i] * arrays.rest_density) +
					boundary / math::square(arrays.rest_density)) *
				boundaryKernel;// *(density ? 1.f : 0.f);
			accel += a;
			if (iteration > 0)
				pressure += boundaryPressure;

			auto F = (-arrays.volume[i] * arrays.rest_density * a).val;
			auto torque = math::cross(F, pb - com).val;
			torque.w = 0.f;
			auto angularAccel = arrays.volumeBoundaryTransformMatrix[v] * 
				(arrays.volumeBoundaryInertiaMatrixInverse[v] * (arrays.volumeBoundaryTransformMatrixInverse[v] * torque));
			auto linearAccel = F / (arrays.volumeBoundaryVolume[v] * arrays.volumeBoundaryDensity[v]).val;
			//angularAccel *= 0.f;
			auto F_ptr = (float4*)arrays.volumeBoundaryAcceleration + v;
			auto T_ptr = (float4*)arrays.volumeBoundaryAngularAcceleration + v;

			if (arrays.volumeBoundaryKind[v] == 0) {
#if defined(__CUDA_ARCH__) || defined(__INTELLISENSE__)
				atomicAdd(&F_ptr->x, linearAccel.x);
				atomicAdd(&F_ptr->y, linearAccel.y);
				atomicAdd(&F_ptr->z, linearAccel.z);
				atomicAdd(&F_ptr->w, linearAccel.w);
				atomicAdd(&T_ptr->x, -angularAccel.x);
				atomicAdd(&T_ptr->y, -angularAccel.y);
				atomicAdd(&T_ptr->z, -angularAccel.z);
				atomicAdd(&T_ptr->w, -angularAccel.w);
#endif
			}
		}
	}
	


	arrays.predictedAcceleration[i] = accel;
	arrays.boundaryPressure[i] = boundaryP;
}

neighFunction(computeAlpha, computeAlphaKernel, "DFSPH: compute alpha", caches<float4, float>{});
neighFunction(computeSourceTerm, computeSourceTermKernel, "DFSPH: compute alpha", caches<float4, float>{});
cellFunction(computeBoundaryPressure, boundaryMLSPressureKernel, "DFSPH: MLS", caches<float4, float>{});
neighFunction(computeAcceleration, computeAccelerationKernel, "IISPH17: jacobi First", caches<float4, float, float>{});
neighFunction(updatePressure, updatePressureKernel, "IISPH17: jacobi First", caches<float4, float4>{});


basicFunctionType predictVelocityKernel(SPH::DFSPH::Memory& arrays) {
	checkedParticleIdx(i);
	arrays.velocity.second[i] = arrays.velocity.first[i];
	arrays.velocity.first[i] += arrays.timestep * arrays.acceleration[i];
	arrays.apparentVolume[i] = arrays.volume[i] / arrays.density[i];
}

basicFunctionType updateVelocityKernel(SPH::DFSPH::Memory& arrays, float* lifeTime, float delay) {
	checkedParticleIdx(i);
	arrays.velocity.first[i] = arrays.velocity.second[i];
	//if(lifeTime[i] > delay * 0.95f)
		arrays.acceleration[i] += arrays.predictedAcceleration[i];
}

cellFunctionType boundaryFrictionKernel(SPH::DFSPH::Memory& arrays, bool density, int32_t iteration) {
	checkedParticleIdx(i);
	alias_arrays((pos, position));
	uFloat4<SI::m> x_i = arrays.position[i];
	uFloat4<SI::acceleration> a_i = arrays.acceleration[i];
	uFloat4<SI::velocity> v_i = arrays.velocity.first[i] + a_i * arrays.timestep;
	uFloat<SI::kg> m_i = arrays.volume[i] * arrays.rest_density;
	uFloat<SI::Pa> p_i = arrays.pressure.first[i];
	uFloat<SI::density> rho_i = arrays.density[i] * arrays.rest_density;
	uFloat<SI::s> dt = arrays.timestep;

	uFloat<SI::velocity> cs = 1.f;
	uFloat<> boundaryViscosity{ (arrays.boundaryViscosity / cs).val };

	auto boundaryDistance = pDistance.val;
	if (boundaryDistance.w < 1e20f) {
		auto pb = arrays.position[i] - boundaryDistance.w * boundaryDistance;
		float_u<SI::Pa> boundaryPressure{ calculateBoundaryPressure<hash_width, order, structure>(i, pb, arrays, density, -1, boundary::kind::plane) };
		auto boundaryKernel = boundary::splineGradient(arrays.position[i], arrays.volume[i], LOOKUP_DENSITY, arrays, boundary::kind::plane, -1);
		float4_u<SI::N> F_p = -arrays.volume[i] * arrays.rest_density * arrays.rest_density* (arrays.pressure.second[i] / math::square(arrays.density[i] * arrays.rest_density) + boundaryPressure / math::square(arrays.rest_density)) * 
			(boundaryKernel);

		uFloat4<SI::velocity> v_b{ 0.f, 0.f, 0.f, 0.f };
		uFloat4<> n_i = math::normalize3(F_p);
		uFloat4<SI::velocity> v_ib = v_i - v_b;
		uFloat4<SI::velocity> t_i = v_ib - math::dot3(v_ib, n_i) * n_i;
		uFloat4<> t_i_norm = math::normalize3(v_ib - math::dot3(v_ib, n_i) * n_i);

		uFloat4<SI::N> f_if1 = - boundaryViscosity * math::length3(F_p) * t_i_norm;
		uFloat4<SI::N> f_if2 = - m_i / dt * t_i;
		bool condition = -dt / m_i * math::dot3(f_if1, t_i_norm) < math::dot3(v_ib, t_i_norm);
		auto a_i = 1.f / m_i * (condition ? f_if1 : f_if2);
		//a_i = f_if2 / m_i;
		if (a_i.val.x == a_i.val.x && a_i.val.y == a_i.val.y && a_i.val.z == a_i.val.z) {
			arrays.acceleration[i] += a_i;
			v_i += a_i * arrays.timestep;
		}
	}
}
cellFunctionType volumeFrictionKernel(SPH::DFSPH::Memory& arrays, bool density, int32_t iteration, int32_t b) {
	checkedParticleIdx(i);
	alias_arrays((pos, position));
	if (volumeDistance(b).val.w >= support_from_volume(arrays.volume[i])*kernelSize())
		return;
	uFloat4<SI::m> x_i = arrays.position[i];
	uFloat4<SI::acceleration> a_i = arrays.acceleration[i];
	uFloat4<SI::velocity> v_i = arrays.velocity.first[i];
	uFloat<SI::kg> m_i = arrays.volume[i] * arrays.rest_density;
	uFloat<SI::Pa> p_i = arrays.pressure.first[i];
	uFloat<SI::density> rho_i = arrays.density[i] * arrays.rest_density;
	uFloat<SI::s> dt = arrays.timestep;

	uFloat<SI::velocity> cs = 1.f;
	uFloat<> boundaryViscosity{ (arrays.boundaryViscosity / cs).val };
	auto pb = arrays.position[i];
	auto tp = pb * 0.f;
	tp.val.w = 1e21f;
	auto volumeDistance = volumeBoundary::distance_fn(pb, arrays.volume[i], arrays, b).val;
	if (volumeDistance.w < 1e20f) {
		pb = pb - volumeDistance.w * volumeDistance;
		pb.val.w = 1.f;
		tp = float4_u<SI::m>{ arrays.volumeBoundaryTransformMatrix[b] * float4{0.f,0.f,0.f,1.f} };
		tp.val.w = volumeDistance.w;
	}
	pb.val.w = arrays.position[i].val.w;
	auto com = tp;
	auto d = volumeDistance.w;
	com.val.w = 1.f;
	if (d < 1e20f) {
		float_u<SI::Pa> boundaryPressure{ calculateBoundaryPressure<hash_width, order, structure>(i, pb, arrays, density, b, boundary::kind::volume) };
		auto fluid = math::max(0.0_Pa, arrays.pressure.second[i]);
		auto boundary = math::max(0.0_Pa, boundaryPressure);
		auto boundaryKernel = gradRho(arrays, x_i, arrays.volume[i], arrays.fluidDensity[i], boundary::kind::volume, b);

		auto F_p = -arrays.volume[i] * arrays.rest_density * arrays.rest_density *
			(fluid / math::square(arrays.density[i] * arrays.rest_density) +
				boundary / math::square(arrays.rest_density)) *
			boundaryKernel;// *(density ? 1.f : 0.f);


		auto v_b = arrays.volumeBoundaryVelocity[b].val 
			-math::cross(pb - com, arrays.volumeBoundaryAngularVelocity[b].val);
		//v_b = -v_b;
		uFloat4<> n_i = math::normalize3(F_p);
		uFloat4<SI::velocity> v_ib = v_i - v_b;
		//arrays.lifetime[i].val = math::length3(v_ib.val);
		uFloat4<SI::velocity> t_i = v_ib - math::dot3(v_ib, n_i) * n_i;
		uFloat4<> t_i_norm = math::normalize3(v_ib - math::dot3(v_ib, n_i) * n_i);

		uFloat4<SI::N> f_if1 = -boundaryViscosity * math::length3(F_p) * t_i_norm;
		uFloat4<SI::N> f_if2 = -m_i / dt * t_i;
		bool condition = -dt / m_i * math::dot3(f_if1, t_i_norm) < math::dot3(v_ib, t_i_norm);
		auto a_i = 1.f / m_i * (condition ? f_if1 : f_if2);
		if (a_i.val.x == a_i.val.x && a_i.val.y == a_i.val.y && a_i.val.z == a_i.val.z) {
			arrays.acceleration[i] += a_i;

			auto F = -(m_i * a_i).val;
			auto torque = math::cross(F, pb - com).val;
			torque.w = 0.f;
			auto angularAccel = arrays.volumeBoundaryTransformMatrix[b]
				* (arrays.volumeBoundaryInertiaMatrixInverse[b] * (arrays.volumeBoundaryTransformMatrixInverse[b] * torque));
			auto linearAccel = F / (arrays.volumeBoundaryVolume[b] * arrays.volumeBoundaryDensity[b]).val;
			//angularAccel *= 0.f;
			auto F_ptr = (float4*)arrays.volumeBoundaryAcceleration + b;
			auto T_ptr = (float4*)arrays.volumeBoundaryAngularAcceleration + b;

//			if (arrays.volumeBoundaryKind[b] != 1) {
//#if defined(__CUDA_ARCH__) || defined(__INTELLISENSE__)
//				atomicAdd(&F_ptr->x, linearAccel.x);
//				atomicAdd(&F_ptr->y, linearAccel.y);
//				atomicAdd(&F_ptr->z, linearAccel.z);
//				atomicAdd(&F_ptr->w, linearAccel.w);
//				atomicAdd(&T_ptr->x, -angularAccel.x);
//				atomicAdd(&T_ptr->y, -angularAccel.y);
//				atomicAdd(&T_ptr->z, -angularAccel.z);
//				atomicAdd(&T_ptr->w, -angularAccel.w);
//#endif
//			}
		}
	}
}
cellFunction(boundaryFriction, boundaryFrictionKernel, "DFSPH: compute alpha");
cellFunction(volumeFriction, volumeFrictionKernel, "DFSPH: compute alpha");
basicFunction(predictVelocity, predictVelocityKernel, "DFSPH: compute alpha");
basicFunction(updateVelocity, updateVelocityKernel, "DFSPH: compute alpha");

void SPH::DFSPH::density_solve(Memory arrays) {
	if (arrays.num_ptcls == 0) return;
	static auto prepareTimer = TimerManager::createGPUTimer("Incompressible: prepare", Color::anjou_pear, true);
	static auto predictTimer = TimerManager::createGPUTimer("Incompressible: predict Velocity", Color::anjou_pear, true);
	static auto alphaTimer = TimerManager::createGPUTimer("Incompressible: compute alpha", Color::anjou_pear, true);
	static auto sourceTimer = TimerManager::createGPUTimer("Incompressible: compute source", Color::anjou_pear, true);
	static auto solveTimer = TimerManager::createGPUTimer("Incompressible: solve", Color::anjou_pear, true);
	static auto boundaryTimer = TimerManager::createGPUTimer("Incompressible: boundary pressure", Color::anjou_pear, true);
	static auto accelTimer = TimerManager::createGPUTimer("Incompressible: compute accel", Color::anjou_pear, true);
	static auto velocityTimer = TimerManager::createGPUTimer("Incompressible: update velocity", Color::anjou_pear, true);
	static auto frictionTimer = TimerManager::createGPUTimer("Incompressible: friction", Color::anjou_pear, true);
	static auto totalTimer = TimerManager::createGPUTimer("Incompressible: total", Color::anjou_pear, true);
	totalTimer->start();
	prepareTimer->start();
	static bool once = true;
	static float4* torqueTemp = nullptr;
	static float4* forceTemp = nullptr;
	auto nb = get<parameters::volumeBoundaryCounter>();
	if (nb > 0 && once) {
		once = false;
		cudaMalloc(&torqueTemp, sizeof(float4) * nb);
		cudaMalloc(&forceTemp, sizeof(float4) * nb);
	}
	if (nb > 0) {
		cudaMemcpy(torqueTemp, arrays::volumeBoundaryAcceleration::ptr, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
		cudaMemcpy(forceTemp, arrays::volumeBoundaryAngularAcceleration::ptr, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
	}
	cuda::sync();
	prepareTimer->stop();
	predictTimer->start();
	launch<predictVelocity>(arrays.num_ptcls, arrays);
	cuda::sync();
	predictTimer->stop();
	alphaTimer->start();
	launch<computeAlpha>(arrays.num_ptcls, arrays, true, true);
	cuda::sync();
	alphaTimer->stop();
	sourceTimer->start();
	launch<computeSourceTerm>(arrays.num_ptcls, arrays, true, true);
	cuda::sync();
	sourceTimer->stop();
	solveTimer->start();
	auto& error = get<parameters::densityError>();
	auto& itCounter = get<parameters::densitySolverIterations>();
	itCounter = 0;
	float totalVolume = math::getValue(algorithm::reduce_sum(arrays.volume, arrays.num_ptcls));
	//std::cout << "Total particle volume: " << totalVolume << std::endl;
	cuda::sync();
	do {
		if (nb > 0) {
			cudaMemcpy(arrays::volumeBoundaryAngularAcceleration::ptr, torqueTemp, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
			cudaMemcpy(arrays::volumeBoundaryAcceleration::ptr, forceTemp, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
		}
		cuda::sync();
		launch<computeBoundaryPressure>(arrays.num_ptcls, arrays, true, true, itCounter);
		cuda::sync();
		launch<computeAcceleration>(arrays.num_ptcls, arrays, true, true);
		cuda::sync();
		launch<updatePressure>(arrays.num_ptcls, arrays, true, true);
		cuda::sync();
		error = math::getValue(algorithm::reduce_sum(arrays.dfsphRhoStar, arrays.num_ptcls)) / totalVolume;
		cuda::sync();
		//std::cout << "Density: " << itCounter << " -> " << error << std::endl;
	} while (itCounter++ < 7 || (error > parameters::densityEta{} && itCounter < 256));
	if (nb > 0) {
		cudaMemcpy(arrays::volumeBoundaryAngularAcceleration::ptr, torqueTemp, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
		cudaMemcpy(arrays::volumeBoundaryAcceleration::ptr, forceTemp, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
	}
	cuda::sync();
	solveTimer->stop();
	boundaryTimer->start();
	launch<computeBoundaryPressure>(arrays.num_ptcls, arrays, true, true, itCounter);
	cuda::sync();
	boundaryTimer->stop();
	accelTimer->start();
	launch<computeAcceleration>(arrays.num_ptcls, arrays, true, true);
	cuda::sync();
	accelTimer->stop();
	velocityTimer->start();
	launch<updateVelocity>(arrays.num_ptcls, arrays, arrays::lifetime::ptr, -FLT_MAX);
	cuda::sync();
	velocityTimer->stop();
	frictionTimer->start();
	launch<boundaryFriction>(arrays.num_ptcls, arrays, true, itCounter);
	for (int32_t b = 0; b < nb; ++b) {
		cuda::sync();
		//launch<volumeFriction>(arrays.num_ptcls, arrays, true, itCounter, b);
	}
	cuda::sync();
	frictionTimer->stop();
	totalTimer->stop();
}
void SPH::DFSPH::divergence_solve(Memory arrays) {
	if (arrays.num_ptcls == 0) return;
	bool planeDivergence = get<parameters::movingPlanes>().size() == 0;
	planeDivergence = false;
	//return;
	static bool once = true;
	static float4* torqueTemp = nullptr;
	static float4* forceTemp = nullptr;
	auto nb = get<parameters::volumeBoundaryCounter>();
	if (nb > 0 && once) {
		once = false;
		cudaMalloc(&torqueTemp, sizeof(float4) * nb);
		cudaMalloc(&forceTemp, sizeof(float4) * nb);
	}
	if (nb > 0) {
		cudaMemcpy(torqueTemp, arrays::volumeBoundaryAngularAcceleration::ptr, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
		cudaMemcpy(forceTemp, arrays::volumeBoundaryAcceleration::ptr, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
	}
	//return;
	cuda::sync();
	launch<predictVelocity>(arrays.num_ptcls, arrays);
	cuda::sync();
	launch<computeAlpha>(arrays.num_ptcls, arrays, false, planeDivergence);
	cuda::sync();
	launch<computeSourceTerm>(arrays.num_ptcls, arrays, false, planeDivergence);
	cuda::sync();
	auto& error = get<parameters::divergenceError>();
	auto& itCounter = get<parameters::divergenceSolverIterations>();
	itCounter = 0;
	float totalVolume = math::getValue(algorithm::reduce_sum(arrays.volume, arrays.num_ptcls));
	cuda::sync();
	//std::cout << "Total volume in frame " << get<parameters::frame>() << " = " << totalVolume << std::endl;
	do {
		if (nb > 0) {
			cudaMemcpy(arrays::volumeBoundaryAngularAcceleration::ptr, torqueTemp, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
			cudaMemcpy(arrays::volumeBoundaryAcceleration::ptr, forceTemp, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
		}
		cuda::sync();
		launch<computeBoundaryPressure>(arrays.num_ptcls, arrays, false, planeDivergence, itCounter);
		cuda::sync();
		launch<computeAcceleration>(arrays.num_ptcls, arrays, false, planeDivergence);
		cuda::sync();
		launch<updatePressure>(arrays.num_ptcls, arrays, false, planeDivergence);
		cuda::sync();
		error = math::getValue(algorithm::reduce_sum(arrays.dfsphDpDt, arrays.num_ptcls)) / totalVolume;
		//std::cout << "Divergence: " << itCounter << " -> " << error << std::endl;
		itCounter++;
	} while (itCounter < 2 || error > parameters::divergenceEta{} && itCounter < 256);
	if (nb > 0) {
		cudaMemcpy(arrays::volumeBoundaryAngularAcceleration::ptr, torqueTemp, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
		cudaMemcpy(arrays::volumeBoundaryAcceleration::ptr, forceTemp, sizeof(float4) * nb, cudaMemcpyDeviceToDevice);
	}
	launch<computeBoundaryPressure>(arrays.num_ptcls, arrays, false, planeDivergence, itCounter);
	cuda::sync();
	launch<computeAcceleration>(arrays.num_ptcls, arrays, false, planeDivergence);
	cuda::sync();
	launch<updateVelocity>(arrays.num_ptcls, arrays, arrays::lifetime::ptr, FLT_MAX);// *parameters::adaptive::delay::ptr);
	cuda::sync();
	//launch<boundaryFriction>(arrays.num_ptcls, arrays, false, itCounter);
	//for (int32_t b = 0; b < nb; ++b) {
	//	cuda::sync();
	//	launch<volumeFriction>(arrays.num_ptcls, arrays, false, itCounter, b);
	//}
	//cuda::sync();
}
