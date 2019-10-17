#include <SPH/IISPH17/iisph17.cuh>
#include <utility/include_all.h>
//
//basicFunctionType IISPH_predictAdvection2(SPH::IISPH17::Memory arrays) {
//	checkedParticleIdx(i);
//	arrays.velocity[i] = arrays.velocity[i] + arrays.timestep * arrays.acceleration[i];
//	arrays.apparentVolume[i] = arrays.volume[i] / arrays.density[i];
//}
//
//neighFunctionType IISPH_prepareSolver2(SPH::IISPH17::Memory arrays) {
//	checkedParticleIdx(i);
//	cache_arrays((pos, position), (v_adv, velocity), (i_vol, apparentVolume), (vol, volume));
//	//auto s_i = 1.f / arrays.density[i] - 1.f;
//	auto s_i = 1.f - arrays.density[i];
//
//	auto V_i = uFloat<SI::volume>(1.f);
//	auto m_i = V_i * arrays.rest_density;
//	auto boundaryKernel = SWH::spikyGradient(pos[i], arrays);
//	auto boundaryKernelSpline = SWH::splineGradient(pos[i], arrays);
//
//	auto a_ii = 1.f / m_i * math::dot3(boundaryKernelSpline, boundaryKernelSpline)*0.f;
//	auto kernelSum = boundaryKernel * 0.f;
//	s_i = s_i - arrays.timestep * math::dot3(v_adv[i], boundaryKernel);
//
//	iterateNeighbors(j) {
//		auto spikyGradient = GPW_ij;
//		auto m_j = vol[j] * arrays.rest_density;
//
//		kernelSum += i_vol[j] * spikyGradient;
//		a_ii += i_vol[j] * i_vol[j] / m_j * math::dot3(spikyGradient, spikyGradient);
//		s_i = s_i - arrays.timestep * i_vol[j] * math::dot3(v_adv[i] - v_adv[j], spikyGradient);
//	}
//
//	a_ii += math::dot3(kernelSum, kernelSum) / (arrays.apparentVolume[i] * arrays.rest_density);
//
//	arrays.sourceTerm[i] = s_i;
//	arrays.Aii[i] = -math::square(arrays.timestep) * i_vol[i] * a_ii;
//	arrays.pressure.second[i] = 0.75f * arrays.pressure.first[i];
//	arrays.boundaryPressure[i] = 0.f;
//	if (arrays.pressure.second[i] != arrays.pressure.second[i] || arrays.pressure.second[i] < 0.f)
//		arrays.pressure.second[i] = 0.f;
//	arrays.pressure.second[i] = 0.f;
//	arrays.pressure.first[i] = arrays.pressure.second[i];
//}
//
//
//neighFunctionType IISPH_jacobiFirst2(const SPH::IISPH17::Memory arrays) {
//	checkedParticleIdx(i);
//	cache_arrays((pos, position), (i_vol, apparentVolume), (pressure, pressure.second));
//
//
//	float4_u<SI::acceleration> kernelSum{ 0.f,0.f,0.f,0.f };
//	iterateNeighbors(j)
//		kernelSum += -1.f * arrays.volume[j] * arrays.rest_density *
//		(pressure[i] / math::square(arrays.density[i] * arrays.rest_density) + pressure[j] / math::square(arrays.density[j] * arrays.rest_density)) *
//		GPW_ij;
//
//	arrays.predictedAcceleration[i] += kernelSum;
//	arrays.pressure.first[i] = arrays.pressure.second[i];
//}
//
//neighFunctionType IISPH_jacobiSecond2(const SPH::IISPH17::Memory arrays) {
//	checkedParticleIdx(i);
//	cache_arrays((pos, position), (acc, predictedAcceleration));
//	alias_arrays((vol, volume), (i_vol, apparentVolume));
//
//	auto boundaryKernel = SWH::spikyGradient(pos[i], arrays);
//
//	auto kernelSum = math::square(arrays.timestep) * math::dot3(2.f * acc[i], boundaryKernel);
//
//	iterateNeighbors(j)
//		kernelSum += math::square(arrays.timestep) * i_vol[j] * math::dot3(acc[i] - acc[j], GPW_ij);;
//
//	float omega = math::max(0.5f, 0.5f * i_vol[i] / vol[i]);
//	omega = 0.5f;
//
//	auto pressure = math::max(arrays.pressure.first[i] + omega * (arrays.sourceTerm[i] - kernelSum) / arrays.Aii[i], 0.f);
//	auto residual = kernelSum - arrays.sourceTerm[i];
//
//	if (math::abs(arrays.Aii[i]) < 1e-10f || pressure != pressure || pressure > 1e16f) {
//		pressure = 0._Pa;
//		residual = decltype(residual)(0.f);
//	}
//
//	arrays.pressure.second[i] = pressure;
//	arrays.volumeError[i] = residual;
//	//arrays.volumeError[i] = math::max(residual, 0.f).val;
//}
//
//basicFunctionType updateAccelerations2(SPH::IISPH17::Memory arrays) {
//	checkedParticleIdx(i);
//	arrays.velocity[i] = arrays.velocity[i] - arrays.timestep * arrays.acceleration[i];
//	arrays.acceleration[i] += arrays.predictedAcceleration[i];
//}
//#include <utility/SVD.h>
//template <hash_length hash_width, cell_ordering order, cell_structuring structure> hostDeviceInline auto
//calculateBoundaryPressure(int32_t i, float4_u<SI::m> pb, const SPH::IISPH17::Memory& arrays) {
//	float4 vecSum{ 0.f,0.f,0.f,0.f };
//	SVD::Mat3x3 M{ 0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f };
//	float sumA = 0.f;
//	float sumB = 0.f;
//
//	auto x_b = pb.val;
//	x_b.w = arrays.position[i].val.w;
//	iterateCells(pb, j) {
//		auto x_j = arrays.position[j].val;
//		auto V_j = arrays.apparentVolume[j].val;
//		auto p_j = arrays.pressure.second[j].val;
//
//		auto Wbbf = kernel(x_b, x_j);
//		Wbbf = SWH::spiky(arrays.position[j], arrays).val;
//
//		M = M + SVD::Mat3x3{
//			x_j.x * x_j.x, x_j.x * x_j.y, x_j.x * x_j.z,
//			x_j.x * x_j.y, x_j.y * x_j.y, x_j.y * x_j.z ,
//			x_j.z * x_j.x, x_j.z * x_j.y, x_j.z * x_j.z
//		} *V_j * Wbbf;
//		vecSum += x_j * p_j * V_j * Wbbf;
//		sumA += p_j * V_j * Wbbf;
//		sumB += V_j * Wbbf;
//	}
//	auto svd = SVD::svd(M);
//	auto U = svd.U;
//	auto S = svd.S;
//	auto V = svd.V;
//	S.m_00 = (S.m_00 > 1e-6f ? 1.f / S.m_00 : 0.f);
//	S.m_11 = (S.m_11 > 1e-6f ? 1.f / S.m_11 : 0.f);
//	S.m_22 = (S.m_22 > 1e-6f ? 1.f / S.m_22 : 0.f);
//	S.m_01 = S.m_02 = S.m_10 = S.m_12 = S.m_20 = S.m_21 = 0.f;
//	auto Mp = V * S * U.transpose();
//	auto MpM = Mp * M;
//	auto alpha = sumA / sumB;
//	auto beta = Mp.m_00 * vecSum.x + Mp.m_01 * vecSum.y + Mp.m_02 * vecSum.z;
//	auto gamma = Mp.m_10 * vecSum.x + Mp.m_11 * vecSum.y + Mp.m_12 * vecSum.z;
//	auto delta = Mp.m_20 * vecSum.x + Mp.m_21 * vecSum.y + Mp.m_22 * vecSum.z;
//
//
//	auto det = M.det();
//	if (math::abs(det) < 1e-12f || svd.S.m_00 < 1e-4f || svd.S.m_11 < 1e-4f || svd.S.m_22 < 1e-4f) {
//		beta = gamma = delta = 0.f;
//	}
//	auto pressure = alpha * 1.f + beta * x_b.x + gamma * x_b.y + delta * x_b.z;
//
//	if (pressure != pressure)
//		pressure = 0.f;
//	pressure = math::max(pressure, 0.f);
//	return pressure;
//	//return float_u<SI::Pa>(sumA);
//}
//
//cellFunctionType boundaryMLSPressure(const SPH::IISPH17::Memory arrays) {
//	checkedParticleIdx(i);
//
//	arrays.predictedAcceleration[i] = float4_u <SI::acceleration>{ 0.f,0.f,0.f,0.f };
//
//	float4_u <SI::acceleration> accel{ 0.f,0.f,0.f,0.f };
//	float_u<SI::Pa> boundaryP{ 0.f };
//	arrays.boundaryPressure[i] = boundaryP;
//
//	auto boundaryDistance = boundary::POSfunction(arrays.position[i], arrays);
//	auto volumeDistance = volumeBoundary::volumeDistanceFn(arrays.position[i], arrays, -1);
//	if (boundaryDistance.val.w >= 1e20f && volumeDistance.val.w >= 1e20f)
//		return;
//
//	auto factor = -1.f;// *arrays.apparentVolume[i] / (arrays.volume[i] * arrays.rest_density);
//	//auto factor * (pressure[i] + arrays.boundaryPressure[i]) / 2.f * boundaryKernel
//	auto pb = arrays.position[i];
//	float_u<SI::Pa> boundaryPressure{ 0.f };
//	if (boundaryDistance.val.w < 1e20f && volumeDistance.val.w >= 1e20f) {
//		pb += boundaryDistance.val.w * boundaryDistance.val;
//	}
//	else if (volumeDistance.val.w < 1e20f && boundaryDistance.val.w >= 1e20f) {
//		pb += volumeDistance.val.w * volumeDistance.val;
//	}
//	else {
//		auto p1 = pb + boundaryDistance.val.w * boundaryDistance.val;
//		auto p2 = pb + volumeDistance.val.w * volumeDistance.val;
//		pb = (p1 + p2) * 0.5f;
//	}
//	boundaryPressure = float_u<SI::Pa>{ calculateBoundaryPressure<hash_width, order, structure>(i, pb, arrays) };
//	auto boundaryKernel = SWH::boundaryGradientLUT(arrays.spikyGradientLUT, arrays.position[i], arrays);
//	auto volumeKernel = SWH::volumeGradientLUT(arrays.spikyGradientLUT, arrays.position[i], arrays);
//	//if (boundaryDistance.val.w < 1e20f && volumeDistance.val.w >= 1e20f) {
//	accel += factor * arrays.rest_density *
//		(arrays.pressure.second[i] / math::square(arrays.density[i] * arrays.rest_density) + boundaryPressure / math::square(arrays.rest_density)) *
//		(boundaryKernel);
//	//}
//	//else if (volumeDistance.val.w < 1e20f && boundaryDistance.val.w >= 1e20f) {
//	accel += factor * arrays.rest_density *
//		(arrays.pressure.second[i] / math::square(arrays.density[i] * arrays.rest_density) + boundaryPressure / math::square(arrays.rest_density)) *
//		(volumeKernel);
//	//}
//	//else {
//		//accel += factor * arrays.rest_density * (arrays.pressure.second[i] / math::square(arrays.density[i] * arrays.rest_density) + boundaryPressure / math::square(arrays.rest_density)) * (boundaryKernel + volumeKernel);
//	//}
//	arrays.predictedAcceleration[i] = accel;
//	arrays.boundaryPressure[i] = boundaryPressure;// +volumePressure;
//}
//basicFunctionType boundaryFrictionKerneliisph(SPH::IISPH17::Memory& arrays, float boundaryViscosityA) {
//	checkedParticleIdx(i);
//	alias_arrays((pos, position));
//	uFloat4<SI::m> x_i = arrays.position[i];
//	uFloat4<SI::acceleration> a_i = arrays.acceleration[i];
//	uFloat4<SI::velocity> v_i = arrays.velocity[i] + a_i * arrays.timestep;
//	uFloat<SI::kg> m_i = arrays.volume[i] * arrays.rest_density;
//	uFloat<SI::Pa> p_i = arrays.pressure.first[i];
//	uFloat<SI::density> rho_i = arrays.density[i] * arrays.rest_density;
//	uFloat<SI::s> dt = arrays.timestep;
//
//	uFloat<SI::velocity> cs = 1.f;
//	uFloat<> boundaryViscosity{ (boundaryViscosityA / cs).val };
//
//	uFloat4<SI::m> volumeDistance{ boundary::POSfunction(arrays.position[i], arrays).val };
//	if (volumeDistance.val.w >= 1e20f)
//		return;
//	uFloat4<SI::m> x_b = x_i + volumeDistance * (volumeDistance.val.w - 1.f * HforV1);
//	uFloat4<SI::velocity> v_b{ 0.f,0.f,0.f,0.f };
//
//	uFloat4<SI::kg, SI::m_4> G = arrays.rest_density *  GPW_ibk(SWH::boundaryKind::boundary);
//	uFloat<SI::kg, SI::m_4> Gnorm = math::length3(G);
//	uFloat4<> n_i = math::normalize3(volumeDistance);// G / Gnorm;
//	uFloat4<SI::velocity> v_ib = v_i - v_b;
//	uFloat4<SI::m> x_ib = x_i - x_b;
//
//	// m/s 
//	uFloat4<> t_i = math::normalize3(v_ib - math::dot3(v_ib, n_i) * n_i);
//
//	//  [] * [kg] * [kg m-1 s-2] / [kg2 m-6] * [kg m-4] * [m s-1] != [kg m s-2]
//	//  [] * [kg] * [kg m-1 s-2] * [kg-2 m6] * [kg m-4] * [m s-1]
//	// -> kg * kg * kg-2 * kg = [kg]
//	// -> m-1 * m6 * m-4 * m = [m2]
//	// -> s-2 s-1 = [s-3]
//	uFloat4<SI::N> f_if = -boundaryViscosity * m_i * (math::max(0.0_Pa, p_i) / math::square(rho_i) + math::max(0.0_Pa, arrays.boundaryPressure[i]) / math::square(arrays.rest_density))* Gnorm * t_i;
//	// kg /s * m/s * ? * m/s
//	// kg m s-2
//	// -> kg m2 s-3
//	uFloat4<SI::N> f_if2 = -m_i / dt * math::dot3(v_ib, t_i) * t_i;
//
//
//	bool condition = -dt / m_i * math::dot3(f_if, t_i) < math::dot3(v_ib, t_i);
//
//	uFloat4<SI::N> f_ifd = condition ? f_if : f_if2;
//	a_i = f_ifd / m_i;
//	if (a_i.val.x == a_i.val.x && a_i.val.y == a_i.val.y && a_i.val.z == a_i.val.z)
//		arrays.acceleration[i] += a_i;
//}
//basicFunction(boundaryFrictioniisph, boundaryFrictionKerneliisph, "DFSPH: compute alpha");
//
//basicFunction(Predict, IISPH_predictAdvection2, "IISPH17: predict Advection");
//neighFunction(Prepare, IISPH_prepareSolver2, "IISPH17: prepare Solver", caches<float4, float4, float, float>{});
//neighFunction(Jacobi1, IISPH_jacobiFirst2, "IISPH17: jacobi First", caches<float4, float, float>{});
//neighFunction(Jacobi2, IISPH_jacobiSecond2, "IISPH17: jacobi Second", caches<float4, float4, float>{});
//basicFunction(Update, updateAccelerations2, "IISPH17: updaate Acceleration");
//cellFunction(MLSPressure, boundaryMLSPressure, "IISPH17: MLS", caches<float4, float>{});

void SPH::IISPH17::pressure_solve(Memory mem) {
	//static auto totalVolume = PI4O3 * mem.radius * mem.radius * mem.radius * mem.num_ptcls;
	//static bool once = true;
	//if (once && mem.num_ptcls > 0) {
	//	once = false;
	//	totalVolume = PI4O3 * mem.radius * mem.radius * mem.radius * mem.num_ptcls;
	//}
	//size_t iteration = 0;
	//float limit = get<parameters::eta>() / 100.f;
	//bool test = true;
	//launch<Predict>(mem.num_ptcls, mem);
	//launch<Prepare>(mem.num_ptcls, mem);

	//do {
	//	iteration++;
	//	launch<MLSPressure>(mem.num_ptcls, mem);
	//	cuda::sync();
	//	launch<Jacobi1>(mem.num_ptcls, mem);
	//	cuda::sync();
	//	launch<Jacobi2>(mem.num_ptcls, mem);
	//	cuda::sync();
	//	if (iteration > 1) {
	//		get_u<parameters::density_error>() = math::getValue(algorithm::reduce_sum(mem.volumeError, mem.num_ptcls) / algorithm::reduce_sum(arrays::volume::ptr, mem.num_ptcls));
	//	}
	//	test = iteration < 2;
	//	test = test || (get<parameters::density_error>() > limit);
	//	test = test && (iteration < 256);
	//} while (test);
	//cuda::sync();
	//launch<MLSPressure>(mem.num_ptcls, mem);
	//cuda::sync();
	//launch<Jacobi1>(mem.num_ptcls, mem);

	//launch<Update>(mem.num_ptcls, mem);
	//launch<boundaryFrictioniisph>(mem.num_ptcls, mem, get<parameters::boundaryViscosity>());
	//get<parameters::iterations>() = (int32_t)iteration;

}
