#include <SPH/surface/surfaceDetection.cuh>
#include <utility/include_all.h>

neighFunctionType estimateNormal(SPH::detection::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume));
	float4 normal{ 0.f,0.f,0.f,0.f };
	iterateNeighbors(j){
		if (i == j) continue;
		auto distance = pos[i] - pos[j];
		normal += math::normalize3(distance);
		//normal += -arrays.volume[j] / arrays.density[j] * GW_ij;
	}
	arrays.particleNormal[i] = math::normalize3(math::castTo<float4>(normal));
}
neighFunctionType detectSurface(SPH::detection::Memory arrays) {
	checkedParticleIdx(i);
	float4 normal = arrays.particleNormal[i];
	bool state = false;
	iterateNeighbors(j) {
		if (i == j) continue;
		auto distance = arrays.position[j] - arrays.position[i];
		auto angle = acosf(math::dot3(normal, math::normalize3(distance)).val);
		state = state || angle <= CUDART_PI_F / 6.f;
	}

	auto V0 = 4.f / 3.f * CUDART_PI_F * math::cubic(arrays.radius);
	auto d = planeBoundary::distance(arrays.position[i], arrays.volume[i], arrays);
	auto db = d;
	//auto dist = math::planeDistance(E, arrays.position[i]);
	auto angle = acosf(math::dot3(normal, -math::normalize3(d)).val);
	auto h0 = support_from_volume(V0) * kernelSize() * 2.f;
	arrays.debugArray[i] = float4{ 0.f, 0.f, 0.f, h0.val / 2.f * 1.1f };
	if (d.val.w < h0) {
		state = state || angle <= CUDART_PI_F / 2.f;
		arrays.debugArray[i] = float4{ d.val.x, d.val.y, d.val.z, d.val.w };

		auto x = d.val.w;
		auto h = support_from_volume(arrays.volume[i]);
		auto H = h.val * kernelSize();
		auto xRel = math::clamp((x + H) / (2.f * H), 0.f, 1.f) * ((float)arrays.boundaryLUTSize - 1.f);
		auto xL = math::floorf(xRel);
		auto xH = math::ceilf(xRel);
		auto xD = xRel - xL;
		int32_t xLi = math::clamp(static_cast<int32_t>(xL), 0, (int32_t)arrays.boundaryLUTSize - 1);
		int32_t xHi = math::clamp(static_cast<int32_t>(xH), 0, (int32_t)arrays.boundaryLUTSize - 1);
		auto lL = arrays.splineLUT[xLi];
		auto lH = arrays.splineLUT[xHi];
		auto val = lL * xD + (1.f - xD) * lH;
		arrays.debugArray[i] = float4{ val.val, xRel, boundary::g(d, h), x };
		arrays.debugArray[i] = db.val;
		//arrays.debugArray[i] = boundary::splineGradient(arrays.position[i], arrays.volume[i], uFloat<>{0.5f}, arrays, boundary::kind::plane, -1).val;
	}

	for (int32_t v = 0; v < arrays.volumeBoundaryCounter; ++v) {
		auto d = volumeBoundary::distance_fn(arrays.position[i], arrays.volume[i], arrays,v);

		//auto dist = math::planeDistance(E, arrays.position[i]);
		auto angle = acosf(math::dot3(normal, -math::normalize3(d)).val);
		auto h0 = support_from_volume(V0) * kernelSize() * 2.f;
		//arrays.debugArray[i] = float4{ 0.f, 0.f, 0.f, h0.val / 2.f * 1.05f };
		if (d.val.w < h0 && d.val.w < db.val.w) {
			//state = state || angle <= CUDART_PI_F / 2.f;

			auto x = d.val.w;
			auto h = support_from_volume(arrays.volume[i]);
			auto H = h.val * kernelSize();
			auto xRel = math::clamp((x + H) / (2.f * H), 0.f, 1.f) * ((float)arrays.boundaryLUTSize - 1.f);
			auto xL = math::floorf(xRel);
			auto xH = math::ceilf(xRel);
			auto xD = xRel - xL;
			int32_t xLi = math::clamp(static_cast<int32_t>(xL), 0, (int32_t)arrays.boundaryLUTSize - 1);
			int32_t xHi = math::clamp(static_cast<int32_t>(xH), 0, (int32_t)arrays.boundaryLUTSize - 1);
			auto lL = arrays.splineLUT[xLi];
			auto lH = arrays.splineLUT[xHi];
			auto val = lL * xD + (1.f - xD) * lH;
			arrays.debugArray[i] = float4{ val.val, xRel, boundary::g(d, h), x };
			arrays.debugArray[i] = d.val;

		//arrays.debugArray[i] = float4{
		//	boundary::g(d,h),
		//	boundary::lookupGradient(arrays, arrays.splineGradientLUT, arrays.boundaryLUTSize, d, arrays.volume[i], uFloat<>{0.5f}, support_from_volume(arrays.volume[i]),0.f).val,
		//	boundary::dg(d,h),
		//	boundary::lookupValue(arrays, arrays.splineLUT, arrays.boundaryLUTSize, d, arrays.volume[i], uFloat<>{0.5f}, support_from_volume(arrays.volume[i]),0.f).val
		//};
		//arrays.debugArray[i] = boundary::internal::lookupGradient(arrays.splineLUT, arrays.splineGradientLUT, arrays.position[i], arrays.volume[i], uFloat<>{0.5f}, arrays, boundary::kind::volume, v).val;

			//arrays.debugArray[i] = boundary::splineGradient(arrays.position[i], arrays.volume[i], uFloat<>{0.5f}, arrays, boundary::kind::volume, v).val;
		}
	}

	if (arrays.neighborListLength[i] < 5) state = false;
	iterateBoundaryPlanes(E) {
		auto dist = math::planeDistance(E, arrays.position[i]);
		if (
		(dist.val < math::unit_get<1>(arrays.surface_distanceFieldDistances).val &&	fabsf(math::dot3(E, float4_u<>{1.f, 0.f, 0.f, 0.f}).val) > 0.5f) ||
		(dist.val < math::unit_get<2>(arrays.surface_distanceFieldDistances).val &&	fabsf(math::dot3(E, float4_u<>{0.f, 1.f, 0.f, 0.f}).val) > 0.5f) ||
		(dist.val < math::unit_get<3>(arrays.surface_distanceFieldDistances).val &&	fabsf(math::dot3(E, float4_u<>{0.f, 0.f, 1.f, 0.f}).val) > 0.5f  && E.val.z > 0.f)) {
			//printf("b");
			state = true;
		}
	}
	//for (int32_t b = 0; b < arrays.volumeBoundaryCounter; ++b) {
	//	auto vos = volumeBoundary::volumeDistanceFn(arrays.position[i], arrays, b);
	//	if (vos.val.w < HforV1 && math::dot3(vos, normal) < 0.f)
	//		state = true;
	//}
	auto r = math::power<ratio<1, 3>>(arrays.volume[i] * PI4O3_1);
	auto phi = state ? arrays.surface_levelLimit : 0.f;
	auto phiOld = arrays.distanceBuffer.first[i];
	phi = state ? arrays.surface_levelLimit : phi;
	phi = math::clamp(phi, arrays.surface_levelLimit, 0.f);
	phi = math::clamp(phi, phiOld - 1.0f * r , phiOld + 1.0f * r );
	arrays.distanceBuffer.second[i] = phi;
	arrays.distanceBuffer.first[i] = phiOld;
}


basicFunctionType correctEstimate(SPH::detection::Memory arrays) {
	checkedParticleIdx(i);
	auto r = math::power<ratio<1, 3>>(arrays.volume[i] * PI4O3_1);

	auto phi = arrays.distanceBuffer.second[i];
	arrays.distanceBuffer.second[i] = phi;
	arrays.decisionBuffer[i] = phi >= -0.85f * r ? 1.f : 0.f;
	arrays.markerBuffer[i] = phi >= -0.85f * r ? 1.f : 0.f;

	if (arrays.markerBuffer[i] < 0.4f) {
		arrays.surface_idxBuffer.second[i] = i;
	}
}
neighFunctionType propagateSurface(SPH::detection::Memory arrays, int32_t threads) {
	checkedThreadIdx(t);
	alias_arrays((pos, position));
	int32_t i = arrays.surface_idxBuffer.first[t];
	if (i == INT_MIN)
		return;

	int32_t partnerIdx = INT_MAX;
	float_u<SI::m> partnerDistance{ FLT_MAX };
	auto partnerPhi = 0.0_m;
	//auto r = math::power<ratio<1, 3>>(arrays.volume[i] * PI4O3_1);
	arrays.markerBuffer[i] = arrays.decisionBuffer[i];
	float marker = arrays.markerBuffer[i];

	iterateNeighbors(j) {
		if (W_ij > 0.f) {
			if (j == i)
				continue;
			float neighbor_decision = arrays.decisionBuffer[j];
			if (neighbor_decision > 0.2f && marker < 0.05f) {
				auto dist = math::abs(math::distance3(pos[i], pos[j]));
				if (dist < partnerDistance) {
					partnerIdx = j;
					partnerDistance = dist;
					partnerPhi = arrays.distanceBuffer.second[j];
				}
			}
		}
		if (partnerIdx != INT_MAX) {
			if (arrays.decisionBuffer[i] < 0.4f) {
				auto phi = partnerPhi - partnerDistance;
				bool changed = phi > arrays.surface_levelLimit.val * 2.f * arrays.radius;
				if (arrays.distanceBuffer.second[i] != phi && changed) {
					cuda_atomic<float> change(arrays.changeBuffer);
					change.add(1.f);
					arrays.distanceBuffer.second[i] = phi;
				}
				arrays.markerBuffer[i] = changed ? 0.5f : 0.1f;
			}
		}
	}
	if (arrays.markerBuffer[i] < 0.4f) {
		arrays.surface_idxBuffer.second[t] = i;
	}
	else {
		arrays.surface_idxBuffer.second[t] = 0xFFFFFFFF;
	}
}
neighFunctionType phiSmooth(SPH::detection::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume));

	arrays.markerBuffer[i] = arrays.decisionBuffer[i];

	auto phiSum = 0.0_m;
	auto counter = 0.f;
	iterateNeighbors(j) {
		counter++;
		phiSum += arrays.distanceBuffer.second[j] * W_ij * vol[j]; // / arrays.density[neigh];
	}

	//SWH2<SPH::detection::Memory> swh(arrays, pos[i], vol[i]);
	auto POS = planeBoundary::distance(pos[i], vol[i], arrays);
	auto r = math::power<ratio<1, 3>>(arrays.volume[i] * PI4O3_1);
	auto phiOld = arrays.distanceBuffer.first[i];

	if (POS.val.w < 1e20f || counter < 5)
		phiSum = arrays.distanceBuffer.second[i];

	phiSum = math::clamp(phiSum, phiOld - 2.0f * r, phiOld + 2.0f * r);
	phiSum = math::clamp(phiSum, arrays.surface_levelLimit, 0.f);

	arrays.distanceBuffer.first[i] = math::max(phiSum, arrays.surface_levelLimit);
}
basicFunction(correct, correctEstimate, "Surface: correct Distance");
neighFunction(propagate, propagateSurface, "Surface: Distance iteration");
neighFunction(smooth, phiSmooth, "Surface: smooth Distance", caches<float4, float>{});

struct is_set {
	hostDeviceInline bool operator()(const int x) { return x != -1; }
};

neighFunction(estimate, estimateNormal, "Surface: estimate Normal", caches<float4, float>{});
neighFunction(detect, detectSurface, "Surface: detect surface");

void SPH::detection::distance(Memory mem) {
	if (mem.num_ptcls == 0) return;
	int32_t diff = 0;
	auto compact_idx = [&]() {
		diff = (int32_t)algorithm::copy_if(arrays::surface_idxBuffer::rear_ptr, arrays::surface_idxBuffer::ptr, mem.num_ptcls, is_set());
		cuda::Memset(mem.surface_idxBuffer.second, 0xFF, sizeof(int32_t) * mem.num_ptcls);
	};
	cuda::Memset(mem.surface_idxBuffer.second, 0xFF, sizeof(int32_t) * mem.num_ptcls);
	launch<estimate>(mem.num_ptcls, mem);
	launch<detect>(mem.num_ptcls, mem);
	launch<correct>(mem.num_ptcls, mem);
	compact_idx();
	int32_t it = 0;
	do {
		cuda::Memset(mem.changeBuffer, 0x00, sizeof(float));
		launch<propagate>(diff, mem, diff);
		cuda::memcpy(&mem.surface_phiChange, mem.changeBuffer, sizeof(float), cudaMemcpyDeviceToHost);
		cuda::memcpy(mem.decisionBuffer, mem.markerBuffer, sizeof(float) * mem.num_ptcls);
		it++;
		if (it % 4 == 0)
			compact_idx();
	} while (mem.surface_phiChange >= 0.5f);
	*parameters::surface_iterations::ptr = it;
	launch<smooth>(mem.num_ptcls, mem);
	*parameters::surface_phiMin::unit_ptr = algorithm::reduce_max(mem.distanceBuffer.first, mem.num_ptcls);
}