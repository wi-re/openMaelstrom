#include <SPH/surface/surfaceDistance.cuh>
#include <utility/include_all.h>

neighFunctionType estimateSurface(SPH::distance::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume));

  SWH2<SPH::distance::Memory> swh(arrays, pos[i], vol[i]);
  auto x_i = pos[i];
  auto POS = swh.POS;
  auto normal = POS;
  auto distance = float_u<SI::m>{ - POS.val.w};
  auto xW = x_i + distance * normal;
  auto lutKernel = swh.spline4() / basicVolume;
  auto lutVal = swh.xBar();

  auto xBar_t = -normal * lutVal.first + xW * lutVal.second;
  auto rBar_t = arrays.radius * lutKernel;
  auto kernelSum = lutKernel;
  auto numNeighs = swh.count() * 2.f;

  iterateNeighbors(j) 
  {
    auto w_ij = W_ij;
    if (w_ij > 0.f) 
    {
      kernelSum += w_ij;
      xBar_t += pos[j] * w_ij;
      rBar_t += math::power<ratio<1, 3>>(vol[j] * PI4O3_1) * w_ij;
      numNeighs += 1;
    }
  }

  auto xBar = xBar_t / kernelSum;
  auto rBar = rBar_t / kernelSum;

  auto phi = math::length3(pos[i] - xBar) - rBar;
  auto r = math::power<ratio<1, 3>>(vol[i] * PI4O3_1);

  auto phiOld = arrays.distanceBuffer.first[i];

  phi = math::min(phi, -0.85f * r);

  if (numNeighs < 15)
    phi = -0.85f * r;
  if (numNeighs > arrays.surface_neighborLimit)
    phi = arrays.surface_levelLimit;
  if (numNeighs > kernelNeighbors() * 1.2f)
    phi = arrays.surface_levelLimit;

  phi = math::clamp(phi, phiOld - 1.0f * r, phiOld + 1.0f * r);

  iterateBoundaryPlanes(E) {
    auto dist = math::planeDistance(E, pos[i]);
    if ((dist.val < math::unit_get<1>(arrays.surface_distanceFieldDistances).val &&
         fabsf(math::dot3(E, float4_u<>{1.f, 0.f, 0.f, 0.f}).val) > 0.5f) ||
        (dist.val < math::unit_get<2>(arrays.surface_distanceFieldDistances).val &&
         fabsf(math::dot3(E, float4_u<>{0.f, 1.f, 0.f, 0.f}).val) > 0.5f)) {
      phi = arrays.surface_levelLimit;
    }
  }

  arrays.distanceBuffer.second[i] = phi;
  arrays.distanceBuffer.first[i] = phiOld;
}

basicFunctionType correctEstimate(SPH::distance::Memory arrays) {
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
neighFunctionType propagateSurface(SPH::distance::Memory arrays, int32_t threads) {
  checkedThreadIdx(t);
  alias_arrays((pos, position));
  int32_t i = arrays.surface_idxBuffer.first[t];
  if (i == INT_MIN)
    return;

  int32_t partnerIdx = INT_MAX;
  float_u<SI::m> partnerDistance{FLT_MAX};
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
  } else {
    arrays.surface_idxBuffer.second[t] = 0xFFFFFFFF;
  }
}
neighFunctionType phiSmooth(SPH::distance::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume));

  arrays.markerBuffer[i] = arrays.decisionBuffer[i];

  auto phiSum = 0.0_m;
  auto counter = 0.f;
  iterateNeighbors(j) {
    counter++;
    phiSum += arrays.distanceBuffer.second[j] * W_ij * vol[j]; // / arrays.density[neigh];
  }

  SWH2<SPH::distance::Memory> swh(arrays, pos[i], vol[i]);
  auto POS = swh.POS;
  auto r = math::power<ratio<1, 3>>(arrays.volume[i] * PI4O3_1);
  auto phiOld = arrays.distanceBuffer.first[i];

  if (POS.val.w < 1e20f || counter < 5)
    phiSum = arrays.distanceBuffer.second[i];

  phiSum = math::clamp(phiSum, phiOld - 1.0f * r, phiOld + 1.0f * r);

  arrays.distanceBuffer.first[i] = math::max(phiSum, arrays.surface_levelLimit);
}
neighFunction(estimate, estimateSurface, "Surface: estimate Distance", caches<float4, float>{});
basicFunction(correct, correctEstimate, "Surface: correct Distance");
neighFunction(propagate, propagateSurface, "Surface: Distance iteration");
neighFunction(smooth, phiSmooth, "Surface: smooth Distance", caches<float4, float>{});

struct is_set {
  hostDeviceInline bool operator()(const int x) { return x != -1; }
};

void SPH::distance::distance(Memory mem) {
  // TODO: FIXME!
  int32_t diff = 0;
  auto compact_idx = [&]() {
    diff = (int32_t) algorithm::copy_if(arrays::surface_idxBuffer::rear_ptr, arrays::surface_idxBuffer::ptr, mem.num_ptcls, is_set());
    cuda::Memset(mem.surface_idxBuffer.second, 0xFF, sizeof(int32_t) * mem.num_ptcls);
  };
  cuda::Memset(mem.surface_idxBuffer.second, 0xFF, sizeof(int32_t) * mem.num_ptcls);
  launch<estimate>(mem.num_ptcls, mem);
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
  if (parameters::adaptiveRatio{} > 1.f)
	  parameters::surface_neighborLimit{} = 25;
}
