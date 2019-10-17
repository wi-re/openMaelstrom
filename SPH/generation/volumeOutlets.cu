#include <SPH/generation/volumeOutlets.cuh>
#include <utility/include_all.h>

basicFunctionDeviceType delete_particles(SPH::Outlet::Memory arrays, float t, float o_t) {
  checkedParticleIdx(i);
  auto p = arrays.position[i];
  auto H = p.w * Kernel<kernel_kind::spline4>::kernel_size();
  for (int32_t b = 0; b < arrays.volumeOutletCounter; ++b) {
    cuda_atomic<float> res_atomic(arrays.volumeOutletRateAccumulator + b);

    float4 d_min = arrays.volumeOutletMin[b];
    float4 d_max = arrays.volumeOutletMax[b];

    if ((d_min.x < p.x) && (d_min.y < p.y) && (d_min.z < p.z) && (d_max.x > p.x) && (d_max.y > p.y) &&
        (d_max.z > p.z)) {
      float4 d_p = (p - d_min) / (d_max - d_min);
      float4 n = tex3D<float4>(arrays.volumeOutletVolumes[b], d_p.x, d_p.y, d_p.z);
      float d = n.w;
      n.w = 0.f;
      math::normalize3(n);
      n.w = d;

      if (d < 0.f){
        float d_t = t - o_t;
        float v = arrays.volume[i];
        float rate = arrays.volumeOutletRate[b];
				float snapped_mlm, old = res_atomic.val();
        bool removed = true;
				do {
					snapped_mlm = old;
          float new_rate = snapped_mlm + v;
					if (new_rate  >= d_t * rate){
            removed = false;
						break;
          }
					old = res_atomic.CAS(snapped_mlm, new_rate);
				} while (old != snapped_mlm);
        if(removed)
          arrays.position[i].w = FLT_MAX;
      }
    }
  }
}

basicFunctionDevice(deleteParticles, delete_particles, "Delete Particles");

void SPH::Outlet::remove(Memory mem) {
  static std::vector<float> old_rates;
  std::vector<float> rates;
  for(auto vol : get<parameters::outletVolumes>()){
    rates.push_back(vol.flowRate.value);
  }
  if(rates.size() != old_rates.size() || rates != old_rates){
    old_rates = rates;
    cudaMemcpy(arrays::volumeOutletRate::ptr, old_rates.data(), sizeof(float) * old_rates.size(), cudaMemcpyHostToDevice);
  }
  if (get<parameters::simulationTime>() > get<parameters::volumeOutletTime>() + 1.f) {
    cudaMemset(arrays::volumeOutletRateAccumulator::ptr, 0x00, sizeof(float) * mem.volumeOutletCounter);
    get<parameters::volumeOutletTime>() = get<parameters::simulationTime>();
  }
  launchDevice<deleteParticles>(mem.num_ptcls, mem, get<parameters::simulationTime>(), get<parameters::volumeOutletTime>());
}