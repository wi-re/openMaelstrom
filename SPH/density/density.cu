#define NEW_STYLE
#include <SPH/density/density.cuh>
#include <utility/include_all.h>
#include <utility/SPH/boundaryFunctions.h>


template<typename Func, typename T>
hostDeviceInline auto iterateMasked2(int32_t i, T& arrays, Func fn) {
	auto x_i = arrays.position[i].val;
	auto factor = powf(0.5f, (float)arrays.MLMResolution[i]);
	auto resolution = arrays.MLMResolution[i];
	int3 cell = pToIdx3(x_i, arrays.min_coord.val, arrays.cell_size.val.x * factor);
	printf("-> %d %d %d\n", cell.x, cell.y, cell.z);
	for (int32_t ci = 0; ci < 27; ++ci) {
		auto c = arrays.neighborMask[i + ci * arrays.max_numptcls];
		if (c.mask == 0) continue;
		printf("<- %d %d %d [%d %d %d]\n", cell.x + (int32_t)c.xi - 2, cell.y + (int32_t)c.yi - 2, cell.z + (int32_t) c.zi - 2, c.xi, c.yi, c.zi);
		auto morton = idx3D_to_morton(cell + int3{ (int32_t)c.xi - 2,(int32_t)c.yi - 2,(int32_t)c.zi - 2 });
		auto s = arrays.compactHashMap[idx3D_to_hash(cell + int3{ (int32_t)c.xi - 2,(int32_t)c.yi - 2,(int32_t)c.zi - 2 }, arrays.hash_entries) + arrays.hash_entries * resolution];
		printf("%d %d %d\n", s.compacted, s.beginning, s.length);
		cell_span cs;
		if (s.compacted) {
			cs = cell_span{ (int32_t)s.beginning, s.length };
		}
		else {
			for (int32_t si = (int32_t)s.beginning; si < (int32_t)s.beginning + s.length; ++si) {
				cs = arrays.compactCellSpan[si + arrays.max_numptcls * resolution];
				if (position_to_morton(arrays.position[cs.beginning], arrays, factor) != morton)
					continue;
			}
		}
		printf("%d\n", c.mask);
		printf("%d %d\n", cs.beginning, cs.length);
		if (c.xi == 0 && c.yi == 0) {
			for (int32_t j = cs.beginning; j < cs.beginning + cs.length; ++j) {
				fn(j);
				printf("%d, ", j);
				//printf("m: %d - " _VECSTR " -> %f\n", j, _VEC(arrays.position[j].val), math::distance3(arrays.position[i], arrays.position[j]).val);
			}
		}
		else {
			while (c.mask != 0) {
				int32_t j = cs.beginning + ffs(c.mask) - 1;
				c.mask = c.mask ^ (1 << (ffs(c.mask) - 1));
				fn(j);
				printf("%d, ", j);
				//printf("m: %d - " _VECSTR " -> %f\n", j, _VEC(arrays.position[j].val), math::distance3(arrays.position[i], arrays.position[j]).val);
				//printf("m: %d - " _VECSTR " -> %f : %f\n", j, _VEC(pos[j].val), math::distance3(pos[i], pos[j]).val, (vol[j] * W_ij).val);
			}
		}
		printf("%\n");
	}
}


/** This function calculates a density estimate using SPH for all particles and additionally uses a
 * lookup table to improve the boundary density contribution of flat boundary planes (no
 * source/novel). Only calculates unit density not actual density  **/
 neighFunctionType estimate_density(SPH::Density::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume));	 
	float_u<> unit_density = 0.f;
	// SWH::spline(pos[i], arrays);
	//unit_density = math::clamp(unit_density, 0.f, 1.f - vol[i] * kernel(pos[i], pos[i]));
	int32_t ctr = 0;
	//arrays.lifetime[i] = unit_density.val;
	iterateNeighbors(j) {
		///VERY STRANGE CONDITIONS, CHECK IT
		if (arrays.particle_type[i] == FLUID_PARTICLE ||
			(arrays.particle_type[i] != FLUID_PARTICLE &&
			(arrays.particle_type[j] == FLUID_PARTICLE || arrays.particle_type[j] == arrays.particle_type[i])))
			unit_density += vol[j] * W_ij;
		ctr++;
	}

	arrays.fluidDensity[i] = unit_density;
	auto boundary_density = boundary::spline(arrays.position[i], arrays.volume[i], unit_density, arrays, boundary::kind::plane, -1);
	for (int32_t b = 0; b < arrays.volumeBoundaryCounter; ++b)
		boundary_density += boundary::spline(arrays.position[i], arrays.volume[i], unit_density, arrays, boundary::kind::volume, b);

	//if (arrays.lifetime[i] < 0.f) {
	//	if (unit_density < 1.f)
	//		unit_density = math::min(1.05f, unit_density + boundary_density);
	//}
	//else
		unit_density += boundary_density;

	arrays.density[i] = unit_density;
#ifdef DEBUG_INVALID_PARITLCES
	if (unit_density != unit_density || unit_density == 0.f) {
		printf("%s: Invalid particle %d: %f %f %f " _VECSTR " %d\n",
			__FUNCTION__, i, unit_density.val, vol[i].val, SWH::spline(pos[i], arrays).val, _VEC(pos[i].val), ctr);
		iterateNeighbors(j) {
			printf("i: %d - " _VECSTR " -> %f : %f\n", j, _VEC(pos[j].val), math::distance3(pos[i], pos[j]).val / kernelSize(), (vol[j] * W_ij).val);
		}
		//float density = 0.f;
		////iterateMasked2(i, arrays, [&](int32_t j) {
		////	density += (vol[j] * W_ij).val;
		////});
		//printf("m: %f\n", density);
	}
#endif
	//if (i == 0) {
	//	printf("%f %f %f %f\n %f %f %f %f\n %f %f %f %f \n%f %f %f %f\n",
	//		arrays.matrixTest[0](0, 0), arrays.matrixTest[0](0, 1), arrays.matrixTest[0](0, 2), arrays.matrixTest[0](0, 3),
	//		arrays.matrixTest[0](1, 0), arrays.matrixTest[0](1, 1), arrays.matrixTest[0](1, 2), arrays.matrixTest[0](1, 3),
	//		arrays.matrixTest[0](2, 0), arrays.matrixTest[0](2, 1), arrays.matrixTest[0](2, 2), arrays.matrixTest[0](2, 3),
	//		arrays.matrixTest[0](3, 0), arrays.matrixTest[0](3, 1), arrays.matrixTest[0](3, 2), arrays.matrixTest[0](3, 3));
	//}
}

neighFunction(estimateDensity, estimate_density, "Estimate Density", caches<float4, float>{});

#include <fstream>
void SPH::Density::estimate_density(Memory mem) {
	launch<estimateDensity>(mem.num_ptcls, mem); 
	//auto mat = std::make_unique<Matrix4x4[]>(1);
	//cudaMemcpy(mat.get(), mem.matrixTest, sizeof(Matrix4x4) * 1, cudaMemcpyDeviceToHost);
	//*mat.get() = Matrix4x4::identity();
	//mat.get()->operator()(1, 2) = get<parameters::frame>();
	//cudaMemcpy(mem.matrixTest, mat.get(), sizeof(Matrix4x4) * 1, cudaMemcpyHostToDevice);
}
 