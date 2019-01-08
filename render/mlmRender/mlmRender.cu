#define NO_QT
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <render/mlmRender/mlmRender.h>
#include <sstream>
#include <texture_types.h>
#include <utility/math.h>
#include <vector_functions.h>
#include <vector_types.h>
//#define DEBUG_AABB

__device__ __constant__ SceneInformation cScene;
__device__ __constant__ FluidSystem fluidSystem;
__device__ __constant__ FluidMemory fluidMemory;

surface<void, cudaSurfaceType2D> surfaceWriteOut; 

struct Ray {
  float3 orig; // ray origin
  float3 dir;  // ray direction
  __device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
};

enum Refl_t { DIFF, METAL, SPEC, REFR, COAT }; // material types

struct Sphere {
  float rad;
  float3 pos, emi, col;
  Refl_t refl;
  __device__ float intersect(const Ray &r) const {
    float3 op = pos - r.orig; //
    float t, epsilon = 0.01f;
    float b = math::dot(op, r.dir);
    float disc = b * b - math::dot(op, op) + rad * rad; // discriminant
    if (disc < 0)
      return 0;
    else
      disc = sqrtf(disc);
    return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
  }
};

__device__ Sphere spheres[] = {
    {16, {128.0f, 128, 128}, {6, 4, 2}, {0.f, 0.f, 0.f}, DIFF},

    {10000, {50.0f, 40.8f, -1060}, {0.55, 0.55, 0.55}, {0.175f, 0.175f, 0.175f}, DIFF},

    {100000, {0.0f, 0, -100000.}, {0, 0, 0}, {0.5f, 0.0f, 0.0f}, DIFF},
    {100000, {0.0f, 0, -100000.1}, {0, 0, 0}, {0.3f, 0.3f, 0.3f}, DIFF}
};

__device__ __host__ auto sgn(float x) {
	return x > 0.f ? 1 : (x < 0.f ? -1 : 0);
}

struct rayHit {
	float3 position;
	float depth;
	float3 normal;
	bool status;
};

struct AABBHit {
	bool hit;
	float tmin;
	float tmax;
};

__device__ AABBHit rayIntersectFluidAABB(Ray worldRay) {
	float tmin, tmax, tymin, tymax, tzmin, tzmax;

	float invdirx = 1.f / worldRay.dir.x;
	float invdiry = 1.f / worldRay.dir.y;
	float invdirz = 1.f / worldRay.dir.z;
	
	tmin = (fluidSystem.bounds[invdirx < 0.f].x - worldRay.orig.x) * invdirx;
	tmax = (fluidSystem.bounds[1 - (invdirx < 0.f)].x - worldRay.orig.x) * invdirx;
	tymin = (fluidSystem.bounds[invdiry < 0.f].y - worldRay.orig.y) * invdiry;
	tymax = (fluidSystem.bounds[1 - (invdiry < 0.f)].y - worldRay.orig.y) * invdiry;

	if ((tmin > tymax) || (tymin > tmax))
		return AABBHit{ false };
	if (tymin > tmin) 
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	tzmin = (fluidSystem.bounds[invdirz < 0.f].z - worldRay.orig.z) * invdirz;
	tzmax = (fluidSystem.bounds[1 - (invdirz < 0.f)].z - worldRay.orig.z) * invdirz;

	if ((tmin > tzmax) || (tzmin > tmax))
		return AABBHit{ false };
	if (tzmin > tmin) 
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;
	return AABBHit{ (tmin < 0.f && tmax > 0.f ) || (tmin > 0.f && tmax > 0.f), tmin, tmax };
}

__device__ auto lookup_cell(int3 idx) {
	//auto idx = position_to_idx3D_i(voxelPosition, fluidMemory.min_coord, math::unit_get<1>(fluidMemory.cell_size));
	uint3 cell = uint3{ (uint32_t)idx.x, (uint32_t)idx.y, (uint32_t)idx.z };
	if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
		return -1;
	auto morton = idx3D_to_morton(cell);
	auto s = fluidMemory.hashMap[idx3D_to_hash(cell, fluidMemory.hash_entries)];
	auto ii = s.beginning;
	if (s.beginning == -1)
		return -1;
	for (; ii < s.beginning + s.length;) {
		auto cs = fluidMemory.cellSpan[ii];
		++ii;
		auto jj = cs.beginning;
		if(position_to_morton(fluidMemory.position[jj], fluidMemory, 1.f) == morton)
			return ii;
	}
	return -1;
}

template<typename Func> __device__ void iterateVoxels(Func&& fn, float3 start, float3 dir){
	auto intBound = [](auto s, auto ds) { 
		auto sIsInteger = (roundf(s) == s);
		if (ds < 0 && sIsInteger)
			return 0.f;
		return (ds > 0 ? math::ceilf(s) - s : s - math::floorf(s)) / math::abs(ds);
	};
	auto cs = fluidMemory.cell_size.x;
	auto half_cell = fluidMemory.cell_size * 0.5f;
	constexpr auto scale = 1.f;
	constexpr auto range = 512.f;
	int3 voxelPosition = position_to_idx3D_i(start, fluidMemory.min_coord, math::unit_get<1>(fluidMemory.cell_size));
	int3 step{ (int32_t)sgn(dir.x), (int32_t)sgn(dir.y), (int32_t)sgn(dir.z) };
	
	auto offset = (start - fluidMemory.min_coord) / fluidMemory.cell_size;

	float3 tMax{ intBound(offset.x, dir.x), intBound(offset.y, dir.y), intBound(offset.z, dir.z) };
	float3 tDelta{ ((float)step.x) / dir.x * cs, ((float)step.y) / dir.y * cs, ((float)step.z) / dir.z * cs};
		
	if (dir.x == 0.f && dir.y == 0.f && dir.z == 0.f)
		return;
	auto radius = 10.f / math::length(dir);
	int3 face = step;
	int32_t counter = 0;
	while (true) {
		float3 vPos = fluidMemory.min_coord + math::castTo<float3>(voxelPosition) / scale * fluidMemory.cell_size.x;
		auto morton = position_to_morton(vPos, fluidMemory, scale);
		auto hash = position_to_hash(vPos, fluidMemory, scale);
		auto cell_idx = lookup_cell(voxelPosition);
		if (cell_idx != -1) { // hit occupied cell
			auto t = math::max_elem(tMax);
			auto pos = start + t * dir;
			auto res = fn(voxelPosition, pos, cell_idx, face);
			if (res == true)
				return;
		}
		if (tMax.x < tMax.y) {
			if (tMax.x < tMax.z) {
				if (tMax.x > range) return;
				voxelPosition.x += step.x;
				tMax.x += tDelta.x;
				face = int3{ -step.x,0,0 };
			}
			else {
				if (tMax.z > range) return;
				voxelPosition.z += step.z;
				tMax.z += tDelta.z;
				face = int3{ 0, 0, -step.z};
			}
		}
		else {
			if (tMax.y < tMax.z) {
				if (tMax.y > range) return;
				voxelPosition.y += step.y;
				tMax.y += tDelta.y;
				face = int3{ 0,-step.y,0 };
			}
			else {
				if (tMax.z > range) return;
				voxelPosition.z += step.z;
				tMax.z += tDelta.z;
				face = int3{ 0, 0, -step.z };
			}
		}
	}
}


__device__  rayHit rayIntersectFluid(Ray worldRay) {
#ifdef DEBUG_AABB
	auto aabb_center = (fluidSystem.bounds[1] - fluidSystem.bounds[0]) / 2.f;
	auto aabb_normal = [](auto v) {
		constexpr auto epsilon = 1e-5f;
		auto c = (fluidSystem.bounds[0] + fluidSystem.bounds[1]) * 0.5f;
		auto prel = v - c;
		auto d = math::abs((fluidSystem.bounds[0] - fluidSystem.bounds[1]) * 0.5f);
		auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
		return float3{ (float) n.x, (float) n.y, (float) n.z };
	};
#endif
	auto aabb = rayIntersectFluidAABB(worldRay);
	if (aabb.hit == true) {
#ifdef DEBUG_AABB
		float3 aabb_min = worldRay.orig + aabb.tmin * worldRay.dir;
		float3 aabb_max = worldRay.orig + aabb.tmax * worldRay.dir;
		// DEBUG render for AABB
		if (aabb.tmin >= 0.f)
			return rayHit{ aabb_min, aabb.tmin, math::abs(aabb_normal(aabb_min)), true };
		else
			return rayHit{ aabb_max, aabb.tmax, math::abs(aabb_normal(aabb_max)), true };
#endif
		bool hit = false;
		float3 hitPosition;
		float3 normal;

		iterateVoxels([&](int3 voxel, float3 vPos, int32_t voxelIdx, int3 face) {
			hit = true;
			hitPosition = vPos;
			normal = math::castTo<float3>(face);
			
			return true;
		}, worldRay.orig + math::max(aabb.tmin - 2.f, 0.f) * worldRay.dir, worldRay.dir);
		if(hit == true) 
			return rayHit{hitPosition, math::distance3(worldRay.orig, hitPosition), -normal, true};
	}
	return rayHit{ float3{FLT_MAX, FLT_MAX, FLT_MAX}, FLT_MAX, float3{1.f,0.f,0.f}, false };
}

__device__ float3 path_trace(curandState *randstate, float3 originInWorldSpace, float3 rayInWorldSpace) {
  float3 mask = float3{1.0f, 1.0f, 1.0f};
  float3 accucolor = float3{0.0f, 0.0f, 0.0f};

  for (int32_t bounces = 0; bounces < 5; bounces++) {
    int32_t sphere_id = -1;
    int32_t geomtype = -1;
    float3 pointHitInWorldSpace;

    float d = 1e20f;
    float scene_t = 1e20f;
    float3 f = float3{0, 0, 0};
    float3 emit = float3{0, 0, 0};
    float3 x;  // intersection point
    float3 n;  // normal
    float3 nl; // oriented normal
    float3 dw; // ray direction of next path segment

    float3 rayorig = float3{originInWorldSpace.x, originInWorldSpace.y, originInWorldSpace.z};
    float3 raydir = float3{rayInWorldSpace.x, rayInWorldSpace.y, rayInWorldSpace.z};

    //float numspheres = sizeof(spheres) / sizeof(Sphere);
    //for (int32_t i = int32_t(numspheres); i--;) {
    //  if ((d = spheres[i].intersect(Ray(rayorig, raydir))) && d < scene_t) {
    //    scene_t = d;
    //    sphere_id = i;
    //    geomtype = 1;
    //  }
    //}

	auto fluidHit = rayIntersectFluid(Ray{ rayorig , raydir });
	if (fluidHit.status && fluidHit.depth < scene_t) {
		scene_t = fluidHit.depth;
		geomtype = 3;
	}

    if (scene_t > 1e20f)
      return float3{0, 0, 0};

    if (geomtype == 1) {
      Sphere &sphere = spheres[sphere_id];
      x = originInWorldSpace + rayInWorldSpace * scene_t;
      n = math::normalize(float3{x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z});
      nl = math::dot(n, rayInWorldSpace) < 0 ? n : n * -1;
      f = float3{sphere.col.x, sphere.col.y, sphere.col.z};
      emit = float3{sphere.emi.x, sphere.emi.y, sphere.emi.z};
      accucolor += (mask * emit);
    }

	if (geomtype == 3) {
		x = fluidHit.position;
		n = fluidHit.normal;;
		nl = math::dot(n, rayInWorldSpace) < 0 ? n : n * -1;
		f = float3{ 0.05098f, 0.23137f, 0.494177f };
		f = math::abs(fluidHit.normal);
		return f;
		emit = float3{ 0.f, 0.f, 0.f };
		accucolor += (mask * emit);
	}

    float phi = 2 * CUDART_PI_F * curand_uniform(randstate);
    float r2 = curand_uniform(randstate);
    float r2s = sqrtf(r2);
    float3 w = math::normalize(nl);
    float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{0, 1, 0} : float3{1, 0, 0}), w));
    float3 v = math::cross(w, u);

    dw = math::normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));
    pointHitInWorldSpace = x + w * 0.01f;
    mask *= f;

    originInWorldSpace = pointHitInWorldSpace;
    rayInWorldSpace = dw;
  }
  return float3{accucolor.x, accucolor.y, accucolor.z};
}

__global__ void CoreLoopPathTracingKernel(float3 *accumbuffer, unsigned int framenumber,
                                          unsigned int hashedframenumber) {
  int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cScene.width)
    return;
  if (y >= cScene.height)
    return;

  int32_t threadId =
      (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

  curandState randState;
  curand_init(hashedframenumber + threadId, 0, 0, &randState);

  float3 rendercampos = float3{cScene.m_camera.position.x, cScene.m_camera.position.y, cScene.m_camera.position.z};

  int32_t i = (cScene.height - y - 1) * cScene.width + x;
  int32_t pixelx = x;
  int32_t pixely = cScene.height - y - 1;

  float3 finalcol = float3{0.0f, 0.0f, 0.0f};
  float3 rendercamview =
      math::normalize(float3{cScene.m_camera.view.x, cScene.m_camera.view.y, cScene.m_camera.view.z});
  float3 rendercamup = math::normalize(float3{cScene.m_camera.up.x, cScene.m_camera.up.y, cScene.m_camera.up.z});
  float3 horizontalAxis = math::normalize(math::cross(rendercamview, rendercamup));
  float3 verticalAxis = math::normalize(math::cross(horizontalAxis, rendercamview));

  float3 middle = rendercampos + rendercamview;
  float3 horizontal = horizontalAxis * tanf(cScene.m_camera.fov.x * 0.5f * (CUDART_PI_F / 180));
  float3 vertical = -verticalAxis * tanf(-cScene.m_camera.fov.y * 0.5f * (CUDART_PI_F / 180));

  float jitterValueX = curand_uniform(&randState) - 0.5f;
  float jitterValueY = curand_uniform(&randState) - 0.5f;
  float sx = (jitterValueX + pixelx) / (cScene.width - 1);
  float sy = (jitterValueY + pixely) / (cScene.height - 1);

  // compute pixel on screen
  float3 pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
  float3 pointOnImagePlane =
      rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * cScene.m_camera.focalDistance);

  float3 aperturePoint;
  if (cScene.m_camera.apertureRadius > 0.00001f) {
    float random1 = curand_uniform(&randState);
    float random2 = curand_uniform(&randState);
    float angle = 2.f * CUDART_PI_F * random1;
    float distance = cScene.m_camera.apertureRadius * sqrtf(random2);
    float apertureX = cos(angle) * distance;
    float apertureY = sin(angle) * distance;

    aperturePoint = rendercampos + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
  } else {
    aperturePoint = rendercampos;
  }
  float3 apertureToImagePlane = pointOnImagePlane - aperturePoint;
  apertureToImagePlane = math::normalize(apertureToImagePlane);
  float3 rayInWorldSpace = math::normalize(apertureToImagePlane);
  float3 originInWorldSpace = aperturePoint;

  finalcol += path_trace(&randState, originInWorldSpace, rayInWorldSpace);
  accumbuffer[i] += finalcol;
  float3 tempcol = accumbuffer[i] / framenumber;
  float3 colour = float3{math::clamp(tempcol.x, 0.0f, 1.0f), math::clamp(tempcol.y, 0.0f, 1.0f),
                         math::clamp(tempcol.z, 0.0f, 1.0f)};
  float4 out{(powf(colour.x, 1 / 2.2f)), (powf(colour.y, 1 / 2.2f)), (powf(colour.z, 1 / 2.2f)), 1.f};
  surf2Dwrite(out, surfaceWriteOut, x * sizeof(float4), y, cudaBoundaryModeClamp);
}

void cudaMLMRender(SceneInformation scene, cudaGraphicsResource_t resource, FluidMemory fmem, FluidSystem fsys, float3 *acc, unsigned framenumber,
                   unsigned hashedframes) {
  static bool once = true;

  if (once) {
    cudaArray_t color_arr;
    cudaGraphicsMapResources(1, &resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&color_arr, resource, 0, 0);
    cudaBindSurfaceToArray(surfaceWriteOut, color_arr);
    once = false;
  }

  cudaMemcpyToSymbol(cScene, &scene, sizeof(SceneInformation));
  cudaMemcpyToSymbol(fluidSystem, &fsys, sizeof(FluidSystem));
  cudaMemcpyToSymbol(fluidMemory, &fmem, sizeof(FluidMemory));

  dim3 texturedim((uint32_t)scene.width, (uint32_t)scene.height, 1);
  dim3 blockdim(16, 16, 1);
  dim3 griddim(texturedim.x / blockdim.x, texturedim.y / blockdim.y, 1);
  if (texturedim.x % blockdim.x != 0)
    griddim.x += 1;
  if (texturedim.y % blockdim.y != 0)
    griddim.y += 1;

  CoreLoopPathTracingKernel<<<griddim, blockdim>>>((float3 *)acc, framenumber, hashedframes);
  cudaDeviceSynchronize();
}