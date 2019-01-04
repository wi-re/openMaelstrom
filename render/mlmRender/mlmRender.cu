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

__device__ __constant__ SceneInformation cScene;
__device__ __constant__ FluidSystem fluidSystem;
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

auto sgn(float x) {
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

	float invdirx = worldRay.dir.x;
	float invdiry = worldRay.dir.y;
	float invdirz = worldRay.dir.z;


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

	return AABBHit{ true, tmin, tmax };
}

__device__  rayHit rayIntersectFluid(Ray worldRay) {
	auto aabb = rayIntersectFluidAABB(worldRay);
	if (aabb.hit == true) {
		float t = aabb.tmax < 0.f ? aabb.tmin : aabb.tmax;
		float3 n;
		if (abs(worldRay.dir.x) >= abs(worldRay.dir.y) && abs(worldRay.dir.x) >= abs(worldRay.dir.z)) n = float3{ 1.f,0.f,0.f };
		if (abs(worldRay.dir.y) >= abs(worldRay.dir.x) && abs(worldRay.dir.y) >= abs(worldRay.dir.z)) n = float3{ 0.f,1.f,0.f };
		if (abs(worldRay.dir.z) >= abs(worldRay.dir.x) && abs(worldRay.dir.z) >= abs(worldRay.dir.y)) n = float3{ 0.f,0.f,1.f };
		return rayHit{ worldRay.orig + t * worldRay.dir, t, n, true };
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

    float numspheres = sizeof(spheres) / sizeof(Sphere);
    for (int32_t i = int32_t(numspheres); i--;) {
      if ((d = spheres[i].intersect(Ray(rayorig, raydir))) && d < scene_t) {
        scene_t = d;
        sphere_id = i;
        geomtype = 1;
      }
    }

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
		emit = float3{ 0.f, 0.f, 0.f };
		accucolor += (mask * emit);
		return n;
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

void cudaMLMRender(SceneInformation scene, cudaGraphicsResource_t resource, FluidSystem fsys, float3 *acc, unsigned framenumber,
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