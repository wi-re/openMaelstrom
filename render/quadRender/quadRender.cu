#define NO_QT
#include <utility/math.h>
#include <render/quadRender/quadRender.h>

#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <math_functions.h>
#include <sstream>
#include <texture_types.h>
#include <vector_functions.h>
#include <vector_types.h>

//#define M_PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define NUDGE_FACTOR 1e-3f // epsilon
#define samps 1            // samples
#define BVH_STACK_SIZE 32

__device__ __constant__ SceneInformation cScene;
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
    {8, {24.0f, 0, 0}, {6, 4, 2}, {0.f, 0.f, 0.f}, DIFF},
    {2, {0.f, 24.f, 0}, {6, 4, 2}, {0.f, 0.f, 0.f}, DIFF},
    {1, {0.f, 0.f, 24.f}, {6, 4, 2}, {0.f, 0.f, 0.f}, DIFF},

    {10000, {50.0f, 40.8f, -1060}, {0.1, 0.3, 0.55}, {0.175f, 0.175f, 0.25f}, DIFF},
    {100000, {0.0f, 0, -100001.1}, {0, 0, 0}, {0.5f, 0.0f, 0.0f}, COAT},
    {100000, {0.0f, 0, -100001.2}, {0, 0, 0}, {0.3f, 0.3f, 0.3f}, DIFF}, 

    {1.1, {1.6, 1.0, 0}, {0, 0.0, 0}, {0.9f, .9f, 0.9f}, SPEC},
    {0.3, {0.0f, -0.4, 4}, {.0, 0., .0}, {0.9f, 0.9f, 0.9f}, DIFF},
};

__device__ bool RayIntersectsBox(const gpuBVH& bvh, const float3 &originInWorldSpace, const float3 &rayInWorldSpace, int boxIdx) {
  float Tnear, Tfar;
  Tnear = -FLT_MAX;
  Tfar = FLT_MAX;

  float2 limits;

#define CHECK_NEAR_AND_FAR_INTERSECTION(c)                                                                             \
  if (rayInWorldSpace.##c == 0.f) {                                                                                    \
    if (originInWorldSpace.##c < limits.x)                                                                             \
      return false;                                                                                                    \
    if (originInWorldSpace.##c > limits.y)                                                                             \
      return false;                                                                                                    \
  } else {                                                                                                             \
    float T1 = (limits.x - originInWorldSpace.##c) / rayInWorldSpace.##c;                                              \
    float T2 = (limits.y - originInWorldSpace.##c) / rayInWorldSpace.##c;                                              \
    if (T1 > T2) {                                                                                                     \
      float tmp = T1;                                                                                                  \
      T1 = T2;                                                                                                         \
      T2 = tmp;                                                                                                        \
    }                                                                                                                  \
    if (T1 > Tnear)                                                                                                    \
      Tnear = T1;                                                                                                      \
    if (T2 < Tfar)                                                                                                     \
      Tfar = T2;                                                                                                       \
    if (Tnear > Tfar)                                                                                                  \
      return false;                                                                                                    \
    if (Tfar < 0.f)                                                                                                    \
      return false;                                                                                                    \
  }
  auto lim = bvh.cudaBVHlimits[boxIdx];
   limits = float2{ lim.bottom.x, lim.top.x };
  CHECK_NEAR_AND_FAR_INTERSECTION(x)
   limits = float2{ lim.bottom.y, lim.top.y };
  CHECK_NEAR_AND_FAR_INTERSECTION(y)
   limits = float2{ lim.bottom.z, lim.top.z };
  CHECK_NEAR_AND_FAR_INTERSECTION(z)
  return true;
}

__device__ bool BVH_IntersectTriangles(gpuBVH bvh, const float3 &origin, const float3 &ray,
                                       unsigned avoidSelf, int &pBestTriIdx, float3 &pointHitInWorldSpace, float &kAB,
                                       float &kBC, float &kCA, float &hitdist, float3 &boxnormal) {
  pBestTriIdx = -1;
  float bestTriDist;
  bestTriDist = FLT_MAX;
  int32_t stack[BVH_STACK_SIZE];
  int32_t stackIdx = 0;
  stack[stackIdx++] = 0;
  while (stackIdx) {
    int32_t boxIdx = stack[stackIdx - 1];
    stackIdx--;
	uint4 data = bvh.cudaBVHindexesOrTrilists[boxIdx];
    if (!(data.x & 0x80000000)) { // INNER NODE
      if (RayIntersectsBox(bvh, origin, ray, boxIdx)) {
        stack[stackIdx++] = data.y; 
        stack[stackIdx++] = data.z; 
        if (stackIdx > BVH_STACK_SIZE) {
          return false;
        }
      }
    } else { 
      for (uint32_t i = data.w; i < data.w + (data.x & 0x7fffffff); i++) {
         int32_t idx = bvh.cudaTriIdxList[i];
        if (avoidSelf == idx)
          continue;
		float4 normal = bvh.cudaTriangleIntersectionData[idx].normal;
        float k = math::dot3(normal, ray);
        if (k == 0.0f)
          continue; 
        float s = (normal.w - math::dot3(normal, origin)) / k;
        if (s <= 0.0f)
          continue;
        if (s <= NUDGE_FACTOR)
          continue;
        float3 hit = ray * s;
        hit += origin;

		float4 ee1 = bvh.cudaTriangleIntersectionData[idx].e1d1;
        float kt1 = math::dot3(ee1, hit) - ee1.w;
        if (kt1 < 0.0f)
          continue;
		float4 ee2 = bvh.cudaTriangleIntersectionData[idx].e2d2;
        float kt2 = math::dot3(ee2, hit) - ee2.w;
        if (kt2 < 0.0f)
          continue;
		float4 ee3 = bvh.cudaTriangleIntersectionData[idx].e3d3;
        float kt3 = math::dot3(ee3, hit) - ee3.w;
        if (kt3 < 0.0f)
          continue;
        {
          float hitZ = math::sqdistance(origin, hit);
          if (hitZ < bestTriDist) {
            bestTriDist = hitZ;
            hitdist = sqrtf(bestTriDist);
            pBestTriIdx = idx;
            pointHitInWorldSpace = hit;
            kAB = kt1;
            kBC = kt2;
            kCA = kt3;
          }
        }
      }
    }
  }

  return pBestTriIdx != -1;
}


__device__ float3 path_trace(curandState *randstate, float3 originInWorldSpace, float3 rayInWorldSpace, int avoidSelf,
                             const gpuBVH &sceneBVH) {
  float3 mask = float3{1.0f, 1.0f, 1.0f};
  float3 accucolor = float3{0.0f, 0.0f, 0.0f};

  for (int32_t bounces = 0; bounces < 5; bounces++) {
    int32_t sphere_id = -1;
    int32_t triangle_id = -1;
    int32_t pBestTriIdx = -1;
    int32_t geomtype = -1;
    const Triangle *pBestTri = NULL;
    float3 pointHitInWorldSpace;
    float kAB = 0.f, kBC = 0.f, kCA = 0.f;

    float d = 1e20f;
    float scene_t = 1e20f;
    float hitdistance = 1e20f;
    float3 f = float3{0, 0, 0};
    float3 emit = float3{0, 0, 0};
    float3 x;  // intersection point
    float3 n;  // normal
    float3 nl; // oriented normal
    float3 boxnormal = float3{0, 0, 0};
    float3 dw; // ray direction of next path segment
    Refl_t refltype;

    float3 rayorig = float3{originInWorldSpace.x, originInWorldSpace.y, originInWorldSpace.z};
    float3 raydir = float3{rayInWorldSpace.x, rayInWorldSpace.y, rayInWorldSpace.z};

    BVH_IntersectTriangles(sceneBVH, originInWorldSpace, rayInWorldSpace, avoidSelf, pBestTriIdx, pointHitInWorldSpace,
                           kAB, kBC, kCA, hitdistance, boxnormal);

    float numspheres = sizeof(spheres) / sizeof(Sphere);
    for (int32_t i = int32_t(numspheres); i--;) {
      if ((d = spheres[i].intersect(Ray(rayorig, raydir))) && d < scene_t) {
        scene_t = d;
        sphere_id = i;
        geomtype = 1;
      }
    }
    avoidSelf = pBestTriIdx;

    if (hitdistance < scene_t && hitdistance > 0.002f) // EPSILON
    {
      scene_t = hitdistance;
      triangle_id = pBestTriIdx;
      geomtype = 2;
    }

    if (scene_t > 1e20f)
      return float3{0, 0, 0};

    if (geomtype == 1) {
      Sphere &sphere = spheres[sphere_id];
      x = originInWorldSpace + rayInWorldSpace * scene_t;
      n = math::normalize(float3{x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z});
      nl = math::dot(n, rayInWorldSpace) < 0 ? n : n * -1;
      f = float3{sphere.col.x, sphere.col.y, sphere.col.z};
      refltype = sphere.refl;
      emit = float3{sphere.emi.x, sphere.emi.y, sphere.emi.z};
      accucolor += (mask * emit);
    }
    if (geomtype == 2) {
      pBestTri = &sceneBVH.pTriangles[triangle_id];
      x = pointHitInWorldSpace;
      n = math::normalize(math::castTo<float3>(pBestTri->normal));
      nl = math::dot(n, rayInWorldSpace) < 0 ? n : n * -1;
      float3 colour = float3{0.9f, 0.3f, 0.0f};
      refltype = COAT;
      f = colour;
      emit = float3{0, 0, 0};
      accucolor += (mask * emit);
    }

    if (refltype == DIFF) {
      float phi = 2 * CUDART_PI_F * curand_uniform(randstate);
      float r2 = curand_uniform(randstate);
      float r2s = sqrtf(r2);
      float3 w = math::normalize(nl);
      float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{0, 1, 0} : float3{1, 0, 0}), w));
      float3 v = math::cross(w, u);

      dw = math::normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));
      pointHitInWorldSpace = x + w * 0.01f;
      mask *= f;
    }
    if (refltype == METAL) {
      float phi = 2.f * CUDART_PI_F * curand_uniform(randstate);
      float r2 = curand_uniform(randstate);
      float phongexponent = 20;
      float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
      float sinTheta = sqrtf(1 - cosTheta * cosTheta);
      float3 w = math::normalize(rayInWorldSpace - n * 2.0f * math::dot(n, rayInWorldSpace));
      float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{0, 1, 0} : float3{1, 0, 0}), w));
      float3 v = math::cross(w, u);

      dw = math::normalize(u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta);
      pointHitInWorldSpace = x + w * 0.01f;
      mask *= f;
    }
    if (refltype == SPEC) {
      dw = rayInWorldSpace - n * 2.0f * math::dot(n, rayInWorldSpace);
      pointHitInWorldSpace = x + nl * 0.01f;
      mask *= f;
    }
    if (refltype == COAT) {
      float rouletteRandomFloat = curand_uniform(randstate);
      float threshold = 0.05f;
      float3 specularColor = float3{1, 1, 1}; // hard-coded
      bool reflectFromSurface = (rouletteRandomFloat < threshold);

      if (reflectFromSurface) {
        mask *= specularColor;
        dw = rayInWorldSpace - n * 2.0f * math::dot(n, rayInWorldSpace);
        pointHitInWorldSpace = x + nl * 0.01f;
      } else {
        float r1 = 2.f * CUDART_PI_F * curand_uniform(randstate);
        float r2 = curand_uniform(randstate);
        float r2s = sqrtf(r2);
        float3 w = math::normalize(nl);
        float3 u = math::normalize(math::cross((fabs(w.x) > .1f ? float3{0, 1, 0} : float3{1, 0, 0}), w));
        float3 v = math::cross(w, u);

        dw = math::normalize(u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1 - r2));
        pointHitInWorldSpace = x + nl * 0.01f;
        mask *= f;
      }
    }
    if (refltype == REFR) {
      bool into = math::dot(n, nl) > 0;
      float nc = 1.0f;
      float nt = 1.5f;
      float nnt = into ? nc / nt : nt / nc;
      float ddn = math::dot(rayInWorldSpace, nl);
      float cos2t = 1.0f - nnt * nnt * (1.f - ddn * ddn);
      if (cos2t < 0.0f) {
        dw = rayInWorldSpace;
        dw -= n * 2.0f * math::dot(n, rayInWorldSpace);
        pointHitInWorldSpace = x + nl * 0.01f;
      } else {
        float3 tdir = rayInWorldSpace * nnt;
        tdir -= n * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t)));
        tdir = math::normalize(tdir);

        float R0 = (nt - nc) * (nt - nc) / (nt + nc) * (nt + nc);
        float c = 1.f - (into ? -ddn : math::dot(tdir, n));
        float Re = R0 + (1.f - R0) * c * c * c * c * c;
        float Tr = 1 - Re; // Transmission
        float P = .25f + .5f * Re;
        float RP = Re / P;
        float TP = Tr / (1.f - P);

        if (curand_uniform(randstate) < 0.25f) {
          mask *= RP;
          dw = rayInWorldSpace;
          dw -= n * 2.0f * math::dot(n, rayInWorldSpace);

          pointHitInWorldSpace = x + nl * 0.01f;
        } else {
          mask *= TP;
          dw = tdir;
          pointHitInWorldSpace = x + nl * 0.001f;
        }
      }
    }
    originInWorldSpace = pointHitInWorldSpace;
    rayInWorldSpace = dw;
  }
  return float3{accucolor.x, accucolor.y, accucolor.z};
}

__global__ void CoreLoopPathTracingKernel(float3 *accumbuffer, gpuBVH sceneBVH, unsigned int framenumber,
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
  for (int32_t s = 0; s < 1; s++) {
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
      float angle = TWO_PI * random1;
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

    finalcol += path_trace(&randState, originInWorldSpace, rayInWorldSpace, -1, sceneBVH) * (1.0f / samps);
  }
  accumbuffer[i] += finalcol;
  float3 tempcol = accumbuffer[i] / framenumber;
  float3 colour = float3{math::clamp(tempcol.x, 0.0f, 1.0f), math::clamp(tempcol.y, 0.0f, 1.0f),
                         math::clamp(tempcol.z, 0.0f, 1.0f)};
  float4 out{(powf(colour.x, 1 / 2.2f)), (powf(colour.y, 1 / 2.2f)), (powf(colour.z, 1 / 2.2f)), 1.f};
  surf2Dwrite(out, surfaceWriteOut, x * sizeof(float4), y, cudaBoundaryModeClamp);
}

void cudaRender(SceneInformation scene, cudaGraphicsResource_t resource, objectLoader &sceneMeshes, float3 *acc,
                unsigned framenumber, unsigned hashedframes) {
  static bool once = true;
  if (once) {
    cudaArray_t color_arr;
    cudaGraphicsMapResources(1, &resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&color_arr, resource, 0, 0);
    cudaBindSurfaceToArray(surfaceWriteOut, color_arr);
    once = false;
  }
  cudaMemcpyToSymbol(cScene, &scene, sizeof(SceneInformation));
  dim3 texturedim((uint32_t)scene.width, (uint32_t)scene.height, 1);
  dim3 blockdim(16, 16, 1);
  dim3 griddim(texturedim.x / blockdim.x, texturedim.y / blockdim.y, 1);
  if (texturedim.x % blockdim.x != 0)
    griddim.x += 1;
  if (texturedim.y % blockdim.y != 0)
    griddim.y += 1;
  CoreLoopPathTracingKernel<<<griddim, blockdim>>>((float3 *)acc, sceneMeshes.getGPUArrays(), framenumber,
                                                   hashedframes);
  cudaDeviceSynchronize();
}