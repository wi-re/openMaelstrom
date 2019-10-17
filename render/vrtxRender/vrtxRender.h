#pragma once
#include <utility/math.h> 
#include <render/vrtxRender/voxelBVH.h>
#include <render/vrtxRender/fluidSystem.h> 
#include <render/vrtxRender/bvh.h>
#include <render/vrtxRender/geometry.h> 
#include <render/vrtxRender/loader.h> 
#include <render/qGLWidget/base_renderer.h>
#include <utility/include_all.h>     
#define BVH_STACK_SIZE 32    
namespace vrtx{
	struct Ray {
		float3 orig, dir;
		__host__ __device__ float3 operator()(float t) const { return orig + dir * t; }
	};
	struct worldRay {
		float3 orig;
		float depth;
		float3 dir;
		uint32_t internal : 1;
		uint32_t bounces : 7;
		uint32_t index : 24;
		__host__ __device__ float3 operator()(float t) { return orig + dir * t; }
		__host__ __device__ __inline__ operator Ray() { return Ray{ orig, dir };  };
	};
	struct AABBHit {
		bool hit;
		float tmin;
		float tmax;
	};
	struct Pixel {
		float3 color;
		float3 mask;
	};
	struct Box {
		float3 min, max, emi, col;
		Refl_t refl;
		__device__ float intersect(const Ray &worldRay) const {
			float tmin = ((worldRay.dir.x < 0.f ? max.x : min.x) - worldRay.orig.x) / worldRay.dir.x;
			float tmax = ((worldRay.dir.x < 0.f ? min.x : max.x) - worldRay.orig.x) / worldRay.dir.x;
			float tymin = ((worldRay.dir.y < 0.f ? max.y : min.y) - worldRay.orig.y) / worldRay.dir.y;
			float tymax = ((worldRay.dir.y < 0.f ? min.y : max.y) - worldRay.orig.y) / worldRay.dir.y;

			if ((tmin > tymax) || (tymin > tmax))
				return 0.f;
			if (tymin > tmin)
				tmin = tymin;
			if (tymax < tmax)
				tmax = tymax;

			float tzmin = ((worldRay.dir.z < 0.f ? max.z : min.z) - worldRay.orig.z) / worldRay.dir.z;
			float tzmax = ((worldRay.dir.z < 0.f ? min.z : max.z) - worldRay.orig.z) / worldRay.dir.z;

			if ((tmin > tzmax) || (tzmin > tmax))
				return 0.f;
			if (tzmin > tmin)
				tmin = tzmin;
			if (tzmax < tmax)
				tmax = tzmax;
			return (tmin < 0.f && tmax > 0.f) || (tmin > 0.f && tmax > 0.f) ? (tmin < 0.f ? tmax : tmin) : 0.f;
		}
		__device__ float3 normal(const float3& hitPosition) const {
			if (min.x == -FLT_MAX || min.y == -FLT_MAX || min.z == -FLT_MAX) {
				return float3{ min.x == -FLT_MAX ? 0.f : 1.f, min.y == -FLT_MAX ? 0.f : 1.f, min.z == -FLT_MAX ? 0.f : 1.f };
			}
			constexpr auto epsilon = 1e-6f;
			auto c = (min + max) * 0.5f;
			auto prel = hitPosition - c;
			auto d = math::abs((min - max) * 0.5f);
			auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
			auto nc = char3{ static_cast<char>(n.x), static_cast<char>(n.y), static_cast<char>(n.z) };
			auto normal = math::castTo<float3>(nc);
			return normal;
		}
	};
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
}
#ifndef NO_QT
#include <render/qGLWidget/oglwidget.h>

class vRTXrender : public RTXRender {
protected:
  virtual void updateRTX() override;
  virtual void renderRTX(bool pretty, int32_t fn, int32_t s) override;
  float4*color_map = nullptr;
  float4* cu_color_map = nullptr;
  int32_t color_map_elements = 0;
  std::vector<vrtx::Sphere> spheres;
  std::vector<vrtx::Box> boxes;

  void loadSpheres();
  void loadBoxes();
  bool boxesDirty = false;
public:
  vRTXrender(OGLWidget *parent);
  virtual void keyPressEvent(QKeyEvent *event) override;
  virtual std::string getInformation() override;

  vrtx::objectLoader loader;
  vrtx::mesh objects;
  int32_t renderMode = 0;
  int32_t renderModes = 3;
  std::map<std::string, std::vector<float>> timings;
  int32_t bounces = 5;
  bool fluidRender = true;
  vrtx::Refl_t fluidMaterial = vrtx::Refl_t::DIFF;
};
#endif

std::map<std::string, std::vector<float>> cuVRTXRender(SceneInformation scene, cudaGraphicsResource_t resource, vrtx::objectLoader &sceneMeshes, vrtxFluidMemory fmem,
	vrtxFluidArrays farrays,
                  float3 *accumulateBuffer, unsigned framenumber, unsigned hashedframes, int32_t renderMode, int32_t bounces, bool fluidRender, int32_t renderGrid, int32_t surfacingTechnique, vrtx::Refl_t fluidMaterial, bool dirty, std::vector<vrtx::Sphere> spheres, std::vector<vrtx::Box> boxes);