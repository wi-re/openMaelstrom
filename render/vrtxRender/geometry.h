#pragma once
#include <float.h>
#include <utility>
#include <tuple>
#include <vector>
#include <utility/math.h>
namespace vrtx {
	struct Vertex {
		float4 position;
		float4 normal;
		Vertex(float x, float y, float z, float nx, float ny, float nz)
			: position{ x, y, z }, normal{ nx, ny, nz } {}
		Vertex(float4 p, float4 n) :position(p), normal(n) {}
		Vertex(float3 p, float3 n) :position(math::castTo<float4>(p)), normal(math::castTo<float4>(n)) {}
	};

	struct Triangle {
		int32_t i0, i1, i2;
		float4 color{ 255.f,255.f,255.f,255.f };
		float4 center{ 0.f,0.f,0.f,0.f };
		float4 normal{ 0.f,0.f,0.f,0.f };
		bool twoSided = false;
		float d, d1, d2, d3;
		float4 e1, e2, e3;
		float4 bottom{ -FLT_MAX, -FLT_MAX, -FLT_MAX ,-FLT_MAX };
		float4 top{ FLT_MAX, FLT_MAX, FLT_MAX , FLT_MAX };

		Triangle() {}
		Triangle(int32_t i0_, int32_t i1_, int32_t i2_, std::vector<Vertex>& vtx);

		std::tuple<Vertex&, Vertex&, Vertex&> getVertices(std::vector<Vertex>& vtx);
		std::tuple<float3&, float3&, float3&> getVertices(std::vector<float3>& vtx);
		std::tuple<Vertex, Vertex, Vertex> getVertices(const std::vector<Vertex>& vtx);
		std::tuple<float3, float3, float3> getVertices(const std::vector<float3>& vtx);
		std::pair<float4, float4> getAABB();
		void recalculate(std::vector<Vertex>& vtx);
	};

}