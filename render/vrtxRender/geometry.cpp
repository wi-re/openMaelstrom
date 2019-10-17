#include <float.h>
#include <utility>
#include <tuple>
#include <utility/math.h>
#include <render/vrtxRender/geometry.h>
	std::tuple<vrtx::Vertex&, vrtx::Vertex&, vrtx::Vertex&> vrtx::Triangle::getVertices(std::vector<Vertex>& vtx) {
		return std::tie(vtx[i0], vtx[i1], vtx[i2]);
	}
	std::tuple<float3&, float3&, float3&> vrtx::Triangle::getVertices(std::vector<float3>& vtx) {
		return std::tie(vtx[i0], vtx[i1], vtx[i2]);
	}
	std::tuple<vrtx::Vertex, vrtx::Vertex, vrtx::Vertex> vrtx::Triangle::getVertices(const std::vector<Vertex>& vtx) {
		return std::make_tuple(vtx[i0], vtx[i1], vtx[i2]);
	}
	std::tuple<float3, float3, float3> vrtx::Triangle::getVertices(const std::vector<float3>& vtx) {
		return std::make_tuple(vtx[i0], vtx[i1], vtx[i2]);
	}
	std::pair<float4, float4> vrtx::Triangle::getAABB() {
		return std::make_pair(bottom, top);
	}
	vrtx::Triangle::Triangle(int32_t i0_, int32_t i1_, int32_t i2_, std::vector<Vertex>& vtx) :i0(i0_), i1(i1_), i2(i2_) {
		const auto&[vtxA, vtxB, vtxC] = getVertices(vtx);
		center = (vtxA.position + vtxB.position + vtxC.position) / 3.f;

		auto AB = vtxB.position - vtxA.position;
		auto AC = vtxC.position - vtxA.position;
		auto cr = math::normalize(math::cross(AB, AC));
		normal = cr;
		vtx[i0].normal += cr;
		vtx[i1].normal += cr;
		vtx[i2].normal += cr;

		bottom = math::min(bottom, vtxA.position);
		bottom = math::min(bottom, vtxB.position);
		bottom = math::min(bottom, vtxC.position);

		top = math::max(top, vtxA.position);
		top = math::max(top, vtxB.position);
		top = math::max(top, vtxC.position);

		// Algorithm for triangle intersection is taken from Roman Kuchkuda's paper.
		// precompute edge vectors
		auto vc1 = vtxB.position - vtxA.position;
		auto vc2 = vtxC.position - vtxB.position;
		auto vc3 = vtxA.position - vtxC.position;

		// plane of triangle, cross product of edge vectors vc1 and vc2
		normal = math::cross(vc1, vc2);

		// choose longest alternative normal for maximum precision
		auto alt1 = math::cross(vc2, vc3);
		if (math::length(alt1) > math::length(normal))
			normal = alt1; // higher precision when triangle has sharp angles

		auto alt2 = math::cross(vc3, vc1);
		if (math::length(alt2) > math::length(normal))
			normal = alt2;

		normal = math::normalize(normal);

		// precompute dot product between normal and first triangle vertex
		d = math::dot(normal, vtxA.position);

		// edge planes
		e1 = math::cross(normal, vc1);
		e1 = math::normalize(e1);
		d1 = math::dot(e1, vtxA.position);
		e2 = math::cross(normal, vc2);
		e2 = math::normalize(e2);
		d2 = math::dot(e2, vtxB.position);
		e3 = math::cross(normal, vc3);
		e3 = math::normalize(e3);
		d3 = math::dot(e3, vtxC.position);

	}

	void vrtx::Triangle::recalculate(std::vector<Vertex>& vtx) {
	  //STRANGE REFERENCE (WAS WITH & AFTER AUTO)

		auto[vtxA, vtxB, vtxC] = getVertices(vtx);
		center = (vtxA.position + vtxB.position + vtxC.position) / 3.f;

		auto AB = vtxB.position - vtxA.position;
		auto AC = vtxC.position - vtxA.position;
		auto cr = math::normalize(math::cross(AB, AC));
		normal = cr;
		vtxA.normal += cr;
		vtxB.normal += cr;
		vtxC.normal += cr;

		bottom = math::min(bottom, vtxA.position);
		bottom = math::min(bottom, vtxB.position);
		bottom = math::min(bottom, vtxC.position);

		top = math::max(top, vtxA.position);
		top = math::max(top, vtxB.position);
		top = math::max(top, vtxC.position);

		// Algorithm for triangle intersection is taken from Roman Kuchkuda's paper.
		// precompute edge vectors
		auto vc1 = vtxB.position - vtxA.position;
		auto vc2 = vtxC.position - vtxB.position;
		auto vc3 = vtxA.position - vtxC.position;

		// plane of triangle, cross product of edge vectors vc1 and vc2
		normal = math::cross(vc1, vc2);

		// choose longest alternative normal for maximum precision
		auto alt1 = math::cross(vc2, vc3);
		if (math::length(alt1) > math::length(normal))
			normal = alt1; // higher precision when triangle has sharp angles

		auto alt2 = math::cross(vc3, vc1);
		if (math::length(alt2) > math::length(normal))
			normal = alt2;

		normal = math::normalize(normal);

		// precompute dot product between normal and first triangle vertex
		d = math::dot(normal, vtxA.position);

		// edge planes
		e1 = math::cross(normal, vc1);
		e1 = math::normalize(e1);
		d1 = math::dot(e1, vtxA.position);
		e2 = math::cross(normal, vc2);
		e2 = math::normalize(e2);
		d2 = math::dot(e2, vtxB.position);
		e3 = math::cross(normal, vc3);
		e3 = math::normalize(e3);
		d3 = math::dot(e3, vtxC.position);

	}