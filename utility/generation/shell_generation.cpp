#pragma once
#include <utility/include_all.h>
#include <utility/generation.h>

namespace generation{    
std::vector<vdb::Vec4f> ObjToShell(fs::path path, float r, float threshold) {
  auto [points, planes, edges, min, max] = fileToObj(path);
  auto edgeToPos = [&](Edge e) { return std::make_pair(points[e.start], points[e.end]); };
  auto planeToPos = [&](Triangle p) { return std::make_tuple(points[p.i0], points[p.i1], points[p.i2]); };
  auto v = PI4O3 * math::power<3>(r);
  auto [spacing, h, H] = getPacking(r);
  spacing = math::power<ratio<1, 3>>(v) * 1.05f;
  auto [mind, maxd] = getDomain();
  std::vector<vdb::Vec4f> particles;
  std::vector<float4> temp_particles;

  // Emit particles along all edges   
  for (const auto &edge : edges) {
    auto [ps, pe] = edgeToPos(edge);
    if (edge.start > edge.end)
      std::swap(ps, pe);
    auto d = math::distance(pe, ps);
    auto diameter = spacing;
    auto n = math::floorf(d / diameter) - 1;
    auto dis = (pe - ps) / n;
    for (int32_t it = 1; it < n; ++it) {
      auto p = ps + dis * it;
      temp_particles.push_back(float4{p.x, p.y, p.z, h});
    }
  }
  // Emit particles on triangles
  for (const auto &plane : planes) {
    auto [v1, v2, v3] = planeToPos(plane);
    auto v1_v2 = v2 - v1;
    auto v2_v3 = v3 - v2;
    auto v3_v1 = v1 - v3;

    auto len_v1_v2 = math::length3(v1_v2);
    auto len_v2_v3 = math::length3(v2_v3);
    auto len_v3_v1 = math::length3(v3_v1);

    std::vector<sort_Edge> edges;
    edges.emplace_back(v1, v2);
    edges.emplace_back(v2, v3);
    edges.emplace_back(v3, v1);
    std::sort(edges.begin(), edges.end());
    auto e_l = edges[2];
    auto e_m = edges[1];
    auto e_s = edges[0];

    auto e_l_dir = e_l.end - e_l.start;

    if (e_m.start == e_l.end || e_m.end == e_l.end) {

    } else {
      e_l_dir = e_l.start - e_l.end;
    }

    auto e_s_dir = e_s.end - e_s.start;
    auto s = math::normalize(math::cross(e_s_dir, math::cross(e_l_dir, e_s_dir)));
    auto sign = math::dot(s, e_l_dir);
    if (sign < 0.f)
      s = -s;
    auto ht = math::dot(s, e_l_dir);
    auto d = spacing;
    auto nt = floorf(ht / d);

    auto sgn = [](float a) { return (a >= 0.f ? 1.f : -1.f); };

    for (float it = 0; it <= abs(ht); it += d) {
      auto start = e_s.start + /*sgn(ht) * */ it * s;
      sort_Edge emit{start, start + e_s_dir};

      auto intersect = [](sort_Edge e1, sort_Edge e2) {
        auto a = e1.end - e1.start;
        auto b = e2.end - e2.start;
        auto c = e2.start - e1.start;

        auto s = (math::dot(math::cross(c, b), math::cross(a, b))) / math::sqlength(math::cross(a, b));
        auto i = e1.start + a * s;
        return i;
      };
      auto i1 = intersect(emit, e_m);
      auto i2 = intersect(emit, e_l);
      sort_Edge e_i{i1, i2};

      auto len_e_i = math::length(e_i.end - e_i.start);
      auto n_i = floorf(len_e_i / d);
      for (float it = 0; it <= abs(len_e_i); it += d) {
        auto p = ((e_i.start + it * math::normalize(e_i.end - e_i.start)));
        temp_particles.push_back(float4{p.x, p.y, p.z, h});
      }
    }
  }

  // Cleanup overly dense samples
  for (int32_t i = 0; i < temp_particles.size(); ++i) {
    auto p = temp_particles[i];
    bool violating = false;
    for (int32_t j = i + 1; j < temp_particles.size(); ++j) {
      if (math::distance3(p, temp_particles[j]) < spacing * threshold) {
        violating = true;
        break;
      }
    }
    if (!violating)
      particles.push_back(vdb::Vec4f(p.x, p.y, p.z, p.w));
  }
  return particles;
}
}