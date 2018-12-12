#pragma once
#include <utility/include_all.h>
#include <utility/generation.h>

namespace generation {
render_tuple ObjFromVDBWithNormals(fs::path path) {
  auto getNormal = [](auto e0, auto e1, auto e2) {
    auto e1_e0 = e1 - e0;
    auto e2_e0 = e2 - e0;
    auto n = math::cross(e1_e0, e2_e0);
    auto l = math::length3(n);
    if (l > 1e-7f)
      return n / l;
    else
      return float3{0.f, 1.f, 0.f};
  };
 
  std::vector<uint32_t> indices;
  std::vector<vdb::Vec3f> pyramidVertices;
  std::vector<vdb::Vec3f> pyramidNormals;

  auto [grid, min_vdb, max_vdb] = fileToVDB(path);
  float3 min{(float)min_vdb.x(), (float)min_vdb.y(), (float)min_vdb.z()};
  float3 max{(float)max_vdb.x(), (float)max_vdb.y(), (float)max_vdb.z()};

  std::vector<float3> vertices;
  std::vector<float3> normals;
  std::vector<int32_t> normal_counter;
  std::vector<Triangle> planes;
  std::vector<Edge> edges;

  vdbt::VolumeToMesh mesher(grid->getGridClass() == vdb::GRID_LEVEL_SET ? 0.0 : 0.01, 0.1);
  mesher(*grid);
  for (vdb::Index64 n = 0, i = 0, N = mesher.pointListSize(); n < N; ++n) {
    const vdb::Vec3s &p = mesher.pointList()[n];
    vertices.push_back(float3{p.x(), p.y(), p.z()});
    normals.push_back(float3{0.f, 0.f, 0.f});
    normal_counter.push_back(0);
  }
  vdbt::PolygonPoolList &polygonPoolList = mesher.polygonPoolList();
  for (vdb::Index64 n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
    const vdbt::PolygonPool &polygons = polygonPoolList[n];
    for (vdb::Index64 i = 0, I = polygons.numQuads(); i < I; ++i) {
      const vdb::Vec4I &quad = polygons.quad(i);
      auto i0 = (int32_t)quad[0];
      auto i1 = (int32_t)quad[1];
      auto i2 = (int32_t)quad[2];
      auto i3 = (int32_t)quad[3];
      planes.push_back(Triangle{i0, i1, i2});
      planes.push_back(Triangle{i3, i2, i0});
      edges.push_back(Edge{i0, i1});
      edges.push_back(Edge{i1, i2});
      edges.push_back(Edge{i2, i3});
      edges.push_back(Edge{i3, i0});
      edges.push_back(Edge{i0, i2});
      auto e0 = vertices[i0];
      auto e1 = vertices[i1];
      auto e2 = vertices[i2];
      auto e3 = vertices[i3];

      auto n1 = getNormal(e0, e1, e2);
      auto n2 = getNormal(e2, e3, e0);
      normals[i0] += n1 + n2;
      normal_counter[i0] += 2;
      normals[i1] += n1;
      normal_counter[i1] += 1;
      normals[i2] += n1 + n2;
      normal_counter[i2] += 2;
      normals[i3] += n2;
      normal_counter[i3] += 1;
    }
    for (vdb::Index64 i = 0, I = polygons.numTriangles(); i < I; ++i) {
      const vdb::Vec3I &quad = polygons.triangle(i);
      auto i0 = (int32_t)quad[0];
      auto i1 = (int32_t)quad[1];
      auto i2 = (int32_t)quad[2];
      planes.push_back(Triangle{i0, i1, i2});
      edges.push_back(Edge{i0, i1});
      edges.push_back(Edge{i1, i2});
      edges.push_back(Edge{i2, i0});
      auto e0 = vertices[i0];
      auto e1 = vertices[i1];
      auto e2 = vertices[i2];

      auto n1 = getNormal(e0, e1, e2);
      normals[i0] += n1;
      normal_counter[i0]++;
      normals[i1] += n1;
      normal_counter[i1]++;
      normals[i2] += n1;
      normal_counter[i2]++;
    }
  }
  for (int32_t i = 0; i < normals.size(); ++i)
    normals[i] = (normal_counter[i] == 0 ? float3{0.f, 1.f, 0.f} : normals[i] / (float)normal_counter[i]);

  return std::make_tuple(vertices, normals, planes, edges, min, max);
}

render_tuple ObjFromObjWithNormals(fs::path path) {
  auto getNormal = [](auto e0, auto e1, auto e2) {
    auto e1_e0 = e1 - e0;
    auto e2_e0 = e2 - e0;
    auto n = math::cross(e1_e0, e2_e0);
    auto l = math::length3(n);
    if (l > 1e-7f)
      return n / l;
    else
      return float3{0.f, 1.f, 0.f};
  };

  std::ifstream objFile(path);

  std::vector<float3> vertices;
  std::vector<float3> normals;
  std::vector<int32_t> normal_counter;
  std::vector<Triangle> planes;
  std::vector<Edge> edges;

  for (std::string line; std::getline(objFile, line);) {
    auto vs = readValues<float>(line, "v");
    if (vs.size() != 0) {
      if (vs.size() != 3)
        throw std::invalid_argument(std::string("could not parse line ") + line +
                                    " as it did not contain 3 coordinates.");
      vertices.push_back(float3{vs[0], vs[1], vs[2]});
      continue;
    }
    auto fs = readValues<int32_t>(line, "f");
    if (fs.size() != 0) {
      if (fs.size() != 3 && fs.size() != 4)
        throw std::invalid_argument(std::string("could not parse line ") + line +
                                    " as it did not contain 3 or 4 indices.");
      planes.push_back(Triangle{fs[0] - 1, fs[1] - 1, fs[2] - 1});
      if (fs.size() == 4)
        planes.push_back(Triangle{fs[2] - 1, fs[3] - 1, fs[0] - 1});
      continue;
    }
  }
  normals.resize(vertices.size());
  normal_counter.resize(vertices.size());
  for (const auto &plane : planes) {
    edges.push_back(Edge{plane.i0, plane.i1});
    edges.push_back(Edge{plane.i1, plane.i2});
    edges.push_back(Edge{plane.i2, plane.i0});

    auto e0 = vertices[plane.i0];
    auto e1 = vertices[plane.i1];
    auto e2 = vertices[plane.i2];

    auto n1 = -getNormal(e0, e1, e2);
    normals[plane.i0] += n1;
    normal_counter[plane.i0]++;
    normals[plane.i1] += n1;
    normal_counter[plane.i1]++;
    normals[plane.i2] += n1;
    normal_counter[plane.i2]++;
  }

  for (int32_t i = 0; i < normals.size(); ++i)
    normals[i] = (normal_counter[i] == 0 ? float3{0.f, 1.f, 0.f} : normals[i] / (float)normal_counter[i]);

  float3 min = vector_t<float, 3>::min();
  float3 max = vector_t<float, 3>::max();
  for (const auto &vtx : vertices) {
    min = math::min(min, vtx);
    max = math::max(max, vtx);
  }

  return std::make_tuple(vertices, normals, planes, edges, min, max);
}

render_tuple ObjWithNormals(fs::path path) {
  if (path.extension() == ".vdb")
    return ObjFromVDBWithNormals(path);
  if (path.extension() == ".obj")
    return ObjFromObjWithNormals(path);
  else {
    std::cerr << "Unknown file extension in " << path.string() << std::endl;
    throw std::invalid_argument("Unknown extension");
  }
}
} // namespace generation