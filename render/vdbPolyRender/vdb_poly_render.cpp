//#define NO_OPERATORS
#define BOOST_USE_WINDOWS_H
#include <render/vdbPolyRender/vdb_poly_render.h>
#include <render/util/color_map.h>

static const char *vertexShaderSource = R"(#version 450 
in vec4 posAttr;
in vec3 normAttr;

uniform mat4 perspective_matrix;
uniform mat4 view_matrix;

out vec4 eyeSpacePos;
out vec3 normal;

void main() {
	eyeSpacePos = view_matrix * vec4(posAttr.xyz ,1.f);
	gl_Position = perspective_matrix * eyeSpacePos;
	normal = normalize(normAttr);
})";

static const char *fragmentShaderSource = R"(#version 450 
in vec3 normal;
in vec4 eyeSpacePos;

uniform mat4 perspective_matrix;

void main() {
	vec3 lightDir = normalize(vec3(0, 0, -1));
	vec3 color = vec3(1,1,1);
	float diffuse = abs(max(0.f, dot(normal, lightDir)));

	gl_FragColor = vec4(color * diffuse,1.f);
})";

bool volumeBoundaryRender::valid() { return get<parameters::volumeBoundary>() == true; }

void volumeBoundaryRender::update() { colorMap::instance().update(); }
#undef foreach
#ifdef _WIN32
#pragma warning(push, 0)
#endif
#include <openvdb/Platform.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tree/ValueAccessor.h>
#ifdef _WIN32
#pragma warning(pop)
#endif
#ifdef _WIN32
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif
#include <fstream>
#include <utility/include_all.h>


volumeBoundaryRender::volumeBoundaryRender(OGLWidget *parent) {
  initializeOpenGLFunctions();
  m_program = new QOpenGLShaderProgram(parent);
  m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  std::vector<uint32_t> indices;
  std::vector<openvdb::Vec3f> pyramidVertices;
  std::vector<openvdb::Vec3f> pyramidNormals;

  for (auto boundaryVolume : get<parameters::boundary_volumes>()) {
    std::string vdbFile = get<parameters::config_folder>() + boundaryVolume.fileName.value;
    auto std_file = fs::path(vdbFile);
    auto tmp_file = std_file;
    tmp_file.replace_extension(".vdb_tmp");

    openvdb::Vec3d min, max;
    int4 dimension;

    logger(log_level::info) << "Parsing VDB file: " << std_file;
    openvdb::initialize();
    openvdb::io::File file(vdbFile);

    openvdb::FloatGrid::Ptr levelSet = openvdb::FloatGrid::create();
    levelSet->setTransform(openvdb::math::Transform::createLinearTransform(0.5));
    levelSet->setGridClass(openvdb::GRID_LEVEL_SET);
    levelSet->setName("LevelSet");

    file.open();
    auto grids = file.getGrids();
    openvdb::GridBase::Ptr baseGrid;
    for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName();
         ++nameIter) {
      if (nameIter.gridName() == "surface") {
        baseGrid = file.readGrid(nameIter.gridName());
      }
    }
    file.close();

    levelSet = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
    openvdb::FloatGrid::ConstAccessor c_accessor = levelSet->getConstAccessor();

    // openvdb::math::UniformScaleMap map(0.5);

    openvdb::FloatGrid::ConstAccessor naccessor = levelSet->getConstAccessor();
    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>
        normalSampler(naccessor, levelSet->transform());

    openvdb::math::ScaleMap scalemap(levelSet->voxelSize());

    dimension = int4{256, 256, 256, 0};
    openvdb::CoordBBox box = levelSet->evalActiveVoxelBoundingBox();
    min = levelSet->indexToWorld(box.getStart());
    max = levelSet->indexToWorld(box.getEnd());

    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> d_sampler(
        *levelSet);

    openvdb::tools::VolumeToMesh mesher(levelSet->getGridClass() == openvdb::GRID_LEVEL_SET ? 0.0
                                                                                            : 0.01);
    mesher(*levelSet);

    // Copy points and generate point normals.
    std::vector<float> points(mesher.pointListSize() * 3);
    std::vector<float> normals(mesher.pointListSize() * 3);

    openvdb::tree::ValueAccessor<const typename openvdb::FloatGrid::TreeType> acc(levelSet->tree());
    openvdb::math::GenericMap map(levelSet->transform());
    openvdb::Coord ijk;

    for (openvdb::Index64 n = 0, i = 0, N = mesher.pointListSize(); n < N; ++n) {
      const openvdb::Vec3s &p = mesher.pointList()[n];
      points[i++] = p[0];
      points[i++] = p[1];
      points[i++] = p[2];
    }

    // Copy primitives
    openvdb::tools::PolygonPoolList &polygonPoolList = mesher.polygonPoolList();
    openvdb::Index64 numQuads = 0;
    for (openvdb::Index64 n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
      numQuads += polygonPoolList[n].numQuads();
    }

    indices.reserve(numQuads * 4);
    openvdb::Vec3d normal, e1, e2;

    for (openvdb::Index64 n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
      const openvdb::tools::PolygonPool &polygons = polygonPoolList[n];
      for (openvdb::Index64 i = 0, I = polygons.numQuads(); i < I; ++i) {
        const openvdb::Vec4I &quad = polygons.quad(i);
        indices.push_back(quad[0]);
        indices.push_back(quad[1]);
        indices.push_back(quad[2]);
        indices.push_back(quad[3]);

        e1 = mesher.pointList()[quad[1]];
        e1 -= mesher.pointList()[quad[0]];
        e2 = mesher.pointList()[quad[2]];
        e2 -= mesher.pointList()[quad[0]];
        normal = e1.cross(e2);

        const double length = normal.length();
        if (length > 1.0e-7)
          normal *= (1.0 / length);

        for (openvdb::Index64 v = 0; v < 4; ++v) {
          normals[quad[v] * 3] = static_cast<float>(normal[0]);
          normals[quad[v] * 3 + 1] = static_cast<float>(normal[1]);
          normals[quad[v] * 3 + 2] = static_cast<float>(normal[2]);
        }
      }
    }

    for (uint32_t it = 0; it < mesher.pointListSize(); ++it) {
      pyramidVertices.push_back(
          openvdb::Vec3f{points[it * 3 + 0], points[it * 3 + 1], points[it * 3 + 2]});
      pyramidNormals.push_back(
          openvdb::Vec3f{normals[it * 3 + 0], normals[it * 3 + 1], normals[it * 3 + 2]});
    }
    break;
  }

  m_program->link();
  m_posAttr = m_program->attributeLocation("posAttr");
  m_colAttr = m_program->attributeLocation("normAttr");

  std::vector<uint32_t> idx = {// front
                               0, 1, 2, 3,
                               // top
                               1, 5, 6, 2,
                               // back
                               7, 6, 5, 4,
                               // bottom
                               4, 0, 3, 7,
                               // left
                               4, 5, 1, 0,
                               // right
                               3, 2, 6, 7};
  glGenBuffers(1, &IBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), indices.data(),
               GL_STATIC_DRAW);

  parent->bind(m_program);


  glGenBuffers(1, &VXO);
  glBindBuffer(GL_ARRAY_BUFFER, VXO);
  glBufferData(GL_ARRAY_BUFFER, pyramidVertices.size() * sizeof(QVector3D), pyramidVertices.data(),
               GL_STATIC_DRAW);
  // glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_posAttr);
  glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  //GLfloat uvs[] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

  glGenBuffers(1, &VUV);
  glBindBuffer(GL_ARRAY_BUFFER, VUV);
  // glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
  glBufferData(GL_ARRAY_BUFFER, pyramidNormals.size() * sizeof(QVector3D), pyramidNormals.data(),
               GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_colAttr);
  glVertexAttribPointer(m_colAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  colorMap::instance().bind(m_program, 0, "colorRamp");
  update();
}

void volumeBoundaryRender::render() {
  glBindVertexArray(vao);

  m_program->bind();

  int size;
  glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
  glDrawElements(GL_QUADS, size / sizeof(int32_t), GL_UNSIGNED_INT, 0);
  // glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)0,
  // get<parameters::num_ptcls>());

  m_program->release();
  glBindVertexArray(0);
}
