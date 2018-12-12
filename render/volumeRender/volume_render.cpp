//#define NO_OPERATORS
#define BOOST_USE_WINDOWS_H
#include <render/volumeRender/volume_render.h>
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

bool volumeRender::valid() { return get<parameters::volumeBoundary>() == true; }

void volumeRender::update() { colorMap::instance().update(); }
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
#include <utility/generation.h>


volumeRender::volumeRender(OGLWidget *parent, std::string vdbFileName) { 
  initializeOpenGLFunctions();
  m_program = new QOpenGLShaderProgram(parent);
  m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  auto path = resolveFile(vdbFileName, {get<parameters::config_folder>()});
  auto [positions, normals, triangles, edges, min, max] = generation::ObjWithNormals(path);
  
  std::vector<int32_t> indices;
  for(const auto& t : triangles){
    indices.push_back(t.i0);
    indices.push_back(t.i1);
    indices.push_back(t.i2);
  }
  tris = indices.size();
  m_program->link();
  m_posAttr = m_program->attributeLocation("posAttr");
  m_colAttr = m_program->attributeLocation("normAttr");

  glGenBuffers(1, &IBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), indices.data(),
               GL_STATIC_DRAW);

  parent->bind(m_program);


  glGenBuffers(1, &VXO);
  glBindBuffer(GL_ARRAY_BUFFER, VXO);
  glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(QVector3D), positions.data(),
               GL_STATIC_DRAW);
  // glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_posAttr);
  glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  //GLfloat uvs[] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

  glGenBuffers(1, &VUV);
  glBindBuffer(GL_ARRAY_BUFFER, VUV);
  // glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
  glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(QVector3D), normals.data(),
               GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_colAttr);
  glVertexAttribPointer(m_colAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  colorMap::instance().bind(m_program, 0, "colorRamp");
  update();
}

void volumeRender::render() {
  if(!active) return;
  glBindVertexArray(vao);

  m_program->bind();

  //int size;
  //glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
  glDrawElements(GL_TRIANGLES, tris, GL_UNSIGNED_INT, 0);
  // glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)0,
  // get<parameters::num_ptcls>());

  m_program->release();
  glBindVertexArray(0);
}

void volumeRender::toggle(){ active = !active;}