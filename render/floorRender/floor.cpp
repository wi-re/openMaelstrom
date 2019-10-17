#include <render/floorRender/floor.h>
#include <render/util/color_map.h>

static const char *vertexShaderSource = R"(#version 450 
in vec3 posAttr;
in vec3 colorAttr;

uniform vec4 camera_right;
uniform vec4 camera_up;
uniform mat4 perspective_matrix;
uniform mat4 view_matrix;
uniform vec3 minCoord;
uniform vec3 maxCoord;
uniform vec3 render_clamp;

uniform vec3 vrtxDomainMin;
uniform vec3 vrtxDomainMax;
uniform float vrtxDomainEpsilon;

uniform float axes_scale;

uniform sampler1D           colorRamp;

out vec2 uv;
out vec4 color;
out vec4 eyeSpacePos;
flat out int invalid;


void main() {
	color = vec4(colorAttr.xyz,1.0);
	vec4 pos = vec4(posAttr.xyz, 1.f);
	//pos.x = posAttr.x < 0.f ? vrtxDomainMin.x + 0.5f : vrtxDomainMax.x - 0.5f;
	//pos.y = posAttr.y < 0.f ? vrtxDomainMin.y + 0.5f : vrtxDomainMax.y - 0.5f;
	//pos.z = posAttr.z < 0.f ? vrtxDomainMin.z + 0.5f : vrtxDomainMax.z - 0.5f;
	
	pos.x *= 1e4f;
	pos.y *= 1e4f;
	pos.z = vrtxDomainMin.z - 0.1f;
	//pos.z *= abs(pos.z) > 1e10f ? sign(pos.z) * 1e7f : pos.z;

	//color =vec4(1,0,0,1);
	eyeSpacePos = view_matrix * pos;
	gl_Position = perspective_matrix * eyeSpacePos;

})";

static const char *fragmentShaderSource = R"(#version 450 
in vec2 uv;
in vec4 color;
in vec4 eyeSpacePos;
flat in int invalid;

uniform sampler1D           colorRamp;
uniform mat4 perspective_matrix;

void main() {
	gl_FragColor = color;

	vec4 projPos = (perspective_matrix * eyeSpacePos);
	gl_FragDepth = (projPos.z / projPos.w)* 0.5 + 0.5;
})";

bool FloorRenderer::valid() { return true; }

void FloorRenderer::update() { colorMap::instance().update(); }

FloorRenderer::FloorRenderer(OGLWidget *parent) {
  initializeOpenGLFunctions();
  m_program = new QOpenGLShaderProgram(parent);
  m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  LOG_INFO << "Compiling vertex shader for " << "FloorRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
  LOG_INFO << "Compiling vertex shader for " << "FloorRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  m_program->link();
  LOG_INFO << "Linking program for " << "FloorRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  m_posAttr = m_program->attributeLocation("posAttr");
  m_colAttr = m_program->attributeLocation("colorAttr");

  parent->bind(m_program);

  GLfloat vertices[] = {
	  -1.0f, -1.0f, 0.0f, 
	  1.0f, -1.0f, 0.0f, 
	  -1.0f, 1.0f, 0.0f, 
	  -1.0f, 1.0f, 0.0f, 
	  1.0f, -1.0f, 0.0f, 
	  1.0f, 1.0f, 0.0f
  };

  glGenBuffers(1, &VXO);
  glBindBuffer(GL_ARRAY_BUFFER, VXO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_posAttr);
  glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
#define COLOR 204.f / 255.f, 207.f / 255.f, 205.f / 255.f
#define REPEAT6(x) x, x, x, x, x, x
  GLfloat uvs[] = {
	  REPEAT6(COLOR)};

  glGenBuffers(1, &VUV);
  glBindBuffer(GL_ARRAY_BUFFER, VUV);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_colAttr);
  glVertexAttribPointer(m_colAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  update();
}

void FloorRenderer::render(bool pretty) {
	if (get<parameters::floorRender>() != 1) return;
  glBindVertexArray(vao);

  m_program->bind();
  int size;
  glDrawArrays(GL_TRIANGLES,0,18);
  m_program->release();
  glBindVertexArray(0);
}
