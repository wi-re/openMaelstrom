#include <render/axisRender/axes.h>
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

uniform float axesScale;

uniform sampler1D           colorRamp;

out vec2 uv;
out vec4 color;
out vec4 eyeSpacePos;
flat out int invalid;


void main() {
	color = vec4(colorAttr.xyz,1.0);
	//color =vec4(1,0,0,1);
	eyeSpacePos = view_matrix * vec4(posAttr.xyz * axesScale ,1.f);
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

bool AxesRenderer::valid() { return true; }

void AxesRenderer::update() { colorMap::instance().update(); }

AxesRenderer::AxesRenderer(OGLWidget *parent) {
  initializeOpenGLFunctions();
  m_program = new QOpenGLShaderProgram(parent);
  m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  LOG_INFO << "Compiling vertex shader for " << "AxesRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
  LOG_INFO << "Compiling vertex shader for " << "AxesRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  m_program->link();
  LOG_INFO << "Linking program for " << "AxesRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  m_posAttr = m_program->attributeLocation("posAttr");
  m_colAttr = m_program->attributeLocation("colorAttr");

  parent->bind(m_program);

  GLfloat vertices[] = {
	  // X-Axis
	-4.0, 0.0f, 0.0f,
	4.0, 0.0f, 0.0f,
	// arrow
	4.0, 0.0f, 0.0f,
	3.0, 1.0f, 0.0f,
	4.0, 0.0f, 0.0f,
	3.0, -1.0f, 0.0f,
	// Y-Axis
	0.0, -4.0f, 0.0f,
	0.0, 4.0f, 0.0f,
	// arrow
	0.0, 4.0f, 0.0f,
	1.0, 3.0f, 0.0f,
	0.0, 4.0f, 0.0f,
	-1.0, 3.0f, 0.0f,
	// Z-Axis
	0.0, 0.0f ,-4.0f,
	0.0, 0.0f ,4.0f,
	// arrow
	0.0, 0.0f ,4.0f,
	0.0, 1.0f ,3.0f,
	0.0, 0.0f ,4.0f,
	0.0, -1.0f ,3.0f
  };

  glGenBuffers(1, &VXO);
  glBindBuffer(GL_ARRAY_BUFFER, VXO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_posAttr);
  glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
#define XCOLOR 1.f, 0.f, 0.f
#define YCOLOR 0.f, 1.f, 0.f
#define ZCOLOR 0.f, 0.f, 1.f
#define REPEAT6(x) x, x, x, x, x, x
  GLfloat uvs[] = {
	  REPEAT6(XCOLOR),
	  REPEAT6(YCOLOR),
	  REPEAT6(ZCOLOR)};

  glGenBuffers(1, &VUV);
  glBindBuffer(GL_ARRAY_BUFFER, VUV);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_colAttr);
  glVertexAttribPointer(m_colAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  update();
}

void AxesRenderer::render(bool pretty) {
	if (get<parameters::axesRender>() != 1) return;
  glBindVertexArray(vao);

  m_program->bind();
  glPolygonMode(GL_FRONT, GL_LINE);
  glPolygonMode(GL_BACK, GL_LINE);

  int size;
  glDrawArrays(GL_LINES,0,18);

  glPolygonMode(GL_FRONT, GL_FILL);
  glPolygonMode(GL_BACK, GL_FILL);
  m_program->release();
  glBindVertexArray(0);
}
