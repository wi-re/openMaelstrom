#include <render/boundaryRender/bounds.h>
#include <render/util/color_map.h>

static const char *vertexShaderSource = R"(#version 450 
in vec4 posAttr;
in vec2 uvAttr;

uniform vec4 camera_right;
uniform vec4 camera_up;
uniform mat4 perspective_matrix;
uniform mat4 view_matrix;
uniform vec3 min_domain;
uniform vec3 max_domain;
uniform vec3 render_clamp;

uniform sampler1D           colorRamp;

out vec2 uv;
out vec4 color;
out vec4 eyeSpacePos;
flat out int invalid;

void main() {
	uv = uvAttr;
	//color = texture(colorRamp,renderIntensity);
	//color = vec4(renderIntensity,renderIntensity,renderIntensity,1.f);
	color = vec4(0.4,0.4,0.4,1.0);
	vec3 pos;
	pos.x = posAttr.x < 0.f ? min_domain.x : max_domain.x;
	pos.y = posAttr.y < 0.f ? min_domain.y : max_domain.y;
	pos.z = posAttr.z < 0.f ? min_domain.z : max_domain.z;

	eyeSpacePos = view_matrix * vec4(pos.xyz ,1.f);
	//eyeSpacePos += vec4(posAttr.xyz * position.w*2.f,0.f);
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
	if(invalid == 1)
		discard;
	vec3 N;
	N.xy = uv.xy * vec2(2.f, 2.f) - vec2(1.f, 1.f);
	float r2 = dot(N.xy,N.xy);
	//if( r2 > 1.f) discard;
	N.z = sqrt(1.f - r2);

	vec3 lightDir = vec3(0, 0, 1);
	float diffuse = abs(dot(N, lightDir));

	gl_FragColor = color;
	//gl_FragColor = color;
	//gl_FragColor = texture(colorRamp,uv.x);
	//gl_FragColor = vec4(N,1.f) ;

vec4 sphereEyeSpacePos;
sphereEyeSpacePos.xyz = eyeSpacePos.xyz + N * eyeSpacePos.w;
sphereEyeSpacePos.w = 1.0;
vec4 projPos = (perspective_matrix * sphereEyeSpacePos);
//gl_FragDepth = (projPos.z / projPos.w)* 0.5 + 0.5;
})";

bool BoundsRenderer::valid() { return true; }

void BoundsRenderer::update() { colorMap::instance().update(); }

BoundsRenderer::BoundsRenderer(OGLWidget *parent) {
  initializeOpenGLFunctions();
  m_program = new QOpenGLShaderProgram(parent);
  m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  m_program->link();
  m_posAttr = m_program->attributeLocation("posAttr");
  m_colAttr = m_program->attributeLocation("uvAttr");

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
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(uint32_t), idx.data(), GL_STATIC_DRAW);

  parent->bind(m_program);

  GLfloat vertices[] = {
      // front
      -1.0,
      -1.0,
      1.0,
      1.0,
      -1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      -1.0,
      1.0,
      1.0,
      // back
      -1.0,
      -1.0,
      -1.0,
      1.0,
      -1.0,
      -1.0,
      1.0,
      1.0,
      -1.0,
      -1.0,
      1.0,
      -1.0,
  };

  glGenBuffers(1, &VXO);
  glBindBuffer(GL_ARRAY_BUFFER, VXO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_posAttr);
  glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  GLfloat uvs[] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

  glGenBuffers(1, &VUV);
  glBindBuffer(GL_ARRAY_BUFFER, VUV);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_colAttr);
  glVertexAttribPointer(m_colAttr, 2, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  colorMap::instance().bind(m_program, 0, "colorRamp");
  update();
}

void BoundsRenderer::render() {
  glBindVertexArray(vao);

  m_program->bind();
  glPolygonMode(GL_FRONT, GL_LINE);
  glPolygonMode(GL_BACK, GL_LINE);

  int size;
  glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
  glDrawElements(GL_QUADS, size / sizeof(int32_t), GL_UNSIGNED_INT, 0);

  glPolygonMode(GL_FRONT, GL_FILL);
  glPolygonMode(GL_BACK, GL_FILL);
  m_program->release();
  glBindVertexArray(0);
}
