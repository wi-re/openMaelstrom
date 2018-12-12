#include <render/particleRender/particle_render.h>
#include <render/util/color_map.h>

static const char *vertexShaderSource = R"(#version 450 
in vec4 posAttr;
in vec2 uvAttr;
in vec4 position;
in float renderIntensity;
in float volume;

uniform vec4 camera_right;
uniform vec4 camera_up;
uniform mat4 perspective_matrix;
uniform mat4 view_matrix;

uniform vec3 min_coord;
uniform vec3 max_coord;
uniform vec3 render_clamp;

uniform int map_flipped;

uniform sampler1D colorRamp;

out vec2 uv;
out vec4 color;
out vec4 eyeSpacePos;
flat out int invalid;

void main() {
	uv = uvAttr;
	if(map_flipped != 0)
		color = texture(colorRamp,1.f - renderIntensity);
	else
		color = texture(colorRamp,renderIntensity);

	float V = pow(volume / (4.f / 3.f * 3.14159265359f),1.f/3.f);
	eyeSpacePos = view_matrix * vec4(position.xyz,1.f);
	eyeSpacePos += vec4(posAttr.xyz * V * 2.f,0.f);
	gl_Position = perspective_matrix * eyeSpacePos;

 	vec3 flowDist = max_coord-min_coord;
 	vec3 clipVecMin = min_coord + abs (render_clamp) * flowDist;
 	vec3 clipVecMax = max_coord - abs (render_clamp) * flowDist;
 	if(
    ((render_clamp.x < 0.0f) ? position.x < clipVecMin.x : position.x > clipVecMax.x) ||
    ((render_clamp.y < 0.0f) ? position.y < clipVecMin.y : position.y > clipVecMax.y) ||
    ((render_clamp.z < 0.0f) ? position.z < clipVecMin.z : position.z > clipVecMax.z) )
			invalid = 1;
	else
			invalid = 0;
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
	if( r2 > 1.f) discard;
	N.z = sqrt(1.f - r2);

	vec3 lightDir = vec3(0, 0, 1);
	float diffuse = abs(dot(N, lightDir));

	gl_FragColor = vec4(diffuse,diffuse,diffuse,1.f) * color;
	//gl_FragColor = color;
	//gl_FragColor = texture(colorRamp,uv.x);
	//gl_FragColor = vec4(N,1.f) ;

vec4 sphereEyeSpacePos;
sphereEyeSpacePos.xyz = eyeSpacePos.xyz + N * eyeSpacePos.w;
sphereEyeSpacePos.w = 1.0;
vec4 projPos = (perspective_matrix * sphereEyeSpacePos);
gl_FragDepth = (projPos.z / projPos.w)* 0.5 + 0.5;
})";

bool ParticleRenderer::valid() { return true; }

void ParticleRenderer::update() { colorMap::instance().update(); }

ParticleRenderer::ParticleRenderer(OGLWidget *parent) {
  initializeOpenGLFunctions();
  m_program = new QOpenGLShaderProgram(parent);
  m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  m_program->link();
  m_posAttr = m_program->attributeLocation("posAttr");
  m_colAttr = m_program->attributeLocation("uvAttr");

  std::vector<uint32_t> idx = {0, 1, 2, 2, 3, 0};
  glGenBuffers(1, &IBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(uint32_t), idx.data(), GL_STATIC_DRAW);

  std::vector<QVector4D> instance_pos = {{0.f, 1.f, 0.f, 1.f},
                                         {1.f, 1.f, 0.f, 1.f},
                                         {0.f, -1.f, 0.f, 1.f},
                                         {-1.f, -1.f, 0.f, 1.f},
                                         {0.f, 0.f, 0.f, 1.f}};

  parent->bind(m_program);

  GLfloat vertices[] = {-0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.5f, 0.5f, 0.0f, -0.5f, 0.5f, 0.0f};

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

void ParticleRenderer::render() {
  glBindVertexArray(vao);

  m_program->bind();

  glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0,
                          get<parameters::num_ptcls>());

  m_program->release();
  glBindVertexArray(0);
}
