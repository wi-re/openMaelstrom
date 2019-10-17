#include <render/particleIndexRender/particleIndexRender.h>
#include <render/util/color_map.h>

static const char *vertexShaderSource = R"(#version 450 compatibility
in vec4 posAttr;
in vec2 uvAttr;
in vec4 position;
in vec4 renderArray;
in float volume;

uniform vec4 camera_right;
uniform vec4 camera_up;
uniform mat4 perspective_matrix;
uniform mat4 view_matrix;

uniform vec3 minCoord;
uniform vec3 maxCoord;
uniform vec3 render_clamp;

uniform int transfer_fn;
uniform int mapping_fn;
uniform float color_map_min;
uniform float color_map_max;

uniform int map_flipped;

uniform sampler1D colorRamp;

out vec2 uv;
out vec4 color;
out vec4 eyeSpacePos;
flat out int invalid;
flat out int instanceID;

float mapValue(float value, float min, float max, int fn){
	if(fn == 0)
		return (value - min) / (max - min);
	if(fn == 1)
		return (sqrt(value) - sqrt(min)) / (sqrt(max) - sqrt(min));
	if(fn == 2)
		return (value * value - min * min) / (max * max - min * min);
	if(fn == 3)
		return (pow(value,1.f/3.f) - pow(min,1.f/3.f)) / (pow(max,1.f/3.f) - pow(min,1.f/3.f));
	if(fn == 4)
		return (value * value * value - min * min * min) / (max * max * max - min * min * min);
	if(fn == 5)
		return (log(value) - log(min)) / (log(max) - log(min));
	return (value - min) / (max - min);
}

void main() {
	instanceID = gl_InstanceID;
	uv = uvAttr;
	float intensity = 0.f;
	float renderIntensity = renderArray.w;
	if(color_map_min < color_map_max)
		intensity = mapValue(renderIntensity, color_map_min, color_map_max, transfer_fn);
	else
		intensity = mapValue(renderIntensity, color_map_max, color_map_min, transfer_fn);
	//intensity = (float(transfer_fn)) / 5.f;
	intensity = clamp(intensity,0.f,1.f);

	if(map_flipped != 0)
		intensity = 1.f - intensity;

	intensity = mapValue(intensity, 0.f, 1.f, mapping_fn);
	color = texture(colorRamp,intensity);

	float V = pow(volume / (4.f / 3.f * 3.14159265359f),1.f/3.f);
	eyeSpacePos = view_matrix * vec4(position.xyz,1.f);
	eyeSpacePos += vec4(posAttr.xyz * V * 2.f,0.f);
	gl_Position = perspective_matrix * eyeSpacePos;

 	vec3 flowDist = maxCoord-minCoord;
 	vec3 clipVecMin = minCoord + abs (render_clamp) * flowDist;
 	vec3 clipVecMax = maxCoord - abs (render_clamp) * flowDist;
 	if(
    ((render_clamp.x < 0.0f) ? position.x < clipVecMin.x : position.x > clipVecMax.x) ||
    ((render_clamp.y < 0.0f) ? position.y < clipVecMin.y : position.y > clipVecMax.y) ||
    ((render_clamp.z < 0.0f) ? position.z < clipVecMin.z : position.z > clipVecMax.z) )
			invalid = 1;
	else
			invalid = 0;
})";

static const char *fragmentShaderSource = R"(#version 450 compatibility
in vec2 uv;
in vec4 eyeSpacePos;
flat in int invalid;
flat in int instanceID;
//out uint myIntOutput;
uniform mat4 perspective_matrix;
uniform mat4 view_matrix;

layout(location = 0) out ivec4 frag_color_uint;

void main() {
	//frag_color_uint.r = 2;
	if(invalid == 1)
		discard;
	vec3 N;
	N.xy = uv.xy * vec2(2.f, 2.f) - vec2(1.f, 1.f);
	float r2 = dot(N.xy,N.xy);
	if( r2 > 1.f) discard;
	frag_color_uint.r = instanceID;
	//frag_color_uint.r = 127;
	//myIntOutput = 100000;
	//myIntOutput = 1;
	N.z = sqrt(1.f - r2);

vec4 sphereEyeSpacePos;
sphereEyeSpacePos.xyz = eyeSpacePos.xyz + N * eyeSpacePos.w;
sphereEyeSpacePos.w = 1.0;
vec4 projPos = (perspective_matrix * sphereEyeSpacePos);
gl_FragDepth = (projPos.z / projPos.w)* 0.5 + 0.5;
})";

bool IndexRenderer::valid() { return true; }

#include <experimental/filesystem>
#include <utility/helpers/pathfinder.h>
void IndexRenderer::update() {}

IndexRenderer::IndexRenderer(OGLWidget *parent) {
  initializeOpenGLFunctions();
  m_program = new QOpenGLShaderProgram(parent);
  m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  LOG_INFO << "Compiling vertex shader for " << "ParticleRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
  LOG_INFO << "Compiling fragment shader for " << "ParticleRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glBindFragDataLocation(m_program->programId(), 0, "frag_color_uint");
  m_program->link();
  LOG_INFO << "Linking " << "ParticleRenderer" << std::endl;
  LOG_INFO << m_program->log().toStdString() << std::endl;
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

  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, 1920, 1080, 0, GL_RED_INTEGER, GL_INT, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  GLuint depthTextureId;
  glGenTextures(1, &depthTextureId);
  glBindTexture(GL_TEXTURE_2D, depthTextureId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1920, 1080, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureId, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  pxlData = (int32_t*)malloc(1920 * 1080 * sizeof(int32_t));
}

void IndexRenderer::render(bool pretty) {
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  unsigned int drawbuffers[] = {GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT };
  glDrawBuffers(sizeof(drawbuffers) / sizeof(unsigned int), drawbuffers);

  // clear buffers
  int32_t clear = -1;
  glClear(GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH);
  glClearBufferiv(GL_COLOR, 0, &clear);
  

  // render
  glBindVertexArray(vao);
  m_program->bind();
  glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0, get<parameters::num_ptcls>());
  glBindVertexArray(0);
  glUseProgram(0);
  glFlush();
  glReadBuffer(GL_COLOR_ATTACHMENT0);
  glReadPixels(0, 0, 1920, 1080, GL_RED_INTEGER, GL_INT, pxlData);

  //QImage image(1920, 1080, QImage::Format_RGBA8888);   // use the overloaded constructor
  //QRgb value;
  //for (int i = 0; i < 1920; i++){
	 // for (int j = 0; j < 1080; j++){
		//  // set the rgb value based on arrs' value
		//  // set them all the same to make em grayscale?
		//  int32_t pxl = pxlData[i + j * 1920];
		//  value = qRgba((float) pxl, (float)pxl, (float)pxl, 255);
		//  //value = qRgba(256, 256, 256, 256);
		//  //std::cout << image.pixel(i, j) << " : ";
		//  image.setPixel(i, j, value);
		//  //std::cout << image.pixel(i, j) << " | ";
	 // }
  //}
  //image.save("test.png","PNG");

  //std::cout << pxlData[0] << std::endl;

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
