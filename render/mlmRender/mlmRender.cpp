#define QT_NO_KEYWORDS
#include <render/mlmRender/mlmRender.h>
#include <simulation/particleSystem.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <render/util/camera.h>

static const char *vertexShaderSource = R"(#version 450
in vec3 vertexPosition_modelspace;
out vec2 UV;

void main(){
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vertexPosition_modelspace.xy+vec2(1,1))/2.0;
	UV.y = 1.f - UV.y;
}
)";

static const char *fragmentShaderSource = R"(#version 450 
uniform sampler2D renderedTexture;

in vec2 UV;
out vec3 color;

void main(){
	vec4 col = texture( renderedTexture, UV);
	color = vec3(col.xyz) ;
	//gl_FragDepth = col.w;
}
)";

bool MLMRender::valid() { return true; }

void MLMRender::update() {}

MLMRender::MLMRender(OGLWidget *parent) {
  auto h_scene = hostScene();
  cudaMalloc(&accumulatebuffer, h_scene.width * h_scene.height * sizeof(float3));

  initializeOpenGLFunctions();
  quad_programID = new QOpenGLShaderProgram(parent);
  quad_programID->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  quad_programID->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);

  glGenVertexArrays(1, &defer_VAO);
  glBindVertexArray(defer_VAO);
  quad_programID->link();
  parent->bind(quad_programID);
  quad_programID->bind();

  auto m_posAttr = quad_programID->attributeLocation("vertexPosition_modelspace");
  glGenTextures(1, &renderedTextureOut);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, renderedTextureOut);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, h_scene.width, h_scene.height, 0, GL_RGBA, GL_FLOAT, 0);
  cudaGraphicsGLRegisterImage(&renderedResourceOut, renderedTextureOut, GL_TEXTURE_2D,
                              cudaGraphicsRegisterFlagsSurfaceLoadStore);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glActiveTexture(GL_TEXTURE0);
  static const GLfloat g_quad_vertex_bufferdata[] = {
      -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
  };

  GLuint quad_vertexbuffer;
  glGenBuffers(1, &quad_vertexbuffer);
  glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_bufferdata), g_quad_vertex_bufferdata, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_posAttr);
  glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  quad_programID->setUniformValue("renderedTexture", 0);

  quad_programID->release();
  prepCUDAscene();
  update();

  cuda_particleSystem::instance().retainArray("cellSpan");
  cuda_particleSystem::instance().retainArray("hashMap");
  cuda_particleSystem::instance().retainArray("MLMResolution");
}

void MLMRender::render() {
  fsys = FluidSystem{ 
	get<parameters::num_ptcls>(), 
	{
		get<parameters::min_domain>(),
		get<parameters::max_domain>() 
	}
  };

  static std::random_device r;
  static std::default_random_engine e1(r());
  static std::uniform_int_distribution<int32_t> uniform_dist(INT_MIN, INT_MAX);
  static int framenumber = 0;
  auto h_scene = hostScene();
  if (h_scene.dirty) {
    cudaMemset(accumulatebuffer, 1, h_scene.width * h_scene.height * sizeof(float3));
    framenumber = 0;
  }
  framenumber++;
  
  cudaMLMRender(h_scene, renderedResourceOut, fsys, accumulatebuffer, framenumber, uniform_dist(e1));

  glBindVertexArray(defer_VAO);
  quad_programID->bind();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, renderedTextureOut);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
  quad_programID->release();
  glBindVertexArray(0);
}

void MLMRender::prepCUDAscene() {
  LOG_INFO << "Rendering data initialised and copied to CUDA global memory\n" << std::endl;
}