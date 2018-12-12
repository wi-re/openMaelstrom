#include <render/quadRender/quadRender.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>

#include <sstream>
#include <texture_types.h>

static const char *vertexShaderSource = R"(#version 450 
// Input vertex data, different for all executions of this shader.
in vec3 vertexPosition_modelspace;

// Output data ; will be interpolated for each fragment.
out vec2 UV;

void main(){
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vertexPosition_modelspace.xy+vec2(1,1))/2.0;
}
)";

static const char *fragmentShaderSource = R"(#version 450 
in vec2 UV;

uniform sampler2D renderedTexture;
uniform sampler2D renderedDepthTexture;
uniform float time;

out vec3 color;

void main(){
	vec4 col = texture( renderedTexture, UV);
	color = vec3(col.xyz) ;
  //discard;
  //color = vec3(0,255,0);
//	if(col.w == 1.f)
//		discard;
	//gl_FragDepth = col.w;
}
)";

bool QuadRender::valid() { return true; }

void QuadRender::update() {}

QuadRender::QuadRender(OGLWidget *parent) {
  initializeOpenGLFunctions();
  quad_programID = new QOpenGLShaderProgram(parent);
  quad_programID->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
  std::cout << quad_programID->log().toStdString() << std::endl;
  quad_programID->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
  std::cout << quad_programID->log().toStdString() << std::endl;

  auto h_scene = hostScene();
  glGenVertexArrays(1, &defer_VAO);
  glBindVertexArray(defer_VAO);
  quad_programID->link();
  parent->bind(quad_programID);
  quad_programID->bind();

  m_posAttr = quad_programID->attributeLocation("vertexPosition_modelspace");
  std::cout << quad_programID->log().toStdString() << std::endl;
  std::cout << m_posAttr << std::endl;
  std::cout << h_scene.width << " : " << h_scene.height << std::endl;
  glGenTextures(1, &renderedTextureOut);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, renderedTextureOut);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, h_scene.width, h_scene.height, 0, GL_RGBA, GL_FLOAT, 0);
  cudaGraphicsGLRegisterImage(&renderedResourceOut, renderedTextureOut, GL_TEXTURE_2D,
                              cudaGraphicsRegisterFlagsSurfaceLoadStore);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glActiveTexture(GL_TEXTURE0);
  static const GLfloat g_quad_vertex_buffer_data[] = {
      -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
  };

  glGenBuffers(1, &quad_vertexbuffer);
  glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_posAttr);
  glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  std::cout << quad_programID->uniformLocation("renderedTexture") << std::endl;
  quad_programID->setUniformValue("renderedTexture", 0);

  quad_programID->release();
  // glClearColor(.0f, .0f, .0f, 0.0f);
  update();
}

void QuadRender::render() {
  static int framenumber = 0;
  framenumber++;

  static cudaEvent_t start, stop;
  static bool once = true;
  if (once) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    once = false;
  }

  cudaDeviceSynchronize();
  cudaEventRecord(start);
  cudaEventSynchronize(start);
  cudaEventRecord(start);

  auto h_scene = hostScene();

  cudaRender(h_scene, renderedResourceOut);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::stringstream title;
  title << "Light Tracer render time: " << milliseconds << "ms, frame: " << framenumber
        << ", fps: " << 1000.f / (milliseconds);
  // std::cout << title.str() << std::endl;
  // glutSetWindowTitle(title.str().c_str());

  glBindVertexArray(defer_VAO);
  // glViewport(0, 0, h_scene.width, h_scene.height);
  // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  quad_programID->bind();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, renderedTextureOut);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
  quad_programID->release();
  glBindVertexArray(0);
}
