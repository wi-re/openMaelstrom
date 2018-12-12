#pragma once
#include <render/qGLWidget/base_renderer.h>
#include <render/qGLWidget/oglwidget.h>

/** This renderer is used to display volume boundary objects (loaded as
 * VDBs). **/
class volumeRender : public Renderer {
public:
  volumeRender(OGLWidget *parent, std::string fileName);
  virtual void update() override;
  virtual void render() override;
  virtual bool valid() override;
  virtual void toggle() override;

  cudaGraphicsResource *cuda_vbo_resource;

  QOpenGLShaderProgram *m_program;

  GLuint m_posAttr;
  GLuint m_colAttr;
  GLuint vao;
  GLuint IBO;
  GLuint VXO;
  GLuint VUV;

  int32_t tris = 0;

  bool active = false;
};
