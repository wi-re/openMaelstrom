#pragma once
#include <render/qGLWidget/base_renderer.h>
#include <render/qGLWidget/oglwidget.h>

/** This renderer is used to display volume boundary objects (loaded as
 * VDBs). **/
class volumeBoundaryRender : public Renderer {
public:
  volumeBoundaryRender(OGLWidget *parent);
  virtual void update() override;
  virtual void render(bool pretty) override;
  virtual bool valid() override;

  cudaGraphicsResource *cuda_vbo_resource;

  QOpenGLShaderProgram *m_program;

  GLuint m_posAttr;
  GLuint m_colAttr;
  GLuint vao;
  GLuint IBO;
  GLuint VXO;
  GLuint VUV;
};
