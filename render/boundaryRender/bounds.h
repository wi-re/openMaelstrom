#pragma once
#include <render/qGLWidget/base_renderer.h>
#include <render/qGLWidget/oglwidget.h>

/** This renderer displays a simple AABB around the simulations initial domain
 * (based on the domain object and not on the current simulaltion size). If
 * parts of the simulation are open the AABB will still be drawn fully. **/
class BoundsRenderer : public Renderer {
public:
  BoundsRenderer(OGLWidget *parent);
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
