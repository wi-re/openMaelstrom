#pragma once
#include <render/qGLWidget/base_renderer.h>
#include <render/qGLWidget/oglwidget.h>

/** This renderer displays all particles in the simulation color coded according
 * to the visualization module and a color map. **/
class ColorMapRenderer : public Renderer {
public:
  bool m_valid = true;
  ColorMapRenderer(OGLWidget *parent);
  virtual void update() override;
  virtual void render(bool pretty) override;
  virtual bool valid() override;

  GLuint defer_VAO;
  QOpenGLShaderProgram *quad_programID; 

  GLuint m_posAttr;
  GLuint m_colAttr;
  GLuint vao;
  GLuint IBO;
  GLuint VXO;
  GLuint VUV;
};
