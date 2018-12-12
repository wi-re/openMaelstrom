#pragma once
#include <render/qGLWidget/base_renderer.h>
#ifndef NO_QT
#include <render/qGLWidget/oglwidget.h>
/** This renderer displays a simple AABB around the simulations initial domain
 * (based on the domain object and not on the current simulaltion size). If
 * parts of the simulation are open the AABB will still be drawn fully. **/
class QuadRender : public Renderer {
public:
  QuadRender(OGLWidget *parent);
  virtual void update() override;
  virtual void render() override;
  virtual bool valid() override;
  
GLuint defer_VAO;         ///< VAO used for the final output
GLuint quad_vertexbuffer; ///< VBO containing the vertex information for a
                          ///< screen quad
                          GLint m_posAttr;
QOpenGLShaderProgram* quad_programID;    ///< Program to render the final output

GLuint renderedTextureOut; ///< GL Handle to the texture CUDA is rendering into
cudaGraphicsResource_t
    renderedResourceOut; ///< CUDA Handle to the texture CUDA is rendering into
};
#endif

void cudaRender(SceneInformation scene, cudaGraphicsResource_t resource);