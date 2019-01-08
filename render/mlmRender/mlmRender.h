#pragma once
#include <utility/math.h> 
#include <render/qGLWidget/base_renderer.h>
#include <utility/include_all.h>     
#include <render/mlmRender/fluidSystem.h>

#ifndef NO_QT 
#include <render/qGLWidget/oglwidget.h>  

class MLMRender : public Renderer {
public:  
	MLMRender(OGLWidget *parent);
  virtual void update() override; 
  virtual void render() override;
  virtual bool valid() override;  
     
  GLuint defer_VAO;
  QOpenGLShaderProgram *quad_programID; 
    
  GLuint renderedTextureOut;
  cudaGraphicsResource_t renderedResourceOut; 
   
  void prepCUDAscene();
  float3 *accumulatebuffer = nullptr;

  FluidSystem fsys;
};
#endif

void cudaMLMRender(SceneInformation scene, cudaGraphicsResource_t resource, FluidMemory fmem, FluidSystem fsys, float3 *accumulateBuffer, unsigned framenumber, unsigned hashedframes);