#pragma once
#include <utility/math.h> 
#include <render/quadRender/bvh.h>
#include <render/quadRender/geometry.h> 
#include <render/quadRender/loader.h> 
#include <render/qGLWidget/base_renderer.h>
#include <utility/include_all.h>      
#define BVH_STACK_SIZE 32    

#ifndef NO_QT
#include "loader.h"    
#include <render/qGLWidget/oglwidget.h>  

class QuadRender : public Renderer {
public:  
  QuadRender(OGLWidget *parent);    
  virtual void update() override; 
  virtual void render() override;
  virtual bool valid() override;  
     
  GLuint defer_VAO;
  QOpenGLShaderProgram *quad_programID; 
   
  GLuint renderedTextureOut;
  cudaGraphicsResource_t renderedResourceOut; 
   
  void prepCUDAscene();
  float3 *accumulatebuffer = nullptr;
   
  objectLoader loader;
  objectLoader fluidLoader = objectLoader(false,false);
  mesh objects;     
};
#endif

void cudaRender(SceneInformation scene, cudaGraphicsResource_t resource, objectLoader &sceneMeshes, objectLoader &fluidMeshes,
                float3 *accumulateBuffer, unsigned framenumber, unsigned hashedframes); 