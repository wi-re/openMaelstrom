#pragma once
#include <utility/math.h> 
#include <render/qGLWidget/base_renderer.h>
#include <utility/include_all.h>     
#include <render/mlmRender/voxelBVH.h>
#include <render/mlmRender/fluidSystem.h>

#ifndef NO_QT 
#include <render/qGLWidget/oglwidget.h>  

class MLMRender : public RTXRender {
protected:  
  virtual void updateRTX() override; 
  virtual void renderRTX(bool pretty, int32_t fn, int32_t s) override;
public:
	MLMRender(OGLWidget *parent);

  virtual void keyPressEvent(QKeyEvent *event) override;
  virtual std::string getInformation() override;

  int32_t renderMode = 0;
  int32_t renderModes = 5;

  FluidSystem fsys;
  BVHManager bvhManager;
};
#endif

void cudaMLMRender(SceneInformation scene, cudaGraphicsResource_t resource, FluidMemory fmem, FluidSystem fsys, float3 *accumulateBuffer, unsigned framenumber, unsigned hashedframes, int32_t renderMode);