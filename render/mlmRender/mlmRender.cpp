#define QT_NO_KEYWORDS
#include <render/mlmRender/mlmRender.h>
#include <simulation/particleSystem.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <render/util/camera.h>
#include <bitset>
#include <QApplication>

void MLMRender::updateRTX() {}
MLMRender::MLMRender(OGLWidget *parent): RTXRender(parent) {
  cuda_particleSystem::instance().retainArray("auxHashMap");
  cuda_particleSystem::instance().retainArray("auxCellSpan");
  //cuda_particleSystem::instance().retainArray("auxCellInformation");
  //cuda_particleSystem::instance().retainArray("auxCellSurface");

  cuda_particleSystem::instance().retainArray("compactCellSpan");
  cuda_particleSystem::instance().retainArray("compactHashMap");
  cuda_particleSystem::instance().retainArray("MLMResolution");
  //cuda_particleSystem::instance().retainArray("auxDistance");
  renderMode = get<parameters::renderMode>() % renderModes;
  //get<parameters::renderMode>() = renderMode;
}
std::string MLMRender::getInformation() {
	if (!valid()) return "";
	switch (renderMode) {
	case 0: return std::string("Render mode: Ray Scheduler [") + std::to_string(framenumber) + "]\n"; break;
	case 1: return std::string("Render mode: Merged Voxel Tracing [") + std::to_string(framenumber) + "]\n"; break;
	case 2: return std::string("Render mode: Voxel Tracing [") + std::to_string(framenumber) + "]\n"; break;
	case 3: return std::string("Render mode: BVH Tracing [") + std::to_string(framenumber) + "]\n"; break;
	case 4: return std::string("Render mode: Voxel + Bounce [") + std::to_string(framenumber) + "]\n"; break;
	default: return std::string("Render mode unknown [") + std::to_string(framenumber) + "]\n"; break;
	}
}
void MLMRender::keyPressEvent(QKeyEvent *event) {
	switch (event->key()) {
	case Qt::Key_F11: {
		if (QApplication::keyboardModifiers().testFlag(Qt::ShiftModifier))
			renderMode = (renderMode + renderModes - 1) % renderModes;
		else
			renderMode = (renderMode + 1) % renderModes;
		dirty = true;
		break;
	}
	}
}
void MLMRender::renderRTX(bool prettyRender, int32_t framenumber, int32_t seed) {
	fsys = FluidSystem{
	  get<parameters::num_ptcls>(),
	  {
		  get<parameters::min_coord>(),
		  get<parameters::max_coord>() + get<parameters::cell_size>() * get<parameters::auxScale>()
	}
  };
  //std::cout << fsys.bounds[0].x << ", " << fsys.bounds[0].y << ", " << fsys.bounds[0].z << std::endl;
	if (renderMode == 3) {
		auto[bvh, size] = bvhManager.getGPUData();
		fsys.fluidBVH = bvh;
	}

  FluidMemory fmem;
  fmem.grid_size = parameters::grid_size{};
  fmem.cell_size = parameters::cell_size{};
  fmem.min_coord = parameters::min_coord{};

  fmem.hash_entries = parameters::hash_entries{};
  fmem.mlm_schemes = parameters::mlm_schemes{};

  fmem.num_ptcls = parameters::num_ptcls{};
  fmem.max_numptcls = parameters::max_numptcls{};
  fmem.timestep = parameters::timestep{};
  fmem.radius = parameters::radius{};
  fmem.rest_density = parameters::rest_density{};

  fmem.compactHashMap = arrays::compactHashMap::ptr;
  fmem.compactCellSpan = arrays::compactCellSpan::ptr;
  fmem.cellSpan = arrays::auxCellSpan::ptr;
  fmem.hashMap = arrays::auxHashMap::ptr;
  fmem.MLMResolution = arrays::MLMResolution::ptr;
  fmem.position = arrays::position::ptr;
  fmem.volume = arrays::volume::ptr;

  fmem.auxScale = parameters::auxScale{};
  fmem.cell_size *= fmem.auxScale;
  fmem.grid_size /= static_cast<int32_t>(fmem.auxScale);
  fmem.grid_size += 1;
  fmem.auxDistance = arrays::auxDistance::ptr;

  auto h_scene = hostScene();
  cudaMLMRender(h_scene, renderedResourceOut, fmem, fsys, accumulatebuffer, framenumber, seed, renderMode);  
}
