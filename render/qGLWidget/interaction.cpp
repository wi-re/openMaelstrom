#include <IO/config/config.h>
#include <QApplication>
#include <iostream>
#include <render/qGLWidget/oglwidget.h>
#include <simulation/particleSystem.h>
#include <utility/identifier/uniform.h>
#include <IO/particle/particle.h>
#include <render/colorMapRender/particle_render.h>

void OGLWidget::resizeGL([[maybe_unused]] int w, [[maybe_unused]] int h) {
  Camera::instance().width = w;
  Camera::instance().height = h;
  Camera::instance().updateAspectRatio(static_cast<float>(w) / static_cast<float>(h));
}
void OGLWidget::setKeyboardModifiers([[maybe_unused]] QInputEvent *event) {
  Camera::instance().setKeyboardModifiers(event);
}
void OGLWidget::resizeEvent(QResizeEvent *event) {
  QGLWidget::resizeEvent(event);
  Camera::instance().resizeEvent(event);
}
void OGLWidget::keyPressEvent(QKeyEvent *event) {
	static std::string map = get<parameters::render_colormap>();
	static std::string buf = get<parameters::render_buffer>();
	static bool direction = get<parameters::render_auto>();
	static bool scaling = get<parameters::visualizeDirection>();
	static bool flipped = get<parameters::map_flipped>();
	static float min = get<parameters::render_min>();
	static float max = get<parameters::render_max>();
	static bool vScaling = get<parameters::vectorScaling>();
	static float vScale = get<parameters::vectorScale>();
	static std::string vMode = get<parameters::vectorMode>();
	
  setKeyboardModifiers(event);
  auto set_colormap = [&](float min, float max, bool auto_scale, bool visualizeVectors, bool vectorScaling, float vectorScale, std::string render_buffer, std::string render_colormap, int flipped = 0, std::string vectorMode = std::string("length")) {
	  if (!(event->modifiers() & Qt::AltModifier)) {
		  *parameters::render_min::ptr = min;
		  *parameters::render_max::ptr = max;
		  *parameters::render_buffer::ptr = render_buffer;
		  *parameters::render_auto::ptr = auto_scale; 
		  *parameters::visualizeDirection::ptr = visualizeVectors;
		  *parameters::vectorScale::ptr = vectorScale;
		  *parameters::vectorScaling::ptr = vectorScaling;
		  *parameters::vectorMode::ptr = vectorMode;
	  }
	  *parameters::render_colormap::ptr = render_colormap;
	  *parameters::map_flipped::ptr = flipped;
  };
  switch (event->key()) {
  case Qt::Key_J:{
	  ((ColorMapRenderer*)colorMapRender)->m_valid = !((ColorMapRenderer*)colorMapRender)->m_valid;
  } break;
  case Qt::Key_H: {
	  m_showText = !m_showText;
  } break;
  case Qt::Key_P:
	  if (QApplication::keyboardModifiers().testFlag(Qt::ShiftModifier))
		  cuda_particleSystem::instance().single = true;
	  else
		cuda_particleSystem::instance().running = !cuda_particleSystem::instance().running;
    break;
  case Qt::Key_G: {
	  std::lock_guard<std::mutex> guard(cuda_particleSystem::instance().simulation_lock);
	  auto cfg = parameters::target{};
	  if (cfg == launch_config::device) 
		  *parameters::target::ptr = launch_config::host;
	  else if (cfg == launch_config::host)
		  *parameters::target::ptr = launch_config::device;
	  break;
  }
  case Qt::Key_X: {
    std::lock_guard<std::mutex> guard(cuda_particleSystem::instance().simulation_lock);
    IO::config::take_snapshot();
  } break;
  case Qt::Key_Z: {
    std::lock_guard<std::mutex> guard(cuda_particleSystem::instance().simulation_lock);
    IO::config::load_snapshot();
  } break; 
  case Qt::Key_C: {
    std::lock_guard<std::mutex> guard(cuda_particleSystem::instance().simulation_lock);
    IO::config::clear_snapshot();
  } break;
  case Qt::Key_U: {
	  get<parameters::dumpForSSSPH>() = 1;
  } break;
  case Qt::Key_M: { 
	  int flip = *parameters::map_flipped::ptr;
	  if (flip == 1)	*parameters::map_flipped::ptr = 0;
	  else				*parameters::map_flipped::ptr = 1;
} break;
  case Qt::Key_V: { 
    for(auto& render : m_volumeRenderFunctions)
      render->toggle();
} break;
  case Qt::Key_T: {
	  static float old_min = parameters::render_min{};
	  static float old_max = parameters::render_max{};
	  if (parameters::render_auto{} == 1) {
		  *parameters::render_min::ptr = old_min;
		  *parameters::render_max::ptr = old_max;
		  *parameters::render_auto::ptr = false;
	  }
	  else {
		  old_min = parameters::render_min{};
		  old_max = parameters::render_max{};
		  *parameters::render_auto::ptr = true;
	  }
  }break;
  case Qt::Key_1: { set_colormap(0.f, 1.f, true, false, false, 1.f, "densityBuffer", "inferno",1); }break;
  case Qt::Key_2: { set_colormap(0.f, 1.f, true, false, false, 1.f, "neighborListLength", "jet",1); }break;
  case Qt::Key_3: { set_colormap(0.f, 1.f, true, false, false, 1.f, "MLMResolution", "Dark2",1); }break;
  case Qt::Key_4: { set_colormap(0.f, 30.f, false, true, false, 1.f, "velocity", "Blues",1); }break;
  case Qt::Key_5: { set_colormap(-0.25f, 0.75f, true, false, false, 1.f, "particle_type", "tab20c"); }break;
  case Qt::Key_6: { set_colormap(0.f, 1.f, true, false, false, 1.f, "lifetime", "plasma"); }break;
  case Qt::Key_7: { set_colormap(0.f, 1.f, true, false, false, 1.f, "volume", "gist_heat",1); }break;
  case Qt::Key_8: { set_colormap(0.f, 1.f, true, true, true, 5.f, "debugArray", "magma", 1, "w"); }break;
  case Qt::Key_9: { set_colormap(0.f, 1.f, true, false, false, 1.f, "distanceBuffer", "viridis"); }break;
  case Qt::Key_0: { set_colormap(-2.f, 2.f, false, false, false, 1.f, "adaptive.classification", "RdBu"); }break;
  case Qt::Key_Minus: { set_colormap(min, max, direction, vScaling, vScale, scaling, buf, map, flipped, vMode); }break;
  case Qt::Key_O: { *parameters::dumpNextframe::ptr = 1; } break;
  case Qt::Key_I: { IO::particle::saveParticles(); } break;
  case Qt::Key_F7: { Camera::instance().tracking = !Camera::instance().tracking; } break;
  case Qt::Key_F8: { pickerActive = !pickerActive; } break;
  //case Qt::Key_F8: {mlmTracer->bValid = !mlmTracer->bValid; vrtxTracer->bValid = false; } break;
  case Qt::Key_F9: {vrtxTracer->bValid = !vrtxTracer->bValid; mlmTracer->bValid = false; } break;
  case Qt::Key_F10: {
	  if (event->modifiers() & Qt::NoModifier)
		  get<parameters::visualizeDirection>() = !get<parameters::visualizeDirection>();
	  if (event->modifiers() & Qt::AltModifier)
		  get<parameters::vectorScale>() = -get<parameters::vectorScale>();
	  if (event->modifiers() & Qt::ShiftModifier)
		  get<parameters::vectorScaling>() = -get<parameters::vectorScaling>();
  
  } break;
  }
  for (const auto& renderer : m_renderFunctions)
	  renderer->keyPressEvent(event);
  for (const auto& renderer : m_volumeRenderFunctions)
	  renderer->keyPressEvent(event);
  for (const auto& renderer : m_rayTracingFunctions)
	  renderer->keyPressEvent(event);
  Camera::instance().keyPressEvent(event);
}
void OGLWidget::keyReleaseEvent(QKeyEvent *event) {
  setKeyboardModifiers(event);
  Camera::instance().keyReleaseEvent(event);
}
void OGLWidget::mousePressEvent(QMouseEvent *event) {
  setKeyboardModifiers(event);
  Camera::instance().mousePressEvent(event);
}
void OGLWidget::mouseReleaseEvent(QMouseEvent *event) {
  setKeyboardModifiers(event);
  switch (event->button()) {
  case Qt::LeftButton:
    break;
  case Qt::MidButton:
    break;
  case Qt::RightButton:
    break;
  default:
    break;
  }
  Camera::instance().mouseReleaseEvent(event);
}
void OGLWidget::mouseDoubleClickEvent(QMouseEvent *event) {
  setKeyboardModifiers(event);
  Camera::instance().mouseDoubleClickEvent(event);
}

#include <render/particleIndexRender/particleIndexRender.h>

void OGLWidget::mouseMoveEvent(QMouseEvent *event) { 
	Camera::instance().mouseMoveEvent(event); 

	//bool pickerActive = false;
	//int32_t pickedParticle = -1;
	if (!pickerActive) return;
	QPoint globalPos = (rect().topLeft());
	QPoint globalPosMax = (rect().bottomRight());

	IndexRenderer* idr = (IndexRenderer*)indexRender;
	auto p = event->pos();

	double xpos, ypos;
	xpos = p.x();// -globalPos.x();
	ypos = globalPosMax.y() - p.y();// +globalPos.y();
	pickerX = (int32_t)xpos;
	pickerY = (int32_t)ypos;
	//std::cout << xpos << " : " << ypos << " -> " << idr->pxlData[(int32_t)xpos + (int32_t)ypos * 1920] << std::endl;
}
void OGLWidget::wheelEvent(QWheelEvent *event) { setKeyboardModifiers(event); }
