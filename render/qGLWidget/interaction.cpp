#include <IO/config/config.h>
#include <QApplication>
#include <iostream>
#include <render/qGLWidget/oglwidget.h>
#include <simulation/particleSystem.h>
#include <utility/identifier/uniform.h>

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
	static bool scaling = get<parameters::render_auto>();
	static bool flipped = get<parameters::map_flipped>();
	static float min = get<parameters::render_min>();
	static float max = get<parameters::render_max>();
	
  setKeyboardModifiers(event);
  auto set_colormap = [&](float min, float max, bool auto_scale, std::string render_buffer, std::string render_colormap, int flipped = 0) {
	  if (!(event->modifiers() & Qt::AltModifier)) {
		  *parameters::render_min::ptr = min;
		  *parameters::render_max::ptr = max;
		  *parameters::render_buffer::ptr = render_buffer;
		  *parameters::render_auto::ptr = auto_scale; 
	  }
	  *parameters::render_colormap::ptr = render_colormap;
	  *parameters::map_flipped::ptr = flipped;
  };
  switch (event->key()) {
  case Qt::Key_H: {
	  m_showText = !m_showText;
  } break;
  case Qt::Key_P:
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
	  if (parameters::render_auto{} == true) {
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
  case Qt::Key_1: { set_colormap(0.f, 1.f, false, "densityBuffer", "jet"); }break;
  case Qt::Key_2: { set_colormap(0.f, 1.f, true, "neighborListLength", "inferno"); }break;
  case Qt::Key_3: { set_colormap(0.f, 1.f, true, "MLMResolution", "viridi"); }break;
  case Qt::Key_4: { set_colormap(0.f, 30.f, false, "velocity", "Blues",1); }break;
  case Qt::Key_5: { set_colormap(-0.25f, 0.75f, false, "lifetime", "plasma"); }break;
  case Qt::Key_6: { set_colormap(0.f, 1.f, true, "lifetime", "plasma"); }break;
  case Qt::Key_7: { set_colormap(0.f, 1.f, true, "volume", "jet"); }break;
  case Qt::Key_8: { set_colormap(0.f, 1.f, true, "supportEstimate", "jet"); }break;
  case Qt::Key_9: { set_colormap(0.f, 1.f, true, "distanceBuffer", "plasma"); }break;
  case Qt::Key_0: { set_colormap(-2.f, 2.f, false, "adaptiveClassification", "RdBu"); }break;
  case Qt::Key_Minus: { set_colormap(min, max, scaling, buf, map, flipped); }break;
  case Qt::Key_O: { *parameters::dumpNextframe::ptr = 1; } break;
  }
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
void OGLWidget::mouseMoveEvent(QMouseEvent *event) { Camera::instance().mouseMoveEvent(event); }
void OGLWidget::wheelEvent(QWheelEvent *event) { setKeyboardModifiers(event); }
