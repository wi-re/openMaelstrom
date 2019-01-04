#define BOOST_USE_WINDOWS_H
#include <QApplication>
#include <QDebug>
#include <QOpenGLTimerQuery>
#include <iostream>
#include <render/qGLWidget/oglwidget.h>
#include <utility/helpers/arguments.h>
#include <utility/identifier/uniform.h>
#include <render/qGLWidget/base_renderer.h>
#include <render/boundaryRender/bounds.h>
#include <render/particleRender/particle_render.h>
#include <render/volumeRender/volume_render.h>
#include <render/vdbPolyRender/vdb_poly_render.h>
#include <render/quadRender/quadRender.h>
#include <utility/template/tuple_for_each.h>
#include <boost/format.hpp>
#include <boost/type_traits/is_assignable.hpp>
#include <boost/type_traits/is_volatile.hpp>
#include <chrono>
#include <mutex>
#include <simulation/particleSystem.h>
#include <sstream>
#include <utility/helpers/arguments.h>
#include <utility/helpers/timer.h>
#include <utility/identifier/arrays.h>
#include <utility/identifier/uniform.h>
#include <utility/math.h>

OGLWidget::OGLWidget(QWidget *parent) : QGLWidget(parent) {
  connect(&m_timer, SIGNAL(timeout()), this, SLOT(update()));
  m_timer.setInterval(0);
  m_timer.start(0);
  setMouseTracking(true);
  QGLFormat format;
  format.setSwapInterval(0);
  setFormat(format);
#ifdef WIN32
  if (get<parameters::gl_record>()) {
    std::stringstream sstream;
    sstream << "ffmpeg -r " << get<parameters::camera_fps>()
            << (!arguments::cmd::instance().vm.count("verbose")
                    ? " -hide_banner -nostats -loglevel 0"
                    : "")
            << " -f rawvideo -pix_fmt rgba -s 1920x1080 -i - "
               "-threads 0 -preset ultrafast -y -vf vflip -c:v libx264 "
               "-pix_fmt yuv420p -b:v 50M "
            << get<parameters::config_folder>() << get<parameters::gl_file>();
    m_ffmpegPipe = _popen(sstream.str().c_str(), "wb");
  }
#endif
}
OGLWidget::~OGLWidget() {
  for (auto& [e,arr] : m_arrayMappings){
    delete arr;
    arr = nullptr;
  }
#ifdef WIN32
  if (m_ffmpegPipe != nullptr)
    _pclose(m_ffmpegPipe);
#endif
}
void OGLWidget::bind(QOpenGLShaderProgram *program) {
  for (auto arr : m_arrayMappings)
    arr.second->bind(program);
  for (auto arr : m_parameterMappings)
    arr.second->add_uniform(program);
  for (auto arr : m_uniformMappings)
    arr.second->add_uniform(program);
}
void OGLWidget::initializeGL() {
  initializeOpenGLFunctions();
  if (get<parameters::gl_record>())
    setFixedSize(get<parameters::camera_resolution>().x,
                 get<parameters::camera_resolution>().y);
  else
    setMinimumSize(1280, 720);
  // Create a mapping for every numeric buffer, this requires no additional
  // memory or overhead as cuda can directly bind to openGL.
  for_each(arrays_list, [&](auto x) {
    using T = decltype(x);
    if constexpr (math::dimension<typename T::type>::value != 0xDEADBEEF)
      m_arrayMappings[T::identifier] = new cuda_buffer<T>();
  });
  // Create a mapping for every numeric parameter of the simulation
  for_each(uniforms_list, [&](auto x) {
    using T = decltype(x);
    if constexpr (math::dimension<typename T::type>::value != 0xDEADBEEF)
      m_parameterMappings[T::identifier] = new gl_uniform<T>();
  });
  // Create two custom uniforms for the view and perspective matrix of the
  // default camera.
  m_uniformMappings["view_matrix"] = new gl_uniform_custom<QMatrix4x4>(
      &Camera::instance().matrices.view, "view_matrix");
  m_uniformMappings["perspective_matrix"] = new gl_uniform_custom<QMatrix4x4>(
      &Camera::instance().matrices.perspective, "perspective_matrix");
  // Setup the global camera instance according to some basic values that are a
  // good starting point.
  Camera::instance().type = Camera::CameraType::firstperson;
  Camera::instance().movementSpeed = 25.0f;
  auto p = get<parameters::camera_position>();
  auto r = get<parameters::camera_angle>();
  Camera::instance().position = QVector3D(p.x, p.y, p.z);
  Camera::instance().setRotation(QVector3D(r.x, r.y, r.z));
  Camera::instance().setPerspective(60.0f, (float)1920 / (float)1080, 0.1f, 64000.0f);
  Camera::instance().updateViewMatrix();

  auto& hScene = hostScene();
  hScene.width = get<parameters::camera_resolution>().x;
  hScene.height = get<parameters::camera_resolution>().y;
  auto[b,cam] = Camera::instance().prepareDeviceCamera();
  hScene.m_camera = cam;

  // Setup the basic render functions that currently exist
  //constexpr auto rayTracing = true;
  //if (!rayTracing) {
	   if (get<parameters::volumeBoundary>() == true)
	     m_renderFunctions.push_back(new volumeBoundaryRender(this));
	   m_renderFunctions.push_back(new ParticleRenderer(this));
	   m_renderFunctions.push_back(new BoundsRenderer(this));
	   for(auto fluidVolume : get<parameters::boundary_volumes>()){
	     auto render = new volumeRender(this, fluidVolume.fileName.value);
	     render->toggle();
	     m_renderFunctions.push_back(render);
	   }
	   for (auto fluidVolume : get<parameters::particle_volumes>()) {
	     m_volumeRenderFunctions.push_back(new volumeRender(this, fluidVolume.fileName.value));
	   }
	   for (auto &fluidVolume : get<parameters::inlet_volumes>()) {
	     m_volumeRenderFunctions.push_back(new volumeRender(this, fluidVolume.fileName.value));
	   }
	   if(get<parameters::volumeOutlets>() == true)
	   for (auto &fluidVolume : get<parameters::outlet_volumes>()) {
	     m_volumeRenderFunctions.push_back(new volumeRender(this, fluidVolume.fileName.value));
	   }
  //}
  //else {
	   rayTracer = new QuadRender(this);
  //}
}

SceneInformation& hostScene(){ 
  static SceneInformation m_instance;
  return m_instance;
}

void OGLWidget::renderFunctions() {
  std::lock_guard<std::mutex> guard(
      cuda_particleSystem::instance().simulation_lock);
      auto [dirty, cam] = Camera::instance().prepareDeviceCamera();
      hostScene().m_camera = cam;
      hostScene().dirty = dirty;
      Camera::instance().dirty = false;

  // Setup the correct basic mode for openGL. This has to be done every frame as
  // Qt will change the openGL state and not restore it to the previous state
  // when rendering some UI elements.
  {
	  glPushAttrib(GL_ALL_ATTRIB_BITS);
	  glMatrixMode(GL_PROJECTION);
	  glPushMatrix();
	  glLoadIdentity();
	  glEnable(GL_MULTISAMPLE);
	  glClearColor(0.2f, 0.2f, 0.2f, 1.f);
	  glEnable(GL_DEPTH_TEST);
	  glEnable(GL_LIGHT0);
	  glEnable(GL_LIGHTING);
	  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	  glEnable(GL_COLOR_MATERIAL);
	  glFlush();
	  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  }
  auto tStart = std::chrono::high_resolution_clock::now();
  // Update all bindings
  for (auto arr : m_arrayMappings)
    arr.second->update();
  for (auto uni : m_parameterMappings)
    uni.second->update();
  for (auto uni : m_uniformMappings)
    uni.second->update();
  // Start the actual rendering process
  if (!rayTracing) {
	  for (auto r : m_renderFunctions)
		  if (r->valid())
			  r->render();
	  for (auto r : m_volumeRenderFunctions)
		  if (r->valid())
			  r->render();
  }
  else
	  if(rayTracer != nullptr)
		rayTracer->render();
  glFlush();
  glFinish();
  // Wait for the frame to finish rendering
  auto tEnd = std::chrono::high_resolution_clock::now();
  auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
  m_frameTimer = (float)tDiff / 1000.0f;
  Camera::instance().update(m_frameTimer);
  // Revert back to a clean openGL state to avoid issues with Qt overdrawing data.
  {
	  glBindVertexArray(0);
	  glMatrixMode(GL_MODELVIEW);
	  glPopMatrix();
	  glMatrixMode(GL_PROJECTION);
	  glPopMatrix();
	  glPopAttrib();
  }
  ++m_frame;
  // Check if the simulation should be quit as it has reached it's goal
  auto &cmd_line = arguments::cmd::instance();
  if (cmd_line.end_simulation_frame &&
      get<parameters::frame>() >= cmd_line.timesteps)
    QApplication::quit();
  if (cmd_line.end_simulation_time &&
      get<parameters::simulationTime>() >= cmd_line.time_limit)
    QApplication::quit();
}
void OGLWidget::drawText(QPainter &p) {
  p.setFont(QFont("Courier New", 8, QFont::Bold));
  QPen penHText(QColor("#00e0fc"));
  p.setPen(penHText);
  std::stringstream sstream;
  float time = m_frameTimer * 1000.f;
  float sim_time = 0.f;

  static Timer *frame_timer = nullptr;
  static bool once = true;
  if (once || frame_timer == nullptr) {
    auto& timers = TimerManager::getTimers();
    for (auto t : timers) {
      if (t->getDecriptor() == "Frame")
        frame_timer = t;
    }
    once = false;
  }
  if (m_showText) {
    if (frame_timer != nullptr && frame_timer->getSamples().size() != 0)
      sim_time = frame_timer->getSamples().at(frame_timer->getSamples().size() - 1).second;

    sstream << "frame time: " << boost::format("%f") % time << std::endl;
	sstream << "context   : " << (parameters::target{} == launch_config::device ? "device" : "host") << std::endl;
    sstream << "sim time  : " << boost::format("%f") % sim_time << std::endl;
    sstream << "frame     : " << boost::format("%d") % get<parameters::frame>() << std::endl;
    sstream << "time      : " << boost::format("%f") % get<parameters::simulationTime>() << std::endl;
    sstream << "timestep  : " << boost::format("%f") % get<parameters::timestep>() << std::endl;
    sstream << "particles : " << boost::format("%d") % get<parameters::num_ptcls>() << std::endl;
    sstream << "IISPH     : " << boost::format("%d") % get<parameters::iterations>() << std::endl;
    sstream << "color arr : " << boost::format("%s") % get<parameters::render_buffer>() << std::endl;
    sstream << "color min : " << boost::format("%f") % get<parameters::render_min>() << std::endl;
    sstream << "color max : " << boost::format("%f") % get<parameters::render_max>() << std::endl;
    if (get<parameters::adaptive>() == true) {
    sstream << "split     : " << boost::format("%d") % get<parameters::split_ptcls>() << std::endl;
    sstream << "merged    : " << boost::format("%d") % get<parameters::merged_ptcls>() << std::endl;
    sstream << "shared    : " << boost::format("%d") % get<parameters::shared_ptcls>() << std::endl;
    sstream << "blending  : " << boost::format("%d") % get<parameters::blend_ptcls>() << std::endl;
    sstream << "achieved  : " << boost::format("%f") % get<parameters::adaptiveRatio>() << std::endl;
    }
    if (get<parameters::surfaceDistance>() == true) {
    sstream << "surface : " << boost::format("%d") % get<parameters::surface_iterations>() << std::endl;
    }
    if (get<parameters::neighborhood>() == "constrained") {
    sstream << "support it: " << boost::format("%d") % get<parameters::support_current_iteration>() << std::endl;
    sstream << "changed  h: " << boost::format("%d") % get<parameters::adjusted_particles>() << std::endl;
    }
    if (get<parameters::sorting>() == "hashed_cell" || get<parameters::sorting>() == "MLM") {
    sstream << "cells     : " << boost::format("%d") % get<parameters::valid_cells>() << std::endl;
    sstream << "collisions: " << boost::format("%d") % get<parameters::collision_cells>() << std::endl;
    sstream << "domain    : " << boost::format("[%d %d %d]") % get<parameters::grid_size>().x % get<parameters::grid_size>().y % get<parameters::grid_size>().z << std::endl;
    }
  }
  p.drawText(QRect(0, 0, width(), height()), Qt::AlignLeft | Qt::AlignTop,
             QString::fromStdString(sstream.str()));
}
void OGLWidget::paintEvent(QPaintEvent *) {
  static bool once2 = true;
  if (once2) {
    if (arguments::cmd::instance().end_simulation_frame ||
        arguments::cmd::instance().end_simulation_time)
      cuda_particleSystem::instance().running = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    once2 = false;
  }
  QPainter p(this);
  p.beginNativePainting();
  for (auto r : m_renderFunctions)
    if (r->valid())
      r->update();
  for (auto r : m_volumeRenderFunctions)
    if (r->valid())
      r->update();
  if (rayTracer != nullptr)
	  rayTracer->update();

  renderFunctions();

  p.endNativePainting();
  drawText(p);
  p.end();

  static float last_time = 0.f - 1.f / get<parameters::camera_fps>();
  if (get<parameters::gl_record>() &&
      get<parameters::simulationTime>() >
          last_time + 1.f / get<parameters::camera_fps>()) {
    last_time += 1.f / get<parameters::camera_fps>();
    auto frame = grabFrameBuffer();
    auto GL_frame = QGLWidget::convertToGLFormat(frame);
    
#ifdef WIN32
    fwrite(GL_frame.bits(), sizeof(int) * GL_frame.width() * GL_frame.height(),
           1, m_ffmpegPipe);
#endif
  }
}
void OGLWidget::draw() {}
