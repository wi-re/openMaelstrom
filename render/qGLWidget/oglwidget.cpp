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
#include <render/floorRender/floor.h>
#include <render/axisRender/axes.h>
#include <render/particleDirectionRender/particleDirectionRender.h>
#include <render/particleRender/particle_render.h>
#include <render/particleIndexRender/particleIndexRender.h>
#include <render/colorMapRender/particle_render.h>
#include <render/rigidRender/rigid_render.h>
#include <render/volumeRender/volume_render.h>
#include <render/vrtxRender/vrtxRender.h>
#include <render/vdbPolyRender/vdb_poly_render.h>
#include <render/quadRender/quadRender.h>
#include <render/mlmRender/mlmRender.h>
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
#include <boost/exception/diagnostic_information.hpp> 

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
template<typename T>
struct mapUniform {
	void operator()(std::map<uniforms, gl_uniform_base *>& map) {
		if constexpr (math::dimension<typename T::type>::value != 0xDEADBEEF)
			map[T::identifier] = new gl_uniform<T>();
	}
};

void OGLWidget::initializeGL() {
  initializeOpenGLFunctions();
  if (get<parameters::gl_record>() || arguments::cmd::instance().renderToFile)
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
  callOnTypes<mapUniform>(uniforms_list, m_parameterMappings);
  //for_each(uniforms_list, [&](auto x) {
  //  using T = decltype(x);
  //  if constexpr (math::dimension<typename T::type>::value != 0xDEADBEEF)
  //    m_parameterMappings[T::identifier] = new gl_uniform<T>();
  //});
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
	   //if (get<parameters::volumeBoundary>() == true)
	   //  m_renderFunctions.push_back(new volumeBoundaryRender(this));

  indexRender = new IndexRenderer(this);
	   //m_renderFunctions.push_back(indexRender);

	   m_renderFunctions.push_back(new ParticleRenderer(this));
	   m_renderFunctions.push_back(new ParticleDirectionRenderer(this));
	   m_renderFunctions.push_back(new BoundsRenderer(this));
	   m_renderFunctions.push_back(new AxesRenderer(this));
	   m_renderFunctions.push_back(new FloorRenderer(this));
	   if (get<parameters::rayTracing>()) {
		   colorMapRender = new ColorMapRenderer(this);
		   ((ColorMapRenderer*)colorMapRender)->m_valid = !arguments::cmd::instance().renderToFile;
		   m_renderFunctions.push_back(colorMapRender);
	   }
	   int32_t i = 0;
	   for(auto fluidVolume : get<parameters::boundaryVolumes>()){
	     auto render = new rigidRender(this, i++);
	     render->toggle();
	     m_renderFunctions.push_back(render);
	   }
	   for (auto fluidVolume : get<parameters::particleVolumes>()) {
	     m_volumeRenderFunctions.push_back(new volumeRender(this, fluidVolume.fileName.value));
	   }
	   for (auto &fluidVolume : get<parameters::inletVolumes>()) {
	     m_volumeRenderFunctions.push_back(new volumeRender(this, fluidVolume.fileName.value));
	   }
	   if(get<parameters::volumeOutlets>() == true)
	   for (auto &fluidVolume : get<parameters::outletVolumes>()) {
	     m_volumeRenderFunctions.push_back(new volumeRender(this, fluidVolume.fileName.value));
	   }
  //}
  //else {
	   //rayTracer = new QuadRender(this);
	   if (get<parameters::rayTracing>()) {
		   //mlmTracer = new QuadRender(this);
		   mlmTracer = new MLMRender(this);
		   vrtxTracer = new vRTXrender(this);
		   mlmTracer->bValid = false;
		   vrtxTracer->bValid = false;
		   m_rayTracingFunctions.push_back(mlmTracer);
		   m_rayTracingFunctions.push_back(vrtxTracer);
		   if (arguments::cmd::instance().rtx)
			   vrtxTracer->bValid = true;
	   }

  //}
}

SceneInformation& hostScene(){ 
  static SceneInformation m_instance;
  return m_instance;
}

void OGLWidget::renderFunctions() {
	static bool mutexOnce = true;
	if (mutexOnce) {
		//std::cout << "MutexOnce" << std::endl;
		cuda_particleSystem::instance().simulation_lock.unlock();

		if (arguments::cmd::instance().end_simulation_frame ||
			arguments::cmd::instance().end_simulation_time)
			cuda_particleSystem::instance().running = true;
		mutexOnce = false;
	}
	//std::cout << ". " << std::endl;
  std::lock_guard<std::mutex> guard(
      cuda_particleSystem::instance().simulation_lock);
  cuda_particleSystem::instance().renderFlag = true;
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
	 // glClearColor(1.f, 1.f, 1.f, 1.f);
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
  if (pickerActive)
	  indexRender->render(false);
	  if (!get<parameters::rayTracing>() || (!vrtxTracer->valid() && !mlmTracer->valid())) {
		  for (auto r : m_renderFunctions)
			  if (r->valid())
				  r->render(false);
		  for (auto r : m_volumeRenderFunctions)
			  if (r->valid())
				  r->render(false);
	  }
	  else {
		  for (auto& rt : m_rayTracingFunctions)
			  rt->render(false);
		  colorMapRender->render(false);
		  //indexRender->render(false);
  }
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
#include <utility/template/for_struct.h>
void OGLWidget::drawText(QPainter &p) {
  p.setFont(QFont("Courier New", 8, QFont::Bold));
 // QPen penHText(QColor("#4542f4"));
  QPen penHTextw(QColor("#ffffff"));
  QPen penHTextb(QColor("#00000"));
  if ((mlmTracer != nullptr && mlmTracer->bValid == true) || (vrtxTracer != nullptr && vrtxTracer->bValid == true)) {
	  p.setPen(penHTextb);
  }
  else{
	  p.setPen(penHTextw);
  }
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

	auto fmtf = [](auto f) {
		std::stringstream sstream;
		sstream << (f < 0.f ? '-' : '+' ) << std::setw(10) << std::setprecision(7) << std::fixed << fabsf(f);
		return sstream.str();
	};
	auto fmte = [](auto f) {
		std::stringstream sstream;
		sstream << (f < 0.f ? '-' : '+') << std::setw(6) << std::setprecision(4) << std::scientific << fabsf(f);
		return sstream.str();
	};

    sstream << "frame time : " << fmtf(time) << std::endl;
	sstream << "context    : " << std::setw(11) << (parameters::target{} == launch_config::device ? "device" : "host") << std::endl;
    sstream << "sim time   : " << fmtf(sim_time) << std::endl;
    sstream << "frame      : " << std::setw(11) << get<parameters::frame>() << std::endl;
    sstream << "time       : " << fmtf(get<parameters::simulationTime>()) << std::endl;
    sstream << "timestep   : " << fmtf(get<parameters::timestep>()) << std::endl;
    sstream << "particles  : " << std::setw(11) << get<parameters::num_ptcls>() << std::endl;
	if (get<parameters::modules::pressure>() == "IISPH17") {
		sstream << "cIterations: " << std::setw(10) << get<parameters::iterations>() << std::endl;
		sstream << "compression: " << fmte(get<parameters::iisph_settings::density_error>()) << std::endl;
	}
	if (get<parameters::modules::pressure>() == "DFSPH"){
		sstream << "dIterations: " << std::setw(11) << get<parameters::dfsph_settings::divergenceSolverIterations>() << std::endl;
		sstream << "divergence : " << fmte(get<parameters::dfsph_settings::divergenceError>()) << std::endl;
		sstream << "cIterations: " << std::setw(11) << get<parameters::dfsph_settings::densitySolverIterations>() << std::endl;
		sstream << "compression: " << fmte(get<parameters::dfsph_settings::densityError>()) << std::endl;
	}
    sstream << "color arr  : " << get<parameters::render_buffer>() << std::endl;
    sstream << "color min  : " << fmte(get<parameters::render_min>()) << std::endl;
    sstream << "color max  : " << fmte(get<parameters::render_max>()) << std::endl;
    if (get<parameters::modules::adaptive>() == true) {
		static std::vector<int32_t> s_split(16), s_merged(16), s_shared(16);
		static float last_t = -1.f;
		static float last_time = -1.f;
		
		if (get<parameters::simulationTime>() > last_t + 1.f / (float)get<parameters::camera_fps>() + get<parameters::timestep>()) {
			//std::cout << get<parameters::simulationTime>() << " : " << last_t + 1.f / (float)get<parameters::camera_fps>() << std::endl;
			last_t = get<parameters::simulationTime>();
			for (int32_t i = 0; i < 16; ++i)
				s_split[i] = s_merged[i] = s_shared[i] = 0;
		}
		if (last_time != get<parameters::simulationTime>()) {
			last_time = get<parameters::simulationTime>();
			auto split = get<parameters::adaptive::splitPtcls>();
			auto merged = get<parameters::adaptive::mergedPtcls>();
			auto shared = get<parameters::adaptive::sharedPtcls>();
			for (int32_t i = 0; i < 16; ++i) {
				s_split[i] += split[i];
				s_merged[i] += merged[i];
				s_shared[i] += shared[i];
			}
		}
		auto splitPtcls = std::accumulate(s_split.begin(), s_split.end(), 0);
		auto mergedPtcls = std::accumulate(s_merged.begin(), s_merged.end(), 0);
		auto sharedPtcls = std::accumulate(s_shared.begin(), s_shared.end(), 0);

		auto vectorStats = [&](auto vec) {
			if (!get<parameters::adaptive::detailedAdaptiveStatistics>()) return;
			sstream << "[ ";
			for (auto e : vec)
				sstream << std::setw(5) << e << " ";
			sstream << "]\n";
		};

    sstream << "split      : " << std::setw(11) << splitPtcls << std::endl;
	vectorStats(s_split);
    sstream << "merged     : " << std::setw(11) << mergedPtcls << std::endl;
	//vectorStats(s_merged);
    sstream << "shared     : " << std::setw(11) << sharedPtcls << std::endl;
	//vectorStats(s_shared);
    sstream << "blending   : " << std::setw(11) << get<parameters::adaptive::blendedPtcls>() << std::endl;
    sstream << "achieved   : " << fmtf(get<parameters::adaptive::ratio>()) << std::endl;
	if(get<parameters::adaptive::detailedAdaptiveStatistics>())
	sstream << "min volume : " << fmtf(get<parameters::adaptive::minVolume>()) << std::endl;
    }
    if (get<parameters::modules::surfaceDistance>() == true) {
    sstream << "surface  : " << std::setw(11) << get<parameters::surface_iterations>() << std::endl;
    }
    if (get<parameters::modules::neighborhood>() == "constrained") {
    sstream << "support it : " << std::setw(11) << get<parameters::support_current_iteration>() << std::endl;
    sstream << "changed  h : " << std::setw(11) << get<parameters::adjusted_particles>() << std::endl;
    }
    if (get<parameters::modules::sorting>() == "hashed_cell" || get<parameters::sorting>() == "MLM") {
    sstream << "cells      : " << std::setw(11) << get<parameters::valid_cells>() << std::endl;
    sstream << "collisions : " << std::setw(11) << get<parameters::collision_cells>() << std::endl;
    sstream << "domain     : " << "[" << get<parameters::grid_size>().x  << " " << get<parameters::grid_size>().y << " " << get<parameters::grid_size>().z << "]" << std::endl;
    }
	if (get<parameters::modules::rayTracing>() == 1) {
		sstream << "aux occup  : " << std::setw(11) << get<parameters::auxCells>() << std::endl;
		sstream << "aux collis : " << std::setw(11) << get<parameters::auxCollisions>() << std::endl;

	}
  }

  for (const auto& renderer : m_renderFunctions)
	  sstream << renderer->getInformation();
  for (const auto& renderer : m_volumeRenderFunctions)
	  sstream << renderer->getInformation();
  for (const auto& r : m_rayTracingFunctions)
	  sstream << r->getInformation();

  p.drawText(QRect(0, 0, width(), height()), Qt::AlignLeft | Qt::AlignTop,
             QString::fromStdString(sstream.str()));

  if (m_showText) {
	  if (pickerActive) {
		  pickedParticle = ((IndexRenderer*)indexRender)->pxlData[(int32_t)pickerX + (int32_t)pickerY * 1920];
		  std::stringstream sstream2;
		  if (get<parameters::debug>()) {
			  sstream2 << "Picked particle: " << pickedParticle << "\n";
			  auto dba = cuda_particleSystem::instance().getDebugArrays();
			  auto pIdx = pickedParticle;
			  for_struct(dba, [&sstream2, pIdx](auto& arr) {
				  using Ta = std::decay_t<decltype(arr)>;
				  using Ty = typename Ta::value_type;
				  if (arr.valid) {
					  sstream2 << Ty::variableName << ": ";
					  if (pIdx == -1) sstream2 << "\n";
					  else sstream2 << arr.ptr[pIdx] << "\n";
				  }
				});
		  }
		  else {
			  sstream2 << "Picked particle: " << pickedParticle << "\n";
		  }
		  auto s = sstream2.str();
		  s = s.substr(0, s.length() - 1);
		  p.drawText(QRect(0, 0, width(), height()), Qt::AlignLeft | Qt::AlignBottom,
			  QString::fromStdString(s));
	  }
  }
  if(get<parameters::rayTracing>())
  if (((ColorMapRenderer*)colorMapRender)->m_valid) {
	  p.setFont(QFont("Courier New", 14, QFont::Bold));
	   
	  char buf[256];
	  sprintf(buf, "%8.6e", get<parameters::render_max>());

	  p.drawText(QRect(width()*0.8f, 0, width()*0.18f, height()), Qt::AlignRight | Qt::AlignTop,
		  QString::fromStdString(std::string(buf)));
	  sprintf(buf, "%8.6e", get<parameters::render_min>());
	  p.drawText(QRect(width()*0.8f, height()*0.88f, width()*0.18f, height()*0.1f), Qt::AlignRight | Qt::AlignBottom,
		  QString::fromStdString(std::string(buf)));
  }


}
void progressBar(int32_t frame, int32_t frameTarget, float progress) {
	std::ios cout_state(nullptr);
	cout_state.copyfmt(std::cout);
	static auto startOverall = std::chrono::high_resolution_clock::now();
	static auto startFrame = startOverall;
	static auto lastTime = startOverall;
	static int32_t lastFrame = frame;
	if (frame != lastFrame) {
		lastFrame = frame;
		startFrame = std::chrono::high_resolution_clock::now();
	}
	auto now = std::chrono::high_resolution_clock::now();
	lastTime = now;
	//float progress = 0.0;
	//while (progress < 1.0) {
		int barWidth = std::min(get<parameters::renderSteps>(),70);
		std::cout << "Rendering " << std::setw(4) << frame;
		if (frameTarget != -1)
			std::cout << "/" << std::setw(4) << frameTarget;
		std::cout << " [";
		int pos = barWidth * progress;
		for (int i = 0; i < barWidth; ++i) {
			if (i < pos) std::cout << "=";
			else if (i == pos) std::cout << ">";
			else std::cout << " ";
		}
		std::cout << "] " << std::setw(3) << int(progress * 100.0) << " ";
		auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - startFrame);
		if (dur.count() < 100 || progress < 1e-3f) {
			std::cout << " ---/---s  ";
		}
		else {
			auto totalTime = ((float)std::chrono::duration_cast<std::chrono::microseconds>(now - startFrame).count()) / 1000.f / 1000.f;
			std::cout << std::fixed << std::setprecision(0) << " " << std::setw(3) << totalTime << "/" << std::setw(3) << (totalTime / progress) << "s  ";
		}
		if (frameTarget != -1 && frame != 0) {
			auto duration = now - startOverall;
			auto progress = ((float)(frame * get<parameters::camera_fps>())) / ((float)(frameTarget * get<parameters::camera_fps>()));
			auto estimated = duration / progress - duration;
			auto printTime = [](auto tp) {
				std::stringstream sstream;
				auto h = std::chrono::duration_cast<std::chrono::hours>(tp).count();
				auto m = std::chrono::duration_cast<std::chrono::minutes>(tp).count() - h * 60;
				auto s = std::chrono::duration_cast<std::chrono::seconds>(tp).count() - h * 3600 - m * 60;
				sstream << std::setw(2) << h << "h " << std::setw(2) << m << "m " << std::setw(2) << s << "s";
				return sstream.str();
			};
			std::cout << " Elapsed: " << printTime(duration) << " ETA: " << printTime(estimated);
			std::cout << "     ";

			//auto totalTime = ((float)std::chrono::duration_cast<std::chrono::microseconds>(now - startOverall).count()) / 1000.f / 1000.f;
			//progress = ((float)frame) / ((float)frameTarget);
			//std::cout << std::fixed << std::setprecision(0) << " " << totalTime << "/" << (totalTime / progress) << "s  ";
		}

		std::cout << "\r";
		std::cout.flush();
		std::cout.copyfmt(cout_state);

		//progress += 0.16; // for demonstration only
	//}
	//std::cout << std::endl;
}
void progressBar2(int32_t frame, int32_t frameTarget) {
	std::ios cout_state(nullptr);
	cout_state.copyfmt(std::cout);
	static auto startOverall = std::chrono::high_resolution_clock::now();
	if (frameTarget == 0) {
		startOverall = std::chrono::high_resolution_clock::now();
		return;
	}
	auto now = std::chrono::high_resolution_clock::now();

	int barWidth = 70;
	std::cout << "Rendering " << std::setw(4) << frame;
	if (frameTarget != -1)
		std::cout << "/" << std::setw(4) << frameTarget;
	std::cout << " [";

	float progress = ((float)frame) / ((float)frameTarget);
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << std::setw(3) << int(progress * 100.0) << "% ";
	if (frame != 0) {
		auto duration = now - startOverall;
		auto estimated = duration / progress - duration;
		auto printTime = [](auto tp) {
			std::stringstream sstream;
			auto h = std::chrono::duration_cast<std::chrono::hours>(tp).count();
			auto m = std::chrono::duration_cast<std::chrono::minutes>(tp).count() - h * 60;
			auto s = std::chrono::duration_cast<std::chrono::seconds>(tp).count() - h * 3600 - m * 60;
			sstream << std::setw(2) << h << "h " << std::setw(2) << m << "m " << std::setw(2) << s << "s";
			return sstream.str();
		};
		std::cout << " Elapsed: " << printTime(duration) << " ETA: " << printTime(estimated);
		std::cout << "     ";
	}
	std::cout << "\r";
	std::cout.flush();
	std::cout.copyfmt(cout_state);
}

void OGLWidget::paintEvent(QPaintEvent *) {
	try {
		//get<parameters::gl_record>() = false;
		static bool once2 = true;
		if (once2) {
			if (arguments::cmd::instance().end_simulation_frame ||
				arguments::cmd::instance().end_simulation_time)
				//cuda_particleSystem::instance().running = true;
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			once2 = false;
		}
		static int32_t frame = get<parameters::frame>();
		static float last_time = get<parameters::simulationTime>() - 1.f / get<parameters::camera_fps>();
		static float target_time = arguments::cmd::instance().time_limit;
		static float timeToSimulate = target_time - get<parameters::simulationTime>();
		static int32_t timeStepsToSimulate = arguments::cmd::instance().end_simulation_time ? floorf(timeToSimulate * get<parameters::camera_fps>()) : -1;
		//std::cout << target_time << " " << get<parameters::simulationTime>() << " " << timeToSimulate << " " << timeStepsToSimulate << std::endl;

		static int32_t frameCtr = 0;
		static int32_t exportedFrames = 0;
		if ((frame != get<parameters::frame>()) && (get<parameters::gl_record>() || arguments::cmd::instance().renderToFile) &&
			(get<parameters::simulationTime>() >
				last_time + 1.f / get<parameters::camera_fps>())) {
			//std::cout << "Starting pretty render for frame" << frame << " at time " << get<parameters::simulationTime>() << " [" << get<parameters::camera_fps>()  << " fps]"<< std::endl;
			last_time += 1.f / get<parameters::camera_fps>();
			cuda_particleSystem::instance().running = false;
			if (arguments::cmd::instance().renderToFile)
				prettyRender = true;
			else
				writeFlag = true;
			frameCtr = 0;
		}
		if (arguments::cmd::instance().renderToFile && vrtxTracer != nullptr) {
			if (!prettyRender)
				vrtxTracer->bValid = false;
			if (prettyRender)
				vrtxTracer->bValid = true;
		}
		QPainter p(this);
		p.beginNativePainting();
		for (auto r : m_renderFunctions)
			if (r->valid())
				r->update();
		for (auto r : m_volumeRenderFunctions)
			if (r->valid())
				r->update();
		for (const auto& r : m_rayTracingFunctions)r->update();

		renderFunctions();

		p.endNativePainting();
		drawText(p);
		p.end(); 
		if (writeFlag) {
			if (get<parameters::gl_record>()) {
				//std::cout << "Writing frame to ffmpeg" << std::endl;
				std::stringstream sstream2;
				sstream2 << "frame_" << std::setfill('0') << std::setw(3) << exportedFrames++ << ".dump";
				auto frame = grabFrameBuffer();
				auto GL_frame = QGLWidget::convertToGLFormat(frame);
#ifdef WIN32
				fwrite(GL_frame.bits(), sizeof(int) * GL_frame.width() * GL_frame.height(),
					1, m_ffmpegPipe);
#endif
				writeFlag = false;
				prettyRender = false;
				get<parameters::dumpFile>() = (arguments::cmd::instance().renderDirectory / sstream2.str()).string();
				if (arguments::cmd::instance().renderToFile)
					if (exportedFrames % 60 == 0) {
						get<parameters::dumpNextframe>() = 1;
						std::this_thread::sleep_for(std::chrono::milliseconds(16));
					}

				//std::stringstream sstream2;
				//sstream2 << "frame_" << std::setfill('0') << std::setw(3) << exportedFrames << ".dump";

				//auto fs = std::experimental::filesystem::path(get<parameters::working_directory>());
				//get<parameters::dumpFile>() = (arguments::cmd::instance().renderDirectory / sstream2.str()).string();
				//if (exportedFrames % 10 == 0)
				   // get<parameters::dumpNextframe>() = 1;
				//std::this_thread::sleep_for(std::chrono::milliseconds(16));
				cuda_particleSystem::instance().running = true;
				// exportedFrames++;
			}
			if (arguments::cmd::instance().renderToFile) {
				auto frame = grabFrameBuffer();
				std::stringstream sstream;
				std::stringstream sstream2;
				sstream << "frame_" << std::setfill('0') << std::setw(3) << exportedFrames++ << ".png";
				sstream2 << "frame_" << std::setfill('0') << std::setw(3) << exportedFrames << ".dump";

				auto fs = arguments::cmd::instance().renderDirectory;
				fs /= sstream.str();
				//std::cout << "Wrote frame " << std::setw(3) << exportedFrames - 1 << " to file " << fs.string() << "                            \r";
				frame.save(QString::fromStdString(fs.string()));

				writeFlag = false;
				prettyRender = false;
				get<parameters::dumpFile>() = (arguments::cmd::instance().renderDirectory / sstream2.str()).string();
				if (exportedFrames % 30 == 0)
					get<parameters::dumpNextframe>() = 1;
				std::this_thread::sleep_for(std::chrono::milliseconds(16));
				cuda_particleSystem::instance().running = true;
			}
		}
		//std::cout << frameCtr << "\t" << (float)get<parameters::renderSteps>() << "\t" << (float) frameCtr / (float)get<parameters::renderSteps>() << "\t";
		if (prettyRender && arguments::cmd::instance().renderToFile)
			progressBar(exportedFrames, timeStepsToSimulate, (float)frameCtr / (float)get<parameters::renderSteps>());
		if (get<parameters::gl_record>())
			progressBar2(exportedFrames, timeStepsToSimulate);
		if (prettyRender && ++frameCtr > get<parameters::renderSteps>()) {
			// std::cout << std::endl << "Setting write flag" << std::endl;
			writeFlag = true;
		}
	}
	catch (...){
		std::cerr << "Caught exception during openGL rendering" << std::endl;
		std::cerr << boost::current_exception_diagnostic_information() << std::endl;
		//QApplication::quit();
		throw;
	}
}
void OGLWidget::draw() {}
