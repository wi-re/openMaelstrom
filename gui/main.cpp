// There is an issue in the compiler that prevents classes with members with alignas specifiers to
// be linked causes unresolved external symbol on new and delete
#if defined(__MSC_VER__) && defined(__clang__)
#include <host_defines.h>
#undef __builtin_align__
#define __builtin_align__(x)
#endif

#ifndef _WIN32
#include <signal.h>
#endif

#include "qt/mainwindow.h"
#include <simulation/particleSystem.h>
#include <sstream>
#include <thread>
#include <utility/helpers/arguments.h>
#include <config/config.h>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

/** Used to silence Qt warnings which just spam the console, comment out
 * qInstallMessageHandler(myMessageOutput); below to not silence all
 * warnings from Qt. **/
void myMessageOutput(QtMsgType, const QMessageLogContext&, const QString&) {}

int main(int argc, char *argv[]) {
  auto binary_directory = fs::system_complete(fs::path(argv[0])).parent_path();
  auto working_directory = fs::current_path().parent_path();

  *parameters::working_directory::ptr = working_directory.string();
  *parameters::binary_directory::ptr = binary_directory.string();
  *parameters::source_directory::ptr = sourceDirectory;
  *parameters::build_directory::ptr = binaryDirectory;

  qInstallMessageHandler(myMessageOutput);
  // Initialize the simulation based on command line parameters.
  auto &cmd_line = arguments::cmd::instance();
  if (!cmd_line.init(false, argc, argv))
    return 0;

  // Setup the OpenGL state to disable vsync globally
  QSurfaceFormat format;
  format.setSwapInterval(0);
  QSurfaceFormat::setDefaultFormat(format);

  // Initialize the actual simulation 
  cuda_particleSystem::instance().init_simulation();
  cuda_particleSystem::instance().running = false;

  // create a mutex which is used to signal program completion from the GUI to
  // the actual simulation. When the GUI closes the simulation can lock the
  // mutex and will exit the while loop thus stopping the simulation.
  std::mutex render_lock;
  render_lock.lock();
  QApplication *a = nullptr;

  std::thread simulation_thread([&]() {
    a = new QApplication(argc, argv);
	auto w = new MainWindow();
    w->show();
    a->exec();
    render_lock.unlock();
  });

  try {
    while (!render_lock.try_lock()) {
      cuda_particleSystem::instance().step();
#ifndef _WIN32
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
#else
      Sleep(1);
#endif
      if (cmd_line.end_simulation_frame && get<parameters::frame>() >= cmd_line.timesteps)
        break;
      if (cmd_line.end_simulation_time && get<parameters::simulationTime>() >= cmd_line.time_limit)
        break;
	  if (cmd_line.pause_simulation_time && get<parameters::simulationTime>() >= cmd_line.time_pause) {
		  cuda_particleSystem::instance().running = false;
		  cmd_line.pause_simulation_time = false;
	  }
    }
  } catch (std::exception e) {
    std::cerr << "Caught exception while running simulation: " << e.what() << std::endl;
	throw;
  }
  // wait for gui to close and finalize the command line arguments (e.g. printing timers)
  simulation_thread.join();
  cmd_line.finalize();
#ifndef _WIN32
  //raise(SIGINT);
#endif
  return 0;
}
