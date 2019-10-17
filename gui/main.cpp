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
#include <boost/exception/diagnostic_information.hpp> 
namespace fs = boost::filesystem;

// create a mutex which is used to signal program completion from the GUI to
// the actual simulation. When the GUI closes the simulation can lock the
// mutex and will exit the while loop thus stopping the simulation.
std::mutex render_lock;

#ifdef _WIN32
#include <windows.h>
#include <eh.h>
#include <Psapi.h>
class InfoFromSE
{
public:
	typedef unsigned int exception_code_t;

	static const char* opDescription(const ULONG opcode)
	{
		switch (opcode) {
		case 0: return "read";
		case 1: return "write";
		case 8: return "user-mode data execution prevention (DEP) violation";
		default: return "unknown";
		}
	}

	static const char* seDescription(const exception_code_t& code)
	{
		switch (code) {
		case EXCEPTION_ACCESS_VIOLATION:         return "EXCEPTION_ACCESS_VIOLATION";
		case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:    return "EXCEPTION_ARRAY_BOUNDS_EXCEEDED";
		case EXCEPTION_BREAKPOINT:               return "EXCEPTION_BREAKPOINT";
		case EXCEPTION_DATATYPE_MISALIGNMENT:    return "EXCEPTION_DATATYPE_MISALIGNMENT";
		case EXCEPTION_FLT_DENORMAL_OPERAND:     return "EXCEPTION_FLT_DENORMAL_OPERAND";
		case EXCEPTION_FLT_DIVIDE_BY_ZERO:       return "EXCEPTION_FLT_DIVIDE_BY_ZERO";
		case EXCEPTION_FLT_INEXACT_RESULT:       return "EXCEPTION_FLT_INEXACT_RESULT";
		case EXCEPTION_FLT_INVALID_OPERATION:    return "EXCEPTION_FLT_INVALID_OPERATION";
		case EXCEPTION_FLT_OVERFLOW:             return "EXCEPTION_FLT_OVERFLOW";
		case EXCEPTION_FLT_STACK_CHECK:          return "EXCEPTION_FLT_STACK_CHECK";
		case EXCEPTION_FLT_UNDERFLOW:            return "EXCEPTION_FLT_UNDERFLOW";
		case EXCEPTION_ILLEGAL_INSTRUCTION:      return "EXCEPTION_ILLEGAL_INSTRUCTION";
		case EXCEPTION_IN_PAGE_ERROR:            return "EXCEPTION_IN_PAGE_ERROR";
		case EXCEPTION_INT_DIVIDE_BY_ZERO:       return "EXCEPTION_INT_DIVIDE_BY_ZERO";
		case EXCEPTION_INT_OVERFLOW:             return "EXCEPTION_INT_OVERFLOW";
		case EXCEPTION_INVALID_DISPOSITION:      return "EXCEPTION_INVALID_DISPOSITION";
		case EXCEPTION_NONCONTINUABLE_EXCEPTION: return "EXCEPTION_NONCONTINUABLE_EXCEPTION";
		case EXCEPTION_PRIV_INSTRUCTION:         return "EXCEPTION_PRIV_INSTRUCTION";
		case EXCEPTION_SINGLE_STEP:              return "EXCEPTION_SINGLE_STEP";
		case EXCEPTION_STACK_OVERFLOW:           return "EXCEPTION_STACK_OVERFLOW";
		default: return "UNKNOWN EXCEPTION";
		}
	}

	static std::string information(struct _EXCEPTION_POINTERS* ep, bool has_exception_code = false, exception_code_t code = 0)
	{
		HMODULE hm;
		::GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, static_cast<LPCTSTR>(ep->ExceptionRecord->ExceptionAddress), &hm);
		MODULEINFO mi;
		::GetModuleInformation(::GetCurrentProcess(), hm, &mi, sizeof(mi));
		char fn[MAX_PATH];
		::GetModuleFileNameExA(::GetCurrentProcess(), hm, fn, MAX_PATH);

		std::ostringstream oss;
		oss << "SE " << (has_exception_code ? seDescription(code) : "") << " at address 0x" << std::hex << ep->ExceptionRecord->ExceptionAddress << std::dec
			<< " inside " << fn << " loaded at base address 0x" << std::hex << mi.lpBaseOfDll << "\n";

		if (has_exception_code && (
			code == EXCEPTION_ACCESS_VIOLATION ||
			code == EXCEPTION_IN_PAGE_ERROR)) {
			oss << "Invalid operation: " << opDescription(ep->ExceptionRecord->ExceptionInformation[0]) << " at address 0x" << std::hex << ep->ExceptionRecord->ExceptionInformation[1] << std::dec << "\n";
		}

		if (has_exception_code && code == EXCEPTION_IN_PAGE_ERROR) {
			oss << "Underlying NTSTATUS code that resulted in the exception " << ep->ExceptionRecord->ExceptionInformation[2] << "\n";
		}

		return oss.str();
	}
};
void translator(InfoFromSE::exception_code_t code, struct _EXCEPTION_POINTERS* ep)
{
	throw std::exception(InfoFromSE::information(ep, true, code).c_str());
}

#endif

BOOL CtrlHandler(DWORD fdwCtrlType)
{
	std::clog << "Caught signal " << fdwCtrlType << std::endl;
	switch (fdwCtrlType)
	{
		//Cleanup exit
	case CTRL_CLOSE_EVENT:
		QApplication::quit();
		render_lock.unlock();
		std::this_thread::sleep_for(std::chrono::milliseconds(2000));
		return(TRUE);

	default:
		return FALSE;
	}
}

/** Used to silence Qt warnings which just spam the console, comment out
 * qInstallMessageHandler(myMessageOutput); below to not silence all
 * warnings from Qt. **/
void myMessageOutput(QtMsgType, const QMessageLogContext&, const QString&) {}

#include <utility/template/for_struct.h>

QApplication *a = nullptr;
MainWindow* w = nullptr;

int main(int argc, char *argv[])
try {
	if (!SetConsoleCtrlHandler((PHANDLER_ROUTINE)CtrlHandler, TRUE)) {
		std::cerr << "Could not set CtrlHandler. Exiting." << endl;
		return 0;
	} 
#ifdef _WIN32
	_set_se_translator(translator);
	//_set_se_translator([](unsigned int u, _EXCEPTION_POINTERS *pExp) {
	//	std::string error = "SE Exception: ";
	//	
	//	switch (u) {
	//	case 0xC0000005:
	//		error += "Access Violation";
	//		break;
	//	default:
	//		char result[11];
	//		sprintf_s(result, 11, "0x%08X", u);
	//		error += result;
	//	};
	//	throw std::exception(error.c_str());
	//});
#endif
  auto binary_directory = fs::system_complete(fs::path(argv[0])).remove_filename();
  auto working_directory = fs::current_path();

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
  cuda_particleSystem::instance().step();

  render_lock.lock();
  cuda_particleSystem::instance().simulation_lock.lock();

  std::thread simulation_thread([&]() {
	  try{
    a = new QApplication(argc, argv);
	w = new MainWindow();
    w->show();
    a->exec();
    render_lock.unlock();
	  }
	  catch (...) {
		  std::cerr << "Caught exception inside render loop" << std::endl;
		  std::cerr << boost::current_exception_diagnostic_information() << std::endl;
		  QApplication::quit();
		  ///std::cerr << "Caught exception while running simulation: " << e.what() << std::endl;
		  throw;
	  }

  });

  //std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
  } catch (...) {
	  std::cerr << "Caught exception while running simulation" << std::endl;
	  std::cerr << boost::current_exception_diagnostic_information() << std::endl;
	  QApplication::quit();
    ///std::cerr << "Caught exception while running simulation: " << e.what() << std::endl;
	//throw;
  }
  // wait for gui to close and finalize the command line arguments (e.g. printing timers)
  simulation_thread.join();
  cmd_line.finalize();
#ifndef _WIN32
  //raise(SIGINT);
#endif
  return 0;
}
catch (...) {
	std::cerr << "Caught exception" << std::endl;
	std::cerr << boost::current_exception_diagnostic_information() << std::endl;
	if (w != nullptr)
		QApplication::quit();
	throw;
}