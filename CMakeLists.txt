cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
# CMake policy settings
cmake_policy(VERSION 3.9)												# required for boost 1.65.1
cmake_policy(SET CMP0071 NEW)											# automoc setting policy introduced in CMake 3.10
   
# CMake settings to build cuda projects properly and to work well with other depdencies, e.g. boost
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake") # add local CMake files (e.g. for openVDB)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
SET(Boost_USE_STATIC_LIBS        OFF)
SET(Boost_USE_MULTITHREADED      ON)
SET(Boost_USE_STATIC_RUNTIME     OFF)
SET(BOOST_ALL_DYN_LINK           ON)   
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON) 						
SET(CMAKE_INCLUDE_CURRENT_DIR ON)	# needed for QT 
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/lib) 
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib) 
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
    

option(USE_CLANG "Use Clang as the compiler for everything" OFF)

# Project settings
SET(NAME openGPUSPH)													# name of the Project, variable for convenience
IF(${USE_CLANG})
SET(CMAKE_C_COMPILER "clang")
SET(CMAKE_CXX_COMPILER "clang++")
project(${NAME} LANGUAGES CXX)
ELSE()
project(${NAME} LANGUAGES CXX CUDA)
ENDIF()

# clang-tidy for code linting if desired
find_program(
	CLANG_TIDY_EXE
	NAMES "clang-tidy"
	DOC "Path to clang-tidy executable" 
	)
if(NOT CLANG_TIDY_EXE)
	message(STATUS "clang-tidy not found.")
else()
	message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
	SET(DO_CLANG_TIDY 
	"${CLANG_TIDY_EXE}" 
	"-checks=*,clang-analyzer-alpha.*,-fuchsia-*,-bugprone-exception-escape" 
	)
endif()

# steps to determine the GPU architecture of the system to only build one architecture
SET(OUTPUTFILE ${CMAKE_CURRENT_BINARY_DIR}/cuda_script) # No suffix required
SET(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_cuda.cu)
if(${USE_CLANG})
	list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)
	SET(CUDA_DIR "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0")
	execute_process(COMMAND clang -fuse-ld=lld -fexceptions --cuda-gpu-arch=sm_62 --cuda-path=${CUDA_DIR} -L${CUDA_DIR}\\lib\\x64 -lcudart ${CUDAFILE} -o ${OUTPUTFILE})
	execute_process(COMMAND clang --version)
else()
execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
execute_process(COMMAND nvcc --version)
endif()
execute_process(COMMAND ${OUTPUTFILE} RESULT_VARIABLE CUDA_RETURN_CODE OUTPUT_VARIABLE ARCH)
if(${CUDA_RETURN_CODE} EQUAL 0)
	SET(CUDA_SUCCESS "TRUE")
else() 
	SET(CUDA_SUCCESS "FALSE")
endif() 
if (${CUDA_SUCCESS})
	message(STATUS "CUDA Architecture: ${ARCH}")
else()
    message(WARNING ${ARCH})
endif()
 
#SET(ARCH 62)

add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)

# Package finding for all components required should be build as per the readme.md file
FIND_PACKAGE ( Boost 		REQUIRED COMPONENTS iostreams system thread program_options filesystem)
FIND_PACKAGE ( Bullet   REQUIRED)											# Rigid body physics
FIND_PACKAGE ( ALEMBIC 		REQUIRED )									# Required for particle based I/O
FIND_PACKAGE ( OpenVDB 		REQUIRED )									# Required for volume based I/O
FIND_PACKAGE ( ILMBase 		REQUIRED )									# Required for openVDB/alembic
FIND_PACKAGE ( OpenEXR 		REQUIRED )  								# Required for openVDB/alembic
FIND_PACKAGE ( BLOSC		REQUIRED )									# Required for openVDB/alembic
FIND_PACKAGE ( ZLIB			REQUIRED )									# Required for openVDB/alembic
FIND_PACKAGE ( TBB			REQUIRED )									# Required for openVDB/alembic and parallel CPU code
FIND_PACKAGE ( Qt5Widgets	REQUIRED )									# Required for the GUI application
FIND_PACKAGE ( Qt5Core		REQUIRED )									# Required for the GUI application
FIND_PACKAGE ( Qt5OpenGL	REQUIRED )									# Required for the GUI application
  
# Configure file to store the binary and source path for resolving file paths
configure_file(cmake/config.h.in config/config.h)
 
# Set Include paths, due to everything needing almost everything this is done here
SET(INCLUDE_DIRS
	${ALEMBIC_INCLUDE_DIR} 
	${ALEMBIC_INCLUDE_DIRS} 
	${OPENVDB_INCLUDE_DIR} 	
	${OPENEXR_INCLUDE_DIR} 	
	${OPENEXR_INCLUDE_DIR}/OpenEXR
	${Boost_INCLUDE_DIR} 		
	${ILMBASE_INCLUDE_DIR} 	
	${ILMBASE_INCLUDE_DIR}/OpenEXR
	${CMAKE_SOURCE_DIR} 	
	${TBB_INCLUDE_DIR} 
	${ZLIB_INCLUDE_DIR} 
	${BLOSC_INCLUDE_DIR}  
	${CMAKE_SOURCE_DIR} 		
	${CMAKE_BINARY_DIR}   
	${BULLET_INCLUDE_DIRS}
)
INCLUDE_DIRECTORIES(${INCLUDE_DIRS})

IF(WIN32)
	IF(${USE_CLANG})
		SET(CMAKE_CXX_STANDARD 20)
		SET(CUDA_DIR "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0")
		add_compile_options("-std=++2a;-Wno-unused-command-line-argument;-Wno-unused-result;-Wno-deprecated-declarations;-Wno-unsupported-target-opt")
		add_compile_options("$<$<CONFIG:DEBUG>:-g;-gcodeview;>")
		add_compile_options("$<$<CONFIG:RELWITHDEBINFO>:-g;-gcodeview;>")
		add_compile_options("-fms-extensions;-fms-compatibility;-fdeclspec;-ftemplate-backtrace-limit=0")
		add_compile_definitions(BOOST_TT_DISABLE_INTRINSICS)
		add_compile_options("-fuse-ld=lld;-fexceptions;--cuda-gpu-arch=sm_${ARCH};--cuda-path=${CUDA_DIR}")
		add_compile_definitions(OPENEXR_DLL OPENVDB_DLL OPENVDB_USE_BLOSC NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS BOOST_ALL_DYN_LINK)
		INCLUDE_DIRECTORIES(${CUDA_DIR}\\include "C:\\dev\\rocm\\HIP\\include\\")
		link_directories(${CUDA_DIR}\\lib\\x64)
	ELSE(${USE_CLANG})
		SET(CUDA_SEPARABLE_COMPILATION ON)								 		
		SET(CUDA_HOST_COMPILATION_CPP ON)	
		SET(CUDA_ATTACH_VS_BUILD_RULE_TO_FILE ON) 	
		INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
		LINK_DIRECTORIES(${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		LINK_DIRECTORIES(${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
		#CMAKE is broken with respect to CMAKE_CXX_STANDARD and clang-tidy as it generates -std=c++17, which is unknown, instead of /std:c++17
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /EHa /bigobj /openmp")
		add_compile_definitions(OPENEXR_DLL OPENVDB_DLL OPENVDB_USE_BLOSC NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS BOOST_ALL_DYN_LINK)
		SET(CUDA_FLAGS_FOR_RELEASE 
		"${CMAKE_CUDA_FLAGS} -O3
		-gencode=arch=compute_${ARCH},code=sm_${ARCH} -gencode=arch=compute_${ARCH},code=compute_${ARCH} 
		--expt-extended-lambda --expt-relaxed-constexpr -ftz=true -keep -use_fast_math -src-in-ptx -lineinfo -Xptxas \"-warn-double-usage\"")
		SET(CUDA_FLAGS_FOR_DEBUG
		"${CMAKE_CUDA_FLAGS} -g -G
		-gencode=arch=compute_${ARCH},code=sm_${ARCH} -gencode=arch=compute_${ARCH},code=compute_${ARCH} 
		--expt-extended-lambda --expt-relaxed-constexpr -ftz=true -keep -use_fast_math -src-in-ptx -lineinfo -Xptxas \"-warn-double-usage\"")
		SET(CMAKE_CUDA_FLAGS "${CUDA_FLAGS_FOR_RELEASE}")
	ENDIF(${USE_CLANG})
ELSE(WIN32)  
	SET(CUDA_SEPARABLE_COMPILATION ON)								 		
	SET(CUDA_HOST_COMPILATION_CPP ON)	
	SET(CUDA_ATTACH_VS_BUILD_RULE_TO_FILE ON) 	
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")  
	SET(CMAKE_CXX_STANDARD 17)
	SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
	SET(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "")
	SET(CMAKE_CUDA_FLAGS "")
	SET(CMAKE_CUDA_FLAGS_RELEASE "")
	SET(cuda_flags "-O3 -gencode=arch=compute_${ARCH},code=sm_${ARCH} -gencode=arch=compute_${ARCH},code=compute_${ARCH} --expt-extended-lambda --expt-relaxed-constexpr -keep -src-in-ptx -use_fast_math -lineinfo -Xptxas -warn-double-usage -g -DNDEBUG -std=c++14 -Xcompiler -std=c++14 --compiler-options \"-Wall -Wextra\"")
	SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${cuda_flags}")
ENDIF(WIN32)   



   
# Boost and Intels TBB require certain libraries to be available with no path to function properly so set this up
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS} )							
LINK_DIRECTORIES(${TBB_LIBRARY_DIRS} )
  
ADD_DEFINITIONS(-DBT_USE_DOUBLE_PRECISION)

# Macro to get all directories within a subdirectory, e.g. for the SPH modules
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child}) 
      LIST(APPEND dirlist ${child})
    ENDIF() 
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()    
  
# Get list of all directories in SPH and metaCode subfolders
SUBDIRLIST(SPH_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/SPH)
SUBDIRLIST(RENDER_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/render)
SUBDIRLIST(META_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/metaCode)
 
# Macro useful to gather all sources for cuda, c++ and qt code and sort them in the correct source group
macro(_GLOB_SOURCES LIBRARY_NAME)
	file(GLOB_RECURSE CUDA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRARY_NAME}/*.cu)
	file(GLOB_RECURSE CUDA_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRARY_NAME}/*.cuh)

	file(GLOB_RECURSE CPP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRARY_NAME}/*.cpp)
	file(GLOB_RECURSE CPP_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRARY_NAME}/*.h ${LIBRARY_NAME}/*.hpp)

	file(GLOB UIS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRARY_NAME}/*.ui)
	file(GLOB QT_SOURCES  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRARY_NAME}/qt/*.cpp  ${EXECUTABLE_NAME}/qt/*.cu)
	file(GLOB QT_HEADERS  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRARY_NAME}/qt/*.h  ${EXECUTABLE_NAME}/qt/*.cuh)
	  
	IF(${USE_CLANG})	
		set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CXX)
	ENDIF()

	SET(source_list ${CUDA_SOURCES} ${CUDA_HEADERS} ${CPP_SOURCES} ${CPP_HEADERS})
	   
	foreach(source IN LISTS source_list)
		get_filename_component(source_path "${source}" PATH) 
		string(REPLACE "/" "\\" source_path_msvc "${source_path}")
		source_group("${source_path_msvc}" FILES "${source}")
	endforeach()  
endmacro(_GLOB_SOURCES)
  
# Macro used to link all system libraries, e.g. openVDB, boost and TBB but NOT cuda
macro(_SETUP_DEPENDENCIES MODULE_NAME)
	target_link_libraries(${MODULE_NAME}
		${ALEMBIC_ABC_LIBRARY}
		${OPENVDB_LIBRARY}
		${OPENEXR_ILMIMF_LIBRARY_RELEASE}
		${ALEMBIC_ILMBASE_IMATH_LIB} 
		${ALEMBIC_ILMBASE_ILMTHREAD_LIB}
		${ALEMBIC_ILMBASE_IEX_LIB}
		${ALEMBIC_ILMBASE_IEXMATH_LIB}
		${ALEMBIC_ILMBASE_HALF_LIB}
		${ILMBASE_HALF_LIBRARY_RELEASE}  
  		${Boost_SYSTEM_LIBRARY_RELEASE}
  		${Boost_THREAD_LIBRARY_RELEASE} 
		${Boost_LIBRARIES}  
		${BLOSC_blosc_LIBRARY}
		${TBB_LIBRARY}   
		${ZLIB_LIBRARY}	 
		${ARGN}
		${BULLET_DYNAMICS_LIBRARY}
		${BULLET_COLLISION_LIBRARY}
		${BULLET_MATH_LIBRARY}
		${BULLET_LIBRARIES}
	)  
	IF(${USE_CLANG})
		target_link_libraries(${MODULE_NAME}
				cudart
			)
	ELSE()
		target_link_libraries(${MODULE_NAME}
			${CUDA_CUDART_LIBRARY}
			${CUDA_LIBRARIES}
			${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES}
			${CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES}	
			)
	ENDIF()
	target_include_directories(${MODULE_NAME} PUBLIC ${INCLUDE_DIRECTORIES})
	IF(NOT WIN32)
		target_link_libraries(${MODULE_NAME} stdc++fs) 
	ENDIF(NOT WIN32)  
endmacro(_SETUP_DEPENDENCIES)  
  
function(buildLibrary LIBRARY_NAME TARGETLIST PREFIX ) 
	_GLOB_SOURCES("${PREFIX}/${LIBRARY_NAME}")
	add_library( ${LIBRARY_NAME} ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS}) 
	_SETUP_DEPENDENCIES(${LIBRARY_NAME} ${ARGN})
	#qt5_use_modules(${LIBRARY_NAME} Widgets OpenGL Core )    
	IF(WIN32)
		IF("${CPP_HEADERS}" MATCHES "dummy\.h")
			add_custom_command(
				OUTPUT ${PREFIX}/${LIBRARY_NAME}/dummy.h
				COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/dummy.h ${PREFIX}/${LIBRARY_NAME}/dummy.h
				DEPENDS ${CUDA_SOURCES})
		ENDIF()
	ENDIF()
	#MESSAGE("local target list: ${${LOCAL_TARGETS}}")
endfunction(buildLibrary) 
function(buildApplication LIBRARY_NAME PREFIX)
	_GLOB_SOURCES("${PREFIX}/${LIBRARY_NAME}")
	add_executable(${LIBRARY_NAME} ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS})
	_SETUP_DEPENDENCIES(${LIBRARY_NAME} ${ARGN})
endfunction(buildApplication) 
function(buildQTLibrary LIBRARY_NAME PREFIX)
	_GLOB_SOURCES("${PREFIX}/${LIBRARY_NAME}")
	QT5_WRAP_UI( UI_HEADERS ${UIS} )
	IF(CUDA_SOURCES AND NOT WIN32)
		add_library("${LIBRARY_NAME}_cuda" ${CUDA_SOURCES} ${CUDA_HEADERS}) 
		_SETUP_DEPENDENCIES("${LIBRARY_NAME}_cuda" ${ARGN})
		add_library( ${LIBRARY_NAME} ${CPP_SOURCES} ${CPP_HEADERS} ${RES_SOURCES} ${UI_HEADERS}) 
		_SETUP_DEPENDENCIES(${LIBRARY_NAME} ${ARGN} "${LIBRARY_NAME}_cuda")
		add_dependencies("${LIBRARY_NAME}_cuda" utility)
		add_dependencies(${LIBRARY_NAME} "${LIBRARY_NAME}_cuda")
		qt5_use_modules(${LIBRARY_NAME} Widgets OpenGL Core )    
	ELSE(CUDA_SOURCES AND NOT WIN32)
		add_library( ${LIBRARY_NAME} ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS} ${RES_SOURCES} ${UI_HEADERS}  ) 
		_SETUP_DEPENDENCIES(${LIBRARY_NAME} ${QT_LIBRARIES} ${ARGN})
		qt5_use_modules(${LIBRARY_NAME} Widgets OpenGL Core )    
	ENDIF(CUDA_SOURCES AND NOT WIN32) 
	IF(WIN32) 
		IF("${CPP_HEADERS}" MATCHES "dummy\.h")
			add_custom_command(
				OUTPUT ${PREFIX}/${LIBRARY_NAME}/dummy.h
				COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/dummy.h ${PREFIX}/${LIBRARY_NAME}/dummy.h
				DEPENDS ${CUDA_SOURCES})
		ENDIF()
	ENDIF()
	IF(LIBRARY_NAME STREQUAL "vrtxRender")
		#target_compile_options("${LIBRARY_NAME}" PUBLIC -g -G)
		#MESSAGE("Enabling debug for ${LIBRARY_NAME}")
	ENDIF()
endfunction(buildQTLibrary)
function(buildQTApplication EXECUTABLE_NAME)
	set(CMAKE_AUTOMOC ON)													# automoc makes life easier with QT
	_GLOB_SOURCES(${EXECUTABLE_NAME})
	QT5_WRAP_UI( UI_HEADERS ${UIS} )
	IF(WIN32)
	add_executable( ${EXECUTABLE_NAME} ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS} ${RES_SOURCES} ${UI_HEADERS}  ${CMAKE_SOURCE_DIR}/app.rc) 
	ELSE()
	add_executable( ${EXECUTABLE_NAME} ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS} ${RES_SOURCES} ${UI_HEADERS})
	ENDIF()
	_SETUP_DEPENDENCIES(${LIBRARY_NAME} ${QT_LIBRARIES} ${ARGN})
	qt5_use_modules(${EXECUTABLE_NAME} Widgets OpenGL Core )    
	set(CMAKE_AUTOMOC OFF)
endfunction(buildQTApplication)

# Build all meta code modules 
foreach(MODULE ${META_MODULES})  
	buildApplication(${MODULE} "metaCode")
	set_target_properties(${MODULE} PROPERTIES FOLDER util)
endforeach(MODULE)	

# adds custom target that contains the meta functionality
add_custom_target(config ALL
	COMMAND ${CMAKE_BINARY_DIR}/bin/arrays ${CMAKE_SOURCE_DIR}/utility/identifier/arrays
	COMMAND ${CMAKE_BINARY_DIR}/bin/functions ${CMAKE_SOURCE_DIR}/functions.json ${CMAKE_SOURCE_DIR}/simulation/functions.h ${CMAKE_SOURCE_DIR}
	COMMAND ${CMAKE_BINARY_DIR}/bin/uniforms ${CMAKE_SOURCE_DIR}/utility/identifier/uniform
)  
SET(SPHModules "")
#SET(VOIDED_LIBRARIES "")
#SET(MODULE_LIST "")   
# Next build the utility system
buildLibrary(utility DISCARDED_LIBRARIES "${CMAKE_SOURCE_DIR}") 
#buildLibrary(Mesh DISCARDED_LIBRARIES "${CMAKE_SOURCE_DIR}") 
# Next build all SPH modules
foreach(MODULE ${SPH_MODULES})   
	buildLibrary(${MODULE} SPHModules "SPH")
endforeach(MODULE)	
#MESSAGE("modules: ${SPHModules}")
foreach(MODULE ${SPH_MODULES})   
	set_target_properties(${MODULE} PROPERTIES FOLDER SPH)
	add_dependencies(${MODULE} utility)
endforeach(MODULE)	
# Build IO Modules and the simulaltion wrapper next
buildLibrary(IO	DISCARDED_LIBRARIES "${CMAKE_SOURCE_DIR}" utility) 
buildLibrary(simulation DISCARDED_LIBRARIES "${CMAKE_SOURCE_DIR}" IO utility) 
# Next build all rendering modules
foreach(MODULE ${RENDER_MODULES})   
	buildQTLibrary(${MODULE} "render")
	set_target_properties(${MODULE} PROPERTIES FOLDER render)
	add_dependencies(${MODULE} utility qGLWidget)
	if ("${MODULE}" STREQUAL "mlmRender" OR "${MODULE}" STREQUAL "vrtxRender")
		TARGET_LINK_LIBRARIES(${MODULE} qGLWidget)
	endif()
endforeach(MODULE)	
# Build the UI part of the code last
#buildQTLibrary(openGLRenderer "${CMAKE_SOURCE_DIR}") 
buildQTApplication(gui utility simulation) 
buildApplication(consoleParticles "${CMAKE_SOURCE_DIR}" utility simulation)
 
#set(CMAKE_VERBOSE_MAKEFILE ON)

# The config target depends on the meta modules and the IO target depends on the LUT and config element
add_dependencies(config arrays uniforms functions)
#add_dependencies(IO LUTCode2)
add_dependencies(IO LUTCode)
add_dependencies(IO config)
 
# Folder setting up for visual studio
set_target_properties (config PROPERTIES FOLDER util)
set_target_properties (utility PROPERTIES FOLDER util)
set_target_properties (consoleParticles PROPERTIES FOLDER UI)
#set_target_properties (openGLRenderer PROPERTIES FOLDER UI)  
set_target_properties (gui PROPERTIES FOLDER UI)  
   
# Dependencies to make sure the UI elements are properly rebuilt

#add_dependencies(consoleParticles simulation ${SPH_MODULES})
#add_dependencies(gui simulation ${SPH_MODULES})
IF(WIN32)
	get_filename_component(BULLET_ROOT ${BULLET_INCLUDE_DIRS} DIRECTORY)
	#MESSAGE(${BULLET_ROOT})
	SET( BULLET_LIBS 
			${BULLET_ROOT}/bin/BulletCollision_vs2010_x64_release.lib
			${BULLET_ROOT}/bin/BulletDynamics_vs2010_x64_release.lib
			${BULLET_ROOT}/bin/BulletSoftBody_vs2010_x64_release.lib
			${BULLET_ROOT}/bin/LinearMath_vs2010_x64_release.lib
			${BULLET_ROOT}/bin/Bullet3Common_vs2010_x64_release.lib) 
			  
	target_link_libraries(gui ${BULLET_LIBS})  
	target_link_libraries(consoleParticles ${BULLET_LIBS})
ELSE()
	SET(BULLET_LIB_ROOT "/usr/local/lib")
	#MESSAGE(${BULLET_ROOT})
	SET( BULLET_LIBS 
		BulletCollision
		BulletDynamics
		BulletSoftBody
		LinearMath
		Bullet3Common)
		target_link_libraries(utility ${BULLET_LIBS})
	target_link_libraries(gui ${BULLET_LIBS})
	target_link_libraries(consoleParticles ${BULLET_LIBS})
ENDIF()
target_link_libraries(gui simulation )
target_link_libraries(utility ${BULLET_LIBRARIES}) 
target_link_libraries(IO ${BULLET_LIBRARIES}) 

foreach(MODULE ${SPH_MODULES})  
	target_link_libraries(gui ${MODULE})  
	target_link_libraries(consoleParticles ${MODULE}) 
endforeach(MODULE)	 
foreach(MODULE ${RENDER_MODULES})  
	target_link_libraries(gui ${MODULE})
	IF(NOT "${MODULE}" STREQUAL "qGLWidget")
		target_link_libraries("qGLWidget" ${MODULE})
	ENDIF(NOT "${MODULE}" STREQUAL "qGLWidget")
	IF(NOT "${MODULE}" STREQUAL "util")
		target_link_libraries(${MODULE} "util")
	ENDIF(NOT "${MODULE}" STREQUAL "util")
endforeach(MODULE)	

# Clang tidy on select targets to not overwhelm with warnings
#if(CLANG_TIDY_EXE)
#set_target_properties(LUTCode PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}")
#set_target_properties(utility PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}")
#endif()