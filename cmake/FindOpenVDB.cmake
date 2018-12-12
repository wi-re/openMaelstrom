# Module to find OpenVDB

# This module will first look into the directories defined by the variables:
#   - OPENVDB_HOME
#
# To use a custom OpenEXR
#   - Set the variable OPENEXR_CUSTOM to True
#   - Set the variable OPENEXR_CUSTOM_LIBRARY to the name of the library to
#     use, e.g. "SpiIlmImf"
#
# This module defines the following variables:
#
# OPENVDB_INCLUDE_DIR - where to find ImfRgbaFile.h, OpenEXRConfig, etc.
# OPENVDB_LIBRARIES   - list of libraries to link against when using OpenEXR.
#                       This list does NOT include the IlmBase libraries.
#                       These are defined by the FindIlmBase module.
# OPENVDB_FOUND       - True if OpenEXR was found.


IF ( BASE_BUILD_PATH ) 
  IF ( NOT OPENVDB_HOME)
    message ( STATUS "Searching ${BASE_BUILD_PATH} for OpenVDB.." ) 
    FILE ( GLOB children RELATIVE ${BASE_BUILD_PATH} ${BASE_BUILD_PATH}/*)
    UNSET ( OPENVDB_HOME )
    FOREACH(subdir ${children})
        IF ( "${subdir}" MATCHES "OpenVDB(.*)" )
          SET ( OPENVDB_HOME ${BASE_BUILD_PATH}/${subdir}/ CACHE PATH "Path to OpenVDB")
       ENDIF()
    ENDFOREACH()     
  ENDIF ()
ENDIF ()

IF (WIN32)
	FIND_PATH( OPENVDB_INCLUDE_DIR openvdb.h
		$ENV{PROGRAMFILES}/openvdb/include
		${PROJECT_SOURCE_DIR}/src/nvgl/openvdb/include
    ${OPENVDB_HOME}/include/
		DOC "The directory where openvdb.h resides")
	FIND_LIBRARY( OPENVDB_LIBRARY
		NAMES openvdb
		PATHS
		$ENV{PROGRAMFILES}/openvdb/lib
		${PROJECT_SOURCE_DIR}/src/nvgl/openvdb/bin
		${PROJECT_SOURCE_DIR}/src/nvgl/openvdb/lib
    ${OPENVDB_HOME}/lib    
		DOC "The OpenVDB library")
ELSE (WIN32)
	FIND_PATH( OPENVDB_INCLUDE_DIR zlib.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where openvdb.h resides")
	FIND_LIBRARY( OPENVDB_LIBRARY
		NAMES zlib ZLIB zlibd zlibstatic
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The OpenVDB library")
ENDIF (WIN32)

UNSET ( OPENVDB_FOUND CACHE )

IF (NOT EXISTS ${OPENVDB_INCLUDE_DIR} )   
   SET ( OPENVDB_INCLUDE_DIR "" )
endif ()

IF (NOT EXISTS ${OPENVDB_LIBRARY} )   
   SET ( OPENVDB_LIBRARY "" )
endif ()

UNSET ( OPENVDB_FOUND CACHE)

IF ( (NOT ${OPENVDB_INCLUDE_DIR} STREQUAL "") AND (NOT ${OPENVDB_LIBRARY} STREQUAL "") )
	SET( OPENVDB_FOUND TRUE CACHE BOOL "")
  message ( STATUS "OpenVDB Library: Found at ${OPENVDB_LIBRARY}" )
ELSE ()
  SET ( OPENVDB_FOUND FALSE CACHE BOOL "")
  message ( "OpenVDB Library: Not found. Confirm that OPENVDB_HOME is correct." )
ENDIF ()

MARK_AS_ADVANCED( OPENVDB_FOUND )
