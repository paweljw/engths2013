# Try to find OpenCL
# This module tries to find an OpenCL implementation on your system. It supports
# AMD / ATI, NVIDIA, INTEL, IBM implementations
#
# To SET manually OpenCL Vendor, define OPENCL_VENDOR variable in cmake
# -DOPENCL_VENDOR = "AMD/NVIDIA/INTEL/IBM"
#
# Once done this will define
#  OPENCL_FOUND        - system has OpenCL
#  OPENCL_INCLUDE_DIRS  - the OpenCL include directory
#  OPENCL_LIBRARIES    - link these to use OpenCL

SET(PLATFORM 32)
IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
	SET(PLATFORM 64)
ENDIF()

IF(OPENCL_VENDOR MATCHES "AMD")
	MESSAGE( STATUS "Checking AMD APP SDK" )
	FIND_PATH(
		OPENCL_INCLUDE_DIR
		NAMES CL/cl.h
		PATHS $ENV{AMDAPPSDKROOT}/include
		NO_DEFAULT_PATH
    )

	IF("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
		IF(PLATFORM EQUAL 32)
			SET(
				OPENCL_LIB_SEARCH_PATH
				${OPENCL_LIB_SEARCH_PATH}
				$ENV{AMDAPPSDKROOT}/lib/x86
			)
		ELSE()
			SET(
				OPENCL_LIB_SEARCH_PATH
				${OPENCL_LIB_SEARCH_PATH}
				$ENV{AMDAPPSDKROOT}/lib/x64
			)
		ENDIF()
	ENDIF("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
  
	IF(PLATFORM EQUAL 32)
		FIND_LIBRARY(
			OPENCL_LIBRARY
			NAMES OpenCL
			PATHS $ENV{AMDAPPSDKROOT}/lib/x86/
		)
	ELSE()
		FIND_LIBRARY(
			OPENCL_LIBRARY
			NAMES OpenCL
			PATHS $ENV{AMDAPPSDKROOT}/lib/x64/
		)
	ENDIF()
	
ELSEIF(OPENCL_VENDOR MATCHES "NVIDIA")
	MESSAGE( STATUS "Checking NVIDIA CUDA SDK" )
	IF("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
		FIND_LIBRARY(OPENCL_LIBRARY NAMES OpenCL
			IF(PLATFORM EQUAL 32)
				PATHS $ENV{CUDA_PATH}/lib
			ELSE()
				PATHS $ENV{CUDA_PATH}/lib64
			ENDIF()
			)
	ELSE()
		FIND_LIBRARY(OPENCL_LIBRARY NAMES OpenCL
			IF(PLATFORM EQUAL 32)
				PATHS $ENV{CUDA_PATH}/lib/Win32
			ELSE()
				PATHS $ENV{CUDA_PATH}/lib/x64
			ENDIF()
			)
	ENDIF()

	FIND_PATH(
		OPENCL_INCLUDE_DIR
		NAMES CL/cl.h
		PATHS $ENV{CUDA_PATH}/include
		NO_CMAKE_SYSTEM_PATH
    	)
	
ELSEIF(OPENCL_VENDOR MATCHES "INTEL")
	MESSAGE( STATUS "Checking INTEL OPENCL SDK" )

	FIND_PATH(
		OPENCL_INCLUDE_DIR
		NAMES CL/cl.h
		PATHS $ENV{INTELOCLSDKROOT}/include
    )
	
	IF(PLATFORM EQUAL 32)
		FIND_LIBRARY(
			OPENCL_LIBRARY
			NAMES OpenCL
			PATHS $ENV{INTELOCLSDKROOT}/lib/x86
		)
	ELSE()
		FIND_LIBRARY(
			OPENCL_LIBRARY
			NAMES OpenCL
			PATHS $ENV{INTELOCLSDKROOT}/lib/x64/
		)
	ENDIF()

ELSEIF(OPENCL_VENDOR MATCHES "IBM")
  MESSAGE( STATUS "Check for IBM SDK" )
  FIND_LIBRARY(
    OPENCL_LIBRARY
    NAMES OpenCL
    PATHS /opt/ibm/OpenCLCommonRuntime/lib64
    )

  FIND_PATH(
    OPENCL_INCLUDE_DIR
    NAMES CL/cl.h
    PATHS /data/blades/chassis_1/blade14/root/usr/include
    )
ENDIF(OPENCL_VENDOR MATCHES "AMD")

INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
	OPENCL
	DEFAULT_MSG
	OPENCL_LIBRARY OPENCL_INCLUDE_DIR
	)

IF(OPENCL_FOUND)
	SET(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
ELSE(OPENCL_FOUND)
	SET(OPENCL_LIBRARIES)
ENDIF(OPENCL_FOUND)

mark_as_advanced(
	OPENCL_INCLUDE_DIR
	OPENCL_LIBRARY
)
