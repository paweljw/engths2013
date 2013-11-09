CMAKE_MINIMUM_REQUIRED (VERSION 2.6)
project (pjwsolver)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "Debug")
endif()

EXECUTE_PROCESS( COMMAND hostname OUTPUT_VARIABLE CMAKE_HOSTNAME )
IF(NOT "${CMAKE_HOSTNAME}" EQUAL "jack")
	SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
ENDIF()

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "C:/SharedLibs/boost")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "C:/SharedLibs/boost/stage/lib")

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "C:/SharedLibs/CUDA/include")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "C:/SharedLibs/CUDA/lib/x64")

SET(OPENCL_VENDOR "NVIDIA" CACHE STRING "Use specicic vendor SDK: NVIDIA, ATI, INTEL, IBM (default: NVIDIA)")
FIND_PACKAGE(OpenCL REQUIRED)
INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})

FIND_PACKAGE(Boost COMPONENTS log log_setup thread date_time filesystem system timer REQUIRED)
IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
	LINK_DIRECTORIES ( ${Boost_LIBRARY_DIRS})
    ADD_DEFINITIONS( "-DHAS_BOOST" )
	
#	message(STATUS ${Boost_SYSTEM_LIBRARY})
#	message(STATUS ${Boost_LIBRARIES})
ENDIF()

find_package(Doxygen)
if(DOXYGEN_FOUND)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
	add_custom_target(doc ALL
	${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Generating API documentation with Doxygen" VERBATIM
)
	file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/doc/html")
endif(DOXYGEN_FOUND)

include_directories("src/")

IF(MSVC)
	# Enable catching C++ exception
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
ENDIF()

# Testy

enable_testing()

#1 opencl_sanity
add_executable(test_openclsanity src/tests/openclsanity.cpp)

target_link_libraries(test_openclsanity
			${OPENCL_LIBRARIES} 
)

add_test(OPENCL_SANITY ${CMAKE_CURRENT_BINARY_DIR}/test_openclsanity)

#2 shortmatrix_sanity

add_executable(test_shortmatrixsanity src/tests/shortmatrixsanity.cpp)

target_link_libraries(test_shortmatrixsanity ${OPENCL_LIBRARIES})

add_test(SHORTMATRIX_SANITY ${CMAKE_CURRENT_BINARY_DIR}/test_shortmatrixsanity)

#3 solver instantiation

add_executable(test_solution src/tests/solution.cpp)

target_link_libraries(test_solution ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})

add_test(SOLUTION ${CMAKE_CURRENT_BINARY_DIR}/test_solution)

#4-n solver stresstests

add_executable(mtx_loader src/tests/mtx_loader.cpp)

target_link_libraries(mtx_loader ${OPENCL_LIBRARIES})

add_test(MTX_LDR ${CMAKE_CURRENT_BINARY_DIR}/mtx_loader C:/Users/PJW/Documents/gpufs/build/Debug/sherman1.mtx C:/Users/PJW/Documents/gpufs/build/Debug/sherman1_rhs1.mtx)

