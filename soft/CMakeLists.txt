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

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "C:/SharedLibraries/boost")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "C:/SharedLibraries/boost/stage/lib")

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "/usr/include")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "/usr/lib")

SET(OPENCL_VENDOR "NVIDIA" CACHE STRING "Use specicic vendor SDK: NVIDIA, ATI, INTEL, IBM (default: NVIDIA)")
FIND_PACKAGE(OpenCL REQUIRED)
INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})

SET(Boost_USE_STATIC_LIBS ON)

FIND_PACKAGE(Boost 1.55.0 COMPONENTS system thread date_time filesystem timer chrono REQUIRED)
IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
	LINK_DIRECTORIES ( ${Boost_LIBRARY_DIRS})
    ADD_DEFINITIONS( -DHAS_BOOST )
    add_definitions(-DBOOST_THREAD_USE_LIB)
    add_definitions(-DBOOST_DATE_TIME_NO_LIB)
    ADD_DEFINITIONS (-DBOOST_CHRONO_INLINED )
    ADD_DEFINITIONS (-DBOOST_CHRONO_HEADER_ONLY )
	
	message(STATUS ${Boost_SYSTEM_LIBRARY})
	message(STATUS ${Boost_LIBRARIES})
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

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories("src/")

IF(MSVC)
	# Enable catching C++ exception
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
ENDIF()

# Testy

enable_testing()

#4-n solver stresstests

add_executable(mtx_loader src/tests/mtx_loader.cpp)

target_link_libraries(mtx_loader ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} rt)

# kernele - wreszcie bedzie spokoj

configure_file(src/twokernels.cl ${CMAKE_CURRENT_BINARY_DIR}/twokernels.cl)
