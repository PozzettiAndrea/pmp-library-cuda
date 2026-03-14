# Install script for directory: /home/shadeform/cudageom/pmp-library-cuda/src/pmp

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpmp.so.3.0.0" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpmp.so.3.0.0")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpmp.so.3.0.0"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/shadeform/cudageom/pmp-library-cuda/build-cuda/libpmp.so.3.0.0")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpmp.so.3.0.0" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpmp.so.3.0.0")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpmp.so.3.0.0"
         OLD_RPATH "/usr/local/cuda-13.2/targets/x86_64-linux/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpmp.so.3.0.0")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/shadeform/cudageom/pmp-library-cuda/build-cuda/libpmp.so")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pmp" TYPE FILE FILES
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./bounding_box.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./exceptions.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./mat_vec.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./memory_usage.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./properties.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./stop_watch.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./surface_mesh.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./types.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pmp/algorithms" TYPE FILE FILES
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/barycentric_coordinates.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/curvature.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/decimation.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/differential_geometry.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/distance_point_triangle.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/fairing.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/features.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/geodesics.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/hole_filling.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/laplace.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/normals.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/numerics.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/parameterization.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/remeshing.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/remeshing_cuda.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/remeshing_cuda_checkpoint.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/remeshing_cuda_host.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/shapes.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/smoothing.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/subdivision.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/triangulation.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/utilities.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pmp/algorithms/cuda" TYPE FILE FILES
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/cuda/gpu_bvh.cuh"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./algorithms/cuda/gpu_trimesh.cuh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pmp/io" TYPE FILE FILES
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./io/helpers.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./io/io.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./io/io_flags.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./io/read_obj.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./io/read_off.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./io/read_pmp.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./io/read_stl.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./io/write_obj.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./io/write_off.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./io/write_pmp.h"
    "/home/shadeform/cudageom/pmp-library-cuda/src/pmp/./io/write_stl.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/shadeform/cudageom/pmp-library-cuda/build-cuda/src/pmp/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
