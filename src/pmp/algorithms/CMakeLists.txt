# make sure to install headers, use relative path from top-level since we're
# included from above
file(GLOB HEADERS ./algorithms/*.h)
file(GLOB SOURCES ./algorithms/*.cpp)

if(PMP_INSTALL)
  install(FILES ${HEADERS} DESTINATION include/pmp/algorithms)
endif()

target_sources(pmp PRIVATE "${SOURCES}" "${HEADERS}")

# CUDA remeshing kernels
if(PMP_BUILD_CUDA)
  file(GLOB CUDA_SOURCES ./algorithms/cuda/*.cu)
  file(GLOB CUDA_HEADERS ./algorithms/cuda/*.cuh ./algorithms/cuda/*.h)
  target_sources(pmp PRIVATE "${CUDA_SOURCES}" "${CUDA_HEADERS}")
  set_target_properties(pmp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17)
  target_link_libraries(pmp PRIVATE CUDA::cudart)
  target_compile_definitions(pmp PUBLIC PMP_HAS_CUDA)
  target_include_directories(pmp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/cuda)

  if(PMP_INSTALL)
    install(FILES ${CUDA_HEADERS} DESTINATION include/pmp/algorithms/cuda)
  endif()
endif()
