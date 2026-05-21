# ==========================================================================
# Generator tests
# ==========================================================================
#
# File layout:
#   tests/generator/*.cc            shared tests (compiled for both CPU and CUDA)
#   tests/generator/cpu_only/*.cc   CPU-only tests (test_generator_cpu_only)
#   tests/generator/cuda_only/*.cc  CUDA-only tests (test_generator_cuda_only,
#                                   USE_CUDA=ON only)

file(GLOB GENERATOR_SHARED_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)
infini_train_add_test_suite(test_generator SOURCES ${GENERATOR_SHARED_SOURCES})

file(GLOB GENERATOR_CPU_ONLY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cpu_only/test_*.cc)
if(GENERATOR_CPU_ONLY_SOURCES)
  infini_train_add_test(test_generator_cpu_only
    SOURCES ${GENERATOR_CPU_ONLY_SOURCES}
    LABELS cpu
  )
endif()

if(USE_CUDA)
  file(GLOB GENERATOR_CUDA_ONLY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_only/test_*.cc)
  if(GENERATOR_CUDA_ONLY_SOURCES)
    infini_train_add_test(test_generator_cuda_only
      SOURCES ${GENERATOR_CUDA_ONLY_SOURCES}
      LABELS cuda
    )
  endif()
endif()
