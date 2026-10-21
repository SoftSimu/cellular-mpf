cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA architectures")
endif()
project(phase_field_gh200 VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PF_BUILD_3D "Build the three-dimensional solver" ON)
option(PF_BUILD_TOOLS "Build initializer and boundary-export tools" ON)
option(PF_BUILD_TWO_GPU_SUBSTRATE "Build the two-GPU slab/channel solver" OFF)
option(PF_DIAGNOSTIC_ALARMS "Enable the high-frequency support-edge counter" OFF)
option(BUILD_TESTING "Build the small CPU reference and smoke tests" OFF)
option(PF_ENABLE_GPU_TESTS "Register smoke tests requiring a CUDA GPU" OFF)
option(PF_ENABLE_CONTACT_TESTS "Register optional Python contact/contour tests" OFF)
if(PF_BUILD_TWO_GPU_SUBSTRATE AND NOT PF_BUILD_3D)
  message(FATAL_ERROR "PF_BUILD_TWO_GPU_SUBSTRATE requires PF_BUILD_3D")
endif()
if(PF_BUILD_3D)
  if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(FATAL_ERROR "The 3D CUDA implementation requires NVIDIA nvcc")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.5)
    message(FATAL_ERROR "3D tensor-map loading requires CUDA 12.5 or newer")
  endif()
  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(NOT arch MATCHES "^([0-9]+)([af])?(-(real|virtual))?$")
      message(FATAL_ERROR "3D requires explicit CUDA architectures >= 90; use -DCMAKE_CUDA_ARCHITECTURES=90")
    endif()
    if(CMAKE_MATCH_1 LESS 90)
      message(FATAL_ERROR "3D bulk-copy instructions require CUDA architecture >= 90")
    endif()
  endforeach()
endif()

# These options specify the validated arithmetic; do not add fast-math.
set(PF_CUDA_FLAGS --expt-relaxed-constexpr -fmad=true -lineinfo -Xptxas=-v)
if(MSVC)
  set(PF_CUDA_RUNTIME Shared)
else()
  set(PF_CUDA_RUNTIME Static)
endif()

function(pf_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-Wpedantic>")
  endif()
endfunction()

function(pf_cuda_target target)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)
  set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY ${PF_CUDA_RUNTIME})
  pf_enable_warnings(${target})
endfunction()

add_library(pf_shape_config INTERFACE)
if(PF_DIAGNOSTIC_ALARMS)
  target_compile_definitions(pf_shape_config INTERFACE PF_ALARMS=1)
endif()
find_package(Threads REQUIRED)
find_path(PF_ZSTD_INCLUDE_DIR zstd.h)
find_library(PF_ZSTD_LIBRARY NAMES zstd libzstd)

function(pf_optional_zstd target)
  if(PF_ZSTD_INCLUDE_DIR AND PF_ZSTD_LIBRARY)
    target_include_directories(${target} PRIVATE ${PF_ZSTD_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE ${PF_ZSTD_LIBRARY})
    target_compile_definitions(${target} PRIVATE PF_BOUNDARY_ZSTD=1)
  endif()
endfunction()

add_library(pf_core STATIC
  src/kernels.cu src/sim.cu src/checkpoint.cu src/trajectory.cpp
  src/output_file.cpp src/velocity_moments.cu src/velocity_output.cu src/boundary.cu)
target_include_directories(pf_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(pf_core PUBLIC pf_shape_config PRIVATE Threads::Threads)
if(MSVC)
  target_compile_definitions(pf_core PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
pf_cuda_target(pf_core)
pf_optional_zstd(pf_core)

add_executable(cell_gh200 src/main.cu)
target_link_libraries(cell_gh200 PRIVATE pf_core)
pf_cuda_target(cell_gh200)

# Host placement uses the same contraction policy in CUDA and C++ translation
# units. nvcc's host compiler can differ from CMake's C++ compiler.
function(pf_no_contraction compiler output)
  if(compiler STREQUAL "NVHPC")
    set(${output} -Mnofma PARENT_SCOPE)
  elseif(compiler MATCHES "^(GNU|Clang|AppleClang)$")
    set(${output} -ffp-contract=off PARENT_SCOPE)
  elseif(compiler STREQUAL "MSVC")
    set(${output} /fp:strict PARENT_SCOPE)
  else()
    set(${output} "" PARENT_SCOPE)
  endif()
endfunction()
pf_no_contraction("${CMAKE_CXX_COMPILER_ID}" PF_HOST_NO_CONTRACTION)
set(PF_CUDA_HOST_ID "${CMAKE_CUDA_HOST_COMPILER_ID}")
if(NOT PF_CUDA_HOST_ID AND CMAKE_CUDA_SIMULATE_ID)
  set(PF_CUDA_HOST_ID "${CMAKE_CUDA_SIMULATE_ID}")
endif()
if(NOT PF_CUDA_HOST_ID)
  set(PF_CUDA_HOST_ID "${CMAKE_CXX_COMPILER_ID}")
endif()
pf_no_contraction("${PF_CUDA_HOST_ID}" PF_CUDA_HOST_NO_CONTRACTION)

if(PF_BUILD_3D)
  if(NOT PF_HOST_NO_CONTRACTION OR NOT PF_CUDA_HOST_NO_CONTRACTION)
    message(FATAL_ERROR "No verified host no-contraction flags for this compiler pair")
  endif()
  add_library(pf3d_boundary STATIC src/pf3d/boundary_output.cu src/pf3d/surface_output.cu)
  target_include_directories(pf3d_boundary PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/common)
  pf_cuda_target(pf3d_boundary)
  pf_optional_zstd(pf3d_boundary)

  add_library(pf3d_core STATIC
    src/pf3d/adaptive_storage.cu src/pf3d/kernels.cu
    src/pf3d/kernels_periodic_slab_walk.cu src/pf3d/sim.cu src/pf3d/checkpoint.cu)
  target_include_directories(pf3d_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/common)
  target_link_libraries(pf3d_core PUBLIC pf_shape_config PRIVATE pf3d_boundary)
  target_compile_options(pf3d_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${PF_HOST_NO_CONTRACTION}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${PF_CUDA_HOST_NO_CONTRACTION}>)
  pf_cuda_target(pf3d_core)

  add_executable(cell_gh200_3d src/pf3d/main.cu)
  target_link_libraries(cell_gh200_3d PRIVATE pf3d_core)
  target_compile_options(cell_gh200_3d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${PF_HOST_NO_CONTRACTION}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${PF_CUDA_HOST_NO_CONTRACTION}>)
  pf_cuda_target(cell_gh200_3d)
  if(PF_BUILD_TWO_GPU_SUBSTRATE)
    add_executable(cell_gh200_3d_two_gpu src/pf3d/main.cu src/pf3d/two_gpu_sim.cu)
    target_link_libraries(cell_gh200_3d_two_gpu PRIVATE pf3d_core)
    target_compile_definitions(cell_gh200_3d_two_gpu PRIVATE PF_TWO_GPU_SUBSTRATE=1)
    target_compile_options(cell_gh200_3d_two_gpu PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${PF_CUDA_HOST_NO_CONTRACTION}>)
    pf_cuda_target(cell_gh200_3d_two_gpu)
  endif()
endif()

if(PF_BUILD_TOOLS)
  add_executable(boundary_export tools/boundary_export.cu)
  target_link_libraries(boundary_export PRIVATE pf_core)
  pf_cuda_target(boundary_export)
  if(PF_HOST_NO_CONTRACTION)
    add_executable(palmieri_centres tools/palmieri_centres.cpp)
    target_include_directories(palmieri_centres PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(palmieri_centres PRIVATE ${PF_HOST_NO_CONTRACTION})
    if(PF_BUILD_3D)
      add_executable(palmieri_centres_3d tools/palmieri_centres_3d.cpp)
      target_include_directories(palmieri_centres_3d PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/common)
      target_compile_options(palmieri_centres_3d PRIVATE ${PF_HOST_NO_CONTRACTION})
    endif()
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
  add_executable(common_primitives_cpu_test tests/common_primitives_cpu_test.cpp)
  target_include_directories(common_primitives_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
  pf_enable_warnings(common_primitives_cpu_test)
  add_test(NAME common_primitives_cpu COMMAND common_primitives_cpu_test)

  if(PF_HOST_NO_CONTRACTION)
    add_executable(pf2d_mobility_model_test tests/mobility_model_cpu_test.cpp)
    target_include_directories(pf2d_mobility_model_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/common
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_options(pf2d_mobility_model_test PRIVATE ${PF_HOST_NO_CONTRACTION})
    pf_enable_warnings(pf2d_mobility_model_test)
    add_test(NAME pf2d_mobility_model COMMAND pf2d_mobility_model_test)
  endif()
  if(PF_BUILD_3D)
    add_executable(pf3d_checksum_cpu_test tests/3d/checksum_cpu_test.cpp)
    target_include_directories(pf3d_checksum_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
    pf_enable_warnings(pf3d_checksum_cpu_test)
    add_test(NAME pf3d_checksum_cpu COMMAND pf3d_checksum_cpu_test)
    add_executable(pf3d_one_step_cpu_test tests/3d/model_one_step_cpu_test.cpp)
    target_include_directories(pf3d_one_step_cpu_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(pf3d_one_step_cpu_test PRIVATE ${PF_HOST_NO_CONTRACTION})
    pf_enable_warnings(pf3d_one_step_cpu_test)
    add_test(NAME pf3d_one_step_cpu COMMAND pf3d_one_step_cpu_test)
  endif()
  if(PF_ENABLE_GPU_TESTS)
    add_test(NAME gh200_smoke COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_gpu_smoke.py --executable $<TARGET_FILE:cell_gh200>)
    set_tests_properties(gh200_smoke PROPERTIES TIMEOUT 180)
    if(PF_BUILD_3D)
      add_test(NAME gh200_3d_smoke COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/3d/run_example.py
        --executable $<TARGET_FILE:cell_gh200_3d>
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/example-3d-output)
      set_tests_properties(gh200_3d_smoke PROPERTIES TIMEOUT 180)
      if(PF_BUILD_TWO_GPU_SUBSTRATE)
        add_test(NAME gh200_two_gpu_smoke COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/3d/run_two_gpu_smoke.py
          --reference $<TARGET_FILE:cell_gh200_3d>
          --executable $<TARGET_FILE:cell_gh200_3d_two_gpu>)
        set_tests_properties(gh200_two_gpu_smoke PROPERTIES TIMEOUT 180)
      endif()
    endif()
  endif()
  if(PF_ENABLE_CONTACT_TESTS AND PF_BUILD_3D)
    find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy, scipy, skimage"
      RESULT_VARIABLE PF_CONTACT_IMPORT_STATUS)
    if(NOT PF_CONTACT_IMPORT_STATUS EQUAL 0)
      message(FATAL_ERROR "Install tools/requirements-contacts.txt before enabling contact tests")
    endif()
    add_test(NAME pf3d_surface_contract COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/3d/surface_contract_test.py)
    foreach(test IN ITEMS geometry exchange cli catalog export)
      add_test(NAME pf3d_contact_${test} COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/3d/contact_${test}_test.py)
    endforeach()
  endif()
endif()

foreach(target IN ITEMS cell_gh200 cell_gh200_3d cell_gh200_3d_two_gpu
                        boundary_export palmieri_centres palmieri_centres_3d)
  if(TARGET ${target})
    install(TARGETS ${target} RUNTIME DESTINATION bin)
  endif()
endforeach()
install(PROGRAMS
  tools/relaxed_2d_to_slab_centres.py tools/stack_channel_centres.py
  tools/render_pf3d_slab.py tools/velocity_moments.py tools/pf3d_boundary.py
  tools/pf3d_analyze_contacts.py tools/pf3d_export_surfaces.py DESTINATION bin)
install(FILES
  tools/pf3d_checkpoint_header.py tools/pf3d_surface.py tools/pf3d_contact.py
  tools/pf3d_exchange.py tools/pf3d_event_catalog.py DESTINATION bin)
install(FILES README.md CITATION.cff LICENSE.txt COPYING DESTINATION share/doc/phase_field_gh200)
install(DIRECTORY docs/ DESTINATION share/doc/phase_field_gh200/docs
  FILES_MATCHING PATTERN "*.md" PATTERN "*.png" PATTERN "*.gif" PATTERN "*.mp4"
  PATTERN "*.json")
install(DIRECTORY examples/ DESTINATION share/doc/phase_field_gh200/examples
  FILES_MATCHING PATTERN "*.py" PATTERN "*.args" PATTERN "__pycache__" EXCLUDE)
install(FILES tools/requirements-visualization.txt tools/requirements-contacts.txt
  DESTINATION share/doc/phase_field_gh200/tools)

message(STATUS "phase_field_gh200: CUDA arch ${CMAKE_CUDA_ARCHITECTURES}; 3D ${PF_BUILD_3D}")
message(STATUS "phase_field_gh200: optional support-edge counter ${PF_DIAGNOSTIC_ALARMS}")
