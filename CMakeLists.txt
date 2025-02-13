cmake_minimum_required(VERSION 3.22)
project(spreadinterp)
set(CMAKE_BUILD_TYPE Release)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
find_package(BLAS REQUIRED)
find_package(Python 3
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind/
  GIT_TAG        v2.4.0
)
FetchContent_MakeAvailable(nanobind)
FetchContent_Declare(
  uammd
  GIT_REPOSITORY https://github.com/RaulPPelaez/uammd
  GIT_TAG        v2.5.5
)
FetchContent_MakeAvailable(uammd)
nanobind_add_module(
  spreadinterp
  STABLE_ABI
  src/python_wrapper.cu
)
uammd_setup_target(spreadinterp)
install(TARGETS spreadinterp LIBRARY DESTINATION ${Python_SITEARCH}/${PROJECT_NAME})
install(FILES src/__init__.py DESTINATION ${Python_SITEARCH}/${PROJECT_NAME})
