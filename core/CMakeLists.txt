# The compiled core (ADR-017 A): the library cfdcore, its GoogleTest tests and
# the nanobind module _cfdcore.
#
#   cmake -S core -B ~/build/cfd-core -G Ninja
#   cmake --build ~/build/cfd-core
#   ctest --test-dir ~/build/cfd-core --output-on-failure
#
# Python reaches the module through `pip install --no-build-isolation ./core`,
# which scikit-build-core runs through this file with CFD_CORE_TESTS OFF
# (core/pyproject.toml). Builds go out of source, never into the checkout.

cmake_minimum_required(VERSION 3.26)
project(cfdcore VERSION 0.1.0 LANGUAGES CXX)

option(CFD_CORE_TESTS "Build the GoogleTest tests" ON)
option(CFD_CORE_PYTHON "Build the nanobind module _cfdcore" ON)
option(CFD_CORE_CUDA "Build the CUDA kernels (Phase 6 step 7)" OFF)
option(CFD_CORE_WERROR "Treat the project's warnings as errors" OFF)

if(CFD_CORE_CUDA)
    message(FATAL_ERROR "CFD_CORE_CUDA: the CUDA kernels are built at Phase 6 step 7")
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Build out of source: cmake -S core -B ~/build/cfd-core")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# The optimisation level is CLAUDE.md's -O2, not CMake's Release default of -O3.
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")

# The project's flags (CLAUDE.md, C++ Coding Standards). -ffp-contract=off keeps
# a multiply and an add two roundings, as the reference rounds them (ADR-017 A).
add_library(cfdcore_flags INTERFACE)
target_compile_options(cfdcore_flags INTERFACE
    -Wall -Wextra -Wpedantic -O2 -ffp-contract=off)
if(CFD_CORE_WERROR)
    target_compile_options(cfdcore_flags INTERFACE -Werror)
endif()

add_library(cfdcore STATIC
    src/grid.cpp
    src/layout.cpp
    src/reduce.cpp
    src/scalar.cpp)
target_include_directories(cfdcore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(cfdcore PRIVATE cfdcore_flags)

if(CFD_CORE_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
        URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)
    add_executable(cfdcore_tests
        tests/test_grid.cpp
        tests/test_lanes.cpp
        tests/test_reduce.cpp
        tests/test_scalar.cpp)
    target_link_libraries(cfdcore_tests PRIVATE cfdcore cfdcore_flags GTest::gtest_main)
    gtest_discover_tests(cfdcore_tests)
endif()

if(CFD_CORE_PYTHON)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE nanobind_ROOT
        COMMAND_ERROR_IS_FATAL ANY)
    find_package(nanobind 3.1 CONFIG REQUIRED)
    # NOMINSIZE: the module is built at the project's -O2, not nanobind's -Os.
    nanobind_add_module(_cfdcore NOMINSIZE python/bindings.cpp)
    target_link_libraries(_cfdcore PRIVATE cfdcore cfdcore_flags)
    install(TARGETS _cfdcore LIBRARY DESTINATION .)
endif()
