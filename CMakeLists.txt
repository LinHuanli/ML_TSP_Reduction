cmake_minimum_required(VERSION 3.20)

project(MLCut LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# 优先兼容当前机器已经安装好的本地依赖目录，避免每次构建都手动补前缀。
foreach(_prefix
    "$ENV{HOME}/.local/xgboost-3.1.2"
    "$ENV{HOME}/.local/opt/or-tools-v9.15")
  if(EXISTS "${_prefix}")
    list(APPEND CMAKE_PREFIX_PATH "${_prefix}")
  endif()
endforeach()

find_package(Eigen3 REQUIRED)
find_package(xgboost CONFIG REQUIRED)
find_package(ortools CONFIG REQUIRED)
find_package(liblinear REQUIRED)

find_path(NLOHMANN_JSON_INCLUDE_DIR
  NAMES nlohmann/json.hpp
  HINTS /usr/pkg/include /usr/include /usr/local/include
  REQUIRED)

add_library(mlcut_warnings INTERFACE)
target_compile_options(mlcut_warnings INTERFACE
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wshadow)

add_library(mlcut_base
  src/base/filesystem.cpp
  src/base/process.cpp
  src/base/types.cpp
  src/parallel/parallel_for.cpp
  src/tsp/instance.cpp
  src/data/manifest.cpp
  src/data/generator.cpp)

target_include_directories(mlcut_base
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${NLOHMANN_JSON_INCLUDE_DIR}")

target_link_libraries(mlcut_base
  PUBLIC
    Eigen3::Eigen
    mlcut_warnings)

add_library(mlcut_ml
  src/ml/liblinear_model.cpp
  src/ml/streaming_linear_model.cpp
  src/ml/xgboost_model.cpp)

target_include_directories(mlcut_ml
  PRIVATE
    "${NLOHMANN_JSON_INCLUDE_DIR}")

target_link_libraries(mlcut_ml
  PUBLIC
    mlcut_base
    xgboost::xgboost
    liblinear::liblinear)

add_library(mlcut_solver
  src/candidate/candidate_graph.cpp
  src/candidate/lkh_runner.cpp
  src/label/concorde_runner.cpp
  src/label/edge_labels.cpp
  src/feature/feature_matrix.cpp
  src/feature/feature_extractor.cpp
  src/solver/benchmark.cpp
  src/solver/ortools_smoke.cpp)

target_link_libraries(mlcut_solver
  PUBLIC
    mlcut_base
    ortools::ortools)

add_library(mlcut_pipeline
  src/pipeline/artifacts.cpp
  src/pipeline/baselines.cpp
  src/pipeline/prepare.cpp
  src/pipeline/metrics.cpp
  src/pipeline/pruning.cpp
  src/pipeline/training.cpp)

target_include_directories(mlcut_pipeline
  PRIVATE
    "${NLOHMANN_JSON_INCLUDE_DIR}")

target_link_libraries(mlcut_pipeline
  PUBLIC
    mlcut_base
    mlcut_ml
    mlcut_solver)

add_executable(mlcut_generate_data
  apps/generate_data.cpp)

target_link_libraries(mlcut_generate_data
  PRIVATE
    mlcut_base)

add_executable(mlcut_extend_train_split
  apps/extend_train_split.cpp)

target_link_libraries(mlcut_extend_train_split
  PRIVATE
    mlcut_base)

add_executable(mlcut_smoke_deps
  apps/smoke_deps.cpp)

target_link_libraries(mlcut_smoke_deps
  PRIVATE
    mlcut_base
    mlcut_ml
    mlcut_solver)

add_executable(mlcut_gen_candidates
  apps/gen_candidates.cpp)

target_link_libraries(mlcut_gen_candidates
  PRIVATE
    mlcut_base
    mlcut_solver)

add_executable(mlcut_build_labels
  apps/build_labels.cpp)

target_link_libraries(mlcut_build_labels
  PRIVATE
    mlcut_base
    mlcut_solver)

add_executable(mlcut_extract_features
  apps/extract_features.cpp)

target_link_libraries(mlcut_extract_features
  PRIVATE
    mlcut_base
    mlcut_solver)

add_executable(mlcut_prepare_manifest
  apps/prepare_manifest.cpp)

target_link_libraries(mlcut_prepare_manifest
  PRIVATE
    mlcut_pipeline)

add_executable(mlcut_train_model
  apps/train_model.cpp)

target_link_libraries(mlcut_train_model
  PRIVATE
    mlcut_pipeline)

add_executable(mlcut_prune_manifest
  apps/prune_manifest.cpp)

target_link_libraries(mlcut_prune_manifest
  PRIVATE
    mlcut_pipeline)

add_executable(mlcut_baseline_prune_manifest
  apps/baseline_prune_manifest.cpp)

target_link_libraries(mlcut_baseline_prune_manifest
  PRIVATE
    mlcut_pipeline)

add_executable(mlcut_solve_manifest
  apps/solve_manifest.cpp)

target_link_libraries(mlcut_solve_manifest
  PRIVATE
    mlcut_pipeline)

add_executable(mlcut_solver_benchmark_v2
  apps/solver_benchmark_v2.cpp)

target_link_libraries(mlcut_solver_benchmark_v2
  PRIVATE
    mlcut_pipeline)
