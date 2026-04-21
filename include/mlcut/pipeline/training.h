#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mlcut/feature/feature_matrix.h"
#include "mlcut/pipeline/artifacts.h"

namespace mlcut::pipeline {

enum class ModelKind {
  kLogisticRegression,
  kLinearSvm,
  kXgboost,
};

enum class ModelBackend {
  kLibLinear,
  kStreamingLinear,
  kXgboost,
};

std::string ToString(ModelKind kind);
std::optional<ModelKind> ParseModelKind(std::string_view text);
std::string ToString(ModelBackend backend);
std::optional<ModelBackend> ParseModelBackend(std::string_view text);

struct TrainModelOptions {
  std::string run_id;
  std::filesystem::path train_manifest;
  std::filesystem::path val_manifest;
  CandidateMode candidate_mode = CandidateMode::kAlpha;
  ModelKind model_kind = ModelKind::kLogisticRegression;
  std::size_t thread_count = 1;
  double c_value = 1.0;
  int xgb_boosting_rounds = 64;
  int xgb_max_depth = 6;
  double xgb_eta = 0.1;
  std::filesystem::path xgb_cache_root;
};

struct TrainModelSummary {
  std::string run_id;
  std::size_t train_instance_count = 0;
  std::size_t val_instance_count = 0;
  std::size_t train_rows = 0;
  std::size_t val_rows = 0;
  std::size_t train_positive_count = 0;
  std::size_t val_positive_count = 0;
  std::size_t train_negative_count = 0;
  std::size_t val_negative_count = 0;
  double train_positive_rate = 0.0;
  double val_positive_rate = 0.0;
  double positive_class_weight = 1.0;
  std::size_t feature_count = 0;
  double train_average_precision = 0.0;
  double val_average_precision = 0.0;
  double average_precision_gap = 0.0;
};

struct ModelMetadata {
  std::string run_id;
  ModelKind model_kind = ModelKind::kLogisticRegression;
  ModelBackend model_backend = ModelBackend::kLibLinear;
  CandidateMode candidate_mode = CandidateMode::kAlpha;
  std::size_t feature_count = 0;
};

TrainModelSummary TrainModelRun(const TrainModelOptions& options);

ModelMetadata ReadModelMetadata(std::string_view run_id);

std::vector<float> PredictScoresForRun(std::string_view run_id,
                                       const feature::FeatureMatrix& features);

}  // namespace mlcut::pipeline
