#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "mlcut/ml/liblinear_model.h"

namespace mlcut::ml {

struct DenseBatch {
  std::vector<float> features;
  std::vector<float> labels;
  std::size_t row_count = 0;
  std::size_t col_count = 0;
};

struct XgBoostTrainOptions {
  int boosting_rounds = 64;
  int max_depth = 6;
  double eta = 0.1;
  double scale_pos_weight = 1.0;
  std::size_t thread_count = 1;
};

class XgBoostModel {
 public:
  XgBoostModel() = default;
  ~XgBoostModel();

  XgBoostModel(const XgBoostModel&) = delete;
  XgBoostModel& operator=(const XgBoostModel&) = delete;

  XgBoostModel(XgBoostModel&& other) noexcept;
  XgBoostModel& operator=(XgBoostModel&& other) noexcept;

  static XgBoostModel TrainBinaryLogistic(const DenseDataset& dataset,
                                          const XgBoostTrainOptions& options);

  static XgBoostModel TrainBinaryLogisticExternalMemory(
      std::size_t feature_count,
      const XgBoostTrainOptions& options,
      const std::filesystem::path& cache_prefix,
      const std::function<void()>& reset_iterator,
      const std::function<bool(DenseBatch*)>& next_batch);

  static XgBoostModel TrainBinaryLogistic(const DenseDataset& dataset,
                                          int boosting_rounds,
                                          int max_depth,
                                          double eta,
                                          std::size_t thread_count);

  void Save(const std::filesystem::path& path) const;
  static XgBoostModel Load(const std::filesystem::path& path);

  [[nodiscard]] std::vector<float> Predict(const std::vector<float>& features,
                                           std::size_t row_count,
                                           std::size_t col_count) const;

 private:
  void* booster_ = nullptr;
};

}  // namespace mlcut::ml
