#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <linear.h>

namespace mlcut::ml {

enum class LinearModelKind {
  kLogisticRegression,
  kLinearSvm,
};

struct DenseDataset {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<float> features;
  std::vector<int> labels;
};

struct LinearTrainOptions {
  LinearModelKind kind = LinearModelKind::kLogisticRegression;
  double c_value = 1.0;
  bool use_probability = true;
  double positive_class_weight = 1.0;
  double negative_class_weight = 1.0;
  std::size_t thread_count = 1;
};

class LibLinearModel {
 public:
  LibLinearModel() = default;
  ~LibLinearModel();

  LibLinearModel(const LibLinearModel&) = delete;
  LibLinearModel& operator=(const LibLinearModel&) = delete;

  LibLinearModel(LibLinearModel&& other) noexcept;
  LibLinearModel& operator=(LibLinearModel&& other) noexcept;

  static LibLinearModel Train(const DenseDataset& dataset,
                              const LinearTrainOptions& options);

  static LibLinearModel Train(const DenseDataset& dataset,
                              LinearModelKind kind,
                              double c_value,
                              bool use_probability);

  void Save(const std::filesystem::path& path) const;
  static LibLinearModel Load(const std::filesystem::path& path);

  [[nodiscard]] std::vector<double> PredictScores(
      const std::vector<float>& features,
      std::size_t row_count,
      std::size_t col_count,
      std::size_t thread_count = 1) const;

 private:
  ::model* model_ = nullptr;
  std::size_t feature_count_ = 0;
};

}  // namespace mlcut::ml
