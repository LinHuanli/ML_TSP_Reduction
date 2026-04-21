#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "mlcut/pipeline/training.h"

namespace mlcut::ml {

class StreamingLinearModel {
 public:
  StreamingLinearModel() = default;
  StreamingLinearModel(mlcut::pipeline::ModelKind model_kind,
                       std::vector<double> weights,
                       double bias);

  [[nodiscard]] mlcut::pipeline::ModelKind model_kind() const { return model_kind_; }
  [[nodiscard]] std::size_t feature_count() const { return weights_.size(); }

  void Save(const std::filesystem::path& path) const;
  static StreamingLinearModel Load(const std::filesystem::path& path);

  [[nodiscard]] std::vector<float> PredictScores(const std::vector<float>& features,
                                                 std::size_t row_count,
                                                 std::size_t col_count) const;

 private:
  mlcut::pipeline::ModelKind model_kind_ =
      mlcut::pipeline::ModelKind::kLogisticRegression;
  std::vector<double> weights_;
  double bias_ = 0.0;
};

}  // namespace mlcut::ml
