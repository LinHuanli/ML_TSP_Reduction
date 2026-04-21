#include "mlcut/ml/streaming_linear_model.h"

#include <cmath>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "mlcut/base/filesystem.h"

namespace mlcut::ml {

using json = nlohmann::json;

namespace {

double Sigmoid(double value) {
  if (value >= 0.0) {
    const double exp_neg = std::exp(-value);
    return 1.0 / (1.0 + exp_neg);
  }
  const double exp_pos = std::exp(value);
  return exp_pos / (1.0 + exp_pos);
}

}  // namespace

StreamingLinearModel::StreamingLinearModel(
    mlcut::pipeline::ModelKind model_kind,
    std::vector<double> weights,
    double bias)
    : model_kind_(model_kind), weights_(std::move(weights)), bias_(bias) {}

void StreamingLinearModel::Save(const std::filesystem::path& path) const {
  json root{
      {"model_kind", mlcut::pipeline::ToString(model_kind_)},
      {"feature_count", weights_.size()},
      {"bias", bias_},
      {"weights", weights_},
  };
  mlcut::base::AtomicWriteText(path, root.dump(2));
}

StreamingLinearModel StreamingLinearModel::Load(const std::filesystem::path& path) {
  const json root = json::parse(mlcut::base::ReadTextFile(path));
  const auto model_kind =
      mlcut::pipeline::ParseModelKind(root.at("model_kind").get<std::string>())
          .value_or(mlcut::pipeline::ModelKind::kLogisticRegression);
  return StreamingLinearModel(
      model_kind,
      root.at("weights").get<std::vector<double>>(),
      root.at("bias").get<double>());
}

std::vector<float> StreamingLinearModel::PredictScores(
    const std::vector<float>& features,
    std::size_t row_count,
    std::size_t col_count) const {
  if (col_count != weights_.size()) {
    throw std::runtime_error("streaming linear 预测特征列数不匹配");
  }
  if (features.size() != row_count * col_count) {
    throw std::runtime_error("streaming linear 预测数据维度不匹配");
  }
  std::vector<float> scores(row_count, 0.0f);
  for (std::size_t row = 0; row < row_count; ++row) {
    const float* x = features.data() + row * col_count;
    double margin = bias_;
    for (std::size_t col = 0; col < col_count; ++col) {
      margin += weights_[col] * static_cast<double>(x[col]);
    }
    if (model_kind_ == mlcut::pipeline::ModelKind::kLogisticRegression) {
      scores[row] = static_cast<float>(Sigmoid(margin));
    } else {
      scores[row] = static_cast<float>(margin);
    }
  }
  return scores;
}

}  // namespace mlcut::ml
