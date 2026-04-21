#include "mlcut/pipeline/training.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include "mlcut/base/filesystem.h"
#include "mlcut/data/manifest.h"
#include "mlcut/feature/feature_matrix.h"
#include "mlcut/label/edge_labels.h"
#include "mlcut/ml/liblinear_model.h"
#include "mlcut/ml/streaming_linear_model.h"
#include "mlcut/ml/xgboost_model.h"
#include "mlcut/parallel/parallel_for.h"
#include "mlcut/pipeline/metrics.h"

namespace mlcut::pipeline {

using json = nlohmann::json;

namespace {

constexpr std::size_t kStreamingLinearEpochs = 3;
constexpr std::size_t kStreamingTrainApSampleRows = 2'000'000;
constexpr std::size_t kStreamingTrainApSampleInstances = 512;
constexpr std::size_t kStreamingBatchInstancesPerThread = 8;
constexpr std::size_t kXgboostExternalBatchInstances = 128;
constexpr double kStreamingAdamBeta1 = 0.9;
constexpr double kStreamingAdamBeta2 = 0.999;
constexpr double kStreamingAdamEpsilon = 1e-8;
constexpr double kStreamingWeightDecay = 1e-6;
constexpr double kStreamingLogisticLearningRate = 0.05;
constexpr double kStreamingSvmLearningRate = 0.10;

struct LoadedDataset {
  mlcut::ml::DenseDataset dense;
  std::size_t instance_count = 0;
  std::size_t positive_count = 0;
};

struct ManifestDatasetSummary {
  std::size_t instance_count = 0;
  std::size_t row_count = 0;
  std::size_t positive_count = 0;
  std::size_t feature_count = 0;
};

class ManifestDenseBatchIterator {
 public:
  ManifestDenseBatchIterator(std::filesystem::path manifest_path,
                             CandidateMode candidate_mode,
                             std::size_t batch_instances)
      : catalog_(mlcut::data::ReadCatalogJson(manifest_path)),
        candidate_mode_(candidate_mode),
        batch_instances_(std::max<std::size_t>(1, batch_instances)) {}

  void Reset() { current_index_ = 0; }

  bool Next(mlcut::ml::DenseBatch* batch) {
    if (batch == nullptr) {
      throw std::runtime_error("DenseBatch 输出参数不能为空");
    }
    batch->features.clear();
    batch->labels.clear();
    batch->row_count = 0;
    batch->col_count = 0;
    if (current_index_ >= catalog_.entries.size()) {
      return false;
    }

    const std::size_t end =
        std::min<std::size_t>(catalog_.entries.size(),
                              current_index_ + batch_instances_);
    for (; current_index_ < end; ++current_index_) {
      const auto& entry = catalog_.entries[current_index_];
      const PreparedInstanceArtifacts artifacts =
          MakePreparedInstanceArtifacts(entry.split, catalog_.preset_name,
                                        entry.instance_id, candidate_mode_);
      auto features = mlcut::feature::FeatureMatrix::ReadBinary(
          ResolvePath(artifacts.feature_binary_file));
      auto labels = mlcut::label::EdgeLabels::ReadBinary(
          ResolvePath(artifacts.label_binary_file));
      if (features.rows() != labels.values().size()) {
        throw std::runtime_error("特征和标签行数不一致: " + entry.instance_id);
      }
      if (batch->col_count == 0) {
        batch->col_count = features.cols();
      } else if (batch->col_count != features.cols()) {
        throw std::runtime_error("外存 XGBoost 训练遇到不一致的特征维度");
      }
      batch->features.insert(batch->features.end(),
                             features.values().begin(), features.values().end());
      batch->labels.reserve(batch->labels.size() + labels.values().size());
      for (std::uint8_t value : labels.values()) {
        batch->labels.push_back(static_cast<float>(value));
      }
      batch->row_count += features.rows();
    }
    return batch->row_count != 0;
  }

 private:
  mlcut::data::DatasetCatalog catalog_;
  CandidateMode candidate_mode_;
  std::size_t batch_instances_ = 1;
  std::size_t current_index_ = 0;
};

double Sigmoid(double value) {
  if (value >= 0.0) {
    const double exp_neg = std::exp(-value);
    return 1.0 / (1.0 + exp_neg);
  }
  const double exp_pos = std::exp(value);
  return exp_pos / (1.0 + exp_pos);
}

bool UsesStreamingLinearBackend(const TrainModelOptions& options) {
  if (options.candidate_mode != CandidateMode::kComplete) {
    return false;
  }
  return options.model_kind == ModelKind::kLogisticRegression ||
         options.model_kind == ModelKind::kLinearSvm;
}

ModelBackend DefaultBackendForKind(ModelKind kind) {
  switch (kind) {
    case ModelKind::kLogisticRegression:
    case ModelKind::kLinearSvm:
      return ModelBackend::kLibLinear;
    case ModelKind::kXgboost:
      return ModelBackend::kXgboost;
  }
  return ModelBackend::kLibLinear;
}

std::filesystem::path ModelFilePathForBackend(const RunLayout& layout,
                                              ModelKind kind,
                                              ModelBackend backend) {
  switch (backend) {
    case ModelBackend::kLibLinear:
      return ModelFilePath(layout, "liblinear");
    case ModelBackend::kStreamingLinear:
      return ModelFilePath(layout, "linear.json");
    case ModelBackend::kXgboost:
      return ModelFilePath(layout, "xgb.json");
  }
  switch (kind) {
    case ModelKind::kXgboost:
      return ModelFilePath(layout, "xgb.json");
    case ModelKind::kLogisticRegression:
    case ModelKind::kLinearSvm:
      return ModelFilePath(layout, "liblinear");
  }
  return ModelFilePath(layout, "liblinear");
}

std::string MakeDefaultRunId(ModelKind kind, CandidateMode candidate_mode) {
  return mlcut::base::MakeTimestampString() + "_" + ToString(kind) + "_" +
         ToString(candidate_mode);
}

ManifestDatasetSummary SummarizeManifestDataset(
    const std::filesystem::path& manifest_path,
    CandidateMode candidate_mode) {
  const auto catalog = mlcut::data::ReadCatalogJson(manifest_path);
  ManifestDatasetSummary summary;
  summary.instance_count = catalog.entries.size();

  if (candidate_mode == CandidateMode::kComplete) {
    summary.feature_count = mlcut::feature::FeatureMatrix::CoreFeatureNames().size();
    for (const auto& entry : catalog.entries) {
      const std::size_t n = static_cast<std::size_t>(entry.size);
      summary.row_count += n * (n - 1) / 2;
      summary.positive_count += n;
    }
    return summary;
  }

  for (const auto& entry : catalog.entries) {
    const PreparedInstanceArtifacts artifacts =
        MakePreparedInstanceArtifacts(entry.split, catalog.preset_name,
                                      entry.instance_id, candidate_mode);
    const auto feature_header = mlcut::feature::FeatureMatrix::ReadBinaryHeader(
        ResolvePath(artifacts.feature_binary_file));
    const auto label_header = mlcut::label::EdgeLabels::ReadBinaryHeader(
        ResolvePath(artifacts.label_binary_file));
    if (feature_header.row_count != label_header.label_count) {
      throw std::runtime_error("特征和标签行数不一致: " + entry.instance_id);
    }
    if (summary.feature_count == 0) {
      summary.feature_count = static_cast<std::size_t>(feature_header.col_count);
    } else if (summary.feature_count !=
               static_cast<std::size_t>(feature_header.col_count)) {
      throw std::runtime_error("不同实例的特征列数不一致: " + entry.instance_id);
    }
    summary.row_count += static_cast<std::size_t>(label_header.label_count);
    const auto labels = mlcut::label::EdgeLabels::ReadBinary(
        ResolvePath(artifacts.label_binary_file));
    summary.positive_count += static_cast<std::size_t>(std::count_if(
        labels.values().begin(), labels.values().end(),
        [](std::uint8_t value) { return value != 0U; }));
  }
  return summary;
}

template <typename Fn>
void ForEachManifestBlock(const std::filesystem::path& manifest_path,
                          CandidateMode candidate_mode,
                          Fn&& fn) {
  const auto catalog = mlcut::data::ReadCatalogJson(manifest_path);
  for (const auto& entry : catalog.entries) {
    const PreparedInstanceArtifacts artifacts =
        MakePreparedInstanceArtifacts(entry.split, catalog.preset_name,
                                      entry.instance_id, candidate_mode);
    auto features = mlcut::feature::FeatureMatrix::ReadBinary(
        ResolvePath(artifacts.feature_binary_file));
    auto labels = mlcut::label::EdgeLabels::ReadBinary(
        ResolvePath(artifacts.label_binary_file));
    if (features.rows() != labels.values().size()) {
      throw std::runtime_error("特征和标签行数不一致: " + entry.instance_id);
    }
    fn(entry, features, labels);
  }
}

template <typename Fn>
void ForEachManifestBlockLimited(const std::filesystem::path& manifest_path,
                                 CandidateMode candidate_mode,
                                 std::size_t max_instances,
                                 Fn&& fn) {
  const auto catalog = mlcut::data::ReadCatalogJson(manifest_path);
  const std::size_t limit =
      std::min<std::size_t>(max_instances, catalog.entries.size());
  for (std::size_t index = 0; index < limit; ++index) {
    const auto& entry = catalog.entries[index];
    const PreparedInstanceArtifacts artifacts =
        MakePreparedInstanceArtifacts(entry.split, catalog.preset_name,
                                      entry.instance_id, candidate_mode);
    auto features = mlcut::feature::FeatureMatrix::ReadBinary(
        ResolvePath(artifacts.feature_binary_file));
    auto labels = mlcut::label::EdgeLabels::ReadBinary(
        ResolvePath(artifacts.label_binary_file));
    if (features.rows() != labels.values().size()) {
      throw std::runtime_error("特征和标签行数不一致: " + entry.instance_id);
    }
    fn(entry, features, labels);
  }
}

LoadedDataset LoadDatasetForManifest(const std::filesystem::path& manifest_path,
                                     CandidateMode candidate_mode,
                                     std::size_t thread_count) {
  (void)thread_count;
  const auto summary = SummarizeManifestDataset(manifest_path, candidate_mode);
  const auto catalog = mlcut::data::ReadCatalogJson(manifest_path);

  LoadedDataset loaded;
  loaded.instance_count = summary.instance_count;
  loaded.positive_count = summary.positive_count;
  loaded.dense.rows = summary.row_count;
  loaded.dense.cols = summary.feature_count;
  loaded.dense.features.reserve(summary.row_count * summary.feature_count);
  loaded.dense.labels.reserve(summary.row_count);

  for (const auto& entry : catalog.entries) {
    const PreparedInstanceArtifacts artifacts =
        MakePreparedInstanceArtifacts(entry.split, catalog.preset_name,
                                      entry.instance_id, candidate_mode);
    const auto features = mlcut::feature::FeatureMatrix::ReadBinary(
        ResolvePath(artifacts.feature_binary_file));
    const auto labels = mlcut::label::EdgeLabels::ReadBinary(
        ResolvePath(artifacts.label_binary_file));
    loaded.dense.features.insert(loaded.dense.features.end(),
                                 features.values().begin(), features.values().end());
    for (std::uint8_t value : labels.values()) {
      loaded.dense.labels.push_back(static_cast<int>(value));
    }
  }
  return loaded;
}

template <typename Predictor>
BinaryRankingMetrics EvaluateManifestWithPredictor(
    const std::filesystem::path& manifest_path,
    CandidateMode candidate_mode,
    const ManifestDatasetSummary& summary,
    Predictor&& predictor,
    std::size_t max_rows = 0,
    std::size_t max_instances = 0) {
  const bool use_sampling = max_rows > 0 && summary.row_count > max_rows;
  const std::size_t target_rows =
      use_sampling ? max_rows : summary.row_count;

  std::vector<float> scores;
  std::vector<std::uint8_t> labels;
  scores.reserve(target_rows);
  labels.reserve(target_rows);
  std::mt19937_64 rng(0);
  std::size_t seen = 0;

  auto consume_block =
      [&](const mlcut::data::ManifestEntry& /*entry*/,
          const mlcut::feature::FeatureMatrix& features,
          const mlcut::label::EdgeLabels& edge_labels) {
        const auto block_scores = predictor(features);
        if (block_scores.size() != edge_labels.values().size()) {
          throw std::runtime_error("评分与标签长度不一致");
        }
        for (std::size_t row = 0; row < block_scores.size(); ++row) {
          const float score = block_scores[row];
          const std::uint8_t label = edge_labels.values()[row];
          if (!use_sampling) {
            scores.push_back(score);
            labels.push_back(label);
            continue;
          }
          if (seen < max_rows) {
            scores.push_back(score);
            labels.push_back(label);
          } else {
            std::uniform_int_distribution<std::uint64_t> dist(
                0, static_cast<std::uint64_t>(seen));
            const std::uint64_t slot = dist(rng);
            if (slot < max_rows) {
              scores[static_cast<std::size_t>(slot)] = score;
              labels[static_cast<std::size_t>(slot)] = label;
            }
          }
          ++seen;
        }
      };

  if (max_instances > 0) {
    ForEachManifestBlockLimited(manifest_path, candidate_mode, max_instances,
                                consume_block);
  } else {
    ForEachManifestBlock(manifest_path, candidate_mode, consume_block);
  }

  return EvaluateBinaryRanking(scores, labels);
}

mlcut::ml::StreamingLinearModel TrainStreamingLinearModel(
    const std::filesystem::path& manifest_path,
    CandidateMode candidate_mode,
    const ManifestDatasetSummary& summary,
    const TrainModelOptions& options,
    double positive_weight,
    double negative_weight) {
  const std::size_t feature_count = summary.feature_count;
  Eigen::VectorXd params = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(feature_count + 1));
  Eigen::VectorXd first_moment = Eigen::VectorXd::Zero(params.size());
  Eigen::VectorXd second_moment = Eigen::VectorXd::Zero(params.size());
  const double base_learning_rate =
      options.model_kind == ModelKind::kLogisticRegression
          ? kStreamingLogisticLearningRate
          : kStreamingSvmLearningRate;
  std::uint64_t update_step = 0;
  const auto catalog = mlcut::data::ReadCatalogJson(manifest_path);
  const std::size_t batch_instances = std::max<std::size_t>(
      1, options.thread_count * kStreamingBatchInstancesPerThread);

  for (std::size_t epoch = 0; epoch < kStreamingLinearEpochs; ++epoch) {
    for (std::size_t begin = 0; begin < catalog.entries.size();
         begin += batch_instances) {
      const std::size_t end =
          std::min<std::size_t>(catalog.entries.size(), begin + batch_instances);
      const std::size_t task_count = end - begin;
      std::vector<Eigen::VectorXd> local_grads(
          task_count, Eigen::VectorXd::Zero(params.size()));
      std::vector<std::size_t> local_rows(task_count, 0);
      const Eigen::VectorXd current_params = params;

      mlcut::parallel::ParallelFor(
          task_count, std::max<std::size_t>(1, options.thread_count),
          [&](std::size_t task_index) {
            const auto& entry = catalog.entries[begin + task_index];
            const PreparedInstanceArtifacts artifacts =
                MakePreparedInstanceArtifacts(entry.split, catalog.preset_name,
                                              entry.instance_id, candidate_mode);
            const auto features = mlcut::feature::FeatureMatrix::ReadBinary(
                ResolvePath(artifacts.feature_binary_file));
            const auto edge_labels = mlcut::label::EdgeLabels::ReadBinary(
                ResolvePath(artifacts.label_binary_file));
            if (features.rows() != edge_labels.values().size()) {
              throw std::runtime_error("特征和标签行数不一致: " + entry.instance_id);
            }

            Eigen::VectorXd grad = Eigen::VectorXd::Zero(current_params.size());
            const std::size_t row_count = features.rows();
            const std::size_t col_count = features.cols();
            const auto& values = features.values();
            for (std::size_t row = 0; row < row_count; ++row) {
              const float* x = values.data() + row * col_count;
              double margin = current_params[static_cast<Eigen::Index>(feature_count)];
              for (std::size_t col = 0; col < col_count; ++col) {
                margin += current_params[static_cast<Eigen::Index>(col)] *
                          static_cast<double>(x[col]);
              }

              const bool positive = edge_labels.values()[row] != 0U;
              const double class_weight =
                  positive ? positive_weight : negative_weight;
              if (options.model_kind == ModelKind::kLogisticRegression) {
                const double target = positive ? 1.0 : 0.0;
                const double coeff =
                    options.c_value * class_weight * (Sigmoid(margin) - target);
                for (std::size_t col = 0; col < col_count; ++col) {
                  grad[static_cast<Eigen::Index>(col)] +=
                      coeff * static_cast<double>(x[col]);
                }
                grad[static_cast<Eigen::Index>(feature_count)] += coeff;
              } else {
                const double target = positive ? 1.0 : -1.0;
                const double slack = 1.0 - target * margin;
                if (slack <= 0.0) {
                  continue;
                }
                const double coeff =
                    -2.0 * options.c_value * class_weight * target * slack;
                for (std::size_t col = 0; col < col_count; ++col) {
                  grad[static_cast<Eigen::Index>(col)] +=
                      coeff * static_cast<double>(x[col]);
                }
                grad[static_cast<Eigen::Index>(feature_count)] += coeff;
              }
            }
            local_grads[task_index] = std::move(grad);
            local_rows[task_index] = row_count;
          });

      Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
      std::size_t total_rows = 0;
      for (std::size_t task_index = 0; task_index < task_count; ++task_index) {
        grad += local_grads[task_index];
        total_rows += local_rows[task_index];
      }
      if (total_rows == 0) {
        continue;
      }

      grad /= static_cast<double>(total_rows);
      for (std::size_t col = 0; col < feature_count; ++col) {
        grad[static_cast<Eigen::Index>(col)] +=
            kStreamingWeightDecay * params[static_cast<Eigen::Index>(col)];
      }

      ++update_step;
      first_moment =
          kStreamingAdamBeta1 * first_moment + (1.0 - kStreamingAdamBeta1) * grad;
      second_moment =
          kStreamingAdamBeta2 * second_moment +
          (1.0 - kStreamingAdamBeta2) * grad.array().square().matrix();

      const double bias_correction1 =
          1.0 - std::pow(kStreamingAdamBeta1, static_cast<double>(update_step));
      const double bias_correction2 =
          1.0 - std::pow(kStreamingAdamBeta2, static_cast<double>(update_step));
      const Eigen::VectorXd m_hat = first_moment / bias_correction1;
      const Eigen::VectorXd v_hat = second_moment / bias_correction2;

      params.array() -=
          base_learning_rate * m_hat.array() /
          (v_hat.array().sqrt() + kStreamingAdamEpsilon);
    }
  }

  std::vector<double> weights(feature_count, 0.0);
  for (std::size_t col = 0; col < feature_count; ++col) {
    weights[col] = params[static_cast<Eigen::Index>(col)];
  }
  return mlcut::ml::StreamingLinearModel(
      options.model_kind, std::move(weights),
      params[static_cast<Eigen::Index>(feature_count)]);
}

mlcut::ml::XgBoostModel TrainExternalMemoryXgboostModel(
    const std::filesystem::path& manifest_path,
    CandidateMode candidate_mode,
    const ManifestDatasetSummary& summary,
    const TrainModelOptions& options,
    double positive_weight,
    const std::filesystem::path& cache_prefix) {
  mlcut::ml::XgBoostTrainOptions train_options;
  train_options.boosting_rounds = options.xgb_boosting_rounds;
  train_options.max_depth = options.xgb_max_depth;
  train_options.eta = options.xgb_eta;
  train_options.scale_pos_weight = positive_weight;
  train_options.thread_count = options.thread_count;

  ManifestDenseBatchIterator iterator(
      manifest_path, candidate_mode, kXgboostExternalBatchInstances);
  return mlcut::ml::XgBoostModel::TrainBinaryLogisticExternalMemory(
      summary.feature_count, train_options, cache_prefix,
      [&iterator]() { iterator.Reset(); },
      [&iterator](mlcut::ml::DenseBatch* batch) { return iterator.Next(batch); });
}

void WriteTrainingOutputs(const RunLayout& layout,
                          const TrainModelOptions& options,
                          const ModelMetadata& metadata,
                          const TrainModelSummary& summary) {
  json metadata_json{
      {"run_id", metadata.run_id},
      {"model_kind", ToString(metadata.model_kind)},
      {"model_backend", ToString(metadata.model_backend)},
      {"candidate_mode", ToString(metadata.candidate_mode)},
      {"feature_count", metadata.feature_count},
      {"train_manifest", options.train_manifest.string()},
      {"val_manifest", options.val_manifest.string()},
      {"thread_count", options.thread_count},
      {"c_value", options.c_value},
      {"xgb_boosting_rounds", options.xgb_boosting_rounds},
      {"xgb_max_depth", options.xgb_max_depth},
      {"xgb_eta", options.xgb_eta},
  };
  json metrics_json{
      {"run_id", summary.run_id},
      {"train_instance_count", summary.train_instance_count},
      {"val_instance_count", summary.val_instance_count},
      {"train_rows", summary.train_rows},
      {"val_rows", summary.val_rows},
      {"train_positive_count", summary.train_positive_count},
      {"val_positive_count", summary.val_positive_count},
      {"train_negative_count", summary.train_negative_count},
      {"val_negative_count", summary.val_negative_count},
      {"train_positive_rate", summary.train_positive_rate},
      {"val_positive_rate", summary.val_positive_rate},
      {"positive_class_weight", summary.positive_class_weight},
      {"feature_count", summary.feature_count},
      {"train_average_precision", summary.train_average_precision},
      {"val_average_precision", summary.val_average_precision},
      {"average_precision_gap", summary.average_precision_gap},
  };
  mlcut::base::AtomicWriteText(ModelMetadataPath(layout), metadata_json.dump(2));
  mlcut::base::AtomicWriteText(ValidationMetricsPath(layout), metrics_json.dump(2));
}

std::filesystem::path ResolveXgboostCacheRoot(const TrainModelOptions& options,
                                              const RunLayout& layout) {
  if (!options.xgb_cache_root.empty()) {
    return options.xgb_cache_root;
  }
  if (const char* env = std::getenv("MLCUT_XGB_CACHE_ROOT")) {
    if (*env != '\0') {
      return std::filesystem::path(env);
    }
  }
  return layout.analysis_dir / "xgb_extmem_cache";
}

}  // namespace

std::string ToString(ModelKind kind) {
  switch (kind) {
    case ModelKind::kLogisticRegression:
      return "lr";
    case ModelKind::kLinearSvm:
      return "svm";
    case ModelKind::kXgboost:
      return "xgb";
  }
  return "lr";
}

std::optional<ModelKind> ParseModelKind(std::string_view text) {
  if (text == "lr") {
    return ModelKind::kLogisticRegression;
  }
  if (text == "svm") {
    return ModelKind::kLinearSvm;
  }
  if (text == "xgb") {
    return ModelKind::kXgboost;
  }
  return std::nullopt;
}

std::string ToString(ModelBackend backend) {
  switch (backend) {
    case ModelBackend::kLibLinear:
      return "liblinear";
    case ModelBackend::kStreamingLinear:
      return "streaming_linear";
    case ModelBackend::kXgboost:
      return "xgboost";
  }
  return "liblinear";
}

std::optional<ModelBackend> ParseModelBackend(std::string_view text) {
  if (text == "liblinear") {
    return ModelBackend::kLibLinear;
  }
  if (text == "streaming_linear") {
    return ModelBackend::kStreamingLinear;
  }
  if (text == "xgboost") {
    return ModelBackend::kXgboost;
  }
  return std::nullopt;
}

TrainModelSummary TrainModelRun(const TrainModelOptions& options) {
  if (options.train_manifest.empty() || options.val_manifest.empty()) {
    throw std::runtime_error("训练和验证 manifest 都不能为空");
  }
  if (options.thread_count == 0) {
    throw std::runtime_error("thread_count 必须大于 0");
  }

  const ManifestDatasetSummary train_summary_manifest =
      SummarizeManifestDataset(options.train_manifest, options.candidate_mode);
  const ManifestDatasetSummary val_summary_manifest =
      SummarizeManifestDataset(options.val_manifest, options.candidate_mode);
  if (train_summary_manifest.positive_count == 0 ||
      val_summary_manifest.positive_count == 0) {
    throw std::runtime_error("训练或验证集不包含正样本");
  }

  TrainModelSummary summary;
  summary.run_id =
      options.run_id.empty() ? MakeDefaultRunId(options.model_kind, options.candidate_mode)
                             : options.run_id;
  summary.train_instance_count = train_summary_manifest.instance_count;
  summary.val_instance_count = val_summary_manifest.instance_count;
  summary.train_rows = train_summary_manifest.row_count;
  summary.val_rows = val_summary_manifest.row_count;
  summary.train_positive_count = train_summary_manifest.positive_count;
  summary.val_positive_count = val_summary_manifest.positive_count;
  summary.feature_count = train_summary_manifest.feature_count;

  const std::size_t train_negative =
      summary.train_rows >= summary.train_positive_count
          ? summary.train_rows - summary.train_positive_count
          : 0;
  const std::size_t val_negative =
      summary.val_rows >= summary.val_positive_count
          ? summary.val_rows - summary.val_positive_count
          : 0;
  const double positive_weight =
      summary.train_positive_count == 0
          ? 1.0
          : static_cast<double>(train_negative) /
                static_cast<double>(summary.train_positive_count);
  summary.train_negative_count = train_negative;
  summary.val_negative_count = val_negative;
  summary.train_positive_rate =
      summary.train_rows == 0
          ? 0.0
          : static_cast<double>(summary.train_positive_count) /
                static_cast<double>(summary.train_rows);
  summary.val_positive_rate =
      summary.val_rows == 0
          ? 0.0
          : static_cast<double>(summary.val_positive_count) /
                static_cast<double>(summary.val_rows);
  summary.positive_class_weight = positive_weight;

  const RunLayout layout = MakeRunLayout(summary.run_id);
  mlcut::base::EnsureDirectory(layout.config_dir);
  mlcut::base::EnsureDirectory(layout.metrics_dir);
  mlcut::base::EnsureDirectory(layout.logs_dir);
  mlcut::base::EnsureDirectory(layout.artifacts_dir);
  mlcut::base::EnsureDirectory(layout.analysis_dir);

  const ModelBackend backend =
      UsesStreamingLinearBackend(options)
          ? ModelBackend::kStreamingLinear
          : DefaultBackendForKind(options.model_kind);

  if (backend == ModelBackend::kStreamingLinear) {
    auto model = TrainStreamingLinearModel(
        options.train_manifest, options.candidate_mode, train_summary_manifest, options,
        positive_weight, 1.0);
    model.Save(ModelFilePathForBackend(layout, options.model_kind, backend));
    const auto train_metrics = EvaluateManifestWithPredictor(
        options.train_manifest, options.candidate_mode, train_summary_manifest,
        [&](const mlcut::feature::FeatureMatrix& features) {
          return model.PredictScores(features.values(), features.rows(), features.cols());
        },
        kStreamingTrainApSampleRows,
        kStreamingTrainApSampleInstances);
    const auto val_metrics = EvaluateManifestWithPredictor(
        options.val_manifest, options.candidate_mode, val_summary_manifest,
        [&](const mlcut::feature::FeatureMatrix& features) {
          return model.PredictScores(features.values(), features.rows(), features.cols());
        });
    summary.train_average_precision = train_metrics.average_precision;
    summary.val_average_precision = val_metrics.average_precision;
    summary.average_precision_gap =
        summary.train_average_precision - summary.val_average_precision;
  } else if (options.candidate_mode == CandidateMode::kComplete &&
             options.model_kind == ModelKind::kXgboost) {
    const std::filesystem::path cache_root =
        ResolveXgboostCacheRoot(options, layout) / summary.run_id;
    std::error_code cache_ec;
    std::filesystem::remove_all(cache_root, cache_ec);
    std::filesystem::create_directories(cache_root);
    auto cleanup_cache = [&]() {
      std::error_code ec;
      std::filesystem::remove_all(cache_root, ec);
    };

    try {
      auto model = TrainExternalMemoryXgboostModel(
          options.train_manifest, options.candidate_mode, train_summary_manifest,
          options, positive_weight, cache_root / "train.cache");
      model.Save(ModelFilePathForBackend(layout, options.model_kind, backend));
      const auto train_metrics = EvaluateManifestWithPredictor(
          options.train_manifest, options.candidate_mode, train_summary_manifest,
          [&](const mlcut::feature::FeatureMatrix& features) {
            return model.Predict(features.values(), features.rows(), features.cols());
          },
          kStreamingTrainApSampleRows,
          kStreamingTrainApSampleInstances);
      const auto val_metrics = EvaluateManifestWithPredictor(
          options.val_manifest, options.candidate_mode, val_summary_manifest,
          [&](const mlcut::feature::FeatureMatrix& features) {
            return model.Predict(features.values(), features.rows(), features.cols());
          });
      summary.train_average_precision = train_metrics.average_precision;
      summary.val_average_precision = val_metrics.average_precision;
      summary.average_precision_gap =
          summary.train_average_precision - summary.val_average_precision;
      cleanup_cache();
    } catch (...) {
      cleanup_cache();
      throw;
    }
  } else {
    const LoadedDataset train =
        LoadDatasetForManifest(options.train_manifest, options.candidate_mode,
                               options.thread_count);
    std::vector<float> train_scores;
    switch (options.model_kind) {
      case ModelKind::kLogisticRegression: {
        mlcut::ml::LinearTrainOptions train_options;
        train_options.kind = mlcut::ml::LinearModelKind::kLogisticRegression;
        train_options.c_value = options.c_value;
        train_options.use_probability = true;
        train_options.positive_class_weight = positive_weight;
        train_options.negative_class_weight = 1.0;
        train_options.thread_count = options.thread_count;
        auto model = mlcut::ml::LibLinearModel::Train(train.dense, train_options);
        model.Save(ModelFilePathForBackend(layout, options.model_kind, backend));
        const auto train_raw_scores =
            model.PredictScores(train.dense.features, train.dense.rows, train.dense.cols,
                                options.thread_count);
        train_scores.assign(train_raw_scores.begin(), train_raw_scores.end());
        break;
      }
      case ModelKind::kLinearSvm: {
        mlcut::ml::LinearTrainOptions train_options;
        train_options.kind = mlcut::ml::LinearModelKind::kLinearSvm;
        train_options.c_value = options.c_value;
        train_options.use_probability = false;
        train_options.positive_class_weight = positive_weight;
        train_options.negative_class_weight = 1.0;
        train_options.thread_count = options.thread_count;
        auto model = mlcut::ml::LibLinearModel::Train(train.dense, train_options);
        model.Save(ModelFilePathForBackend(layout, options.model_kind, backend));
        const auto train_raw_scores =
            model.PredictScores(train.dense.features, train.dense.rows, train.dense.cols,
                                options.thread_count);
        train_scores.assign(train_raw_scores.begin(), train_raw_scores.end());
        break;
      }
      case ModelKind::kXgboost: {
        mlcut::ml::XgBoostTrainOptions train_options;
        train_options.boosting_rounds = options.xgb_boosting_rounds;
        train_options.max_depth = options.xgb_max_depth;
        train_options.eta = options.xgb_eta;
        train_options.scale_pos_weight = positive_weight;
        train_options.thread_count = options.thread_count;
        auto model =
            mlcut::ml::XgBoostModel::TrainBinaryLogistic(train.dense, train_options);
        model.Save(ModelFilePathForBackend(layout, options.model_kind, backend));
        train_scores =
            model.Predict(train.dense.features, train.dense.rows, train.dense.cols);
        break;
      }
    }

    const auto train_metrics =
        EvaluateBinaryRanking(train_scores, train.dense.labels);
    summary.train_average_precision = train_metrics.average_precision;

    const LoadedDataset val =
        LoadDatasetForManifest(options.val_manifest, options.candidate_mode,
                               options.thread_count);
    std::vector<float> val_scores;
    switch (options.model_kind) {
      case ModelKind::kLogisticRegression:
      case ModelKind::kLinearSvm: {
        auto model = mlcut::ml::LibLinearModel::Load(
            ModelFilePathForBackend(layout, options.model_kind, backend));
        const auto val_raw_scores =
            model.PredictScores(val.dense.features, val.dense.rows, val.dense.cols,
                                options.thread_count);
        val_scores.assign(val_raw_scores.begin(), val_raw_scores.end());
        break;
      }
      case ModelKind::kXgboost: {
        auto model = mlcut::ml::XgBoostModel::Load(
            ModelFilePathForBackend(layout, options.model_kind, backend));
        val_scores =
            model.Predict(val.dense.features, val.dense.rows, val.dense.cols);
        break;
      }
    }
    const auto val_metrics = EvaluateBinaryRanking(val_scores, val.dense.labels);
    summary.val_average_precision = val_metrics.average_precision;
    summary.average_precision_gap =
        summary.train_average_precision - summary.val_average_precision;
  }

  ModelMetadata metadata;
  metadata.run_id = summary.run_id;
  metadata.model_kind = options.model_kind;
  metadata.model_backend = backend;
  metadata.candidate_mode = options.candidate_mode;
  metadata.feature_count = summary.feature_count;
  WriteTrainingOutputs(layout, options, metadata, summary);
  return summary;
}

ModelMetadata ReadModelMetadata(std::string_view run_id) {
  const RunLayout layout = MakeRunLayout(run_id);
  const json root = json::parse(mlcut::base::ReadTextFile(ModelMetadataPath(layout)));
  ModelMetadata metadata;
  metadata.run_id = root.at("run_id").get<std::string>();
  metadata.model_kind = ParseModelKind(root.at("model_kind").get<std::string>())
                            .value_or(ModelKind::kLogisticRegression);
  if (root.contains("model_backend")) {
    metadata.model_backend =
        ParseModelBackend(root.at("model_backend").get<std::string>())
            .value_or(DefaultBackendForKind(metadata.model_kind));
  } else {
    metadata.model_backend = DefaultBackendForKind(metadata.model_kind);
  }
  metadata.candidate_mode =
      ParseCandidateMode(root.at("candidate_mode").get<std::string>())
          .value_or(CandidateMode::kAlpha);
  metadata.feature_count = root.at("feature_count").get<std::size_t>();
  return metadata;
}

std::vector<float> PredictScoresForRun(std::string_view run_id,
                                       const feature::FeatureMatrix& features) {
  const ModelMetadata metadata = ReadModelMetadata(run_id);
  const RunLayout layout = MakeRunLayout(run_id);
  switch (metadata.model_backend) {
    case ModelBackend::kLibLinear: {
      auto model = mlcut::ml::LibLinearModel::Load(
          ModelFilePathForBackend(layout, metadata.model_kind, metadata.model_backend));
      const auto scores =
          model.PredictScores(features.values(), features.rows(), features.cols());
      return std::vector<float>(scores.begin(), scores.end());
    }
    case ModelBackend::kStreamingLinear: {
      auto model = mlcut::ml::StreamingLinearModel::Load(
          ModelFilePathForBackend(layout, metadata.model_kind, metadata.model_backend));
      return model.PredictScores(features.values(), features.rows(), features.cols());
    }
    case ModelBackend::kXgboost: {
      auto model = mlcut::ml::XgBoostModel::Load(
          ModelFilePathForBackend(layout, metadata.model_kind, metadata.model_backend));
      return model.Predict(features.values(), features.rows(), features.cols());
    }
  }
  throw std::runtime_error("未知模型后端");
}

}  // namespace mlcut::pipeline
