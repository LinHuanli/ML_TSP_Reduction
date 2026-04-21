#include "mlcut/ml/liblinear_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <linear.h>

#include "mlcut/parallel/parallel_for.h"

namespace mlcut::ml {

namespace {

void SilenceLibLinear(const char*) {}

int SolverType(LinearModelKind kind) {
  switch (kind) {
    case LinearModelKind::kLogisticRegression:
      return L2R_LR;
    case LinearModelKind::kLinearSvm:
      return L2R_L2LOSS_SVC_DUAL;
  }
  return L2R_LR;
}

std::vector<feature_node> MakeRowNodes(const float* row,
                                       std::size_t col_count) {
  std::vector<feature_node> nodes;
  nodes.reserve(col_count + 1);
  for (std::size_t col = 0; col < col_count; ++col) {
    nodes.push_back(feature_node{
        .index = static_cast<int>(col + 1),
        .value = static_cast<double>(row[col]),
    });
  }
  nodes.push_back(feature_node{-1, 0.0});
  return nodes;
}

}  // namespace

LibLinearModel::~LibLinearModel() {
  if (model_ != nullptr) {
    free_and_destroy_model(&model_);
  }
}

LibLinearModel::LibLinearModel(LibLinearModel&& other) noexcept
    : model_(other.model_), feature_count_(other.feature_count_) {
  other.model_ = nullptr;
  other.feature_count_ = 0;
}

LibLinearModel& LibLinearModel::operator=(LibLinearModel&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (model_ != nullptr) {
    free_and_destroy_model(&model_);
  }
  model_ = other.model_;
  feature_count_ = other.feature_count_;
  other.model_ = nullptr;
  other.feature_count_ = 0;
  return *this;
}

LibLinearModel LibLinearModel::Train(const DenseDataset& dataset,
                                     const LinearTrainOptions& options) {
  if (dataset.rows == 0 || dataset.cols == 0) {
    throw std::runtime_error("liblinear 训练数据为空");
  }
  if (dataset.features.size() != dataset.rows * dataset.cols ||
      dataset.labels.size() != dataset.rows) {
    throw std::runtime_error("liblinear 训练数据维度不匹配");
  }

  set_print_string_function(&SilenceLibLinear);

  // liblinear 训练内核在当前系统库上仍是单线程，但样本展开和校验预测可以并行。
  std::vector<std::vector<feature_node>> rows(dataset.rows);
  mlcut::parallel::ParallelFor(
      dataset.rows, std::max<std::size_t>(1, options.thread_count),
      [&](std::size_t row) {
        rows[row] =
            MakeRowNodes(dataset.features.data() + row * dataset.cols, dataset.cols);
      });
  std::vector<feature_node*> row_ptrs(dataset.rows, nullptr);
  for (std::size_t row = 0; row < dataset.rows; ++row) {
    row_ptrs[row] = rows[row].data();
  }

  std::vector<double> labels(dataset.labels.begin(), dataset.labels.end());

  problem prob{};
  prob.l = static_cast<int>(dataset.rows);
  prob.n = static_cast<int>(dataset.cols);
  prob.y = labels.data();
  prob.x = row_ptrs.data();
  prob.bias = -1.0;

  parameter param{};
  param.solver_type = SolverType(options.kind);
  param.eps = 0.01;
  param.C = options.c_value;
  std::array<int, 2> weight_labels{1, 0};
  std::array<double, 2> weights{options.positive_class_weight,
                                options.negative_class_weight};
  param.nr_weight = 2;
  param.weight_label = weight_labels.data();
  param.weight = weights.data();
  param.p = 0.1;
  param.nu = 0.5;
  param.init_sol = nullptr;
  param.regularize_bias = 1;
  param.w_recalc = false;

  if (const char* error = check_parameter(&prob, &param); error != nullptr) {
    throw std::runtime_error(std::string("liblinear 参数错误: ") + error);
  }
  if (options.use_probability && options.kind == LinearModelKind::kLinearSvm) {
    throw std::runtime_error("Linear SVM 概率输出暂不由 liblinear 原生提供");
  }

  LibLinearModel output;
  output.model_ = train(&prob, &param);
  output.feature_count_ = dataset.cols;
  if (output.model_ == nullptr) {
    throw std::runtime_error("liblinear 训练失败");
  }
  return output;
}

LibLinearModel LibLinearModel::Train(const DenseDataset& dataset,
                                     LinearModelKind kind,
                                     double c_value,
                                     bool use_probability) {
  LinearTrainOptions options;
  options.kind = kind;
  options.c_value = c_value;
  options.use_probability = use_probability;
  return Train(dataset, options);
}

void LibLinearModel::Save(const std::filesystem::path& path) const {
  if (model_ == nullptr) {
    throw std::runtime_error("liblinear 模型为空，不能保存");
  }
  if (save_model(path.string().c_str(), model_) != 0) {
    throw std::runtime_error("保存 liblinear 模型失败: " + path.string());
  }
}

LibLinearModel LibLinearModel::Load(const std::filesystem::path& path) {
  LibLinearModel model;
  model.model_ = load_model(path.string().c_str());
  if (model.model_ == nullptr) {
    throw std::runtime_error("加载 liblinear 模型失败: " + path.string());
  }
  model.feature_count_ = static_cast<std::size_t>(get_nr_feature(model.model_));
  return model;
}

std::vector<double> LibLinearModel::PredictScores(
    const std::vector<float>& features,
    std::size_t row_count,
    std::size_t col_count,
    std::size_t thread_count) const {
  if (model_ == nullptr) {
    throw std::runtime_error("liblinear 模型为空，无法预测");
  }
  if (col_count != feature_count_) {
    throw std::runtime_error("liblinear 预测特征列数不匹配");
  }
  if (features.size() != row_count * col_count) {
    throw std::runtime_error("liblinear 预测数据维度不匹配");
  }
  std::vector<double> outputs(row_count, 0.0);
  const int class_count = get_nr_class(model_);
  std::vector<int> class_labels(static_cast<std::size_t>(class_count), 0);
  if (class_count > 0) {
    get_labels(model_, class_labels.data());
  }
  const bool use_probability = check_probability_model(model_) != 0;
  const auto positive_iter = std::find(class_labels.begin(), class_labels.end(), 1);
  const std::size_t positive_index =
      use_probability
          ? static_cast<std::size_t>(std::distance(class_labels.begin(), positive_iter))
          : 0;
  if (use_probability && positive_iter == class_labels.end()) {
    throw std::runtime_error("liblinear 概率模型中找不到正类标签 1");
  }

  mlcut::parallel::ParallelFor(
      row_count, std::max<std::size_t>(1, thread_count), [&](std::size_t row) {
        auto nodes = MakeRowNodes(features.data() + row * col_count, col_count);
        double score = 0.0;
        if (use_probability) {
          std::vector<double> probabilities(static_cast<std::size_t>(class_count));
          predict_probability(model_, nodes.data(), probabilities.data());
          score = probabilities[positive_index];
        } else {
          std::vector<double> values(static_cast<std::size_t>(class_count), 0.0);
          predict_values(model_, nodes.data(), values.data());
          if (class_count == 2) {
            const double margin = values.front();
            score = class_labels.front() == 1 ? margin : -margin;
          } else {
            score = values.front();
          }
        }
        outputs[row] = score;
      });
  return outputs;
}

}  // namespace mlcut::ml
