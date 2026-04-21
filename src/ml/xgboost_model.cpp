#include "mlcut/ml/xgboost_model.h"

#include <functional>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

#include <xgboost/c_api.h>

namespace mlcut::ml {

using json = nlohmann::json;

namespace {

void CheckXgb(int code) {
  if (code != 0) {
    throw std::runtime_error(XGBGetLastError());
  }
}

DMatrixHandle MakeMatrix(const std::vector<float>& features,
                         std::size_t row_count,
                         std::size_t col_count) {
  DMatrixHandle matrix = nullptr;
  CheckXgb(XGDMatrixCreateFromMat(features.data(),
                                  row_count,
                                  col_count,
                                  std::numeric_limits<float>::quiet_NaN(),
                                  &matrix));
  return matrix;
}

std::string MakeArrayInterfaceJson(const float* data,
                                   std::size_t row_count,
                                   std::size_t col_count) {
  json root;
  root["data"] = {reinterpret_cast<std::uintptr_t>(data), true};
  root["typestr"] = "<f4";
  root["version"] = 3;
  root["strides"] = nullptr;
  if (col_count == 0) {
    root["shape"] = {row_count};
  } else {
    root["shape"] = {row_count, col_count};
  }
  root["mask"] = nullptr;
  return root.dump();
}

struct ExternalMemoryContext {
  std::function<void()> reset_iterator;
  std::function<bool(DenseBatch*)> next_batch;
  DenseBatch batch;
  DMatrixHandle proxy = nullptr;
  std::string feature_json;
  std::string label_json;
  std::string error_message;
};

void ResetExternalMemoryIterator(DataIterHandle handle) {
  auto* context = reinterpret_cast<ExternalMemoryContext*>(handle);
  context->batch = DenseBatch{};
  context->feature_json.clear();
  context->label_json.clear();
  context->error_message.clear();
  context->reset_iterator();
}

int NextExternalMemoryBatch(DataIterHandle handle) {
  auto* context = reinterpret_cast<ExternalMemoryContext*>(handle);
  try {
    context->batch = DenseBatch{};
    if (!context->next_batch(&context->batch)) {
      return 0;
    }
    if (context->batch.row_count == 0 || context->batch.col_count == 0) {
      context->error_message = "XGBoost external-memory batch is empty";
      return -1;
    }
    if (context->batch.features.size() !=
        context->batch.row_count * context->batch.col_count) {
      context->error_message =
          "XGBoost external-memory batch feature shape mismatch";
      return -1;
    }
    if (context->batch.labels.size() != context->batch.row_count) {
      context->error_message =
          "XGBoost external-memory batch label shape mismatch";
      return -1;
    }
    context->feature_json =
        MakeArrayInterfaceJson(context->batch.features.data(),
                               context->batch.row_count,
                               context->batch.col_count);
    context->label_json =
        MakeArrayInterfaceJson(context->batch.labels.data(),
                               context->batch.row_count, 0);
    int code = XGProxyDMatrixSetDataDense(context->proxy,
                                          context->feature_json.c_str());
    if (code != 0) {
      context->error_message = XGBGetLastError();
      return -1;
    }
    code = XGDMatrixSetInfoFromInterface(context->proxy, "label",
                                         context->label_json.c_str());
    if (code != 0) {
      context->error_message = XGBGetLastError();
      return -1;
    }
    return 1;
  } catch (const std::exception& error) {
    context->error_message = error.what();
    return -1;
  }
}

}  // namespace

XgBoostModel::~XgBoostModel() {
  if (booster_ != nullptr) {
    XGBoosterFree(reinterpret_cast<BoosterHandle>(booster_));
  }
}

XgBoostModel::XgBoostModel(XgBoostModel&& other) noexcept : booster_(other.booster_) {
  other.booster_ = nullptr;
}

XgBoostModel& XgBoostModel::operator=(XgBoostModel&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (booster_ != nullptr) {
    XGBoosterFree(reinterpret_cast<BoosterHandle>(booster_));
  }
  booster_ = other.booster_;
  other.booster_ = nullptr;
  return *this;
}

XgBoostModel XgBoostModel::TrainBinaryLogistic(const DenseDataset& dataset,
                                               const XgBoostTrainOptions& options) {
  if (dataset.features.size() != dataset.rows * dataset.cols ||
      dataset.labels.size() != dataset.rows) {
    throw std::runtime_error("XGBoost 训练数据维度不匹配");
  }
  DMatrixHandle dtrain = MakeMatrix(dataset.features, dataset.rows, dataset.cols);
  std::vector<float> labels(dataset.labels.begin(), dataset.labels.end());
  CheckXgb(XGDMatrixSetFloatInfo(dtrain, "label", labels.data(), labels.size()));

  BoosterHandle booster = nullptr;
  CheckXgb(XGBoosterCreate(&dtrain, 1, &booster));
  CheckXgb(XGBoosterSetParam(booster, "objective", "binary:logistic"));
  CheckXgb(XGBoosterSetParam(booster, "eval_metric", "logloss"));
  CheckXgb(XGBoosterSetParam(booster, "max_depth",
                             std::to_string(options.max_depth).c_str()));
  CheckXgb(XGBoosterSetParam(booster, "eta", std::to_string(options.eta).c_str()));
  CheckXgb(XGBoosterSetParam(
      booster, "scale_pos_weight", std::to_string(options.scale_pos_weight).c_str()));
  CheckXgb(XGBoosterSetParam(booster, "nthread",
                             std::to_string(options.thread_count).c_str()));
  CheckXgb(XGBoosterSetParam(booster, "verbosity", "0"));
  for (int round = 0; round < options.boosting_rounds; ++round) {
    CheckXgb(XGBoosterUpdateOneIter(booster, round, dtrain));
  }

  XGDMatrixFree(dtrain);
  XgBoostModel output;
  output.booster_ = booster;
  return output;
}

XgBoostModel XgBoostModel::TrainBinaryLogistic(const DenseDataset& dataset,
                                               int boosting_rounds,
                                               int max_depth,
                                               double eta,
                                               std::size_t thread_count) {
  XgBoostTrainOptions options;
  options.boosting_rounds = boosting_rounds;
  options.max_depth = max_depth;
  options.eta = eta;
  options.thread_count = thread_count;
  return TrainBinaryLogistic(dataset, options);
}

XgBoostModel XgBoostModel::TrainBinaryLogisticExternalMemory(
    std::size_t feature_count,
    const XgBoostTrainOptions& options,
    const std::filesystem::path& cache_prefix,
    const std::function<void()>& reset_iterator,
    const std::function<bool(DenseBatch*)>& next_batch) {
  if (feature_count == 0) {
    throw std::runtime_error("XGBoost external-memory 训练特征数必须大于 0");
  }
  if (!reset_iterator || !next_batch) {
    throw std::runtime_error("XGBoost external-memory 回调不能为空");
  }

  DMatrixHandle proxy = nullptr;
  CheckXgb(XGProxyDMatrixCreate(&proxy));

  ExternalMemoryContext context;
  context.reset_iterator = reset_iterator;
  context.next_batch = next_batch;
  context.proxy = proxy;

  DMatrixHandle dtrain = nullptr;
  const json config = {
      {"missing", -999999.0},
      {"cache_prefix", cache_prefix.string()},
      {"nthread", options.thread_count},
  };

  try {
    CheckXgb(XGDMatrixCreateFromCallback(
        reinterpret_cast<DataIterHandle>(&context), proxy,
        &ResetExternalMemoryIterator, &NextExternalMemoryBatch,
        config.dump().c_str(), &dtrain));
    if (!context.error_message.empty()) {
      throw std::runtime_error(context.error_message);
    }

    BoosterHandle booster = nullptr;
    CheckXgb(XGBoosterCreate(&dtrain, 1, &booster));
    CheckXgb(XGBoosterSetParam(booster, "objective", "binary:logistic"));
    CheckXgb(XGBoosterSetParam(booster, "eval_metric", "logloss"));
    CheckXgb(XGBoosterSetParam(booster, "tree_method", "hist"));
    CheckXgb(XGBoosterSetParam(booster, "max_bin", "256"));
    CheckXgb(XGBoosterSetParam(booster, "max_depth",
                               std::to_string(options.max_depth).c_str()));
    CheckXgb(XGBoosterSetParam(booster, "eta",
                               std::to_string(options.eta).c_str()));
    CheckXgb(XGBoosterSetParam(
        booster, "scale_pos_weight",
        std::to_string(options.scale_pos_weight).c_str()));
    CheckXgb(XGBoosterSetParam(
        booster, "nthread", std::to_string(options.thread_count).c_str()));
    CheckXgb(XGBoosterSetParam(booster, "verbosity", "0"));
    for (int round = 0; round < options.boosting_rounds; ++round) {
      CheckXgb(XGBoosterUpdateOneIter(booster, round, dtrain));
    }

    XGDMatrixFree(dtrain);
    XGDMatrixFree(proxy);
    XgBoostModel output;
    output.booster_ = booster;
    return output;
  } catch (...) {
    if (dtrain != nullptr) {
      XGDMatrixFree(dtrain);
    }
    if (proxy != nullptr) {
      XGDMatrixFree(proxy);
    }
    throw;
  }
}

void XgBoostModel::Save(const std::filesystem::path& path) const {
  if (booster_ == nullptr) {
    throw std::runtime_error("XGBoost 模型为空，不能保存");
  }
  CheckXgb(XGBoosterSaveModel(reinterpret_cast<BoosterHandle>(booster_),
                              path.string().c_str()));
}

XgBoostModel XgBoostModel::Load(const std::filesystem::path& path) {
  BoosterHandle booster = nullptr;
  CheckXgb(XGBoosterCreate(nullptr, 0, &booster));
  CheckXgb(XGBoosterLoadModel(booster, path.string().c_str()));
  XgBoostModel output;
  output.booster_ = booster;
  return output;
}

std::vector<float> XgBoostModel::Predict(const std::vector<float>& features,
                                         std::size_t row_count,
                                         std::size_t col_count) const {
  if (booster_ == nullptr) {
    throw std::runtime_error("XGBoost 模型为空，无法预测");
  }
  DMatrixHandle dmatrix = MakeMatrix(features, row_count, col_count);
  const bst_ulong* shape = nullptr;
  bst_ulong dim = 0;
  const float* result = nullptr;
  CheckXgb(XGBoosterPredictFromDMatrix(
      reinterpret_cast<BoosterHandle>(booster_),
      dmatrix,
      "{\"type\":0,\"training\":false,\"iteration_begin\":0,"
      "\"iteration_end\":0,\"strict_shape\":true}",
      &shape,
      &dim,
      &result));
  std::size_t element_count = 1;
  for (bst_ulong index = 0; index < dim; ++index) {
    element_count *= static_cast<std::size_t>(shape[index]);
  }
  std::vector<float> output(result, result + element_count);
  XGDMatrixFree(dmatrix);
  return output;
}

}  // namespace mlcut::ml
