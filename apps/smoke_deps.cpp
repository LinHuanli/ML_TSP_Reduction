#include <filesystem>
#include <iostream>
#include <vector>

#include "mlcut/ml/liblinear_model.h"
#include "mlcut/ml/xgboost_model.h"
#include "mlcut/solver/ortools_smoke.h"

int main() {
  mlcut::ml::DenseDataset dataset;
  dataset.rows = 4;
  dataset.cols = 2;
  dataset.features = {
      0.0f, 0.0f,
      0.0f, 1.0f,
      1.0f, 0.0f,
      1.0f, 1.0f,
  };
  dataset.labels = {0, 0, 1, 1};

  auto lr_model = mlcut::ml::LibLinearModel::Train(
      dataset, mlcut::ml::LinearModelKind::kLogisticRegression, 1.0, true);
  const auto lr_scores = lr_model.PredictScores(dataset.features, dataset.rows, dataset.cols);

  auto svm_model = mlcut::ml::LibLinearModel::Train(
      dataset, mlcut::ml::LinearModelKind::kLinearSvm, 1.0, false);
  const auto svm_scores =
      svm_model.PredictScores(dataset.features, dataset.rows, dataset.cols);

  auto xgb_model =
      mlcut::ml::XgBoostModel::TrainBinaryLogistic(dataset, 8, 2, 0.3, 1);
  const std::filesystem::path model_path =
      std::filesystem::current_path() / "cache" / "tmp" / "xgb_smoke_model.json";
  std::filesystem::create_directories(model_path.parent_path());
  xgb_model.Save(model_path);
  const auto xgb_scores = xgb_model.Predict(dataset.features, dataset.rows, dataset.cols);

  const double ortools_value = mlcut::solver::SolveOrToolsSmokeModel();

  std::cout << "LR scores:";
  for (double value : lr_scores) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';

  std::cout << "SVM scores:";
  for (double value : svm_scores) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';

  std::cout << "XGBoost scores:";
  for (float value : xgb_scores) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';

  std::cout << "OR-Tools objective: " << ortools_value << '\n';
  return 0;
}

