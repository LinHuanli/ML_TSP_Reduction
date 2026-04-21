#include <iostream>
#include <stdexcept>
#include <string>

#include "mlcut/pipeline/training.h"

int main(int argc, char** argv) {
  try {
    mlcut::pipeline::TrainModelOptions options;
    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--run-id" && index + 1 < argc) {
        options.run_id = argv[++index];
      } else if (arg == "--train-manifest" && index + 1 < argc) {
        options.train_manifest = argv[++index];
      } else if (arg == "--val-manifest" && index + 1 < argc) {
        options.val_manifest = argv[++index];
      } else if (arg == "--candidate-mode" && index + 1 < argc) {
        options.candidate_mode =
            mlcut::pipeline::ParseCandidateMode(argv[++index])
                .value_or(mlcut::pipeline::CandidateMode::kAlpha);
      } else if (arg == "--model" && index + 1 < argc) {
        options.model_kind =
            mlcut::pipeline::ParseModelKind(argv[++index])
                .value_or(mlcut::pipeline::ModelKind::kLogisticRegression);
      } else if (arg == "--threads" && index + 1 < argc) {
        options.thread_count = static_cast<std::size_t>(std::stoull(argv[++index]));
      } else if (arg == "--c" && index + 1 < argc) {
        options.c_value = std::stod(argv[++index]);
      } else if (arg == "--xgb-rounds" && index + 1 < argc) {
        options.xgb_boosting_rounds = std::stoi(argv[++index]);
      } else if (arg == "--xgb-depth" && index + 1 < argc) {
        options.xgb_max_depth = std::stoi(argv[++index]);
      } else if (arg == "--xgb-eta" && index + 1 < argc) {
        options.xgb_eta = std::stod(argv[++index]);
      } else if (arg == "--xgb-cache-root" && index + 1 < argc) {
        options.xgb_cache_root = argv[++index];
      } else {
        throw std::runtime_error("未知参数: " + arg);
      }
    }

    const auto summary = mlcut::pipeline::TrainModelRun(options);
    std::cout << "train 完成: run_id=" << summary.run_id
              << ", train_instances=" << summary.train_instance_count
              << ", val_instances=" << summary.val_instance_count
              << ", train_rows=" << summary.train_rows
              << ", val_rows=" << summary.val_rows
              << ", train_ap=" << summary.train_average_precision
              << ", val_ap=" << summary.val_average_precision
              << ", ap_gap=" << summary.average_precision_gap << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "train 失败: " << error.what() << '\n';
    return 1;
  }
}
