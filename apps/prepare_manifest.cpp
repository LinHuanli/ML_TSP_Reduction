#include <iostream>
#include <stdexcept>
#include <string>

#include "mlcut/pipeline/prepare.h"

int main(int argc, char** argv) {
  try {
    mlcut::pipeline::PrepareManifestOptions options;
    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--manifest" && index + 1 < argc) {
        options.manifest_path = argv[++index];
      } else if (arg == "--candidate-mode" && index + 1 < argc) {
        options.candidate_mode =
            mlcut::pipeline::ParseCandidateMode(argv[++index])
                .value_or(mlcut::pipeline::CandidateMode::kAlpha);
      } else if (arg == "--threads" && index + 1 < argc) {
        options.thread_count = static_cast<std::size_t>(std::stoull(argv[++index]));
      } else if (arg == "--max-candidates" && index + 1 < argc) {
        options.max_candidates = std::stoi(argv[++index]);
      } else if (arg == "--knn-k" && index + 1 < argc) {
        options.knn_k = static_cast<std::size_t>(std::stoull(argv[++index]));
      } else if (arg == "--skip-labels") {
        options.build_labels = false;
      } else if (arg == "--overwrite") {
        options.overwrite = true;
      } else {
        throw std::runtime_error("未知参数: " + arg);
      }
    }

    const auto summary = mlcut::pipeline::PrepareManifest(options);
    std::cout << "prepare 完成: instances=" << summary.instance_count
              << ", candidates=" << summary.generated_candidate_count
              << ", labels=" << summary.generated_label_count
              << ", features=" << summary.generated_feature_count << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "prepare 失败: " << error.what() << '\n';
    return 1;
  }
}
