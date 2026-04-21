#pragma once

#include <cstddef>
#include <filesystem>

#include "mlcut/pipeline/artifacts.h"

namespace mlcut::pipeline {

struct PrepareManifestOptions {
  std::filesystem::path manifest_path;
  CandidateMode candidate_mode = CandidateMode::kAlpha;
  std::size_t thread_count = 1;
  int max_candidates = 32;
  std::size_t knn_k = 10;
  bool build_labels = true;
  bool overwrite = false;
};

struct PrepareManifestSummary {
  std::size_t instance_count = 0;
  std::size_t generated_candidate_count = 0;
  std::size_t generated_label_count = 0;
  std::size_t generated_feature_count = 0;
};

PrepareManifestSummary PrepareManifest(const PrepareManifestOptions& options);

}  // namespace mlcut::pipeline
