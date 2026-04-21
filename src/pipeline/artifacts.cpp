#include "mlcut/pipeline/artifacts.h"

#include <stdexcept>

namespace mlcut::pipeline {

namespace {

std::string NormalizeDatasetTag(std::string_view dataset_tag) {
  if (dataset_tag.empty()) {
    return "default";
  }
  return std::string(dataset_tag);
}

}  // namespace

std::string ToString(CandidateMode mode) {
  switch (mode) {
    case CandidateMode::kComplete:
      return "complete";
    case CandidateMode::kAlpha:
      return "alpha";
    case CandidateMode::kPopmusic:
      return "popmusic";
    case CandidateMode::kUnion:
      return "union";
  }
  return "alpha";
}

std::optional<CandidateMode> ParseCandidateMode(std::string_view text) {
  if (text == "complete") {
    return CandidateMode::kComplete;
  }
  if (text == "alpha") {
    return CandidateMode::kAlpha;
  }
  if (text == "popmusic") {
    return CandidateMode::kPopmusic;
  }
  if (text == "union") {
    return CandidateMode::kUnion;
  }
  return std::nullopt;
}

PreparedInstanceArtifacts MakePreparedInstanceArtifacts(std::string_view split_name,
                                                        std::string_view dataset_tag,
                                                        std::string_view instance_id,
                                                        CandidateMode mode) {
  const std::string mode_name = ToString(mode);
  const std::string dataset_name = NormalizeDatasetTag(dataset_tag);
  const std::filesystem::path dataset_dir = std::filesystem::path("cache") / dataset_name;
  const std::filesystem::path split_dir = std::string(split_name);
  PreparedInstanceArtifacts artifacts;
  artifacts.candidate_parameter_file =
      dataset_dir / "candidates" / mode_name / split_dir /
      (std::string(instance_id) + ".par");
  artifacts.candidate_text_file =
      dataset_dir / "candidates" / mode_name / split_dir /
      (std::string(instance_id) + ".cand.txt");
  artifacts.candidate_binary_file =
      dataset_dir / "candidates" / mode_name / split_dir /
      (std::string(instance_id) + ".cand.bin");
  artifacts.candidate_log_file =
      dataset_dir / "candidates" / mode_name / split_dir /
      (std::string(instance_id) + ".log");
  artifacts.tour_file =
      dataset_dir / "tours" / "concorde" / split_dir /
      (std::string(instance_id) + ".tour");
  artifacts.tour_log_file =
      dataset_dir / "tours" / "concorde" / split_dir /
      (std::string(instance_id) + ".log");
  artifacts.label_binary_file =
      dataset_dir / "labels" / mode_name / split_dir /
      (std::string(instance_id) + ".labels.bin");
  artifacts.feature_binary_file =
      (mode == CandidateMode::kUnion ? dataset_dir / "features" / "core_plus_source"
                                     : dataset_dir / "features" / "core") /
      mode_name / split_dir / (std::string(instance_id) + ".feat.bin");
  return artifacts;
}

RunLayout MakeRunLayout(std::string_view run_id) {
  RunLayout layout;
  layout.root_dir = std::filesystem::path("results/runs") / std::string(run_id);
  layout.config_dir = layout.root_dir / "config";
  layout.metrics_dir = layout.root_dir / "metrics";
  layout.logs_dir = layout.root_dir / "logs";
  layout.artifacts_dir = layout.root_dir / "artifacts";
  layout.analysis_dir = layout.root_dir / "analysis";
  return layout;
}

std::filesystem::path ResolvePath(const std::filesystem::path& path) {
  if (path.is_absolute()) {
    return path;
  }
  return std::filesystem::absolute(path);
}

std::filesystem::path ModelFilePath(const RunLayout& layout, std::string_view extension) {
  return layout.artifacts_dir / ("model." + std::string(extension));
}

std::filesystem::path ModelMetadataPath(const RunLayout& layout) {
  return layout.config_dir / "model_metadata.json";
}

std::filesystem::path ValidationMetricsPath(const RunLayout& layout) {
  return layout.metrics_dir / "validation.json";
}

std::filesystem::path PrunedCandidatePath(const RunLayout& layout,
                                          std::string_view prune_tag,
                                          std::string_view split_name,
                                          std::string_view instance_id) {
  return layout.artifacts_dir / "pruned" / std::string(prune_tag) /
         std::string(split_name) /
         (std::string(instance_id) + ".cand.bin");
}

std::filesystem::path PruneMetricsPath(const RunLayout& layout,
                                       std::string_view prune_tag,
                                       std::string_view split_name) {
  return layout.metrics_dir /
         ("prune_" + std::string(prune_tag) + "_" + std::string(split_name) + ".tsv");
}

}  // namespace mlcut::pipeline
