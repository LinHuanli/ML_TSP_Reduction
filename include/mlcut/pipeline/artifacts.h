#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace mlcut::pipeline {

enum class CandidateMode {
  kComplete,
  kAlpha,
  kPopmusic,
  kUnion,
};

std::string ToString(CandidateMode mode);
std::optional<CandidateMode> ParseCandidateMode(std::string_view text);

struct PreparedInstanceArtifacts {
  std::filesystem::path candidate_parameter_file;
  std::filesystem::path candidate_text_file;
  std::filesystem::path candidate_binary_file;
  std::filesystem::path candidate_log_file;
  std::filesystem::path tour_file;
  std::filesystem::path tour_log_file;
  std::filesystem::path label_binary_file;
  std::filesystem::path feature_binary_file;
};

PreparedInstanceArtifacts MakePreparedInstanceArtifacts(std::string_view split_name,
                                                        std::string_view dataset_tag,
                                                        std::string_view instance_id,
                                                        CandidateMode mode);

struct RunLayout {
  std::filesystem::path root_dir;
  std::filesystem::path config_dir;
  std::filesystem::path metrics_dir;
  std::filesystem::path logs_dir;
  std::filesystem::path artifacts_dir;
  std::filesystem::path analysis_dir;
};

RunLayout MakeRunLayout(std::string_view run_id);

std::filesystem::path ResolvePath(const std::filesystem::path& path);

std::filesystem::path ModelFilePath(const RunLayout& layout, std::string_view extension);
std::filesystem::path ModelMetadataPath(const RunLayout& layout);
std::filesystem::path ValidationMetricsPath(const RunLayout& layout);
std::filesystem::path PrunedCandidatePath(const RunLayout& layout,
                                          std::string_view prune_tag,
                                          std::string_view split_name,
                                          std::string_view instance_id);
std::filesystem::path PruneMetricsPath(const RunLayout& layout,
                                       std::string_view prune_tag,
                                       std::string_view split_name);

}  // namespace mlcut::pipeline
