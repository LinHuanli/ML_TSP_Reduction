#pragma once

#include <filesystem>
#include <string>

#include "mlcut/candidate/candidate_graph.h"

namespace mlcut::candidate {

enum class CandidateSetKind {
  kAlpha,
  kPopmusic,
};

struct LkhCandidateOptions {
  std::filesystem::path lkh_binary = "LKH-3.0.13/LKH";
  std::filesystem::path problem_file;
  std::filesystem::path parameter_file;
  std::filesystem::path candidate_text_file;
  std::filesystem::path candidate_binary_file;
  std::filesystem::path log_file;
  CandidateSetKind kind = CandidateSetKind::kAlpha;
  int max_candidates = 32;
  int trace_level = 0;
  bool overwrite = false;
};

std::string ToLkhKeyword(CandidateSetKind kind);

CandidateGraph RunLkhAndLoadCandidates(const LkhCandidateOptions& options);

CandidateGraph ParseLkhCandidateFile(const std::filesystem::path& path);

}  // namespace mlcut::candidate

