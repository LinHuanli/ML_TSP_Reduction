#include "mlcut/candidate/lkh_runner.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "mlcut/base/filesystem.h"
#include "mlcut/base/process.h"

namespace mlcut::candidate {

namespace {

std::filesystem::path Abs(const std::filesystem::path& path) {
  return std::filesystem::absolute(path);
}

}  // namespace

std::string ToLkhKeyword(CandidateSetKind kind) {
  switch (kind) {
    case CandidateSetKind::kAlpha:
      return "ALPHA";
    case CandidateSetKind::kPopmusic:
      return "POPMUSIC";
  }
  return "ALPHA";
}

CandidateGraph ParseLkhCandidateFile(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("无法打开 LKH 候选文件: " + path.string());
  }
  std::uint32_t node_count = 0;
  in >> node_count;
  if (node_count == 0) {
    throw std::runtime_error("LKH 候选文件节点数非法: " + path.string());
  }

  std::vector<std::vector<std::pair<std::uint32_t, std::int32_t>>> adjacency(node_count);
  std::vector<std::uint32_t> mst_parent(node_count, node_count);

  while (true) {
    int node_id = -1;
    in >> node_id;
    if (!in) {
      break;
    }
    if (node_id == -1) {
      break;
    }
    int dad = 0;
    int count = 0;
    in >> dad >> count;
    const std::uint32_t node = static_cast<std::uint32_t>(node_id - 1);
    mst_parent.at(node) =
        dad <= 0 ? node_count : static_cast<std::uint32_t>(dad - 1);
    for (int edge_index = 0; edge_index < count; ++edge_index) {
      int to = 0;
      int alpha = 0;
      in >> to >> alpha;
      adjacency.at(node).push_back(
          {static_cast<std::uint32_t>(to - 1), static_cast<std::int32_t>(alpha)});
    }
    std::sort(adjacency.at(node).begin(), adjacency.at(node).end(),
              [](const auto& lhs, const auto& rhs) {
                return std::tie(lhs.first, lhs.second) < std::tie(rhs.first, rhs.second);
              });
  }

  std::vector<std::uint64_t> offsets(node_count + 1, 0);
  for (std::uint32_t node = 0; node < node_count; ++node) {
    offsets[node + 1] = offsets[node] + adjacency[node].size();
  }
  std::vector<std::uint32_t> neighbors;
  std::vector<std::int32_t> alphas;
  neighbors.reserve(offsets.back());
  alphas.reserve(offsets.back());
  for (std::uint32_t node = 0; node < node_count; ++node) {
    for (const auto& [to, alpha] : adjacency[node]) {
      neighbors.push_back(to);
      alphas.push_back(alpha);
    }
  }
  return CandidateGraph(node_count, std::move(offsets), std::move(neighbors),
                        std::move(alphas), std::move(mst_parent));
}

CandidateGraph RunLkhAndLoadCandidates(const LkhCandidateOptions& options) {
  if (!options.overwrite && std::filesystem::exists(options.candidate_binary_file)) {
    return CandidateGraph::ReadBinary(options.candidate_binary_file);
  }

  mlcut::base::EnsureDirectory(options.parameter_file.parent_path());
  mlcut::base::EnsureDirectory(options.candidate_text_file.parent_path());
  mlcut::base::EnsureDirectory(options.candidate_binary_file.parent_path());
  mlcut::base::EnsureDirectory(options.log_file.parent_path());
  const std::filesystem::path scratch_dir =
      options.candidate_text_file.parent_path() /
      (options.problem_file.stem().string() + ".lkh_work");
  mlcut::base::EnsureDirectory(scratch_dir);

  std::ostringstream par;
  par << "PROBLEM_FILE = " << Abs(options.problem_file).string() << '\n';
  par << "CANDIDATE_FILE = " << Abs(options.candidate_text_file).string() << '\n';
  par << "CANDIDATE_SET_TYPE = " << ToLkhKeyword(options.kind) << '\n';
  par << "MAX_CANDIDATES = " << options.max_candidates << " SYMMETRIC\n";
  par << "MAX_TRIALS = 0\n";
  // LKH 要求 RUNS 为正数，但 TOTAL_TIME_LIMIT = 0 会让主搜索循环立刻退出，
  // 从而只保留候选集预处理阶段，避免为生成候选图额外跑完整启发式。
  par << "RUNS = 1\n";
  par << "TOTAL_TIME_LIMIT = 0\n";
  par << "TRACE_LEVEL = " << options.trace_level << '\n';
  if (options.kind == CandidateSetKind::kPopmusic) {
    par << "POPMUSIC_SAMPLE_SIZE = 10\n";
    par << "POPMUSIC_SOLUTIONS = 30\n";
    par << "POPMUSIC_MAX_NEIGHBORS = 5\n";
    par << "POPMUSIC_TRIALS = 0\n";
  }
  mlcut::base::AtomicWriteText(options.parameter_file, par.str());

  const std::string command =
      mlcut::base::ShellQuote(Abs(options.lkh_binary).string()) + " " +
      mlcut::base::ShellQuote(Abs(options.parameter_file).string()) + " > " +
      mlcut::base::ShellQuote(Abs(options.log_file).string()) + " 2>&1";
  if (mlcut::base::RunCommand(command, scratch_dir) != 0) {
    throw std::runtime_error("LKH 候选集生成失败，请检查日志: " +
                             options.log_file.string());
  }
  const CandidateGraph graph = ParseLkhCandidateFile(options.candidate_text_file);
  graph.WriteBinary(options.candidate_binary_file);
  return graph;
}

}  // namespace mlcut::candidate
