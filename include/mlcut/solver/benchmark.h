#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "mlcut/candidate/candidate_graph.h"
#include "mlcut/tsp/instance.h"

namespace mlcut::solver {

enum class SolverKind {
  kLkh,
  kOrtoolsRouting,
  kOrtoolsCpSat,
};

std::string ToString(SolverKind kind);
std::optional<SolverKind> ParseSolverKind(std::string_view text);

struct SolverOptions {
  SolverKind kind = SolverKind::kLkh;
  std::filesystem::path lkh_binary = "LKH-3.0.13/LKH";
  int lkh_runs = 4;
  int lkh_max_trials = 1000;
  int ortools_time_limit_sec = 30;
  bool record_trace = false;
};

struct IncumbentEvent {
  std::size_t event_index = 0;
  double elapsed_cpu_sec = 0.0;
  double elapsed_wall_sec = 0.0;
  std::int64_t objective = 0;
  double gap_pct = 0.0;
  bool hit_optimal_now = false;
};

struct SolverResult {
  std::string status = "unknown";
  std::int64_t objective = 0;
  double cpu_sec = 0.0;
  double wall_sec = 0.0;
  bool hit_optimal = false;
  double time_to_opt_cpu_sec = -1.0;
  double time_to_opt_wall_sec = -1.0;
  std::vector<IncumbentEvent> trace;
};

std::int64_t TourLength(const tsp::TspInstance& instance,
                        std::span<const std::uint32_t> tour);

std::vector<std::uint32_t> ReadTsplibTourFile(const std::filesystem::path& path);

SolverResult SolveWithLkh(const tsp::TspInstance& instance,
                          const std::filesystem::path& problem_file,
                          const candidate::CandidateGraph& graph,
                          const std::filesystem::path& work_dir,
                          const SolverOptions& options);

SolverResult SolveWithOrtoolsRouting(const tsp::TspInstance& instance,
                                     const candidate::CandidateGraph& graph,
                                     const SolverOptions& options);

SolverResult SolveWithOrtoolsRoutingTracked(
    const tsp::TspInstance& instance,
    const candidate::CandidateGraph* graph,
    std::optional<std::int64_t> exact_objective,
    const SolverOptions& options);

SolverResult SolveWithOrtoolsCpSat(
    const tsp::TspInstance& instance,
    const candidate::CandidateGraph* graph,
    std::optional<std::int64_t> exact_objective,
    const SolverOptions& options);

}  // namespace mlcut::solver
