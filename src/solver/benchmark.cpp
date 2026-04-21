#include "mlcut/solver/benchmark.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "ortools/constraint_solver/routing.h"
#include "ortools/constraint_solver/routing_enums.pb.h"
#include "ortools/constraint_solver/routing_parameters.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"

#include "mlcut/base/filesystem.h"
#include "mlcut/base/process.h"

namespace mlcut::solver {

namespace {

std::filesystem::path Abs(const std::filesystem::path& path) {
  return std::filesystem::absolute(path);
}

double CurrentProcessCpuTimeSec() {
  timespec ts{};
  if (::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) {
    throw std::runtime_error("无法读取进程 CPU 时间");
  }
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

bool IsIntegerToken(const std::string& token) {
  if (token.empty()) {
    return false;
  }
  std::size_t start = token[0] == '-' ? 1 : 0;
  if (start >= token.size()) {
    return false;
  }
  return std::all_of(token.begin() + static_cast<std::ptrdiff_t>(start), token.end(),
                     [](unsigned char ch) { return std::isdigit(ch) != 0; });
}

double GapPct(std::int64_t objective, const std::optional<std::int64_t>& exact_objective) {
  if (!exact_objective.has_value() || *exact_objective <= 0) {
    return 0.0;
  }
  return (static_cast<double>(objective - *exact_objective) /
          static_cast<double>(*exact_objective)) *
         100.0;
}

void AppendTraceEvent(SolverResult* result,
                      std::int64_t objective,
                      double elapsed_cpu_sec,
                      double elapsed_wall_sec,
                      const std::optional<std::int64_t>& exact_objective,
                      bool record_trace) {
  const bool hit_optimal_now =
      exact_objective.has_value() && objective == *exact_objective;
  if (hit_optimal_now && !result->hit_optimal) {
    result->hit_optimal = true;
    result->time_to_opt_cpu_sec = elapsed_cpu_sec;
    result->time_to_opt_wall_sec = elapsed_wall_sec;
  }
  if (!record_trace) {
    return;
  }
  IncumbentEvent event;
  event.event_index = result->trace.size();
  event.elapsed_cpu_sec = elapsed_cpu_sec;
  event.elapsed_wall_sec = elapsed_wall_sec;
  event.objective = objective;
  event.gap_pct = GapPct(objective, exact_objective);
  event.hit_optimal_now = hit_optimal_now;
  result->trace.push_back(std::move(event));
}

std::optional<double> ReadLkhCpuTimeFromLog(const std::filesystem::path& log_file) {
  std::ifstream in(log_file);
  if (!in) {
    return std::nullopt;
  }
  std::string line;
  while (std::getline(in, line)) {
    constexpr std::string_view kPrefix = "Time.total = ";
    const std::size_t pos = line.find(kPrefix);
    if (pos == std::string::npos) {
      continue;
    }
    const std::size_t value_begin = pos + kPrefix.size();
    const std::size_t value_end = line.find(" sec", value_begin);
    if (value_end == std::string::npos) {
      continue;
    }
    return std::stod(line.substr(value_begin, value_end - value_begin));
  }
  return std::nullopt;
}

}  // namespace

std::string ToString(SolverKind kind) {
  switch (kind) {
    case SolverKind::kLkh:
      return "lkh";
    case SolverKind::kOrtoolsRouting:
      return "ortools_routing";
    case SolverKind::kOrtoolsCpSat:
      return "ortools_cp_sat";
  }
  return "lkh";
}

std::optional<SolverKind> ParseSolverKind(std::string_view text) {
  if (text == "lkh") {
    return SolverKind::kLkh;
  }
  if (text == "ortools" || text == "ortools_routing") {
    return SolverKind::kOrtoolsRouting;
  }
  if (text == "ortools_cp_sat" || text == "cp_sat" || text == "ortools_cp") {
    return SolverKind::kOrtoolsCpSat;
  }
  return std::nullopt;
}

std::int64_t TourLength(const tsp::TspInstance& instance,
                        std::span<const std::uint32_t> tour) {
  if (tour.empty()) {
    return 0;
  }
  std::int64_t length = 0;
  for (std::size_t index = 0; index < tour.size(); ++index) {
    const std::uint32_t u = tour[index];
    const std::uint32_t v = tour[(index + 1) % tour.size()];
    length += instance.Distance(u, v);
  }
  return length;
}

std::vector<std::uint32_t> ReadTsplibTourFile(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("无法读取 TSPLIB tour 文件: " + path.string());
  }

  std::vector<std::uint32_t> tour;
  bool in_tour_section = false;
  std::string token;
  while (in >> token) {
    if (!in_tour_section) {
      if (token == "TOUR_SECTION") {
        in_tour_section = true;
      }
      continue;
    }
    if (!IsIntegerToken(token)) {
      continue;
    }
    const int node = std::stoi(token);
    if (node == -1) {
      break;
    }
    if (node <= 0) {
      throw std::runtime_error("TSPLIB tour 节点编号非法: " + path.string());
    }
    tour.push_back(static_cast<std::uint32_t>(node - 1));
  }
  if (tour.empty()) {
    throw std::runtime_error("TSPLIB tour 文件不含 TOUR_SECTION: " + path.string());
  }
  return tour;
}

SolverResult SolveWithLkh(const tsp::TspInstance& instance,
                          const std::filesystem::path& problem_file,
                          const candidate::CandidateGraph& graph,
                          const std::filesystem::path& work_dir,
                          const SolverOptions& options) {
  mlcut::base::EnsureDirectory(work_dir);
  const std::filesystem::path candidate_file = work_dir / "candidates.txt";
  const std::filesystem::path parameter_file = work_dir / "solve.par";
  const std::filesystem::path output_tour_file = work_dir / "best.tour";
  const std::filesystem::path log_file = work_dir / "solver.log";

  graph.WriteLkhCandidateText(candidate_file);

  // benchmark 阶段不再让 LKH 自己重新构造候选集，而是显式喂入剪枝后的候选文件，
  // 从而把求解时间差异尽量归因到候选图本身。
  std::ostringstream par;
  par << "PROBLEM_FILE = " << Abs(problem_file).string() << '\n';
  par << "CANDIDATE_FILE = " << Abs(candidate_file).string() << '\n';
  par << "MAX_TRIALS = " << options.lkh_max_trials << '\n';
  par << "RUNS = " << options.lkh_runs << '\n';
  par << "TRACE_LEVEL = 0\n";
  par << "OUTPUT_TOUR_FILE = " << Abs(output_tour_file).string() << '\n';
  mlcut::base::AtomicWriteText(parameter_file, par.str());

  const std::string command =
      mlcut::base::ShellQuote(Abs(options.lkh_binary).string()) + " " +
      mlcut::base::ShellQuote(Abs(parameter_file).string()) + " > " +
      mlcut::base::ShellQuote(Abs(log_file).string()) + " 2>&1";

  const auto start = std::chrono::steady_clock::now();
  const int exit_code = mlcut::base::RunCommand(command, work_dir);
  const auto finish = std::chrono::steady_clock::now();

  SolverResult result;
  result.wall_sec =
      std::chrono::duration<double>(finish - start).count();
  result.status = exit_code == 0 ? "ok" : "error";
  result.cpu_sec = ReadLkhCpuTimeFromLog(log_file).value_or(0.0);
  if (exit_code != 0) {
    return result;
  }

  const auto tour = ReadTsplibTourFile(output_tour_file);
  result.objective = TourLength(instance, tour);
  return result;
}

SolverResult SolveWithOrtoolsRoutingTracked(
    const tsp::TspInstance& instance,
    const candidate::CandidateGraph* graph,
    std::optional<std::int64_t> exact_objective,
    const SolverOptions& options) {
  namespace orr = operations_research;
  using NodeIndex = orr::RoutingIndexManager::NodeIndex;

  orr::RoutingIndexManager manager(static_cast<int>(instance.Size()), 1, NodeIndex{0});
  orr::RoutingModel routing(manager);
  const int transit_callback = routing.RegisterTransitCallback(
      [&](std::int64_t from_index, std::int64_t to_index) -> std::int64_t {
        const std::size_t from = static_cast<std::size_t>(
            manager.IndexToNode(from_index).value());
        const std::size_t to = static_cast<std::size_t>(
            manager.IndexToNode(to_index).value());
        return instance.Distance(from, to);
      });
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback);

  if (graph != nullptr) {
    const std::int64_t end_index = routing.End(0);
    for (std::uint32_t node = 0; node < graph->NodeCount(); ++node) {
      const std::int64_t from_index =
          manager.NodeToIndex(NodeIndex{static_cast<int>(node)});
      std::unordered_set<std::int64_t> allowed;
      for (std::uint32_t neighbor : graph->Neighbors(node)) {
        if (neighbor == 0) {
          allowed.insert(end_index);
        } else {
          allowed.insert(manager.NodeToIndex(NodeIndex{static_cast<int>(neighbor)}));
        }
      }

      for (std::uint32_t candidate_node = 1; candidate_node < graph->NodeCount();
           ++candidate_node) {
        const std::int64_t to_index =
            manager.NodeToIndex(NodeIndex{static_cast<int>(candidate_node)});
        if (!allowed.contains(to_index)) {
          routing.NextVar(from_index)->RemoveValue(to_index);
        }
      }
      if (node != 0 && !allowed.contains(end_index)) {
        routing.NextVar(from_index)->RemoveValue(end_index);
      }
    }
  }

  auto search_parameters = orr::DefaultRoutingSearchParameters();
  search_parameters.set_first_solution_strategy(
      orr::FirstSolutionStrategy::PATH_CHEAPEST_ARC);
  search_parameters.set_local_search_metaheuristic(
      orr::LocalSearchMetaheuristic::GUIDED_LOCAL_SEARCH);
  search_parameters.mutable_time_limit()->set_seconds(options.ortools_time_limit_sec);
  search_parameters.set_log_search(false);

  SolverResult result;
  const auto wall_start = std::chrono::steady_clock::now();
  const double cpu_start = CurrentProcessCpuTimeSec();
  std::int64_t best_objective = std::numeric_limits<std::int64_t>::max();
  if (options.record_trace || exact_objective.has_value()) {
    routing.AddAtSolutionCallback([&]() {
      const std::int64_t objective = routing.CostVar()->Value();
      if (objective >= best_objective) {
        return;
      }
      best_objective = objective;
      const double elapsed_wall =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start)
              .count();
      const double elapsed_cpu = CurrentProcessCpuTimeSec() - cpu_start;
      AppendTraceEvent(&result, objective, elapsed_cpu, elapsed_wall, exact_objective,
                       options.record_trace);
    });
  }

  const orr::Assignment* solution = routing.SolveWithParameters(search_parameters);
  const auto wall_finish = std::chrono::steady_clock::now();
  const double cpu_finish = CurrentProcessCpuTimeSec();

  result.wall_sec =
      std::chrono::duration<double>(wall_finish - wall_start).count();
  result.cpu_sec = cpu_finish - cpu_start;
  if (solution == nullptr) {
    result.status = "no_solution";
    return result;
  }
  result.status = "ok";
  result.objective = solution->ObjectiveValue();
  if (exact_objective.has_value() && result.objective == *exact_objective &&
      !result.hit_optimal) {
    result.hit_optimal = true;
    result.time_to_opt_cpu_sec = result.cpu_sec;
    result.time_to_opt_wall_sec = result.wall_sec;
  }
  return result;
}

SolverResult SolveWithOrtoolsRouting(const tsp::TspInstance& instance,
                                     const candidate::CandidateGraph& graph,
                                     const SolverOptions& options) {
  return SolveWithOrtoolsRoutingTracked(instance, &graph, std::nullopt, options);
}

SolverResult SolveWithOrtoolsCpSat(
    const tsp::TspInstance& instance,
    const candidate::CandidateGraph* graph,
    std::optional<std::int64_t> exact_objective,
    const SolverOptions& options) {
  namespace ors = operations_research::sat;

  const int node_count = static_cast<int>(instance.Size());
  ors::CpModelBuilder model;
  ors::CircuitConstraint circuit = model.AddCircuitConstraint();
  const ors::BoolVar false_var = model.FalseVar();

  ors::LinearExpr objective_expr;
  for (int tail = 0; tail < node_count; ++tail) {
    circuit.AddArc(tail, tail, false_var);
    if (graph == nullptr) {
      for (int head = 0; head < node_count; ++head) {
        if (tail == head) {
          continue;
        }
        const ors::BoolVar arc = model.NewBoolVar();
        circuit.AddArc(tail, head, arc);
        objective_expr += instance.Distance(static_cast<std::size_t>(tail),
                                            static_cast<std::size_t>(head)) *
                          arc;
      }
      continue;
    }

    for (const std::uint32_t head : graph->Neighbors(static_cast<std::uint32_t>(tail))) {
      const ors::BoolVar arc = model.NewBoolVar();
      circuit.AddArc(tail, static_cast<int>(head), arc);
      objective_expr += instance.Distance(static_cast<std::size_t>(tail),
                                          static_cast<std::size_t>(head)) *
                        arc;
    }
  }
  model.Minimize(objective_expr);

  ors::SatParameters params;
  params.set_max_time_in_seconds(static_cast<double>(options.ortools_time_limit_sec));
  params.set_num_search_workers(1);
  params.set_log_search_progress(false);

  operations_research::sat::Model sat_model;
  sat_model.Add(ors::NewSatParameters(params));

  SolverResult result;
  const auto wall_start = std::chrono::steady_clock::now();
  const double cpu_start = CurrentProcessCpuTimeSec();
  if (options.record_trace || exact_objective.has_value()) {
    sat_model.Add(ors::NewFeasibleSolutionObserver(
        [&](const ors::CpSolverResponse& response) {
          const std::int64_t objective =
              static_cast<std::int64_t>(std::llround(response.objective_value()));
          const double elapsed_wall =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start)
                  .count();
          const double elapsed_cpu = CurrentProcessCpuTimeSec() - cpu_start;
          AppendTraceEvent(&result, objective, elapsed_cpu, elapsed_wall, exact_objective,
                           options.record_trace);
        }));
  }

  const ors::CpSolverResponse response = ors::SolveCpModel(model.Build(), &sat_model);
  const auto wall_finish = std::chrono::steady_clock::now();
  const double cpu_finish = CurrentProcessCpuTimeSec();

  result.wall_sec =
      std::chrono::duration<double>(wall_finish - wall_start).count();
  result.cpu_sec = cpu_finish - cpu_start;

  switch (response.status()) {
    case operations_research::sat::CpSolverStatus::OPTIMAL:
      result.status = "optimal";
      break;
    case operations_research::sat::CpSolverStatus::FEASIBLE:
      result.status = "feasible";
      break;
    case operations_research::sat::CpSolverStatus::INFEASIBLE:
      result.status = "infeasible";
      break;
    case operations_research::sat::CpSolverStatus::MODEL_INVALID:
      result.status = "model_invalid";
      break;
    case operations_research::sat::CpSolverStatus::UNKNOWN:
      result.status = "unknown";
      break;
    default:
      result.status = "unknown";
      break;
  }

  if (response.status() == operations_research::sat::CpSolverStatus::OPTIMAL ||
      response.status() == operations_research::sat::CpSolverStatus::FEASIBLE) {
    result.objective =
        static_cast<std::int64_t>(std::llround(response.objective_value()));
    if (exact_objective.has_value() && result.objective == *exact_objective &&
        !result.hit_optimal) {
      result.hit_optimal = true;
      result.time_to_opt_cpu_sec = result.cpu_sec;
      result.time_to_opt_wall_sec = result.wall_sec;
    }
  }
  return result;
}

}  // namespace mlcut::solver
