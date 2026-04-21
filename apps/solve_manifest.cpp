#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlcut/base/filesystem.h"
#include "mlcut/candidate/candidate_graph.h"
#include "mlcut/data/manifest.h"
#include "mlcut/label/concorde_runner.h"
#include "mlcut/parallel/parallel_for.h"
#include "mlcut/pipeline/artifacts.h"
#include "mlcut/solver/benchmark.h"
#include "mlcut/tsp/instance.h"

namespace {

std::filesystem::path ResolveCandidatePath(const mlcut::data::ManifestEntry& entry,
                                           std::string_view dataset_tag,
                                           std::optional<mlcut::pipeline::CandidateMode> candidate_mode,
                                           std::string_view input_run_id,
                                           std::string_view prune_tag) {
  if (!input_run_id.empty()) {
    const auto layout = mlcut::pipeline::MakeRunLayout(input_run_id);
    return mlcut::pipeline::PrunedCandidatePath(layout, prune_tag, entry.split,
                                                entry.instance_id);
  }
  const auto artifacts = mlcut::pipeline::MakePreparedInstanceArtifacts(
      entry.split, dataset_tag, entry.instance_id, *candidate_mode);
  return mlcut::pipeline::ResolvePath(artifacts.candidate_binary_file);
}

std::int64_t ResolveExactObjective(const mlcut::data::ManifestEntry& entry,
                                   std::string_view dataset_tag) {
  const auto artifacts = mlcut::pipeline::MakePreparedInstanceArtifacts(
      entry.split, dataset_tag, entry.instance_id,
      mlcut::pipeline::CandidateMode::kAlpha);
  const auto tour = mlcut::label::ReadConcordeTour(
      mlcut::pipeline::ResolvePath(artifacts.tour_file));
  const auto instance =
      mlcut::tsp::TspInstance::ReadBinary(mlcut::pipeline::ResolvePath(entry.instance_path));
  return mlcut::solver::TourLength(instance, tour);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string output_run_id;
    std::string input_run_id;
    std::string prune_tag;
    std::optional<mlcut::pipeline::CandidateMode> candidate_mode;
    std::filesystem::path manifest_path;
    mlcut::solver::SolverOptions solver_options;
    std::size_t thread_count = 1;

    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--output-run-id" && index + 1 < argc) {
        output_run_id = argv[++index];
      } else if (arg == "--input-run-id" && index + 1 < argc) {
        input_run_id = argv[++index];
      } else if (arg == "--prune-tag" && index + 1 < argc) {
        prune_tag = argv[++index];
      } else if (arg == "--candidate-mode" && index + 1 < argc) {
        candidate_mode = mlcut::pipeline::ParseCandidateMode(argv[++index]);
      } else if (arg == "--manifest" && index + 1 < argc) {
        manifest_path = argv[++index];
      } else if (arg == "--solver" && index + 1 < argc) {
        solver_options.kind =
            mlcut::solver::ParseSolverKind(argv[++index])
                .value_or(mlcut::solver::SolverKind::kLkh);
      } else if (arg == "--threads" && index + 1 < argc) {
        thread_count = static_cast<std::size_t>(std::stoull(argv[++index]));
      } else if (arg == "--lkh-runs" && index + 1 < argc) {
        solver_options.lkh_runs = std::stoi(argv[++index]);
      } else if (arg == "--lkh-max-trials" && index + 1 < argc) {
        solver_options.lkh_max_trials = std::stoi(argv[++index]);
      } else if (arg == "--ortools-time-limit-sec" && index + 1 < argc) {
        solver_options.ortools_time_limit_sec = std::stoi(argv[++index]);
      } else {
        throw std::runtime_error("未知参数: " + arg);
      }
    }

    if (output_run_id.empty() || manifest_path.empty()) {
      throw std::runtime_error("必须同时提供 --output-run-id 和 --manifest");
    }
    if (input_run_id.empty() && !candidate_mode.has_value()) {
      throw std::runtime_error("评测基础候选图时必须提供 --candidate-mode");
    }
    if (!input_run_id.empty() && prune_tag.empty()) {
      throw std::runtime_error("评测剪枝候选图时必须同时提供 --input-run-id 和 --prune-tag");
    }

    const auto output_layout = mlcut::pipeline::MakeRunLayout(output_run_id);
    mlcut::base::EnsureDirectory(output_layout.metrics_dir);
    mlcut::base::EnsureDirectory(output_layout.logs_dir);

    const auto catalog = mlcut::data::ReadCatalogJson(manifest_path);
    std::vector<std::string> lines(catalog.entries.size());
    std::atomic<std::size_t> finished{0};

    mlcut::parallel::ParallelFor(
        catalog.entries.size(), thread_count, [&](std::size_t index) {
          const auto& entry = catalog.entries[index];
          const auto candidate_path = ResolveCandidatePath(
              entry, catalog.preset_name, candidate_mode, input_run_id, prune_tag);
          const auto instance =
              mlcut::tsp::TspInstance::ReadBinary(mlcut::pipeline::ResolvePath(entry.instance_path));
          const auto graph = mlcut::candidate::CandidateGraph::ReadBinary(candidate_path);
          const std::int64_t exact_objective =
              ResolveExactObjective(entry, catalog.preset_name);

          const std::filesystem::path work_dir =
              output_layout.logs_dir / mlcut::solver::ToString(solver_options.kind) /
              entry.split / entry.instance_id;
          mlcut::solver::SolverResult result;
          if (solver_options.kind == mlcut::solver::SolverKind::kLkh) {
            result = mlcut::solver::SolveWithLkh(
                instance, mlcut::pipeline::ResolvePath(entry.tsplib_path), graph, work_dir,
                solver_options);
          } else {
            result = mlcut::solver::SolveWithOrtoolsRouting(instance, graph, solver_options);
          }

          const double gap_pct =
              result.status == "ok" && exact_objective > 0
                  ? (static_cast<double>(result.objective - exact_objective) /
                     static_cast<double>(exact_objective)) *
                        100.0
                  : 0.0;
          std::ostringstream line;
          line << entry.instance_id << '\t' << entry.split << '\t' << instance.Size() << '\t'
               << graph.UniqueEdges().size() << '\t' << result.status << '\t'
               << result.objective << '\t' << exact_objective << '\t' << gap_pct << '\t'
               << result.wall_sec;
          lines[index] = line.str();

          const std::size_t current = finished.fetch_add(1) + 1;
          std::cout << "[solve] " << current << "/" << catalog.entries.size() << ' '
                    << entry.instance_id << '\n';
        });

    std::ostringstream out;
    out << "instance_id\tsplit\tnode_count\tcandidate_edge_count\tstatus\tobjective\t"
           "exact_objective\tgap_pct\twall_sec\n";
    for (const auto& line : lines) {
      out << line << '\n';
    }

    const std::string input_tag =
        input_run_id.empty() ? ("base_" + mlcut::pipeline::ToString(*candidate_mode))
                             : (input_run_id + "_" + prune_tag);
    mlcut::base::AtomicWriteText(
        output_layout.metrics_dir /
            ("solve_" + mlcut::solver::ToString(solver_options.kind) + "_" + input_tag +
             "_" + manifest_path.stem().string() + ".tsv"),
        out.str());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "solve manifest 失败: " << error.what() << '\n';
    return 1;
  }
}
