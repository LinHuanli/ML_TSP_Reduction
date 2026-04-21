#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "mlcut/base/filesystem.h"
#include "mlcut/base/process.h"
#include "mlcut/candidate/candidate_graph.h"
#include "mlcut/data/manifest.h"
#include "mlcut/feature/feature_matrix.h"
#include "mlcut/label/concorde_runner.h"
#include "mlcut/pipeline/artifacts.h"
#include "mlcut/pipeline/pruning.h"
#include "mlcut/pipeline/training.h"
#include "mlcut/solver/benchmark.h"
#include "mlcut/tsp/instance.h"

namespace {

enum class GraphRegimeKind {
  kComplete,
  kBase,
  kMl,
};

struct GraphRegime {
  GraphRegimeKind kind = GraphRegimeKind::kComplete;
  mlcut::pipeline::CandidateMode candidate_mode =
      mlcut::pipeline::CandidateMode::kAlpha;
  std::string name = "complete";
};

std::string JsonEscape(std::string_view text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (const char ch : text) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(ch);
        break;
    }
  }
  return out;
}

std::string ToOptionalNumber(double value) {
  if (value < 0.0) {
    return "";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << value;
  return out.str();
}

std::string ToOptionalInteger(std::int64_t value, bool present) {
  if (!present) {
    return "";
  }
  return std::to_string(value);
}

void WriteGzipText(const std::filesystem::path& output_path,
                   std::string_view content) {
  mlcut::base::EnsureDirectory(output_path.parent_path());
  const std::filesystem::path temp_plain =
      output_path.string() + ".tmp." + mlcut::base::MakeTimestampString();
  const std::filesystem::path temp_gz = temp_plain.string() + ".gz";
  mlcut::base::AtomicWriteText(temp_plain, std::string(content));
  const std::string command =
      "gzip -n -c " + mlcut::base::ShellQuote(temp_plain.string()) + " > " +
      mlcut::base::ShellQuote(temp_gz.string());
  if (mlcut::base::RunCommand(command) != 0) {
    std::filesystem::remove(temp_plain);
    throw std::runtime_error("gzip 压缩失败: " + output_path.string());
  }
  std::filesystem::remove(temp_plain);
  std::filesystem::rename(temp_gz, output_path);
}

GraphRegime ParseGraphRegime(std::string_view text) {
  if (text == "complete") {
    return GraphRegime{GraphRegimeKind::kComplete,
                       mlcut::pipeline::CandidateMode::kAlpha, "complete"};
  }
  if (text == "base_alpha") {
    return GraphRegime{GraphRegimeKind::kBase, mlcut::pipeline::CandidateMode::kAlpha,
                       "base_alpha"};
  }
  if (text == "base_popmusic") {
    return GraphRegime{GraphRegimeKind::kBase,
                       mlcut::pipeline::CandidateMode::kPopmusic, "base_popmusic"};
  }
  if (text == "base_union") {
    return GraphRegime{GraphRegimeKind::kBase, mlcut::pipeline::CandidateMode::kUnion,
                       "base_union"};
  }
  if (text == "ml_alpha") {
    return GraphRegime{GraphRegimeKind::kMl, mlcut::pipeline::CandidateMode::kAlpha,
                       "ml_alpha"};
  }
  if (text == "ml_popmusic") {
    return GraphRegime{GraphRegimeKind::kMl, mlcut::pipeline::CandidateMode::kPopmusic,
                       "ml_popmusic"};
  }
  if (text == "ml_union") {
    return GraphRegime{GraphRegimeKind::kMl, mlcut::pipeline::CandidateMode::kUnion,
                       "ml_union"};
  }
  throw std::runtime_error("未知 graph regime: " + std::string(text));
}

std::filesystem::path ResolveBaseCandidatePath(const mlcut::data::ManifestEntry& entry,
                                               std::string_view dataset_tag,
                                               mlcut::pipeline::CandidateMode mode) {
  const auto artifacts = mlcut::pipeline::MakePreparedInstanceArtifacts(
      entry.split, dataset_tag, entry.instance_id, mode);
  return mlcut::pipeline::ResolvePath(artifacts.candidate_binary_file);
}

std::int64_t ResolveExactObjective(const mlcut::data::ManifestEntry& entry,
                                   std::string_view dataset_tag,
                                   const mlcut::tsp::TspInstance& instance) {
  const auto artifacts = mlcut::pipeline::MakePreparedInstanceArtifacts(
      entry.split, dataset_tag, entry.instance_id,
      mlcut::pipeline::CandidateMode::kAlpha);
  const auto tour = mlcut::label::ReadConcordeTour(
      mlcut::pipeline::ResolvePath(artifacts.tour_file));
  return mlcut::solver::TourLength(instance, tour);
}

mlcut::candidate::CandidateGraph BuildMlGraph(
    const mlcut::data::ManifestEntry& entry,
    std::string_view dataset_tag,
    mlcut::pipeline::CandidateMode candidate_mode,
    std::string_view model_run_id,
    const mlcut::pipeline::LocalMassPruneOptions& prune_options) {
  const auto artifacts = mlcut::pipeline::MakePreparedInstanceArtifacts(
      entry.split, dataset_tag, entry.instance_id, candidate_mode);
  const auto graph = mlcut::candidate::CandidateGraph::ReadBinary(
      mlcut::pipeline::ResolvePath(artifacts.candidate_binary_file));
  const auto features = mlcut::feature::FeatureMatrix::ReadBinary(
      mlcut::pipeline::ResolvePath(artifacts.feature_binary_file));
  const auto scores = mlcut::pipeline::PredictScoresForRun(model_run_id, features);
  const auto keep_mask =
      mlcut::pipeline::SelectEdgeMaskByLocalMass(graph, scores, prune_options);
  return mlcut::candidate::FilterByEdgeMask(graph, keep_mask);
}

std::size_t UniqueEdgeCountForComplete(std::size_t node_count) {
  return node_count < 2 ? 0 : (node_count * (node_count - 1)) / 2;
}

std::string MakeRunJson(const std::filesystem::path& manifest_path,
                        const mlcut::data::DatasetCatalog& catalog,
                        std::string_view solver_name,
                        std::string_view regime_name,
                        int time_cap_sec,
                        std::string_view code_version,
                        std::string_view model_run_id,
                        const mlcut::pipeline::LocalMassPruneOptions& prune_options) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"solver\": \"" << JsonEscape(solver_name) << "\",\n";
  out << "  \"regime\": \"" << JsonEscape(regime_name) << "\",\n";
  out << "  \"preset_name\": \"" << JsonEscape(catalog.preset_name) << "\",\n";
  out << "  \"split\": \"" << JsonEscape(catalog.entries.empty() ? "" : catalog.entries.front().split)
      << "\",\n";
  out << "  \"instance_count\": " << catalog.entries.size() << ",\n";
  out << "  \"node_count\": " << (catalog.entries.empty() ? 0 : catalog.entries.front().size)
      << ",\n";
  out << "  \"time_cap_sec\": " << time_cap_sec << ",\n";
  out << "  \"manifest_path\": \"" << JsonEscape(manifest_path.string()) << "\",\n";
  out << "  \"code_version\": \"" << JsonEscape(code_version) << "\"";
  if (!model_run_id.empty()) {
    out << ",\n  \"model_run_id\": \"" << JsonEscape(model_run_id) << "\",\n";
    out << "  \"eta\": " << std::fixed << std::setprecision(6) << prune_options.eta
        << ",\n";
    out << "  \"temperature\": " << prune_options.temperature << ",\n";
    out << "  \"min_keep\": " << prune_options.min_keep_per_node << '\n';
  } else {
    out << '\n';
  }
  out << "}\n";
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path manifest_path;
    std::filesystem::path output_root = "solver_results_v2";
    std::string code_version = "unknown";
    std::string model_run_id;
    std::string regime_name;
    std::string dataset_tag = "large";
    mlcut::pipeline::LocalMassPruneOptions prune_options;
    mlcut::solver::SolverOptions solver_options;

    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--manifest" && index + 1 < argc) {
        manifest_path = argv[++index];
      } else if (arg == "--output-root" && index + 1 < argc) {
        output_root = argv[++index];
      } else if (arg == "--solver" && index + 1 < argc) {
        solver_options.kind = mlcut::solver::ParseSolverKind(argv[++index])
                                  .value_or(mlcut::solver::SolverKind::kOrtoolsRouting);
      } else if (arg == "--regime" && index + 1 < argc) {
        regime_name = argv[++index];
      } else if (arg == "--dataset-tag" && index + 1 < argc) {
        dataset_tag = argv[++index];
      } else if (arg == "--time-limit-sec" && index + 1 < argc) {
        solver_options.ortools_time_limit_sec = std::stoi(argv[++index]);
      } else if (arg == "--model-run-id" && index + 1 < argc) {
        model_run_id = argv[++index];
      } else if (arg == "--eta" && index + 1 < argc) {
        prune_options.eta = std::stod(argv[++index]);
      } else if (arg == "--temperature" && index + 1 < argc) {
        prune_options.temperature = std::stod(argv[++index]);
      } else if (arg == "--min-keep" && index + 1 < argc) {
        prune_options.min_keep_per_node =
            static_cast<std::size_t>(std::stoull(argv[++index]));
      } else if (arg == "--code-version" && index + 1 < argc) {
        code_version = argv[++index];
      } else if (arg == "--lkh-binary" && index + 1 < argc) {
        solver_options.lkh_binary = argv[++index];
      } else if (arg == "--lkh-runs" && index + 1 < argc) {
        solver_options.lkh_runs = std::stoi(argv[++index]);
      } else if (arg == "--lkh-max-trials" && index + 1 < argc) {
        solver_options.lkh_max_trials = std::stoi(argv[++index]);
      } else {
        throw std::runtime_error("未知参数: " + arg);
      }
    }

    if (manifest_path.empty() || regime_name.empty()) {
      throw std::runtime_error("必须提供 --manifest 和 --regime");
    }

    const GraphRegime regime = ParseGraphRegime(regime_name);
    if (regime.kind == GraphRegimeKind::kMl && model_run_id.empty()) {
      throw std::runtime_error("ML regime 必须提供 --model-run-id");
    }
    solver_options.record_trace = true;
    prune_options.aggregation = mlcut::pipeline::AggregationMode::kUnion;

    const auto catalog = mlcut::data::ReadCatalogJson(manifest_path);
    if (catalog.entries.empty()) {
      throw std::runtime_error("manifest 不包含任何实例");
    }

    const std::filesystem::path run_dir =
        output_root / "runs" / mlcut::solver::ToString(solver_options.kind) / regime.name /
        catalog.entries.front().split;
    mlcut::base::EnsureDirectory(run_dir);

    std::ostringstream instances_out;
    instances_out
        << "instance_id\tsplit\tnode_count\tdistance_type\tdistribution\tregime\tsolver\t"
           "candidate_edge_count\tstatus\tfinal_objective\texact_objective\tfinal_gap_pct\t"
           "hit_optimal\ttime_to_opt_cpu_sec\ttime_to_opt_wall_sec\tfinal_cpu_sec\t"
           "final_wall_sec\ttrace_event_count\n";
    std::ostringstream trace_out;
    trace_out
        << "instance_id\tevent_index\telapsed_cpu_sec\telapsed_wall_sec\tobjective\tgap_pct\t"
           "hit_optimal_now\n";

    for (const auto& entry : catalog.entries) {
      const auto instance = mlcut::tsp::TspInstance::ReadBinary(
          mlcut::pipeline::ResolvePath(entry.instance_path));
      const std::int64_t exact_objective =
          ResolveExactObjective(entry, dataset_tag, instance);

      std::optional<mlcut::candidate::CandidateGraph> graph;
      std::size_t candidate_edge_count = 0;
      if (regime.kind == GraphRegimeKind::kComplete) {
        candidate_edge_count = UniqueEdgeCountForComplete(instance.Size());
      } else if (regime.kind == GraphRegimeKind::kBase) {
        graph = mlcut::candidate::CandidateGraph::ReadBinary(
            ResolveBaseCandidatePath(entry, dataset_tag, regime.candidate_mode));
        candidate_edge_count = graph->UniqueEdges().size();
      } else {
        graph = BuildMlGraph(entry, dataset_tag, regime.candidate_mode, model_run_id,
                             prune_options);
        candidate_edge_count = graph->UniqueEdges().size();
      }

      mlcut::solver::SolverResult result;
      if (solver_options.kind == mlcut::solver::SolverKind::kOrtoolsRouting) {
        result = mlcut::solver::SolveWithOrtoolsRoutingTracked(
            instance, graph ? &*graph : nullptr, exact_objective, solver_options);
      } else if (solver_options.kind == mlcut::solver::SolverKind::kOrtoolsCpSat) {
        result = mlcut::solver::SolveWithOrtoolsCpSat(
            instance, graph ? &*graph : nullptr, exact_objective, solver_options);
      } else {
        const std::filesystem::path work_dir =
            std::filesystem::temp_directory_path() / "mlcut_solver_v2_tmp" /
            mlcut::solver::ToString(solver_options.kind) / regime.name / entry.split /
            entry.instance_id;
        const auto problem_path = mlcut::pipeline::ResolvePath(entry.tsplib_path);
        const auto& graph_ref = graph ? *graph
                                      : throw std::runtime_error(
                                            "LKH benchmark 需要显式 candidate graph");
        result = mlcut::solver::SolveWithLkh(instance, problem_path, graph_ref, work_dir,
                                             solver_options);
        if (std::filesystem::exists(work_dir)) {
          std::error_code ignore;
          std::filesystem::remove_all(work_dir, ignore);
        }
      }

      const bool has_final_solution =
          result.status != "no_solution" && result.status != "infeasible" &&
          result.status != "unknown" && result.status != "error";
      const std::string final_gap =
          has_final_solution ? ToOptionalNumber((static_cast<double>(result.objective -
                                                                      exact_objective) /
                                                 static_cast<double>(exact_objective)) *
                                                100.0)
                             : "";

      instances_out << entry.instance_id << '\t' << entry.split << '\t' << instance.Size()
                    << '\t' << mlcut::base::ToString(entry.distance_type) << '\t'
                    << mlcut::base::ToString(entry.distribution_type) << '\t'
                    << regime.name << '\t' << mlcut::solver::ToString(solver_options.kind)
                    << '\t' << candidate_edge_count << '\t' << result.status << '\t'
                    << ToOptionalInteger(result.objective, has_final_solution) << '\t'
                    << exact_objective << '\t' << final_gap << '\t'
                    << (result.hit_optimal ? 1 : 0) << '\t'
                    << ToOptionalNumber(result.time_to_opt_cpu_sec) << '\t'
                    << ToOptionalNumber(result.time_to_opt_wall_sec) << '\t'
                    << ToOptionalNumber(result.cpu_sec) << '\t'
                    << ToOptionalNumber(result.wall_sec) << '\t' << result.trace.size()
                    << '\n';

      for (const auto& event : result.trace) {
        trace_out << entry.instance_id << '\t' << event.event_index << '\t'
                  << ToOptionalNumber(event.elapsed_cpu_sec) << '\t'
                  << ToOptionalNumber(event.elapsed_wall_sec) << '\t' << event.objective
                  << '\t' << ToOptionalNumber(event.gap_pct) << '\t'
                  << (event.hit_optimal_now ? 1 : 0) << '\n';
      }

      std::cout << "[solver-v2] " << mlcut::solver::ToString(solver_options.kind) << ' '
                << regime.name << ' ' << entry.instance_id << '\n';
    }

    mlcut::base::AtomicWriteText(
        run_dir / "run.json",
        MakeRunJson(manifest_path, catalog, mlcut::solver::ToString(solver_options.kind),
                    regime.name, solver_options.ortools_time_limit_sec, code_version,
                    model_run_id, prune_options));
    WriteGzipText(run_dir / "instances.tsv.gz", instances_out.str());
    WriteGzipText(run_dir / "trace.tsv.gz", trace_out.str());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "solver benchmark v2 失败: " << error.what() << '\n';
    return 1;
  }
}
