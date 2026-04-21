#include <atomic>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "mlcut/base/filesystem.h"
#include "mlcut/candidate/candidate_graph.h"
#include "mlcut/data/manifest.h"
#include "mlcut/feature/feature_matrix.h"
#include "mlcut/label/edge_labels.h"
#include "mlcut/parallel/parallel_for.h"
#include "mlcut/pipeline/artifacts.h"
#include "mlcut/pipeline/baselines.h"
#include "mlcut/pipeline/pruning.h"

int main(int argc, char** argv) {
  try {
    using json = nlohmann::json;

    std::string run_id;
    std::filesystem::path manifest_path;
    mlcut::pipeline::CandidateMode candidate_mode =
        mlcut::pipeline::CandidateMode::kAlpha;
    mlcut::pipeline::BaselinePruneOptions baseline_options;
    std::size_t thread_count = 1;
    bool overwrite = false;

    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--run-id" && index + 1 < argc) {
        run_id = argv[++index];
      } else if (arg == "--manifest" && index + 1 < argc) {
        manifest_path = argv[++index];
      } else if (arg == "--candidate-mode" && index + 1 < argc) {
        candidate_mode =
            mlcut::pipeline::ParseCandidateMode(argv[++index])
                .value_or(mlcut::pipeline::CandidateMode::kAlpha);
      } else if (arg == "--baseline" && index + 1 < argc) {
        baseline_options.kind =
            mlcut::pipeline::ParseBaselineKind(argv[++index])
                .value_or(mlcut::pipeline::BaselineKind::kShortestEdges);
      } else if (arg == "--per-node-k" && index + 1 < argc) {
        baseline_options.per_node_k =
            static_cast<std::size_t>(std::stoull(argv[++index]));
      } else if (arg == "--min-keep" && index + 1 < argc) {
        baseline_options.min_keep_per_node =
            static_cast<std::size_t>(std::stoull(argv[++index]));
      } else if (arg == "--alpha-threshold" && index + 1 < argc) {
        baseline_options.alpha_threshold = std::stoi(argv[++index]);
      } else if (arg == "--aggregation" && index + 1 < argc) {
        baseline_options.aggregation =
            mlcut::pipeline::ParseAggregationMode(argv[++index])
                .value_or(mlcut::pipeline::AggregationMode::kUnion);
      } else if (arg == "--threads" && index + 1 < argc) {
        thread_count = static_cast<std::size_t>(std::stoull(argv[++index]));
      } else if (arg == "--overwrite") {
        overwrite = true;
      } else {
        throw std::runtime_error("未知参数: " + arg);
      }
    }

    if (run_id.empty() || manifest_path.empty()) {
      throw std::runtime_error("必须同时提供 --run-id 和 --manifest");
    }

    const auto layout = mlcut::pipeline::MakeRunLayout(run_id);
    mlcut::base::EnsureDirectory(layout.config_dir);
    mlcut::base::EnsureDirectory(layout.metrics_dir);
    mlcut::base::EnsureDirectory(layout.artifacts_dir);
    mlcut::base::EnsureDirectory(layout.analysis_dir);

    json config_json{
        {"run_id", run_id},
        {"candidate_mode", mlcut::pipeline::ToString(candidate_mode)},
        {"baseline", mlcut::pipeline::ToString(baseline_options.kind)},
        {"per_node_k", baseline_options.per_node_k},
        {"min_keep_per_node", baseline_options.min_keep_per_node},
        {"alpha_threshold", baseline_options.alpha_threshold},
        {"aggregation", mlcut::pipeline::ToString(baseline_options.aggregation)},
        {"thread_count", thread_count},
        {"manifest", manifest_path.string()},
    };
    mlcut::base::AtomicWriteText(layout.config_dir / "baseline_metadata.json",
                                 config_json.dump(2));

    const auto catalog = mlcut::data::ReadCatalogJson(manifest_path);
    const std::string prune_tag = mlcut::pipeline::MakeBaselineTag(baseline_options);
    std::vector<std::string> lines(catalog.entries.size());
    std::atomic<std::size_t> finished{0};

    mlcut::parallel::ParallelFor(
        catalog.entries.size(), thread_count, [&](std::size_t index) {
          const auto& entry = catalog.entries[index];
          const auto artifacts = mlcut::pipeline::MakePreparedInstanceArtifacts(
              entry.split, catalog.preset_name, entry.instance_id, candidate_mode);
          const auto graph = mlcut::candidate::CandidateGraph::ReadBinary(
              mlcut::pipeline::ResolvePath(artifacts.candidate_binary_file));
          const auto features = mlcut::feature::FeatureMatrix::ReadBinary(
              mlcut::pipeline::ResolvePath(artifacts.feature_binary_file));
          const auto labels = mlcut::label::EdgeLabels::ReadBinary(
              mlcut::pipeline::ResolvePath(artifacts.label_binary_file));
          const auto keep_mask =
              mlcut::pipeline::SelectEdgeMaskByBaseline(graph, features, baseline_options);
          const auto pruned_graph = mlcut::candidate::FilterByEdgeMask(graph, keep_mask);
          const auto output_path = mlcut::pipeline::PrunedCandidatePath(
              layout, prune_tag, entry.split, entry.instance_id);
          if (overwrite || !std::filesystem::exists(output_path)) {
            mlcut::base::EnsureDirectory(output_path.parent_path());
            pruned_graph.WriteBinary(output_path);
          }
          const auto metrics = mlcut::pipeline::EvaluateEdgeMask(
              entry.instance_id, entry.split, graph, keep_mask, labels);
          std::ostringstream line;
          line << metrics.instance_id << '\t' << metrics.split << '\t'
               << metrics.node_count << '\t' << metrics.base_edge_count << '\t'
               << metrics.kept_edge_count << '\t' << metrics.positives_in_base << '\t'
               << metrics.positives_kept << '\t' << metrics.m_base << '\t'
               << metrics.m_ml << '\t' << metrics.m_total << '\t' << metrics.rho
               << '\t' << metrics.recall_cond;
          lines[index] = line.str();

          const std::size_t current = finished.fetch_add(1) + 1;
          std::cout << "[baseline-prune] " << current << "/" << catalog.entries.size()
                    << ' ' << entry.instance_id << '\n';
        });

    std::ostringstream out;
    out << "instance_id\tsplit\tnode_count\tbase_edge_count\tkept_edge_count\t"
           "positives_in_base\tpositives_kept\tm_base\tm_ml\tm_total\trho\t"
           "recall_cond\n";
    for (const std::string& line : lines) {
      out << line << '\n';
    }
    mlcut::base::AtomicWriteText(
        mlcut::pipeline::PruneMetricsPath(
            layout, prune_tag,
            std::filesystem::path(manifest_path).stem().string()),
        out.str());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "baseline prune 失败: " << error.what() << '\n';
    return 1;
  }
}
