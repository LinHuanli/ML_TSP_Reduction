#include "mlcut/pipeline/prepare.h"

#include <atomic>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>

#include "mlcut/base/filesystem.h"
#include "mlcut/candidate/candidate_graph.h"
#include "mlcut/candidate/lkh_runner.h"
#include "mlcut/data/manifest.h"
#include "mlcut/feature/feature_extractor.h"
#include "mlcut/label/concorde_runner.h"
#include "mlcut/label/edge_labels.h"
#include "mlcut/parallel/parallel_for.h"
#include "mlcut/tsp/instance.h"

namespace mlcut::pipeline {

namespace {

mlcut::candidate::CandidateSetKind ToCandidateSetKind(CandidateMode mode) {
  switch (mode) {
    case CandidateMode::kComplete:
      break;
    case CandidateMode::kAlpha:
      return mlcut::candidate::CandidateSetKind::kAlpha;
    case CandidateMode::kPopmusic:
      return mlcut::candidate::CandidateSetKind::kPopmusic;
    case CandidateMode::kUnion:
      return mlcut::candidate::CandidateSetKind::kAlpha;
  }
  throw std::runtime_error("complete 模式不应走 LKH 候选生成");
}

struct PreparedGraphs {
  mlcut::candidate::CandidateGraph graph;
  std::optional<mlcut::candidate::CandidateGraph> alpha_graph;
  std::optional<mlcut::candidate::CandidateGraph> popmusic_graph;
  bool generated_target_candidate = false;
};

bool NeedsBuild(const std::filesystem::path& path, bool overwrite) {
  return overwrite || !std::filesystem::exists(path);
}

mlcut::candidate::CandidateGraph LoadOrBuildSingleModeGraph(
    const mlcut::data::ManifestEntry& entry,
    std::string_view dataset_tag,
    CandidateMode mode,
    int max_candidates,
    bool overwrite) {
  const PreparedInstanceArtifacts artifacts =
      MakePreparedInstanceArtifacts(entry.split, dataset_tag, entry.instance_id, mode);
  mlcut::candidate::LkhCandidateOptions candidate_options;
  candidate_options.kind = ToCandidateSetKind(mode);
  candidate_options.problem_file = ResolvePath(entry.tsplib_path);
  candidate_options.parameter_file = ResolvePath(artifacts.candidate_parameter_file);
  candidate_options.candidate_text_file = ResolvePath(artifacts.candidate_text_file);
  candidate_options.candidate_binary_file = ResolvePath(artifacts.candidate_binary_file);
  candidate_options.log_file = ResolvePath(artifacts.candidate_log_file);
  candidate_options.max_candidates = max_candidates;
  candidate_options.overwrite = overwrite;
  return mlcut::candidate::RunLkhAndLoadCandidates(candidate_options);
}

mlcut::candidate::CandidateGraph LoadOrBuildCompleteGraph(
    const mlcut::data::ManifestEntry& entry,
    std::string_view dataset_tag,
    bool overwrite) {
  const PreparedInstanceArtifacts artifacts =
      MakePreparedInstanceArtifacts(entry.split, dataset_tag, entry.instance_id,
                                    CandidateMode::kComplete);
  const std::filesystem::path candidate_binary_path =
      ResolvePath(artifacts.candidate_binary_file);
  if (!overwrite && std::filesystem::exists(candidate_binary_path)) {
    return mlcut::candidate::CandidateGraph::ReadBinary(candidate_binary_path);
  }

  const auto instance =
      mlcut::tsp::TspInstance::ReadBinary(ResolvePath(entry.instance_path));
  auto graph = mlcut::candidate::BuildCompleteGraph(
      static_cast<std::uint32_t>(instance.Size()));
  mlcut::base::EnsureDirectory(candidate_binary_path.parent_path());
  graph.WriteBinary(candidate_binary_path);
  return graph;
}

PreparedGraphs LoadPreparedGraphs(const mlcut::data::ManifestEntry& entry,
                                  std::string_view dataset_tag,
                                  const PrepareManifestOptions& options,
                                  bool need_target_candidate) {
  PreparedGraphs prepared;
  if (options.candidate_mode == CandidateMode::kComplete) {
    prepared.graph =
        LoadOrBuildCompleteGraph(entry, dataset_tag, options.overwrite);
    prepared.generated_target_candidate = need_target_candidate;
    return prepared;
  }
  if (options.candidate_mode != CandidateMode::kUnion) {
    prepared.graph = LoadOrBuildSingleModeGraph(entry, dataset_tag,
                                                options.candidate_mode,
                                                options.max_candidates, options.overwrite);
    prepared.generated_target_candidate = need_target_candidate;
    return prepared;
  }

  const PreparedInstanceArtifacts union_artifacts =
      MakePreparedInstanceArtifacts(entry.split, dataset_tag, entry.instance_id,
                                    CandidateMode::kUnion);
  const std::filesystem::path union_path = ResolvePath(union_artifacts.candidate_binary_file);

  prepared.alpha_graph = LoadOrBuildSingleModeGraph(entry, dataset_tag,
                                                    CandidateMode::kAlpha,
                                                    options.max_candidates, options.overwrite);
  prepared.popmusic_graph = LoadOrBuildSingleModeGraph(entry, dataset_tag,
                                                       CandidateMode::kPopmusic,
                                                       options.max_candidates, options.overwrite);

  // union 模式把两套基础候选缓存都保留下来：目标图用于标签/剪枝，
  // source 图用于额外的来源特征和后续 baseline。
  if (!need_target_candidate && std::filesystem::exists(union_path)) {
    prepared.graph = mlcut::candidate::CandidateGraph::ReadBinary(union_path);
    return prepared;
  }

  prepared.graph = mlcut::candidate::MergeCandidateGraphs(*prepared.alpha_graph,
                                                          *prepared.popmusic_graph);
  mlcut::base::EnsureDirectory(union_path.parent_path());
  prepared.graph.WriteBinary(union_path);
  prepared.generated_target_candidate = need_target_candidate;
  return prepared;
}

}  // namespace

PrepareManifestSummary PrepareManifest(const PrepareManifestOptions& options) {
  if (options.manifest_path.empty()) {
    throw std::runtime_error("manifest 路径不能为空");
  }
  if (options.thread_count == 0) {
    throw std::runtime_error("thread_count 必须大于 0");
  }

  const auto catalog = mlcut::data::ReadCatalogJson(options.manifest_path);
  const std::string dataset_tag = catalog.preset_name;
  PrepareManifestSummary summary;
  summary.instance_count = catalog.entries.size();
  std::atomic<std::size_t> finished{0};
  std::atomic<std::size_t> candidate_done{0};
  std::atomic<std::size_t> label_done{0};
  std::atomic<std::size_t> feature_done{0};

  mlcut::parallel::ParallelFor(
      catalog.entries.size(), options.thread_count, [&](std::size_t index) {
        const auto& entry = catalog.entries[index];
        const PreparedInstanceArtifacts artifacts =
            MakePreparedInstanceArtifacts(entry.split, dataset_tag, entry.instance_id,
                                          options.candidate_mode);
        const std::filesystem::path candidate_binary_path =
            ResolvePath(artifacts.candidate_binary_file);
        const std::filesystem::path label_binary_path =
            ResolvePath(artifacts.label_binary_file);
        const std::filesystem::path feature_binary_path =
            ResolvePath(artifacts.feature_binary_file);
        const bool need_candidate = NeedsBuild(candidate_binary_path, options.overwrite);
        const bool need_label =
            options.build_labels && NeedsBuild(label_binary_path, options.overwrite);
        const bool need_feature = NeedsBuild(feature_binary_path, options.overwrite);

        // 先按实例检查缓存，缺什么补什么，避免为补齐尾部数据而重算整份 manifest。
        if (!need_candidate && !need_label && !need_feature) {
          const std::size_t current = finished.fetch_add(1) + 1;
          std::cout << "[prepare] " << current << "/" << catalog.entries.size() << ' '
                    << entry.instance_id << " cached\n";
          return;
        }

        // union 模式会复用 alpha/popmusic 两套缓存，并把目标候选图合并写到 union 路径。
        const auto prepared =
            LoadPreparedGraphs(entry, dataset_tag, options, need_candidate);
        const auto& graph = prepared.graph;
        if (prepared.generated_target_candidate) {
          candidate_done.fetch_add(1);
        }

        if (need_label) {
          mlcut::label::ConcordeOptions concorde_options;
          concorde_options.problem_file = ResolvePath(entry.tsplib_path);
          concorde_options.output_tour_file = ResolvePath(artifacts.tour_file);
          concorde_options.log_file = ResolvePath(artifacts.tour_log_file);
          concorde_options.overwrite = options.overwrite;
          const auto tour = mlcut::label::RunConcordeAndReadTour(concorde_options);
          const auto labels =
              mlcut::label::BuildEdgeLabels(graph, mlcut::label::TourEdges(tour));
          labels.WriteBinary(ResolvePath(artifacts.label_binary_file));
          label_done.fetch_add(1);
        }

        if (need_feature) {
          const auto instance =
              mlcut::tsp::TspInstance::ReadBinary(ResolvePath(entry.instance_path));
          const auto features = options.candidate_mode == CandidateMode::kUnion
                                    ? mlcut::feature::ExtractCorePlusSourceFeatures(
                                          instance, graph, options.knn_k,
                                          {.alpha_graph = prepared.alpha_graph
                                                              ? &*prepared.alpha_graph
                                                              : nullptr,
                                           .popmusic_graph = prepared.popmusic_graph
                                                                 ? &*prepared.popmusic_graph
                                                                 : nullptr})
                                    : mlcut::feature::ExtractCoreFeatures(
                                          instance, graph, options.knn_k);
          features.WriteBinary(ResolvePath(artifacts.feature_binary_file));
          feature_done.fetch_add(1);
        }

        const std::size_t current = finished.fetch_add(1) + 1;
        std::cout << "[prepare] " << current << "/" << catalog.entries.size() << ' '
                  << entry.instance_id << '\n';
      });

  summary.generated_candidate_count = candidate_done.load();
  summary.generated_label_count = label_done.load();
  summary.generated_feature_count = feature_done.load();
  return summary;
}

}  // namespace mlcut::pipeline
