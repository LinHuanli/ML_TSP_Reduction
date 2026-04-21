#include "mlcut/pipeline/baselines.h"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mlcut::pipeline {

namespace {

constexpr std::size_t kDistanceColumn = 0;
constexpr std::size_t kMutualKnnColumn = 9;
constexpr std::size_t kSourceInBothColumn = 16;

std::vector<std::vector<std::size_t>> BuildIncidentEdges(
    const candidate::CandidateGraph& graph,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> edges) {
  std::vector<std::vector<std::size_t>> incident(graph.NodeCount());
  for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    const auto [u, v] = edges[edge_index];
    incident[u].push_back(edge_index);
    incident[v].push_back(edge_index);
  }
  return incident;
}

std::size_t OtherEndpoint(std::pair<std::uint32_t, std::uint32_t> edge,
                          std::uint32_t node) {
  return edge.first == node ? edge.second : edge.first;
}

void SortByDistance(const feature::FeatureMatrix& features,
                    std::vector<std::size_t>* indices) {
  std::sort(indices->begin(), indices->end(), [&](std::size_t lhs, std::size_t rhs) {
    const float lhs_distance =
        features.values()[lhs * features.cols() + kDistanceColumn];
    const float rhs_distance =
        features.values()[rhs * features.cols() + kDistanceColumn];
    if (lhs_distance != rhs_distance) {
      return lhs_distance < rhs_distance;
    }
    return lhs < rhs;
  });
}

void MarkRequested(std::span<const std::size_t> node_edges,
                   std::size_t max_keep,
                   std::vector<std::uint8_t>* requested) {
  for (std::size_t pos = 0; pos < node_edges.size() && pos < max_keep; ++pos) {
    (*requested)[node_edges[pos]] =
        static_cast<std::uint8_t>((*requested)[node_edges[pos]] + 1U);
  }
}

std::vector<std::uint8_t> FinalizeRequested(std::span<const std::uint8_t> requested,
                                            AggregationMode aggregation) {
  std::vector<std::uint8_t> keep_mask(requested.size(), 0U);
  for (std::size_t index = 0; index < requested.size(); ++index) {
    keep_mask[index] = aggregation == AggregationMode::kUnion
                           ? (requested[index] > 0U ? 1U : 0U)
                           : (requested[index] >= 2U ? 1U : 0U);
  }
  return keep_mask;
}

}  // namespace

std::string ToString(BaselineKind kind) {
  switch (kind) {
    case BaselineKind::kShortestEdges:
      return "shortest";
    case BaselineKind::kMutualKnn:
      return "mutual_knn";
    case BaselineKind::kSourceIntersectionFirst:
      return "source_intersection";
    case BaselineKind::kAlphaThreshold:
      return "alpha_threshold";
  }
  return "shortest";
}

std::optional<BaselineKind> ParseBaselineKind(std::string_view text) {
  if (text == "shortest") {
    return BaselineKind::kShortestEdges;
  }
  if (text == "mutual_knn") {
    return BaselineKind::kMutualKnn;
  }
  if (text == "source_intersection") {
    return BaselineKind::kSourceIntersectionFirst;
  }
  if (text == "alpha_threshold") {
    return BaselineKind::kAlphaThreshold;
  }
  return std::nullopt;
}

std::string MakeBaselineTag(const BaselinePruneOptions& options) {
  std::ostringstream out;
  out << "baseline_" << ToString(options.kind) << "_" << ToString(options.aggregation)
      << "_k" << options.per_node_k << "_m" << options.min_keep_per_node;
  if (options.kind == BaselineKind::kAlphaThreshold) {
    out << "_a" << options.alpha_threshold;
  }
  return out.str();
}

std::vector<std::uint8_t> SelectEdgeMaskByBaseline(
    const candidate::CandidateGraph& graph,
    const feature::FeatureMatrix& features,
    const BaselinePruneOptions& options) {
  const auto edges = graph.UniqueEdges();
  if (features.rows() != edges.size()) {
    throw std::runtime_error("baseline 剪枝时特征行数与候选边数不一致");
  }

  const auto incident = BuildIncidentEdges(graph, edges);
  std::vector<std::uint8_t> requested(edges.size(), 0U);

  for (std::uint32_t node = 0; node < graph.NodeCount(); ++node) {
    const std::size_t max_keep =
        std::max<std::size_t>(options.per_node_k, options.min_keep_per_node);
    auto sorted_by_distance = incident[node];
    SortByDistance(features, &sorted_by_distance);

    if (options.kind == BaselineKind::kShortestEdges) {
      MarkRequested(sorted_by_distance, max_keep, &requested);
      continue;
    }

    std::vector<std::size_t> preferred;
    preferred.reserve(sorted_by_distance.size());
    for (std::size_t edge_index : sorted_by_distance) {
      bool accept = false;
      // baseline 的主逻辑都是“先按各自启发式挑一批，再用最短边补足保底度数”，
      // 这样能和 learned pruning 的 min_keep 约束保持可比。
      if (options.kind == BaselineKind::kMutualKnn) {
        const float mutual =
            features.values()[edge_index * features.cols() + kMutualKnnColumn];
        accept = mutual > 0.5f;
      } else if (options.kind == BaselineKind::kSourceIntersectionFirst) {
        if (features.cols() <= kSourceInBothColumn) {
          throw std::runtime_error("source_intersection baseline 需要 union source 特征");
        }
        const float source_both =
            features.values()[edge_index * features.cols() + kSourceInBothColumn];
        accept = source_both > 0.5f;
      } else if (options.kind == BaselineKind::kAlphaThreshold) {
        const auto [u, v] = edges[edge_index];
        const auto alpha = graph.FindAlpha(node, static_cast<std::uint32_t>(OtherEndpoint({u, v}, node)));
        accept = alpha.has_value() && *alpha <= options.alpha_threshold;
      }
      if (accept) {
        preferred.push_back(edge_index);
      }
    }

    if (preferred.size() < options.min_keep_per_node) {
      for (std::size_t edge_index : sorted_by_distance) {
        if (std::find(preferred.begin(), preferred.end(), edge_index) != preferred.end()) {
          continue;
        }
        preferred.push_back(edge_index);
        if (preferred.size() >= options.min_keep_per_node) {
          break;
        }
      }
    }
    if (preferred.size() > max_keep) {
      preferred.resize(max_keep);
    }
    MarkRequested(preferred, preferred.size(), &requested);
  }

  return FinalizeRequested(requested, options.aggregation);
}

}  // namespace mlcut::pipeline
