#include "mlcut/pipeline/pruning.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mlcut::pipeline {

std::string ToString(AggregationMode mode) {
  switch (mode) {
    case AggregationMode::kUnion:
      return "union";
    case AggregationMode::kIntersection:
      return "intersection";
  }
  return "union";
}

std::optional<AggregationMode> ParseAggregationMode(std::string_view text) {
  if (text == "union") {
    return AggregationMode::kUnion;
  }
  if (text == "intersection") {
    return AggregationMode::kIntersection;
  }
  return std::nullopt;
}

std::string MakePruneTag(const LocalMassPruneOptions& options) {
  std::ostringstream out;
  out << ToString(options.aggregation) << "_eta" << std::fixed << std::setprecision(2)
      << options.eta << "_t" << options.temperature << "_m" << options.min_keep_per_node;
  std::string tag = out.str();
  for (char& ch : tag) {
    if (ch == '.') {
      ch = 'p';
    }
  }
  return tag;
}

std::vector<std::uint8_t> SelectEdgeMaskByLocalMass(
    const candidate::CandidateGraph& graph,
    std::span<const float> edge_scores,
    const LocalMassPruneOptions& options) {
  const auto edges = graph.UniqueEdges();
  if (edge_scores.size() != edges.size()) {
    throw std::runtime_error("edge_scores 长度与候选边数量不一致");
  }
  if (options.temperature <= 0.0) {
    throw std::runtime_error("temperature 必须大于 0");
  }

  std::vector<std::vector<std::size_t>> incident_edges(graph.NodeCount());
  for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    const auto [u, v] = edges[edge_index];
    incident_edges[u].push_back(edge_index);
    incident_edges[v].push_back(edge_index);
  }

  std::vector<std::uint8_t> requested(edges.size(), 0U);
  for (std::uint32_t node = 0; node < graph.NodeCount(); ++node) {
    auto& node_edges = incident_edges[node];
    if (node_edges.empty()) {
      continue;
    }
    std::sort(node_edges.begin(), node_edges.end(), [&](std::size_t lhs, std::size_t rhs) {
      if (edge_scores[lhs] != edge_scores[rhs]) {
        return edge_scores[lhs] > edge_scores[rhs];
      }
      return lhs < rhs;
    });

    const float max_score = edge_scores[node_edges.front()];
    double weight_sum = 0.0;
    std::vector<double> weights(node_edges.size(), 0.0);
    for (std::size_t pos = 0; pos < node_edges.size(); ++pos) {
      const double scaled = (static_cast<double>(edge_scores[node_edges[pos]]) -
                             static_cast<double>(max_score)) /
                            options.temperature;
      weights[pos] = std::exp(scaled);
      weight_sum += weights[pos];
    }

    const std::size_t min_keep =
        std::min(options.min_keep_per_node, node_edges.size());
    double covered = 0.0;
    for (std::size_t pos = 0; pos < node_edges.size(); ++pos) {
      requested[node_edges[pos]] =
          static_cast<std::uint8_t>(requested[node_edges[pos]] + 1U);
      covered += weights[pos] / std::max(weight_sum, 1e-12);
      if (pos + 1 >= min_keep && covered >= options.eta) {
        break;
      }
    }
  }

  std::vector<std::uint8_t> keep_mask(edges.size(), 0U);
  for (std::size_t index = 0; index < edges.size(); ++index) {
    keep_mask[index] = options.aggregation == AggregationMode::kUnion
                           ? (requested[index] > 0U ? 1U : 0U)
                           : (requested[index] >= 2U ? 1U : 0U);
  }
  return keep_mask;
}

EdgePruneMetrics EvaluateEdgeMask(std::string_view instance_id,
                                  std::string_view split,
                                  const candidate::CandidateGraph& graph,
                                  std::span<const std::uint8_t> keep_mask,
                                  const label::EdgeLabels& labels) {
  const auto edges = graph.UniqueEdges();
  if (keep_mask.size() != edges.size()) {
    throw std::runtime_error("keep_mask 长度与候选边数量不一致");
  }
  if (labels.values().size() != edges.size()) {
    throw std::runtime_error("标签长度与候选边数量不一致");
  }

  EdgePruneMetrics metrics;
  metrics.instance_id = std::string(instance_id);
  metrics.split = std::string(split);
  metrics.node_count = graph.NodeCount();
  metrics.base_edge_count = edges.size();

  for (std::size_t index = 0; index < edges.size(); ++index) {
    const bool is_positive = labels.values()[index] != 0U;
    const bool is_kept = keep_mask[index] != 0U;
    metrics.kept_edge_count += is_kept ? 1U : 0U;
    metrics.positives_in_base += is_positive ? 1U : 0U;
    metrics.positives_kept += (is_positive && is_kept) ? 1U : 0U;
  }

  metrics.m_base = metrics.node_count >= metrics.positives_in_base
                       ? metrics.node_count - metrics.positives_in_base
                       : 0;
  metrics.m_ml = metrics.positives_in_base - metrics.positives_kept;
  metrics.m_total = metrics.m_base + metrics.m_ml;
  metrics.rho = metrics.node_count == 0
                    ? 0.0
                    : static_cast<double>(metrics.kept_edge_count) /
                          static_cast<double>(metrics.node_count);
  metrics.recall_cond =
      metrics.positives_in_base == 0
          ? 1.0
          : static_cast<double>(metrics.positives_kept) /
                static_cast<double>(metrics.positives_in_base);
  return metrics;
}

}  // namespace mlcut::pipeline
