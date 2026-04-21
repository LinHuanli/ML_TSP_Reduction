#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "mlcut/candidate/candidate_graph.h"
#include "mlcut/label/edge_labels.h"

namespace mlcut::pipeline {

enum class AggregationMode {
  kUnion,
  kIntersection,
};

std::string ToString(AggregationMode mode);
std::optional<AggregationMode> ParseAggregationMode(std::string_view text);

struct LocalMassPruneOptions {
  double eta = 0.95;
  double temperature = 1.0;
  std::size_t min_keep_per_node = 2;
  AggregationMode aggregation = AggregationMode::kUnion;
};

std::string MakePruneTag(const LocalMassPruneOptions& options);

struct EdgePruneMetrics {
  std::string instance_id;
  std::string split;
  std::size_t node_count = 0;
  std::size_t base_edge_count = 0;
  std::size_t kept_edge_count = 0;
  std::size_t positives_in_base = 0;
  std::size_t positives_kept = 0;
  std::size_t m_base = 0;
  std::size_t m_ml = 0;
  std::size_t m_total = 0;
  double rho = 0.0;
  double recall_cond = 1.0;
};

std::vector<std::uint8_t> SelectEdgeMaskByLocalMass(
    const candidate::CandidateGraph& graph,
    std::span<const float> edge_scores,
    const LocalMassPruneOptions& options);

EdgePruneMetrics EvaluateEdgeMask(std::string_view instance_id,
                                  std::string_view split,
                                  const candidate::CandidateGraph& graph,
                                  std::span<const std::uint8_t> keep_mask,
                                  const label::EdgeLabels& labels);

}  // namespace mlcut::pipeline
