#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "mlcut/candidate/candidate_graph.h"
#include "mlcut/feature/feature_matrix.h"
#include "mlcut/pipeline/pruning.h"

namespace mlcut::pipeline {

enum class BaselineKind {
  kShortestEdges,
  kMutualKnn,
  kSourceIntersectionFirst,
  kAlphaThreshold,
};

std::string ToString(BaselineKind kind);
std::optional<BaselineKind> ParseBaselineKind(std::string_view text);

struct BaselinePruneOptions {
  BaselineKind kind = BaselineKind::kShortestEdges;
  std::size_t per_node_k = 4;
  std::size_t min_keep_per_node = 2;
  AggregationMode aggregation = AggregationMode::kUnion;
  std::int32_t alpha_threshold = 0;
};

std::string MakeBaselineTag(const BaselinePruneOptions& options);

std::vector<std::uint8_t> SelectEdgeMaskByBaseline(
    const candidate::CandidateGraph& graph,
    const feature::FeatureMatrix& features,
    const BaselinePruneOptions& options);

}  // namespace mlcut::pipeline
