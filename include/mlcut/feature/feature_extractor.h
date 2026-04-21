#pragma once

#include "mlcut/candidate/candidate_graph.h"
#include "mlcut/feature/feature_matrix.h"
#include "mlcut/tsp/instance.h"

namespace mlcut::feature {

struct SourceFeatureOptions {
  const candidate::CandidateGraph* alpha_graph = nullptr;
  const candidate::CandidateGraph* popmusic_graph = nullptr;
};

FeatureMatrix ExtractCoreFeatures(const tsp::TspInstance& instance,
                                  const candidate::CandidateGraph& graph,
                                  std::size_t knn_k);

FeatureMatrix ExtractCorePlusSourceFeatures(
    const tsp::TspInstance& instance,
    const candidate::CandidateGraph& graph,
    std::size_t knn_k,
    const SourceFeatureOptions& source_options);

}  // namespace mlcut::feature
