#include "mlcut/feature/feature_extractor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mlcut::feature {

namespace {

struct RowStats {
  float nearest = 1.0f;
  float mean = 0.0f;
  float stddev = 1.0f;
};

std::size_t Index(std::size_t n, std::size_t row, std::size_t col) {
  return row * n + col;
}

std::size_t CountIntersection(std::span<const std::uint32_t> lhs,
                              std::span<const std::uint32_t> rhs) {
  std::size_t count = 0;
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < lhs.size() && j < rhs.size()) {
    if (lhs[i] == rhs[j]) {
      ++count;
      ++i;
      ++j;
    } else if (lhs[i] < rhs[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return count;
}

FeatureMatrix ExtractFeaturesImpl(const tsp::TspInstance& instance,
                                  const candidate::CandidateGraph& graph,
                                  std::size_t knn_k,
                                  const SourceFeatureOptions* source_options) {
  const bool with_source = source_options != nullptr;
  const std::size_t n = instance.Size();
  if (n == 0 || graph.NodeCount() != n) {
    throw std::runtime_error("实例和候选图节点数不一致");
  }
  const auto edges = graph.UniqueEdges();
  const std::size_t feature_count =
      with_source ? FeatureMatrix::CorePlusSourceFeatureNames().size()
                  : FeatureMatrix::CoreFeatureNames().size();

  std::vector<std::int32_t> dist_matrix(n * n, 0);
  std::vector<std::uint16_t> rank_matrix(n * n, 0);
  std::vector<RowStats> stats(n);
  std::vector<std::vector<std::uint32_t>> topk_lists(n);
  std::vector<std::vector<std::uint8_t>> topk_mask(n, std::vector<std::uint8_t>(n, 0));

  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::pair<std::int32_t, std::uint32_t>> order;
    order.reserve(n - 1);
    double sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      const std::int32_t distance = instance.Distance(i, j);
      dist_matrix[Index(n, i, j)] = distance;
      order.emplace_back(distance, static_cast<std::uint32_t>(j));
      sum += distance;
    }
    std::sort(order.begin(), order.end(),
              [](const auto& lhs, const auto& rhs) { return lhs < rhs; });
    if (order.empty()) {
      continue;
    }
    stats[i].nearest = static_cast<float>(std::max(1, order.front().first));
    stats[i].mean = static_cast<float>(sum / static_cast<double>(order.size()));
    double variance_sum = 0.0;
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
      const auto [distance, node] = order[rank];
      rank_matrix[Index(n, i, node)] = static_cast<std::uint16_t>(rank + 1);
      const double centered = static_cast<double>(distance) - stats[i].mean;
      variance_sum += centered * centered;
      if (rank < knn_k) {
        topk_lists[i].push_back(node);
        topk_mask[i][node] = 1U;
      }
    }
    const double variance = variance_sum / static_cast<double>(order.size());
    stats[i].stddev = static_cast<float>(std::sqrt(std::max(variance, 1e-6)));
    std::sort(topk_lists[i].begin(), topk_lists[i].end());
  }

  std::vector<float> features(edges.size() * feature_count, 0.0f);
  for (std::size_t row = 0; row < edges.size(); ++row) {
    const auto [u, v] = edges[row];
    const float distance = static_cast<float>(dist_matrix[Index(n, u, v)]);
    const float rank_u =
        static_cast<float>(rank_matrix[Index(n, u, v)]) / static_cast<float>(n - 1);
    const float rank_v =
        static_cast<float>(rank_matrix[Index(n, v, u)]) / static_cast<float>(n - 1);
    const float rel_u = distance / stats[u].nearest;
    const float rel_v = distance / stats[v].nearest;
    const float z_u = (distance - stats[u].mean) / stats[u].stddev;
    const float z_v = (distance - stats[v].mean) / stats[v].stddev;
    const float mutual = (topk_mask[u][v] && topk_mask[v][u]) ? 1.0f : 0.0f;
    const std::size_t overlap_count = CountIntersection(topk_lists[u], topk_lists[v]);
    const std::size_t union_size =
        topk_lists[u].size() + topk_lists[v].size() - overlap_count;
    const float overlap =
        union_size == 0 ? 0.0f
                        : static_cast<float>(overlap_count) /
                              static_cast<float>(union_size);
    const float degree_u = static_cast<float>(graph.Degree(u));
    const float degree_v = static_cast<float>(graph.Degree(v));
    const float common_neighbors = static_cast<float>(
        CountIntersection(graph.Neighbors(u), graph.Neighbors(v)));

    float* row_ptr = features.data() + row * feature_count;
    row_ptr[0] = distance;
    row_ptr[1] = rank_u;
    row_ptr[2] = rank_v;
    row_ptr[3] = std::min(rank_u, rank_v);
    row_ptr[4] = std::max(rank_u, rank_v);
    row_ptr[5] = rel_u;
    row_ptr[6] = rel_v;
    row_ptr[7] = z_u;
    row_ptr[8] = z_v;
    row_ptr[9] = mutual;
    row_ptr[10] = overlap;
    row_ptr[11] = degree_u;
    row_ptr[12] = degree_v;
    row_ptr[13] = common_neighbors;

    if (with_source) {
      const bool in_alpha =
          source_options->alpha_graph != nullptr &&
          source_options->alpha_graph->ContainsEdge(u, v);
      const bool in_popmusic =
          source_options->popmusic_graph != nullptr &&
          source_options->popmusic_graph->ContainsEdge(u, v);
      row_ptr[14] = in_alpha ? 1.0f : 0.0f;
      row_ptr[15] = in_popmusic ? 1.0f : 0.0f;
      row_ptr[16] = (in_alpha && in_popmusic) ? 1.0f : 0.0f;
      row_ptr[17] = (in_alpha && !in_popmusic) ? 1.0f : 0.0f;
      row_ptr[18] = (!in_alpha && in_popmusic) ? 1.0f : 0.0f;
    }
  }
  return FeatureMatrix(edges.size(), feature_count, std::move(features));
}

}  // namespace

FeatureMatrix ExtractCoreFeatures(const tsp::TspInstance& instance,
                                  const candidate::CandidateGraph& graph,
                                  std::size_t knn_k) {
  return ExtractFeaturesImpl(instance, graph, knn_k, nullptr);
}

FeatureMatrix ExtractCorePlusSourceFeatures(
    const tsp::TspInstance& instance,
    const candidate::CandidateGraph& graph,
    std::size_t knn_k,
    const SourceFeatureOptions& source_options) {
  return ExtractFeaturesImpl(instance, graph, knn_k, &source_options);
}

}  // namespace mlcut::feature
