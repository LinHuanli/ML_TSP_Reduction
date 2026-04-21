#include "mlcut/pipeline/metrics.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mlcut::pipeline {

namespace {

template <typename LabelT>
BinaryRankingMetrics EvaluateBinaryRankingImpl(std::span<const float> scores,
                                               std::span<const LabelT> labels) {
  if (scores.size() != labels.size()) {
    throw std::runtime_error("评分向量与标签向量长度不一致");
  }
  BinaryRankingMetrics metrics;
  metrics.row_count = scores.size();
  metrics.positive_count = static_cast<std::size_t>(
      std::count_if(labels.begin(), labels.end(),
                    [](LabelT value) { return value != static_cast<LabelT>(0); }));
  if (metrics.row_count == 0) {
    return metrics;
  }
  metrics.positive_rate =
      static_cast<double>(metrics.positive_count) / static_cast<double>(metrics.row_count);
  if (metrics.positive_count == 0) {
    return metrics;
  }

  std::vector<std::size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
    if (scores[lhs] != scores[rhs]) {
      return scores[lhs] > scores[rhs];
    }
    return lhs < rhs;
  });

  double precision_sum = 0.0;
  std::size_t true_positive = 0;
  for (std::size_t rank = 0; rank < order.size(); ++rank) {
    if (labels[order[rank]] == static_cast<LabelT>(0)) {
      continue;
    }
    ++true_positive;
    precision_sum += static_cast<double>(true_positive) /
                     static_cast<double>(rank + 1);
  }
  metrics.average_precision =
      precision_sum / static_cast<double>(metrics.positive_count);
  return metrics;
}

}  // namespace

BinaryRankingMetrics EvaluateBinaryRanking(std::span<const float> scores,
                                           std::span<const int> labels) {
  return EvaluateBinaryRankingImpl(scores, labels);
}

BinaryRankingMetrics EvaluateBinaryRanking(std::span<const float> scores,
                                           std::span<const std::uint8_t> labels) {
  return EvaluateBinaryRankingImpl(scores, labels);
}

}  // namespace mlcut::pipeline
