#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace mlcut::pipeline {

struct BinaryRankingMetrics {
  std::size_t row_count = 0;
  std::size_t positive_count = 0;
  double positive_rate = 0.0;
  double average_precision = 0.0;
};

BinaryRankingMetrics EvaluateBinaryRanking(std::span<const float> scores,
                                           std::span<const int> labels);

BinaryRankingMetrics EvaluateBinaryRanking(std::span<const float> scores,
                                           std::span<const std::uint8_t> labels);

}  // namespace mlcut::pipeline
