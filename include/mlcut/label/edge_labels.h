#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "mlcut/candidate/candidate_graph.h"

namespace mlcut::label {

class EdgeLabels {
 public:
  struct BinaryHeader {
    char magic[8];
    std::uint32_t version;
    std::uint64_t label_count;
  };

  EdgeLabels() = default;
  explicit EdgeLabels(std::vector<std::uint8_t> values);

  [[nodiscard]] const std::vector<std::uint8_t>& values() const { return values_; }

  void WriteBinary(const std::filesystem::path& path) const;
  static BinaryHeader ReadBinaryHeader(const std::filesystem::path& path);
  static EdgeLabels ReadBinary(const std::filesystem::path& path);

 private:
  std::vector<std::uint8_t> values_;
};

EdgeLabels BuildEdgeLabels(
    const candidate::CandidateGraph& graph,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& tour_edges);

}  // namespace mlcut::label
