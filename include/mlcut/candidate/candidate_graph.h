#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace mlcut::candidate {

class CandidateGraph {
 public:
  struct BinaryHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t node_count;
    std::uint64_t arc_count;
  };

  CandidateGraph() = default;
  CandidateGraph(std::uint32_t node_count,
                 std::vector<std::uint64_t> offsets,
                 std::vector<std::uint32_t> neighbors,
                 std::vector<std::int32_t> alphas,
                 std::vector<std::uint32_t> mst_parent);

  [[nodiscard]] std::uint32_t NodeCount() const { return node_count_; }
  [[nodiscard]] std::size_t ArcCount() const { return neighbors_.size(); }
  [[nodiscard]] std::size_t Degree(std::uint32_t node) const;
  [[nodiscard]] std::span<const std::uint32_t> Neighbors(std::uint32_t node) const;
  [[nodiscard]] std::span<const std::int32_t> Alphas(std::uint32_t node) const;
  [[nodiscard]] bool ContainsEdge(std::uint32_t lhs, std::uint32_t rhs) const;
  [[nodiscard]] std::optional<std::int32_t> FindAlpha(std::uint32_t from,
                                                      std::uint32_t to) const;
  [[nodiscard]] std::uint32_t MstParent(std::uint32_t node) const {
    return mst_parent_.at(node);
  }

  [[nodiscard]] std::vector<std::pair<std::uint32_t, std::uint32_t>> UniqueEdges() const;

  void WriteBinary(const std::filesystem::path& path) const;
  void WriteLkhCandidateText(const std::filesystem::path& path) const;
  static CandidateGraph ReadBinary(const std::filesystem::path& path);

 private:
  std::uint32_t node_count_ = 0;
  std::vector<std::uint64_t> offsets_;
  std::vector<std::uint32_t> neighbors_;
  std::vector<std::int32_t> alphas_;
  std::vector<std::uint32_t> mst_parent_;
};

CandidateGraph FilterByEdgeMask(const CandidateGraph& graph,
                                std::span<const std::uint8_t> keep_mask);

CandidateGraph MergeCandidateGraphs(const CandidateGraph& lhs,
                                    const CandidateGraph& rhs);

CandidateGraph BuildCompleteGraph(std::uint32_t node_count);

}  // namespace mlcut::candidate
