#include "mlcut/candidate/candidate_graph.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "mlcut/base/filesystem.h"

namespace mlcut::candidate {

namespace {

constexpr std::array<char, 8> kMagic = {'M', 'L', 'C', 'C', 'A', 'N', 'D', '\0'};

std::uint64_t EdgeKey(std::uint32_t u, std::uint32_t v) {
  const std::uint64_t a = std::min<std::uint32_t>(u, v);
  const std::uint64_t b = std::max<std::uint32_t>(u, v);
  return (a << 32U) | b;
}

template <typename T>
void AppendBytes(std::vector<std::byte>& out, const T& value) {
  const auto* ptr = reinterpret_cast<const std::byte*>(&value);
  out.insert(out.end(), ptr, ptr + sizeof(T));
}

template <typename T>
T ReadValue(const std::vector<std::byte>& data, std::size_t* offset) {
  if (*offset + sizeof(T) > data.size()) {
    throw std::runtime_error("读取候选图二进制时越界");
  }
  T value{};
  std::memcpy(&value, data.data() + *offset, sizeof(T));
  *offset += sizeof(T);
  return value;
}

}  // namespace

CandidateGraph::CandidateGraph(std::uint32_t node_count,
                               std::vector<std::uint64_t> offsets,
                               std::vector<std::uint32_t> neighbors,
                               std::vector<std::int32_t> alphas,
                               std::vector<std::uint32_t> mst_parent)
    : node_count_(node_count),
      offsets_(std::move(offsets)),
      neighbors_(std::move(neighbors)),
      alphas_(std::move(alphas)),
      mst_parent_(std::move(mst_parent)) {}

std::size_t CandidateGraph::Degree(std::uint32_t node) const {
  return static_cast<std::size_t>(offsets_.at(node + 1) - offsets_.at(node));
}

std::span<const std::uint32_t> CandidateGraph::Neighbors(std::uint32_t node) const {
  const std::size_t begin = static_cast<std::size_t>(offsets_.at(node));
  const std::size_t end = static_cast<std::size_t>(offsets_.at(node + 1));
  return std::span<const std::uint32_t>(neighbors_.data() + begin, end - begin);
}

std::span<const std::int32_t> CandidateGraph::Alphas(std::uint32_t node) const {
  const std::size_t begin = static_cast<std::size_t>(offsets_.at(node));
  const std::size_t end = static_cast<std::size_t>(offsets_.at(node + 1));
  return std::span<const std::int32_t>(alphas_.data() + begin, end - begin);
}

bool CandidateGraph::ContainsEdge(std::uint32_t lhs, std::uint32_t rhs) const {
  if (lhs >= node_count_ || rhs >= node_count_) {
    return false;
  }
  const auto neighbors = Neighbors(lhs);
  return std::binary_search(neighbors.begin(), neighbors.end(), rhs);
}

std::optional<std::int32_t> CandidateGraph::FindAlpha(std::uint32_t from,
                                                      std::uint32_t to) const {
  if (from >= node_count_ || to >= node_count_) {
    return std::nullopt;
  }
  const auto neighbors = Neighbors(from);
  const auto alphas = Alphas(from);
  const auto iter = std::lower_bound(neighbors.begin(), neighbors.end(), to);
  if (iter == neighbors.end() || *iter != to) {
    return std::nullopt;
  }
  const std::size_t offset = static_cast<std::size_t>(iter - neighbors.begin());
  return alphas[offset];
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> CandidateGraph::UniqueEdges() const {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;
  for (std::uint32_t node = 0; node < node_count_; ++node) {
    for (std::uint32_t neighbor : Neighbors(node)) {
      if (node < neighbor) {
        edges.emplace_back(node, neighbor);
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
}

void CandidateGraph::WriteBinary(const std::filesystem::path& path) const {
  BinaryHeader header{};
  std::memcpy(header.magic, kMagic.data(), kMagic.size());
  header.version = 1;
  header.node_count = node_count_;
  header.arc_count = neighbors_.size();

  std::vector<std::byte> data;
  data.reserve(sizeof(BinaryHeader) + offsets_.size() * sizeof(std::uint64_t) +
               neighbors_.size() * sizeof(std::uint32_t) +
               alphas_.size() * sizeof(std::int32_t) +
               mst_parent_.size() * sizeof(std::uint32_t));
  AppendBytes(data, header);
  for (std::uint64_t value : offsets_) {
    AppendBytes(data, value);
  }
  for (std::uint32_t value : neighbors_) {
    AppendBytes(data, value);
  }
  for (std::int32_t value : alphas_) {
    AppendBytes(data, value);
  }
  for (std::uint32_t value : mst_parent_) {
    AppendBytes(data, value);
  }
  mlcut::base::AtomicWriteBinary(path, data);
}

void CandidateGraph::WriteLkhCandidateText(const std::filesystem::path& path) const {
  std::ostringstream out;
  out << node_count_ << '\n';
  for (std::uint32_t node = 0; node < node_count_; ++node) {
    const auto neighbors = Neighbors(node);
    const auto alphas = Alphas(node);
    const std::uint32_t mst_parent =
        MstParent(node) < node_count_ ? MstParent(node) + 1U : 0U;
    out << node + 1U << ' ' << mst_parent << ' ' << neighbors.size() << '\n';
    for (std::size_t index = 0; index < neighbors.size(); ++index) {
      out << neighbors[index] + 1U << ' ' << alphas[index] << '\n';
    }
  }
  out << -1 << '\n';
  mlcut::base::AtomicWriteText(path, out.str());
}

CandidateGraph CandidateGraph::ReadBinary(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("无法读取候选图二进制文件: " + path.string());
  }
  std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  std::vector<std::byte> data(raw.size());
  std::memcpy(data.data(), raw.data(), raw.size());
  std::size_t offset = 0;
  const BinaryHeader header = ReadValue<BinaryHeader>(data, &offset);
  if (!std::equal(std::begin(header.magic), std::end(header.magic), kMagic.begin())) {
    throw std::runtime_error("候选图二进制 magic 不匹配: " + path.string());
  }
  std::vector<std::uint64_t> offsets(header.node_count + 1);
  for (std::uint64_t& value : offsets) {
    value = ReadValue<std::uint64_t>(data, &offset);
  }
  std::vector<std::uint32_t> neighbors(header.arc_count);
  for (std::uint32_t& value : neighbors) {
    value = ReadValue<std::uint32_t>(data, &offset);
  }
  std::vector<std::int32_t> alphas(header.arc_count);
  for (std::int32_t& value : alphas) {
    value = ReadValue<std::int32_t>(data, &offset);
  }
  std::vector<std::uint32_t> mst_parent(header.node_count);
  for (std::uint32_t& value : mst_parent) {
    value = ReadValue<std::uint32_t>(data, &offset);
  }
  return CandidateGraph(header.node_count, std::move(offsets), std::move(neighbors),
                        std::move(alphas), std::move(mst_parent));
}

CandidateGraph FilterByEdgeMask(const CandidateGraph& graph,
                                std::span<const std::uint8_t> keep_mask) {
  const auto unique_edges = graph.UniqueEdges();
  if (keep_mask.size() != unique_edges.size()) {
    throw std::runtime_error("候选边掩码长度与 UniqueEdges 不一致");
  }

  std::unordered_set<std::uint64_t> kept_edges;
  kept_edges.reserve(unique_edges.size() * 2);
  for (std::size_t index = 0; index < unique_edges.size(); ++index) {
    if (keep_mask[index] != 0U) {
      const auto [u, v] = unique_edges[index];
      kept_edges.insert(EdgeKey(u, v));
    }
  }

  std::vector<std::uint64_t> offsets(graph.NodeCount() + 1, 0);
  std::vector<std::uint32_t> neighbors;
  std::vector<std::int32_t> alphas;
  neighbors.reserve(graph.ArcCount());
  alphas.reserve(graph.ArcCount());

  for (std::uint32_t node = 0; node < graph.NodeCount(); ++node) {
    const auto node_neighbors = graph.Neighbors(node);
    const auto node_alphas = graph.Alphas(node);
    for (std::size_t offset = 0; offset < node_neighbors.size(); ++offset) {
      const std::uint32_t to = node_neighbors[offset];
      if (!kept_edges.contains(EdgeKey(node, to))) {
        continue;
      }
      neighbors.push_back(to);
      alphas.push_back(node_alphas[offset]);
    }
    offsets[node + 1] = neighbors.size();
  }

  std::vector<std::uint32_t> mst_parent(graph.NodeCount(), graph.NodeCount());
  for (std::uint32_t node = 0; node < graph.NodeCount(); ++node) {
    mst_parent[node] = graph.MstParent(node);
  }

  return CandidateGraph(graph.NodeCount(), std::move(offsets), std::move(neighbors),
                        std::move(alphas), std::move(mst_parent));
}

CandidateGraph MergeCandidateGraphs(const CandidateGraph& lhs,
                                    const CandidateGraph& rhs) {
  if (lhs.NodeCount() != rhs.NodeCount()) {
    throw std::runtime_error("无法合并节点数不同的候选图");
  }

  std::vector<std::uint64_t> offsets(lhs.NodeCount() + 1, 0);
  std::vector<std::uint32_t> neighbors;
  std::vector<std::int32_t> alphas;
  neighbors.reserve(lhs.ArcCount() + rhs.ArcCount());
  alphas.reserve(lhs.ArcCount() + rhs.ArcCount());

  for (std::uint32_t node = 0; node < lhs.NodeCount(); ++node) {
    const auto lhs_neighbors = lhs.Neighbors(node);
    const auto lhs_alphas = lhs.Alphas(node);
    const auto rhs_neighbors = rhs.Neighbors(node);
    const auto rhs_alphas = rhs.Alphas(node);
    std::size_t lhs_index = 0;
    std::size_t rhs_index = 0;

    while (lhs_index < lhs_neighbors.size() || rhs_index < rhs_neighbors.size()) {
      const bool lhs_available = lhs_index < lhs_neighbors.size();
      const bool rhs_available = rhs_index < rhs_neighbors.size();
      if (lhs_available && (!rhs_available || lhs_neighbors[lhs_index] < rhs_neighbors[rhs_index])) {
        neighbors.push_back(lhs_neighbors[lhs_index]);
        alphas.push_back(lhs_alphas[lhs_index]);
        ++lhs_index;
        continue;
      }
      if (rhs_available && (!lhs_available || rhs_neighbors[rhs_index] < lhs_neighbors[lhs_index])) {
        neighbors.push_back(rhs_neighbors[rhs_index]);
        alphas.push_back(rhs_alphas[rhs_index]);
        ++rhs_index;
        continue;
      }

      neighbors.push_back(lhs_neighbors[lhs_index]);
      alphas.push_back(std::min(lhs_alphas[lhs_index], rhs_alphas[rhs_index]));
      ++lhs_index;
      ++rhs_index;
    }
    offsets[node + 1] = neighbors.size();
  }

  std::vector<std::uint32_t> mst_parent(lhs.NodeCount(), lhs.NodeCount());
  for (std::uint32_t node = 0; node < lhs.NodeCount(); ++node) {
    const std::uint32_t lhs_parent = lhs.MstParent(node);
    const std::uint32_t rhs_parent = rhs.MstParent(node);
    mst_parent[node] = lhs_parent < lhs.NodeCount() ? lhs_parent : rhs_parent;
  }

  return CandidateGraph(lhs.NodeCount(), std::move(offsets), std::move(neighbors),
                        std::move(alphas), std::move(mst_parent));
}

CandidateGraph BuildCompleteGraph(std::uint32_t node_count) {
  std::vector<std::uint64_t> offsets(node_count + 1, 0);
  std::vector<std::uint32_t> neighbors;
  std::vector<std::int32_t> alphas;
  neighbors.reserve(static_cast<std::size_t>(node_count) *
                    (node_count == 0 ? 0 : node_count - 1));
  alphas.reserve(neighbors.capacity());

  for (std::uint32_t node = 0; node < node_count; ++node) {
    for (std::uint32_t neighbor = 0; neighbor < node_count; ++neighbor) {
      if (neighbor == node) {
        continue;
      }
      neighbors.push_back(neighbor);
      alphas.push_back(0);
    }
    offsets[node + 1] = neighbors.size();
  }

  std::vector<std::uint32_t> mst_parent(node_count, node_count);
  return CandidateGraph(node_count, std::move(offsets), std::move(neighbors),
                        std::move(alphas), std::move(mst_parent));
}

}  // namespace mlcut::candidate
