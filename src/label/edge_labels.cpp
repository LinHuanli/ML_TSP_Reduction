#include "mlcut/label/edge_labels.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

#include "mlcut/base/filesystem.h"

namespace mlcut::label {

namespace {

constexpr std::array<char, 8> kMagic = {'M', 'L', 'C', 'L', 'A', 'B', 'L', '\0'};

std::uint64_t EdgeKey(std::uint32_t u, std::uint32_t v) {
  const std::uint64_t a = std::min(u, v);
  const std::uint64_t b = std::max(u, v);
  return (a << 32U) | b;
}

template <typename T>
void AppendBytes(std::vector<std::byte>& out, const T& value) {
  const auto* ptr = reinterpret_cast<const std::byte*>(&value);
  out.insert(out.end(), ptr, ptr + sizeof(T));
}

}  // namespace

EdgeLabels::EdgeLabels(std::vector<std::uint8_t> values) : values_(std::move(values)) {}

void EdgeLabels::WriteBinary(const std::filesystem::path& path) const {
  BinaryHeader header{};
  std::memcpy(header.magic, kMagic.data(), kMagic.size());
  header.version = 1;
  header.label_count = values_.size();
  std::vector<std::byte> data;
  data.reserve(sizeof(BinaryHeader) + values_.size());
  AppendBytes(data, header);
  for (std::uint8_t value : values_) {
    AppendBytes(data, value);
  }
  mlcut::base::AtomicWriteBinary(path, data);
}

EdgeLabels::BinaryHeader EdgeLabels::ReadBinaryHeader(
    const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("无法读取标签二进制文件: " + path.string());
  }
  BinaryHeader header{};
  in.read(reinterpret_cast<char*>(&header),
          static_cast<std::streamsize>(sizeof(BinaryHeader)));
  if (!in) {
    throw std::runtime_error("读取标签二进制头失败: " + path.string());
  }
  if (!std::equal(std::begin(header.magic), std::end(header.magic), kMagic.begin())) {
    throw std::runtime_error("标签二进制 magic 不匹配: " + path.string());
  }
  if (header.version != 1U) {
    throw std::runtime_error("标签二进制版本不支持: " + path.string());
  }
  return header;
}

EdgeLabels EdgeLabels::ReadBinary(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("无法读取标签二进制文件: " + path.string());
  }
  const BinaryHeader header = ReadBinaryHeader(path);
  in.seekg(static_cast<std::streamoff>(sizeof(BinaryHeader)), std::ios::beg);
  std::vector<std::uint8_t> values(header.label_count);
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(std::uint8_t)));
  if (!in) {
    throw std::runtime_error("读取标签二进制数据失败: " + path.string());
  }
  return EdgeLabels(std::move(values));
}

EdgeLabels BuildEdgeLabels(
    const candidate::CandidateGraph& graph,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& tour_edges) {
  std::unordered_set<std::uint64_t> edge_set;
  edge_set.reserve(tour_edges.size() * 2);
  for (const auto& [u, v] : tour_edges) {
    edge_set.insert(EdgeKey(u, v));
  }

  const auto edges = graph.UniqueEdges();
  std::vector<std::uint8_t> values;
  values.reserve(edges.size());
  for (const auto& [u, v] : edges) {
    values.push_back(edge_set.contains(EdgeKey(u, v)) ? 1U : 0U);
  }
  return EdgeLabels(std::move(values));
}

}  // namespace mlcut::label
