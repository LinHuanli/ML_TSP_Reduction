#include "mlcut/feature/feature_matrix.h"

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "mlcut/base/filesystem.h"

namespace mlcut::feature {

namespace {

constexpr std::array<char, 8> kMagic = {'M', 'L', 'C', 'F', 'E', 'A', 'T', '\0'};

template <typename T>
void AppendBytes(std::vector<std::byte>& out, const T& value) {
  const auto* ptr = reinterpret_cast<const std::byte*>(&value);
  out.insert(out.end(), ptr, ptr + sizeof(T));
}

}  // namespace

FeatureMatrix::FeatureMatrix(std::size_t rows,
                             std::size_t cols,
                             std::vector<float> values)
    : rows_(rows), cols_(cols), values_(std::move(values)) {}

void FeatureMatrix::WriteBinary(const std::filesystem::path& path) const {
  BinaryHeader header{};
  std::memcpy(header.magic, kMagic.data(), kMagic.size());
  header.version = 1;
  header.row_count = rows_;
  header.col_count = cols_;
  std::vector<std::byte> data;
  data.reserve(sizeof(BinaryHeader) + values_.size() * sizeof(float));
  AppendBytes(data, header);
  for (float value : values_) {
    AppendBytes(data, value);
  }
  mlcut::base::AtomicWriteBinary(path, data);
}

FeatureMatrix::BinaryHeader FeatureMatrix::ReadBinaryHeader(
    const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("无法读取特征矩阵文件: " + path.string());
  }
  BinaryHeader header{};
  in.read(reinterpret_cast<char*>(&header),
          static_cast<std::streamsize>(sizeof(BinaryHeader)));
  if (!in) {
    throw std::runtime_error("读取特征矩阵头失败: " + path.string());
  }
  if (!std::equal(std::begin(header.magic), std::end(header.magic), kMagic.begin())) {
    throw std::runtime_error("特征矩阵 magic 不匹配: " + path.string());
  }
  if (header.version != 1U) {
    throw std::runtime_error("特征矩阵版本不支持: " + path.string());
  }
  return header;
}

FeatureMatrix FeatureMatrix::ReadBinary(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("无法读取特征矩阵文件: " + path.string());
  }
  const BinaryHeader header = ReadBinaryHeader(path);
  in.seekg(static_cast<std::streamoff>(sizeof(BinaryHeader)), std::ios::beg);
  std::vector<float> values(header.row_count * header.col_count);
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!in) {
    throw std::runtime_error("读取特征矩阵数据失败: " + path.string());
  }
  return FeatureMatrix(static_cast<std::size_t>(header.row_count),
                       static_cast<std::size_t>(header.col_count),
                       std::move(values));
}

const std::vector<std::string>& FeatureMatrix::CoreFeatureNames() {
  static const std::vector<std::string> kNames = {
      "distance",
      "rank_i_pct",
      "rank_j_pct",
      "rank_min_pct",
      "rank_max_pct",
      "rel_dist_i",
      "rel_dist_j",
      "zscore_i",
      "zscore_j",
      "mutual_knn",
      "overlap_knn",
      "degree_i",
      "degree_j",
      "common_candidate_neighbors",
  };
  return kNames;
}

const std::vector<std::string>& FeatureMatrix::CorePlusSourceFeatureNames() {
  static const std::vector<std::string> kNames = {
      "distance",
      "rank_i_pct",
      "rank_j_pct",
      "rank_min_pct",
      "rank_max_pct",
      "rel_dist_i",
      "rel_dist_j",
      "zscore_i",
      "zscore_j",
      "mutual_knn",
      "overlap_knn",
      "degree_i",
      "degree_j",
      "common_candidate_neighbors",
      "source_in_alpha",
      "source_in_popmusic",
      "source_in_both",
      "source_alpha_only",
      "source_popmusic_only",
  };
  return kNames;
}

}  // namespace mlcut::feature
