#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace mlcut::feature {

class FeatureMatrix {
 public:
  struct BinaryHeader {
    char magic[8];
    std::uint32_t version;
    std::uint64_t row_count;
    std::uint64_t col_count;
  };

  FeatureMatrix() = default;
  FeatureMatrix(std::size_t rows, std::size_t cols, std::vector<float> values);

  [[nodiscard]] std::size_t rows() const { return rows_; }
  [[nodiscard]] std::size_t cols() const { return cols_; }
  [[nodiscard]] const std::vector<float>& values() const { return values_; }

  void WriteBinary(const std::filesystem::path& path) const;
  static BinaryHeader ReadBinaryHeader(const std::filesystem::path& path);
  static FeatureMatrix ReadBinary(const std::filesystem::path& path);

  static const std::vector<std::string>& CoreFeatureNames();
  static const std::vector<std::string>& CorePlusSourceFeatureNames();

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<float> values_;
};

}  // namespace mlcut::feature
