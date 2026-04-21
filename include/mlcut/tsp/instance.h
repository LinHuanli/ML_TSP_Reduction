#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "mlcut/base/types.h"

namespace mlcut::tsp {

struct Point2D {
  double x = 0.0;
  double y = 0.0;
};

struct InstanceMetadata {
  std::string instance_id;
  base::DistanceType distance_type = base::DistanceType::kEuc2D;
  base::DistributionType distribution_type = base::DistributionType::kUniform;
  std::uint64_t seed = 0;
  std::uint32_t size = 0;
  std::string dataset_version = "v1";
};

class TspInstance {
 public:
  struct BinaryHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t node_count;
    std::uint32_t distance_type;
    std::uint32_t distribution_type;
    std::uint64_t seed;
  };

  TspInstance() = default;
  TspInstance(InstanceMetadata metadata, std::vector<Point2D> points);

  [[nodiscard]] const InstanceMetadata& metadata() const { return metadata_; }
  [[nodiscard]] const std::vector<Point2D>& points() const { return points_; }
  [[nodiscard]] std::size_t Size() const { return points_.size(); }

  [[nodiscard]] std::int32_t Distance(std::size_t lhs, std::size_t rhs) const;

  void WriteBinary(const std::filesystem::path& path) const;
  static TspInstance ReadBinary(const std::filesystem::path& path);

  void WriteMetaJson(const std::filesystem::path& path) const;
  void WriteTsplib(const std::filesystem::path& path) const;

 private:
  static std::int32_t Nint(double value);
  static double ToTsplibGeoCoordinate(double decimal_degrees);
  static double GeoRadians(double decimal_degrees);

  InstanceMetadata metadata_;
  std::vector<Point2D> points_;
};

}  // namespace mlcut::tsp

