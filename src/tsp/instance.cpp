#include "mlcut/tsp/instance.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numbers>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "mlcut/base/filesystem.h"

namespace mlcut::tsp {

using json = nlohmann::json;

namespace {

constexpr std::array<char, 8> kMagic = {'M', 'L', 'C', 'T', 'S', 'P', 'B', '\0'};

template <typename T>
void AppendBytes(std::vector<std::byte>& out, const T& value) {
  const auto* ptr = reinterpret_cast<const std::byte*>(&value);
  out.insert(out.end(), ptr, ptr + sizeof(T));
}

template <typename T>
T ReadValue(const std::vector<std::byte>& data, std::size_t* offset) {
  if (*offset + sizeof(T) > data.size()) {
    throw std::runtime_error("读取实例二进制时越界");
  }
  T value{};
  std::memcpy(&value, data.data() + *offset, sizeof(T));
  *offset += sizeof(T);
  return value;
}

double Clamp(double value, double min_value, double max_value) {
  return std::max(min_value, std::min(max_value, value));
}

}  // namespace

TspInstance::TspInstance(InstanceMetadata metadata, std::vector<Point2D> points)
    : metadata_(std::move(metadata)), points_(std::move(points)) {
  metadata_.size = static_cast<std::uint32_t>(points_.size());
}

std::int32_t TspInstance::Nint(double value) {
  return static_cast<std::int32_t>(std::floor(value + 0.5));
}

double TspInstance::ToTsplibGeoCoordinate(double decimal_degrees) {
  const double degrees = std::floor(decimal_degrees);
  const double fractional = decimal_degrees - degrees;
  return degrees + (fractional * 3.0 / 5.0);
}

double TspInstance::GeoRadians(double decimal_degrees) {
  return std::numbers::pi * decimal_degrees / 180.0;
}

std::int32_t TspInstance::Distance(std::size_t lhs, std::size_t rhs) const {
  const Point2D& a = points_.at(lhs);
  const Point2D& b = points_.at(rhs);
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  switch (metadata_.distance_type) {
    case base::DistanceType::kEuc2D:
      return Nint(std::sqrt(dx * dx + dy * dy));
    case base::DistanceType::kMan2D:
      return Nint(std::abs(dx) + std::abs(dy));
    case base::DistanceType::kAtt: {
      const double rij = std::sqrt((dx * dx + dy * dy) / 10.0);
      const auto tij = static_cast<std::int32_t>(std::floor(rij + 0.5));
      return tij < rij ? tij + 1 : tij;
    }
    case base::DistanceType::kGeo: {
      constexpr double kEarthRadius = 6378.388;
      const double lat_a = GeoRadians(a.x);
      const double lon_a = GeoRadians(a.y);
      const double lat_b = GeoRadians(b.x);
      const double lon_b = GeoRadians(b.y);
      const double q1 = std::cos(lon_a - lon_b);
      const double q2 = std::cos(lat_a - lat_b);
      const double q3 = std::cos(lat_a + lat_b);
      const double inner = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3);
      const double safe_inner = Clamp(inner, -1.0, 1.0);
      return static_cast<std::int32_t>(kEarthRadius * std::acos(safe_inner) + 1.0);
    }
  }
  throw std::runtime_error("未知距离类型");
}

void TspInstance::WriteBinary(const std::filesystem::path& path) const {
  BinaryHeader header{};
  std::memcpy(header.magic, kMagic.data(), kMagic.size());
  header.version = 1;
  header.node_count = static_cast<std::uint32_t>(points_.size());
  header.distance_type = static_cast<std::uint32_t>(metadata_.distance_type);
  header.distribution_type = static_cast<std::uint32_t>(metadata_.distribution_type);
  header.seed = metadata_.seed;

  std::vector<std::byte> data;
  data.reserve(sizeof(BinaryHeader) + points_.size() * sizeof(Point2D));
  AppendBytes(data, header);
  for (const Point2D& point : points_) {
    AppendBytes(data, point.x);
    AppendBytes(data, point.y);
  }
  base::AtomicWriteBinary(path, data);
}

TspInstance TspInstance::ReadBinary(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("无法打开实例二进制文件: " + path.string());
  }
  std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  std::vector<std::byte> data(raw.size());
  std::memcpy(data.data(), raw.data(), raw.size());
  std::size_t offset = 0;
  const BinaryHeader header = ReadValue<BinaryHeader>(data, &offset);
  if (!std::equal(std::begin(header.magic), std::end(header.magic), kMagic.begin())) {
    throw std::runtime_error("实例二进制文件 magic 不匹配: " + path.string());
  }
  std::vector<Point2D> points;
  points.reserve(header.node_count);
  for (std::uint32_t index = 0; index < header.node_count; ++index) {
    Point2D point;
    point.x = ReadValue<double>(data, &offset);
    point.y = ReadValue<double>(data, &offset);
    points.push_back(point);
  }
  InstanceMetadata metadata;
  metadata.instance_id = path.stem().string();
  metadata.distance_type =
      static_cast<base::DistanceType>(header.distance_type);
  metadata.distribution_type =
      static_cast<base::DistributionType>(header.distribution_type);
  metadata.seed = header.seed;
  metadata.size = header.node_count;
  return TspInstance(metadata, std::move(points));
}

void TspInstance::WriteMetaJson(const std::filesystem::path& path) const {
  json meta = {
      {"instance_id", metadata_.instance_id},
      {"distance_type", base::ToString(metadata_.distance_type)},
      {"distribution_type", base::ToString(metadata_.distribution_type)},
      {"seed", metadata_.seed},
      {"size", metadata_.size},
      {"dataset_version", metadata_.dataset_version},
  };
  base::AtomicWriteText(path, meta.dump(2));
}

void TspInstance::WriteTsplib(const std::filesystem::path& path) const {
  std::ostringstream out;
  out << "NAME : " << metadata_.instance_id << '\n';
  out << "TYPE : TSP\n";
  out << "COMMENT : distance=" << base::ToString(metadata_.distance_type)
      << ", distribution=" << base::ToString(metadata_.distribution_type)
      << ", seed=" << metadata_.seed << '\n';
  out << "DIMENSION : " << points_.size() << '\n';
  out << "EDGE_WEIGHT_TYPE : " << base::ToTsplibKeyword(metadata_.distance_type)
      << '\n';
  out << "NODE_COORD_SECTION\n";
  for (std::size_t index = 0; index < points_.size(); ++index) {
    const Point2D& point = points_[index];
    if (metadata_.distance_type == base::DistanceType::kGeo) {
      out << index + 1 << ' ' << ToTsplibGeoCoordinate(point.x) << ' '
          << ToTsplibGeoCoordinate(point.y) << '\n';
    } else {
      out << index + 1 << ' ' << point.x << ' ' << point.y << '\n';
    }
  }
  out << "EOF\n";
  base::AtomicWriteText(path, out.str());
}

}  // namespace mlcut::tsp
