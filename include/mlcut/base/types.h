#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace mlcut::base {

enum class DistanceType : std::uint32_t {
  kEuc2D = 0,
  kMan2D = 1,
  kAtt = 2,
  kGeo = 3,
};

enum class DistributionType : std::uint32_t {
  kUniform = 0,
  kClustered = 1,
  kGridJitter = 2,
  kOutlierMixture = 3,
  kCorridor = 4,
};

std::string ToString(DistanceType type);
std::string ToTsplibKeyword(DistanceType type);
std::optional<DistanceType> ParseDistanceType(std::string_view text);

std::string ToString(DistributionType type);
std::optional<DistributionType> ParseDistributionType(std::string_view text);

std::uint64_t StableHash(std::string_view text);

}  // namespace mlcut::base

