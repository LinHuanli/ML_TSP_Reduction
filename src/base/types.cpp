#include "mlcut/base/types.h"

#include <algorithm>
#include <cctype>
#include <functional>

namespace mlcut::base {

namespace {

std::string LowerCopy(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return out;
}

}  // namespace

std::string ToString(DistanceType type) {
  switch (type) {
    case DistanceType::kEuc2D:
      return "euc_2d";
    case DistanceType::kMan2D:
      return "man_2d";
    case DistanceType::kAtt:
      return "att";
    case DistanceType::kGeo:
      return "geo";
  }
  return "unknown";
}

std::string ToTsplibKeyword(DistanceType type) {
  switch (type) {
    case DistanceType::kEuc2D:
      return "EUC_2D";
    case DistanceType::kMan2D:
      return "MAN_2D";
    case DistanceType::kAtt:
      return "ATT";
    case DistanceType::kGeo:
      return "GEO";
  }
  return "UNKNOWN";
}

std::optional<DistanceType> ParseDistanceType(std::string_view text) {
  const std::string lowered = LowerCopy(text);
  if (lowered == "euc_2d" || lowered == "euc2d") {
    return DistanceType::kEuc2D;
  }
  if (lowered == "man_2d" || lowered == "man2d") {
    return DistanceType::kMan2D;
  }
  if (lowered == "att") {
    return DistanceType::kAtt;
  }
  if (lowered == "geo") {
    return DistanceType::kGeo;
  }
  return std::nullopt;
}

std::string ToString(DistributionType type) {
  switch (type) {
    case DistributionType::kUniform:
      return "uniform";
    case DistributionType::kClustered:
      return "clustered";
    case DistributionType::kGridJitter:
      return "grid_jitter";
    case DistributionType::kOutlierMixture:
      return "outlier_mixture";
    case DistributionType::kCorridor:
      return "corridor";
  }
  return "unknown";
}

std::optional<DistributionType> ParseDistributionType(std::string_view text) {
  const std::string lowered = LowerCopy(text);
  if (lowered == "uniform") {
    return DistributionType::kUniform;
  }
  if (lowered == "clustered") {
    return DistributionType::kClustered;
  }
  if (lowered == "grid_jitter" || lowered == "gridjitter") {
    return DistributionType::kGridJitter;
  }
  if (lowered == "outlier_mixture" || lowered == "outliermixture") {
    return DistributionType::kOutlierMixture;
  }
  if (lowered == "corridor") {
    return DistributionType::kCorridor;
  }
  return std::nullopt;
}

std::uint64_t StableHash(std::string_view text) {
  return static_cast<std::uint64_t>(std::hash<std::string_view>{}(text));
}

}  // namespace mlcut::base

