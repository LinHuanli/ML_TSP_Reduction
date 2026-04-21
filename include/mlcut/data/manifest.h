#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "mlcut/base/types.h"

namespace mlcut::data {

struct ManifestEntry {
  std::string instance_id;
  std::string split;
  base::DistanceType distance_type = base::DistanceType::kEuc2D;
  base::DistributionType distribution_type = base::DistributionType::kUniform;
  std::uint32_t size = 0;
  std::uint64_t seed = 0;
  std::filesystem::path instance_path;
  std::filesystem::path meta_path;
  std::filesystem::path tsplib_path;
};

struct DatasetCatalog {
  std::string preset_name;
  std::vector<ManifestEntry> entries;
};

DatasetCatalog ReadCatalogJson(const std::filesystem::path& path);

void WriteManifestJson(const std::filesystem::path& path,
                       std::string_view preset_name,
                       const std::vector<ManifestEntry>& entries);

void WriteCatalogJson(const std::filesystem::path& path,
                      std::string_view preset_name,
                      const DatasetCatalog& catalog);

void WriteCatalogTsv(const std::filesystem::path& path,
                     const DatasetCatalog& catalog);

void WriteSplitSummaryTsv(const std::filesystem::path& path,
                          const DatasetCatalog& catalog);

}  // namespace mlcut::data
