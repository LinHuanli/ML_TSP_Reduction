#include "mlcut/data/generator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "mlcut/base/filesystem.h"
#include "mlcut/parallel/parallel_for.h"

namespace mlcut::data {

namespace {

constexpr double kCartesianScale = 10000.0;

double Clamp(double value, double min_value, double max_value) {
  return std::max(min_value, std::min(max_value, value));
}

std::pair<double, double> MapNormalizedPoint(base::DistanceType distance_type,
                                             double x,
                                             double y) {
  if (distance_type == base::DistanceType::kGeo) {
    const double latitude = -70.0 + x * 140.0;
    const double longitude = -160.0 + y * 320.0;
    return {latitude, longitude};
  }
  return {x * kCartesianScale, y * kCartesianScale};
}

std::vector<tsp::Point2D> GenerateUniform(std::size_t size,
                                          base::DistanceType distance_type,
                                          std::mt19937_64* rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::vector<tsp::Point2D> points;
  points.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    const auto [x, y] = MapNormalizedPoint(distance_type, dist(*rng), dist(*rng));
    points.push_back({x, y});
  }
  return points;
}

std::vector<tsp::Point2D> GenerateClustered(std::size_t size,
                                            base::DistanceType distance_type,
                                            std::mt19937_64* rng) {
  const std::size_t cluster_count = std::clamp<std::size_t>(size / 40, 3, 8);
  std::uniform_real_distribution<double> center_dist(0.15, 0.85);
  std::normal_distribution<double> jitter(0.0, 0.06);
  std::vector<std::pair<double, double>> centers;
  centers.reserve(cluster_count);
  for (std::size_t cluster = 0; cluster < cluster_count; ++cluster) {
    centers.emplace_back(center_dist(*rng), center_dist(*rng));
  }
  std::uniform_int_distribution<std::size_t> cluster_pick(0, cluster_count - 1);
  std::vector<tsp::Point2D> points;
  points.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    const auto& [cx, cy] = centers[cluster_pick(*rng)];
    const double x = Clamp(cx + jitter(*rng), 0.0, 1.0);
    const double y = Clamp(cy + jitter(*rng), 0.0, 1.0);
    const auto [mapped_x, mapped_y] = MapNormalizedPoint(distance_type, x, y);
    points.push_back({mapped_x, mapped_y});
  }
  return points;
}

std::vector<tsp::Point2D> GenerateGridJitter(std::size_t size,
                                             base::DistanceType distance_type,
                                             std::mt19937_64* rng) {
  const std::size_t side = static_cast<std::size_t>(std::ceil(std::sqrt(size)));
  std::normal_distribution<double> jitter(0.0, 0.03);
  std::vector<tsp::Point2D> points;
  points.reserve(size);
  for (std::size_t row = 0; row < side && points.size() < size; ++row) {
    for (std::size_t col = 0; col < side && points.size() < size; ++col) {
      const double base_x = (static_cast<double>(row) + 0.5) / static_cast<double>(side);
      const double base_y = (static_cast<double>(col) + 0.5) / static_cast<double>(side);
      const double x = Clamp(base_x + jitter(*rng), 0.0, 1.0);
      const double y = Clamp(base_y + jitter(*rng), 0.0, 1.0);
      const auto [mapped_x, mapped_y] = MapNormalizedPoint(distance_type, x, y);
      points.push_back({mapped_x, mapped_y});
    }
  }
  return points;
}

std::vector<tsp::Point2D> GenerateOutlierMixture(std::size_t size,
                                                 base::DistanceType distance_type,
                                                 std::mt19937_64* rng) {
  const std::size_t core_count =
      static_cast<std::size_t>(std::round(static_cast<double>(size) * 0.8));
  std::normal_distribution<double> core_x(0.45, 0.1);
  std::normal_distribution<double> core_y(0.55, 0.1);
  std::uniform_real_distribution<double> outlier(0.0, 1.0);
  std::bernoulli_distribution edge_pick(0.5);
  std::vector<tsp::Point2D> points;
  points.reserve(size);
  for (std::size_t index = 0; index < core_count; ++index) {
    const auto [mapped_x, mapped_y] = MapNormalizedPoint(
        distance_type, Clamp(core_x(*rng), 0.0, 1.0), Clamp(core_y(*rng), 0.0, 1.0));
    points.push_back({mapped_x, mapped_y});
  }
  for (std::size_t index = core_count; index < size; ++index) {
    double x = outlier(*rng);
    double y = outlier(*rng);
    if (edge_pick(*rng)) {
      x = x < 0.5 ? x * 0.15 : 0.85 + x * 0.15;
    } else {
      y = y < 0.5 ? y * 0.15 : 0.85 + y * 0.15;
    }
    const auto [mapped_x, mapped_y] = MapNormalizedPoint(distance_type, x, y);
    points.push_back({mapped_x, mapped_y});
  }
  return points;
}

std::vector<tsp::Point2D> GenerateCorridor(std::size_t size,
                                           base::DistanceType distance_type,
                                           std::mt19937_64* rng) {
  std::uniform_real_distribution<double> x_dist(0.0, 1.0);
  std::normal_distribution<double> corridor_noise(0.0, 0.025);
  std::vector<tsp::Point2D> points;
  points.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    const double x = x_dist(*rng);
    const double center_y = 0.5 + 0.18 * std::sin(4.0 * std::numbers::pi * x);
    const double y = Clamp(center_y + corridor_noise(*rng), 0.0, 1.0);
    const auto [mapped_x, mapped_y] = MapNormalizedPoint(distance_type, x, y);
    points.push_back({mapped_x, mapped_y});
  }
  return points;
}

std::vector<tsp::Point2D> GeneratePoints(base::DistanceType distance_type,
                                         base::DistributionType distribution_type,
                                         std::size_t size,
                                         std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  switch (distribution_type) {
    case base::DistributionType::kUniform:
      return GenerateUniform(size, distance_type, &rng);
    case base::DistributionType::kClustered:
      return GenerateClustered(size, distance_type, &rng);
    case base::DistributionType::kGridJitter:
      return GenerateGridJitter(size, distance_type, &rng);
    case base::DistributionType::kOutlierMixture:
      return GenerateOutlierMixture(size, distance_type, &rng);
    case base::DistributionType::kCorridor:
      return GenerateCorridor(size, distance_type, &rng);
  }
  throw std::runtime_error("未知分布类型");
}

std::string MakeInstanceId(base::DistanceType distance_type,
                           base::DistributionType distribution_type,
                           std::uint32_t size,
                           std::uint64_t seed) {
  std::ostringstream out;
  out << "syn_" << base::ToString(distance_type) << '_'
      << base::ToString(distribution_type) << "_n" << size << "_s";
  out.width(6);
  out.fill('0');
  out << seed;
  return out.str();
}

std::filesystem::path InstancesRoot(const GenerationOptions& options) {
  return options.project_root / "data" / "instances" / "synthetic" / "v1";
}

ManifestEntry BuildManifestEntry(const GenerationOptions& options,
                                 const SplitRequest& split,
                                 base::DistanceType distance_type,
                                 base::DistributionType distribution_type,
                                 std::uint64_t seed) {
  const std::string instance_id =
      MakeInstanceId(distance_type, distribution_type, split.size, seed);
  const std::filesystem::path family_dir =
      std::filesystem::relative(
          InstancesRoot(options) / split.split_name / base::ToString(distance_type) /
              base::ToString(distribution_type) / ("n" + std::to_string(split.size)),
          options.project_root);
  ManifestEntry entry;
  entry.instance_id = instance_id;
  entry.split = split.split_name;
  entry.distance_type = distance_type;
  entry.distribution_type = distribution_type;
  entry.size = split.size;
  entry.seed = seed;
  entry.instance_path = family_dir / (instance_id + ".tspb");
  entry.meta_path = family_dir / (instance_id + ".meta.json");
  entry.tsplib_path = family_dir / (instance_id + ".tsp");
  return entry;
}

std::filesystem::path ToAbsolute(const GenerationOptions& options,
                                 const std::filesystem::path& relative_path) {
  return options.project_root / relative_path;
}

}  // namespace

GenerationPreset MakeSmokePreset() {
      return GenerationPreset{
      .preset_name = "smoke",
      .splits = {
          {"train_n100", 100, 10},
          {"test_n50", 50, 10},
          {"test_n100", 100, 10},
      },
  };
}

GenerationPreset MakePilotPreset() {
      return GenerationPreset{
      .preset_name = "pilot",
      .splits = {
          {"train_n100", 100, 30},
          {"test_n50", 50, 30},
          {"test_n100", 100, 30},
          {"test_n200", 200, 30},
      },
  };
}

GenerationPreset MakeFullPreset() {
      return GenerationPreset{
      .preset_name = "full",
      .splits = {
          {"train_n100", 100, 200},
          {"val_n100", 100, 50},
          {"test_n50", 50, 100},
          {"test_n100", 100, 100},
          {"test_n200", 200, 100},
          {"test_n500", 500, 100},
      },
  };
}

GenerationPreset MakeLargePreset() {
      return GenerationPreset{
      .preset_name = "large",
      .splits = {
          {"train_n100", 100, 5000},
          {"val_n100", 100, 1000},
          {"test_n50", 50, 1000},
          {"test_n100", 100, 1000},
          {"test_n200", 200, 1000},
          {"test_n500", 500, 1000},
      },
  };
}

std::vector<base::DistanceType> AllDistanceTypes() {
  return {base::DistanceType::kEuc2D, base::DistanceType::kMan2D,
          base::DistanceType::kAtt, base::DistanceType::kGeo};
}

std::vector<base::DistributionType> AllDistributionTypes() {
  return {base::DistributionType::kUniform, base::DistributionType::kClustered,
          base::DistributionType::kGridJitter,
          base::DistributionType::kOutlierMixture,
          base::DistributionType::kCorridor};
}

tsp::TspInstance GenerateSyntheticInstance(base::DistanceType distance_type,
                                           base::DistributionType distribution_type,
                                           std::uint32_t size,
                                           std::uint64_t seed) {
  tsp::InstanceMetadata metadata;
  metadata.instance_id =
      MakeInstanceId(distance_type, distribution_type, size, seed);
  metadata.distance_type = distance_type;
  metadata.distribution_type = distribution_type;
  metadata.seed = seed;
  metadata.size = size;
  metadata.dataset_version = "v1";
  return tsp::TspInstance(
      metadata, GeneratePoints(distance_type, distribution_type, size, seed));
}

DatasetCatalog GeneratePresetDataset(const GenerationOptions& options) {
  const auto distance_types = AllDistanceTypes();
  const auto distribution_types = AllDistributionTypes();
  std::vector<ManifestEntry> tasks;
  std::uint64_t next_seed = 0;
  for (const SplitRequest& split : options.preset.splits) {
    for (base::DistanceType distance_type : distance_types) {
      for (base::DistributionType distribution_type : distribution_types) {
        for (std::size_t sample = 0; sample < split.count_per_family; ++sample) {
          const std::uint64_t seed = next_seed++;
          tasks.push_back(
              BuildManifestEntry(options, split, distance_type, distribution_type, seed));
        }
      }
    }
  }

  parallel::ParallelFor(tasks.size(), options.thread_count,
                        [&](std::size_t index) {
                          const ManifestEntry& entry = tasks[index];
                          const std::filesystem::path instance_path =
                              ToAbsolute(options, entry.instance_path);
                          const std::filesystem::path meta_path =
                              ToAbsolute(options, entry.meta_path);
                          const std::filesystem::path tsplib_path =
                              ToAbsolute(options, entry.tsplib_path);
                          if (!options.overwrite &&
                              std::filesystem::exists(instance_path) &&
                              std::filesystem::exists(meta_path) &&
                              std::filesystem::exists(tsplib_path)) {
                            return;
                          }
                          const tsp::TspInstance instance = GenerateSyntheticInstance(
                              entry.distance_type, entry.distribution_type, entry.size,
                              entry.seed);
                          instance.WriteBinary(instance_path);
                          instance.WriteMetaJson(meta_path);
                          instance.WriteTsplib(tsplib_path);
                        });

  DatasetCatalog catalog;
  catalog.entries = tasks;

  const std::filesystem::path manifest_root =
      options.project_root / "data" / "manifests" / "synthetic" /
      options.preset.preset_name;
  std::map<std::string, std::vector<ManifestEntry>> split_to_entries;
  for (const ManifestEntry& entry : catalog.entries) {
    split_to_entries[entry.split].push_back(entry);
  }
  for (const auto& [split_name, entries] : split_to_entries) {
    WriteManifestJson(manifest_root / (split_name + ".json"),
                      options.preset.preset_name, entries);
  }
  WriteCatalogJson(manifest_root / "catalog.json", options.preset.preset_name, catalog);
  WriteCatalogTsv(options.project_root / "data" / "manifests" / "catalogs" /
                      "synthetic_instances.tsv",
                  catalog);
  WriteSplitSummaryTsv(options.project_root / "data" / "manifests" / "catalogs" /
                           "split_summary.tsv",
                       catalog);
  return catalog;
}

}  // namespace mlcut::data
