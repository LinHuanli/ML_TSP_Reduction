#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlcut/base/filesystem.h"
#include "mlcut/base/types.h"
#include "mlcut/data/generator.h"
#include "mlcut/data/manifest.h"
#include "mlcut/parallel/parallel_for.h"

namespace {

using FamilyKey =
    std::pair<mlcut::base::DistanceType, mlcut::base::DistributionType>;

std::string MakeSyntheticInstanceId(mlcut::base::DistanceType distance_type,
                                    mlcut::base::DistributionType distribution_type,
                                    std::uint32_t size,
                                    std::uint64_t seed) {
  std::ostringstream out;
  out << "syn_" << mlcut::base::ToString(distance_type) << '_'
      << mlcut::base::ToString(distribution_type) << "_n" << size << "_s";
  out.width(6);
  out.fill('0');
  out << seed;
  return out.str();
}

mlcut::data::ManifestEntry BuildTrainEntry(const std::filesystem::path& project_root,
                                           std::string_view split_name,
                                           std::uint32_t size,
                                           mlcut::base::DistanceType distance_type,
                                           mlcut::base::DistributionType distribution_type,
                                           std::uint64_t seed) {
  const std::string instance_id =
      MakeSyntheticInstanceId(distance_type, distribution_type, size, seed);
  const std::filesystem::path family_dir = std::filesystem::relative(
      project_root / "data" / "instances" / "synthetic" / "v1" /
          std::string(split_name) / mlcut::base::ToString(distance_type) /
          mlcut::base::ToString(distribution_type) / ("n" + std::to_string(size)),
      project_root);

  mlcut::data::ManifestEntry entry;
  entry.instance_id = instance_id;
  entry.split = std::string(split_name);
  entry.distance_type = distance_type;
  entry.distribution_type = distribution_type;
  entry.size = size;
  entry.seed = seed;
  entry.instance_path = family_dir / (instance_id + ".tspb");
  entry.meta_path = family_dir / (instance_id + ".meta.json");
  entry.tsplib_path = family_dir / (instance_id + ".tsp");
  return entry;
}

std::map<std::string, mlcut::data::DatasetCatalog> ReadSplitCatalogs(
    const std::filesystem::path& root) {
  std::map<std::string, mlcut::data::DatasetCatalog> catalogs;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".json" ||
        entry.path().filename() == "catalog.json") {
      continue;
    }
    catalogs.emplace(entry.path().stem().string(),
                     mlcut::data::ReadCatalogJson(entry.path()));
  }
  return catalogs;
}

void EnsureInstanceMaterialized(const std::filesystem::path& project_root,
                                const mlcut::data::ManifestEntry& entry) {
  const std::filesystem::path instance_path = project_root / entry.instance_path;
  const std::filesystem::path meta_path = project_root / entry.meta_path;
  const std::filesystem::path tsplib_path = project_root / entry.tsplib_path;
  if (std::filesystem::exists(instance_path) && std::filesystem::exists(meta_path) &&
      std::filesystem::exists(tsplib_path)) {
    return;
  }

  mlcut::base::EnsureDirectory(instance_path.parent_path());
  mlcut::base::EnsureDirectory(meta_path.parent_path());
  mlcut::base::EnsureDirectory(tsplib_path.parent_path());

  const auto instance = mlcut::data::GenerateSyntheticInstance(
      entry.distance_type, entry.distribution_type, entry.size, entry.seed);
  instance.WriteBinary(instance_path);
  instance.WriteMetaJson(meta_path);
  instance.WriteTsplib(tsplib_path);
}

void SortEntriesBySeed(std::vector<mlcut::data::ManifestEntry>* entries) {
  std::sort(entries->begin(), entries->end(),
            [](const mlcut::data::ManifestEntry& lhs,
               const mlcut::data::ManifestEntry& rhs) {
              if (lhs.seed != rhs.seed) {
                return lhs.seed < rhs.seed;
              }
              if (lhs.distance_type != rhs.distance_type) {
                return mlcut::base::ToString(lhs.distance_type) <
                       mlcut::base::ToString(rhs.distance_type);
              }
              if (lhs.distribution_type != rhs.distribution_type) {
                return mlcut::base::ToString(lhs.distribution_type) <
                       mlcut::base::ToString(rhs.distribution_type);
              }
              return lhs.instance_id < rhs.instance_id;
            });
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path base_root = "data/manifests/synthetic/full";
    std::filesystem::path output_root = "data/manifests/synthetic/full_train1000";
    std::string split_name = "train_n100";
    std::size_t target_count_per_family = 1000;
    std::size_t threads = 16;

    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--base-root" && index + 1 < argc) {
        base_root = argv[++index];
      } else if (arg == "--output-root" && index + 1 < argc) {
        output_root = argv[++index];
      } else if (arg == "--split-name" && index + 1 < argc) {
        split_name = argv[++index];
      } else if (arg == "--target-count-per-family" && index + 1 < argc) {
        target_count_per_family = static_cast<std::size_t>(std::stoull(argv[++index]));
      } else if (arg == "--threads" && index + 1 < argc) {
        threads = static_cast<std::size_t>(std::stoull(argv[++index]));
      } else if (arg == "--help") {
        std::cout << "用法: mlcut_extend_train_split "
                  << "[--base-root PATH] [--output-root PATH] "
                  << "[--split-name train_n100] [--target-count-per-family N] "
                  << "[--threads N]\n";
        return 0;
      } else {
        throw std::runtime_error("未知参数: " + arg);
      }
    }

    if (threads == 0) {
      throw std::runtime_error("threads 必须大于 0");
    }

    const std::filesystem::path project_root = std::filesystem::current_path();
    const auto split_catalogs = ReadSplitCatalogs(base_root);
    const auto split_it = split_catalogs.find(split_name);
    if (split_it == split_catalogs.end()) {
      throw std::runtime_error("找不到目标 split manifest: " +
                               (base_root / (split_name + ".json")).string());
    }

    const auto& base_train_entries = split_it->second.entries;
    if (base_train_entries.empty()) {
      throw std::runtime_error("目标 train split 为空: " + split_name);
    }

    std::uint32_t train_size = base_train_entries.front().size;
    std::map<FamilyKey, std::size_t> family_counts;
    std::uint64_t next_seed = 0;

    // 新增训练样本必须避开所有已有 split 的 seed，避免把验证/测试实例重新塞回训练集。
    for (const auto& [current_split, catalog] : split_catalogs) {
      (void)current_split;
      for (const auto& entry : catalog.entries) {
        next_seed = std::max(next_seed, entry.seed + 1);
      }
    }
    for (const auto& entry : base_train_entries) {
      if (entry.split != split_name) {
        throw std::runtime_error("train manifest 中存在不匹配的 split: " + entry.split);
      }
      if (entry.size != train_size) {
        throw std::runtime_error("train manifest 中存在不一致的 size");
      }
      family_counts[{entry.distance_type, entry.distribution_type}] += 1;
    }

    std::vector<mlcut::data::ManifestEntry> extra_entries;
    for (const auto distance_type : mlcut::data::AllDistanceTypes()) {
      for (const auto distribution_type : mlcut::data::AllDistributionTypes()) {
        const FamilyKey key{distance_type, distribution_type};
        const std::size_t current_count = family_counts[key];
        for (std::size_t sample = current_count; sample < target_count_per_family;
             ++sample) {
          extra_entries.push_back(BuildTrainEntry(project_root, split_name, train_size,
                                                  distance_type, distribution_type,
                                                  next_seed++));
        }
      }
    }

    mlcut::parallel::ParallelFor(extra_entries.size(), threads, [&](std::size_t index) {
      EnsureInstanceMaterialized(project_root, extra_entries[index]);
    });

    const std::string output_preset_name = output_root.filename().string();
    mlcut::data::DatasetCatalog merged_catalog;
    mlcut::base::EnsureDirectory(output_root);
    for (const auto& [current_split, catalog] : split_catalogs) {
      std::vector<mlcut::data::ManifestEntry> entries = catalog.entries;
      if (current_split == split_name) {
        entries.insert(entries.end(), extra_entries.begin(), extra_entries.end());
        SortEntriesBySeed(&entries);
      }
      merged_catalog.entries.insert(merged_catalog.entries.end(), entries.begin(),
                                    entries.end());
      mlcut::data::WriteManifestJson(output_root / (current_split + ".json"),
                                     output_preset_name, entries);
    }
    SortEntriesBySeed(&merged_catalog.entries);
    mlcut::data::WriteCatalogJson(output_root / "catalog.json", output_preset_name,
                                  merged_catalog);

    const std::filesystem::path catalog_dir =
        project_root / "data" / "manifests" / "catalogs";
    mlcut::base::EnsureDirectory(catalog_dir);
    mlcut::data::WriteCatalogTsv(catalog_dir /
                                     (output_preset_name + "_synthetic_instances.tsv"),
                                 merged_catalog);
    mlcut::data::WriteSplitSummaryTsv(
        catalog_dir / (output_preset_name + "_split_summary.tsv"), merged_catalog);

    const std::size_t merged_train_count =
        base_train_entries.size() + extra_entries.size();
    std::cout << "已扩展 " << split_name << ": 原始=" << base_train_entries.size()
              << ", 新增=" << extra_entries.size() << ", 目标总数="
              << merged_train_count << ", 每族目标=" << target_count_per_family
              << ", 输出根目录=" << output_root << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "扩展训练集失败: " << error.what() << '\n';
    return 1;
  }
}
