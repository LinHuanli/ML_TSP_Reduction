#include "mlcut/data/manifest.h"

#include <map>
#include <sstream>

#include <nlohmann/json.hpp>

#include "mlcut/base/filesystem.h"

namespace mlcut::data {

using json = nlohmann::json;

namespace {

json EntryToJson(const ManifestEntry& entry) {
  return json{
      {"instance_id", entry.instance_id},
      {"split", entry.split},
      {"distance_type", base::ToString(entry.distance_type)},
      {"distribution_type", base::ToString(entry.distribution_type)},
      {"size", entry.size},
      {"seed", entry.seed},
      {"instance_path", entry.instance_path.string()},
      {"meta_path", entry.meta_path.string()},
      {"tsplib_path", entry.tsplib_path.string()},
  };
}

ManifestEntry EntryFromJson(const json& node) {
  ManifestEntry entry;
  entry.instance_id = node.at("instance_id").get<std::string>();
  entry.split = node.at("split").get<std::string>();
  entry.distance_type = base::ParseDistanceType(
                            node.at("distance_type").get<std::string>())
                            .value_or(base::DistanceType::kEuc2D);
  entry.distribution_type =
      base::ParseDistributionType(node.at("distribution_type").get<std::string>())
          .value_or(base::DistributionType::kUniform);
  entry.size = node.at("size").get<std::uint32_t>();
  entry.seed = node.at("seed").get<std::uint64_t>();
  entry.instance_path = node.at("instance_path").get<std::string>();
  entry.meta_path = node.at("meta_path").get<std::string>();
  entry.tsplib_path = node.at("tsplib_path").get<std::string>();
  return entry;
}

}  // namespace

DatasetCatalog ReadCatalogJson(const std::filesystem::path& path) {
  const json root = json::parse(base::ReadTextFile(path));
  DatasetCatalog catalog;
  catalog.preset_name = root.value("preset_name", "default");
  for (const json& entry_node : root.at("entries")) {
    catalog.entries.push_back(EntryFromJson(entry_node));
  }
  return catalog;
}

void WriteManifestJson(const std::filesystem::path& path,
                       std::string_view preset_name,
                       const std::vector<ManifestEntry>& entries) {
  json root;
  root["preset_name"] = preset_name;
  root["entry_count"] = entries.size();
  root["entries"] = json::array();
  for (const ManifestEntry& entry : entries) {
    root["entries"].push_back(EntryToJson(entry));
  }
  base::AtomicWriteText(path, root.dump(2));
}

void WriteCatalogJson(const std::filesystem::path& path,
                      std::string_view preset_name,
                      const DatasetCatalog& catalog) {
  json root;
  root["preset_name"] = preset_name;
  root["entry_count"] = catalog.entries.size();
  root["entries"] = json::array();
  for (const ManifestEntry& entry : catalog.entries) {
    root["entries"].push_back(EntryToJson(entry));
  }
  base::AtomicWriteText(path, root.dump(2));
}

void WriteCatalogTsv(const std::filesystem::path& path,
                     const DatasetCatalog& catalog) {
  std::ostringstream out;
  out << "instance_id\tsplit\tdistance_type\tdistribution_type\tsize\tseed\tinstance_path\tmeta_path\ttsplib_path\n";
  for (const ManifestEntry& entry : catalog.entries) {
    out << entry.instance_id << '\t' << entry.split << '\t'
        << base::ToString(entry.distance_type) << '\t'
        << base::ToString(entry.distribution_type) << '\t' << entry.size << '\t'
        << entry.seed << '\t' << entry.instance_path.string() << '\t'
        << entry.meta_path.string() << '\t' << entry.tsplib_path.string() << '\n';
  }
  base::AtomicWriteText(path, out.str());
}

void WriteSplitSummaryTsv(const std::filesystem::path& path,
                          const DatasetCatalog& catalog) {
  std::map<std::string, std::size_t> counts;
  for (const ManifestEntry& entry : catalog.entries) {
    const std::string key =
        entry.split + "|" + base::ToString(entry.distance_type) + "|" +
        base::ToString(entry.distribution_type) + "|n" + std::to_string(entry.size);
    counts[key] += 1;
  }
  std::ostringstream out;
  out << "split\tdistance_type\tdistribution_type\tsize\tcount\n";
  for (const auto& [key, count] : counts) {
    const std::size_t first = key.find('|');
    const std::size_t second = key.find('|', first + 1);
    const std::size_t third = key.find('|', second + 1);
    out << key.substr(0, first) << '\t' << key.substr(first + 1, second - first - 1)
        << '\t' << key.substr(second + 1, third - second - 1) << '\t'
        << key.substr(third + 1) << '\t' << count << '\n';
  }
  base::AtomicWriteText(path, out.str());
}

}  // namespace mlcut::data
