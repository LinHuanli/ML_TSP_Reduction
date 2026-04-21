#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "mlcut/base/types.h"
#include "mlcut/data/manifest.h"
#include "mlcut/tsp/instance.h"

namespace mlcut::data {

struct SplitRequest {
  std::string split_name;
  std::uint32_t size = 0;
  std::size_t count_per_family = 0;
};

struct GenerationPreset {
  std::string preset_name;
  std::vector<SplitRequest> splits;
};

struct GenerationOptions {
  std::filesystem::path project_root;
  std::size_t thread_count = 1;
  bool overwrite = false;
  GenerationPreset preset;
};

GenerationPreset MakeSmokePreset();
GenerationPreset MakePilotPreset();
GenerationPreset MakeFullPreset();
GenerationPreset MakeLargePreset();

std::vector<base::DistanceType> AllDistanceTypes();
std::vector<base::DistributionType> AllDistributionTypes();

// 生成单个 synthetic 实例，便于后续补单例测试、故障复现和局部重跑。
tsp::TspInstance GenerateSyntheticInstance(base::DistanceType distance_type,
                                           base::DistributionType distribution_type,
                                           std::uint32_t size,
                                           std::uint64_t seed);

// 批量生成某个 preset 对应的数据与 manifest。
DatasetCatalog GeneratePresetDataset(const GenerationOptions& options);

}  // namespace mlcut::data
