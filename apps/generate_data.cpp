#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mlcut/data/generator.h"

namespace {

mlcut::data::GenerationPreset PickPreset(const std::string& preset_name) {
  if (preset_name == "smoke") {
    return mlcut::data::MakeSmokePreset();
  }
  if (preset_name == "pilot") {
    return mlcut::data::MakePilotPreset();
  }
  if (preset_name == "full") {
    return mlcut::data::MakeFullPreset();
  }
  if (preset_name == "large") {
    return mlcut::data::MakeLargePreset();
  }
  throw std::runtime_error("未知 preset: " + preset_name);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string preset_name = "smoke";
    std::size_t threads = 1;
    bool overwrite = false;

    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--preset" && index + 1 < argc) {
        preset_name = argv[++index];
      } else if (arg == "--threads" && index + 1 < argc) {
        threads = static_cast<std::size_t>(std::stoull(argv[++index]));
      } else if (arg == "--overwrite") {
        overwrite = true;
      } else if (arg == "--help") {
        std::cout << "用法: mlcut_generate_data "
                  << "[--preset smoke|pilot|full|large] [--threads N] "
                  << "[--overwrite]\n";
        return 0;
      } else {
        throw std::runtime_error("未知参数: " + arg);
      }
    }

    mlcut::data::GenerationOptions options;
    options.project_root = std::filesystem::current_path();
    options.thread_count = threads;
    options.overwrite = overwrite;
    options.preset = PickPreset(preset_name);

    const auto catalog = mlcut::data::GeneratePresetDataset(options);
    std::cout << "已生成 preset=" << preset_name << ", 实例数=" << catalog.entries.size()
              << ", 线程数=" << threads << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "生成数据失败: " << error.what() << '\n';
    return 1;
  }
}
