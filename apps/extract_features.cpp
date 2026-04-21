#include <iostream>
#include <stdexcept>
#include <string>

#include "mlcut/candidate/candidate_graph.h"
#include "mlcut/feature/feature_extractor.h"
#include "mlcut/tsp/instance.h"

int main(int argc, char** argv) {
  try {
    std::filesystem::path instance_path;
    std::filesystem::path candidate_path;
    std::filesystem::path output_path;
    std::size_t knn_k = 10;

    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--instance" && index + 1 < argc) {
        instance_path = argv[++index];
      } else if (arg == "--candidate" && index + 1 < argc) {
        candidate_path = argv[++index];
      } else if (arg == "--output" && index + 1 < argc) {
        output_path = argv[++index];
      } else if (arg == "--knn-k" && index + 1 < argc) {
        knn_k = static_cast<std::size_t>(std::stoull(argv[++index]));
      } else {
        throw std::runtime_error("未知参数: " + arg);
      }
    }

    const auto instance = mlcut::tsp::TspInstance::ReadBinary(instance_path);
    const auto graph = mlcut::candidate::CandidateGraph::ReadBinary(candidate_path);
    const auto features = mlcut::feature::ExtractCoreFeatures(instance, graph, knn_k);
    features.WriteBinary(output_path);
    std::cout << "特征提取完成: rows=" << features.rows()
              << ", cols=" << features.cols() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "提取特征失败: " << error.what() << '\n';
    return 1;
  }
}

