#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mlcut/candidate/lkh_runner.h"

int main(int argc, char** argv) {
  try {
    mlcut::candidate::LkhCandidateOptions options;
    options.kind = mlcut::candidate::CandidateSetKind::kAlpha;
    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--problem" && index + 1 < argc) {
        options.problem_file = argv[++index];
      } else if (arg == "--parameter" && index + 1 < argc) {
        options.parameter_file = argv[++index];
      } else if (arg == "--candidate-text" && index + 1 < argc) {
        options.candidate_text_file = argv[++index];
      } else if (arg == "--candidate-bin" && index + 1 < argc) {
        options.candidate_binary_file = argv[++index];
      } else if (arg == "--log" && index + 1 < argc) {
        options.log_file = argv[++index];
      } else if (arg == "--mode" && index + 1 < argc) {
        const std::string mode = argv[++index];
        options.kind = mode == "popmusic" ? mlcut::candidate::CandidateSetKind::kPopmusic
                                           : mlcut::candidate::CandidateSetKind::kAlpha;
      } else if (arg == "--max-candidates" && index + 1 < argc) {
        options.max_candidates = std::stoi(argv[++index]);
      } else if (arg == "--overwrite") {
        options.overwrite = true;
      } else {
        throw std::runtime_error("未知参数: " + arg);
      }
    }
    const auto graph = mlcut::candidate::RunLkhAndLoadCandidates(options);
    std::cout << "候选图生成完成: nodes=" << graph.NodeCount()
              << ", arcs=" << graph.ArcCount() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "生成候选图失败: " << error.what() << '\n';
    return 1;
  }
}

