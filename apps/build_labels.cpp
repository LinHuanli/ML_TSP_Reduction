#include <iostream>
#include <stdexcept>
#include <string>

#include "mlcut/candidate/candidate_graph.h"
#include "mlcut/label/concorde_runner.h"
#include "mlcut/label/edge_labels.h"

int main(int argc, char** argv) {
  try {
    mlcut::label::ConcordeOptions concorde;
    std::filesystem::path candidate_binary;
    std::filesystem::path label_output;
    bool overwrite = false;

    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--problem" && index + 1 < argc) {
        concorde.problem_file = argv[++index];
      } else if (arg == "--tour" && index + 1 < argc) {
        concorde.output_tour_file = argv[++index];
      } else if (arg == "--log" && index + 1 < argc) {
        concorde.log_file = argv[++index];
      } else if (arg == "--candidate" && index + 1 < argc) {
        candidate_binary = argv[++index];
      } else if (arg == "--output" && index + 1 < argc) {
        label_output = argv[++index];
      } else if (arg == "--overwrite") {
        overwrite = true;
      } else {
        throw std::runtime_error("未知参数: " + arg);
      }
    }
    concorde.overwrite = overwrite;
    const auto graph = mlcut::candidate::CandidateGraph::ReadBinary(candidate_binary);
    const auto tour = mlcut::label::RunConcordeAndReadTour(concorde);
    const auto labels =
        mlcut::label::BuildEdgeLabels(graph, mlcut::label::TourEdges(tour));
    labels.WriteBinary(label_output);
    std::cout << "标签生成完成: count=" << labels.values().size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "生成标签失败: " << error.what() << '\n';
    return 1;
  }
}

