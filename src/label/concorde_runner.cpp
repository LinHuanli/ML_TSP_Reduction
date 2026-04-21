#include "mlcut/label/concorde_runner.h"

#include <algorithm>
#include <system_error>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <unistd.h>

#include "mlcut/base/filesystem.h"
#include "mlcut/base/process.h"

namespace mlcut::label {

namespace {

std::filesystem::path Abs(const std::filesystem::path& path) {
  return std::filesystem::absolute(path);
}

std::string MakeTempSuffix() {
  return ".tmp-" + mlcut::base::MakeTimestampString() + "-" +
         std::to_string(::getpid());
}

}  // namespace

std::vector<std::uint32_t> ReadConcordeTour(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("无法读取 Concorde tour 文件: " + path.string());
  }
  std::size_t node_count = 0;
  in >> node_count;
  std::vector<std::uint32_t> tour;
  tour.reserve(node_count);
  for (std::size_t index = 0; index < node_count; ++index) {
    std::uint32_t node = 0;
    in >> node;
    if (!in) {
      throw std::runtime_error("Concorde tour 文件格式不完整: " + path.string());
    }
    tour.push_back(node);
  }
  return tour;
}

std::vector<std::uint32_t> RunConcordeAndReadTour(const ConcordeOptions& options) {
  if (!options.overwrite && std::filesystem::exists(options.output_tour_file)) {
    return ReadConcordeTour(options.output_tour_file);
  }
  mlcut::base::EnsureDirectory(options.output_tour_file.parent_path());
  mlcut::base::EnsureDirectory(options.log_file.parent_path());
  // 共享 tour 缓存会被不同 candidate mode 复用，因此先写临时文件再原子替换，
  // 避免并发 prepare 时读到半写 tour 或互相覆盖同一个中间文件。
  const std::string temp_suffix = MakeTempSuffix();
  const std::filesystem::path scratch_dir =
      options.log_file.parent_path() /
      (options.problem_file.stem().string() + ".concorde_work" + temp_suffix);
  mlcut::base::EnsureDirectory(scratch_dir);
  const std::filesystem::path temp_tour_file =
      scratch_dir / (options.output_tour_file.filename().string() + temp_suffix);
  const std::filesystem::path temp_log_file =
      scratch_dir / (options.log_file.filename().string() + temp_suffix);
  const std::string command =
      mlcut::base::ShellQuote(Abs(options.concorde_binary).string()) + " -o " +
      mlcut::base::ShellQuote(Abs(temp_tour_file).string()) + " " +
      mlcut::base::ShellQuote(Abs(options.problem_file).string()) + " > " +
      mlcut::base::ShellQuote(Abs(temp_log_file).string()) + " 2>&1";
  if (mlcut::base::RunCommand(command, scratch_dir) != 0) {
    if (std::filesystem::exists(temp_log_file)) {
      std::filesystem::rename(temp_log_file, options.log_file);
    }
    throw std::runtime_error("Concorde 求最优 tour 失败，请检查日志: " +
                             options.log_file.string());
  }
  if (std::filesystem::exists(temp_log_file)) {
    std::filesystem::rename(temp_log_file, options.log_file);
  }
  std::filesystem::rename(temp_tour_file, options.output_tour_file);
  // 成功后删除独占 workdir，避免 test_n500 这类长任务留下海量 Concorde 中间文件。
  std::error_code cleanup_error;
  std::filesystem::remove_all(scratch_dir, cleanup_error);
  return ReadConcordeTour(options.output_tour_file);
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> TourEdges(
    const std::vector<std::uint32_t>& tour) {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;
  if (tour.empty()) {
    return edges;
  }
  edges.reserve(tour.size());
  for (std::size_t index = 0; index < tour.size(); ++index) {
    const std::uint32_t u = tour[index];
    const std::uint32_t v = tour[(index + 1) % tour.size()];
    edges.emplace_back(std::min(u, v), std::max(u, v));
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
}

}  // namespace mlcut::label
