#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace mlcut::label {

struct ConcordeOptions {
  std::filesystem::path concorde_binary = "./concorde";
  std::filesystem::path problem_file;
  std::filesystem::path output_tour_file;
  std::filesystem::path log_file;
  bool overwrite = false;
};

std::vector<std::uint32_t> RunConcordeAndReadTour(const ConcordeOptions& options);

std::vector<std::uint32_t> ReadConcordeTour(const std::filesystem::path& path);

std::vector<std::pair<std::uint32_t, std::uint32_t>> TourEdges(
    const std::vector<std::uint32_t>& tour);

}  // namespace mlcut::label

