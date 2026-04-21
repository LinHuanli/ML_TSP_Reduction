#include "mlcut/base/filesystem.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace mlcut::base {

namespace {

std::filesystem::path MakeTempPath(const std::filesystem::path& path) {
  return path.string() + ".tmp-" + MakeTimestampString() + "-" +
         std::to_string(::getpid());
}

}  // namespace

void EnsureDirectory(const std::filesystem::path& dir) {
  if (dir.empty()) {
    return;
  }
  std::filesystem::create_directories(dir);
}

void AtomicWriteBinary(const std::filesystem::path& path,
                       const std::vector<std::byte>& data) {
  EnsureDirectory(path.parent_path());
  const std::filesystem::path temp_path = MakeTempPath(path);
  {
    std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("无法打开临时二进制文件: " + temp_path.string());
    }
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    if (!out) {
      throw std::runtime_error("写入临时二进制文件失败: " + temp_path.string());
    }
  }
  std::filesystem::rename(temp_path, path);
}

void AtomicWriteText(const std::filesystem::path& path,
                     const std::string& content) {
  EnsureDirectory(path.parent_path());
  const std::filesystem::path temp_path = MakeTempPath(path);
  {
    std::ofstream out(temp_path, std::ios::trunc);
    if (!out) {
      throw std::runtime_error("无法打开临时文本文件: " + temp_path.string());
    }
    out << content;
    if (!out) {
      throw std::runtime_error("写入临时文本文件失败: " + temp_path.string());
    }
  }
  std::filesystem::rename(temp_path, path);
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("无法读取文本文件: " + path.string());
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::string MakeTimestampString() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm tm = *std::localtime(&time);
  std::ostringstream out;
  out << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return out.str();
}

}  // namespace mlcut::base
