#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace mlcut::base {

// 统一的目录创建入口。后续所有写文件逻辑都通过它保证父目录存在。
void EnsureDirectory(const std::filesystem::path& dir);

// 二进制原子写入。先写临时文件再 rename，避免并发或中断时产生半写文件。
void AtomicWriteBinary(const std::filesystem::path& path,
                       const std::vector<std::byte>& data);

// 文本原子写入。适合 manifest、meta.json、日志快照等小文件。
void AtomicWriteText(const std::filesystem::path& path,
                     const std::string& content);

// 读取整个文件到字符串，主要用于测试和轻量配置文件。
std::string ReadTextFile(const std::filesystem::path& path);

// 将当前时间格式化为紧凑字符串，便于构造 run_id 和临时文件名。
std::string MakeTimestampString();

}  // namespace mlcut::base

