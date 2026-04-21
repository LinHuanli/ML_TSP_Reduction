#pragma once

#include <filesystem>
#include <string>

namespace mlcut::base {

// 对路径参数做最小必要的 shell 引号转义，避免外部求解器命令因为空格路径失效。
std::string ShellQuote(const std::string& text);

// 统一的命令执行入口。当前阶段优先简单可靠，后续如需更强控制再升级到 posix_spawn。
int RunCommand(const std::string& command);

// 允许为外部求解器指定工作目录，把临时副产物约束到 cache 目录下。
int RunCommand(const std::string& command, const std::filesystem::path& workdir);

}  // namespace mlcut::base
