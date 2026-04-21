#include "mlcut/base/process.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace mlcut::base {

std::string ShellQuote(const std::string& text) {
  std::string out = "'";
  for (char ch : text) {
    if (ch == '\'') {
      out += "'\"'\"'";
    } else {
      out.push_back(ch);
    }
  }
  out.push_back('\'');
  return out;
}

int RunCommand(const std::string& command) {
  const int code = std::system(command.c_str());
  if (code == -1) {
    throw std::runtime_error("无法执行命令: " + command);
  }
  return code;
}

int RunCommand(const std::string& command, const std::filesystem::path& workdir) {
  const pid_t pid = ::fork();
  if (pid < 0) {
    throw std::runtime_error("fork 失败: " + std::string(std::strerror(errno)));
  }
  if (pid == 0) {
    if (!workdir.empty() && ::chdir(workdir.c_str()) != 0) {
      std::fprintf(stderr, "切换工作目录失败: %s\n", std::strerror(errno));
      std::_Exit(127);
    }
    ::execl("/bin/sh", "sh", "-lc", command.c_str(), static_cast<char*>(nullptr));
    std::fprintf(stderr, "执行 shell 失败: %s\n", std::strerror(errno));
    std::_Exit(127);
  }

  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) {
    throw std::runtime_error("waitpid 失败: " + std::string(std::strerror(errno)));
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return status;
}

}  // namespace mlcut::base
