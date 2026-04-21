#include "mlcut/parallel/parallel_for.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mlcut::parallel {

void ParallelFor(std::size_t task_count,
                 std::size_t thread_count,
                 const std::function<void(std::size_t)>& fn) {
  if (task_count == 0) {
    return;
  }
  const std::size_t workers =
      std::max<std::size_t>(1, std::min(task_count, thread_count));
  std::atomic<std::size_t> next_index{0};
  std::exception_ptr first_error = nullptr;
  std::vector<std::jthread> pool;
  pool.reserve(workers);
  for (std::size_t worker = 0; worker < workers; ++worker) {
    pool.emplace_back([&]() {
      try {
        while (true) {
          const std::size_t index = next_index.fetch_add(1);
          if (index >= task_count) {
            break;
          }
          fn(index);
        }
      } catch (...) {
        if (first_error == nullptr) {
          first_error = std::current_exception();
        }
      }
    });
  }
  for (auto& worker : pool) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  if (first_error != nullptr) {
    std::rethrow_exception(first_error);
  }
}

}  // namespace mlcut::parallel

