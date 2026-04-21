#pragma once

#include <cstddef>
#include <functional>

namespace mlcut::parallel {

// 一个足够轻量的并行 for 封装。当前阶段重点是实例级并行，不追求复杂调度器。
void ParallelFor(std::size_t task_count,
                 std::size_t thread_count,
                 const std::function<void(std::size_t)>& fn);

}  // namespace mlcut::parallel

