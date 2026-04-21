#pragma once

namespace mlcut::solver {

// 一个最小的 OR-Tools 验证入口，用来确认链接和运行时依赖都正常。
double SolveOrToolsSmokeModel();

}  // namespace mlcut::solver

