#include "mlcut/solver/ortools_smoke.h"

#include "ortools/linear_solver/linear_solver.h"

namespace mlcut::solver {

double SolveOrToolsSmokeModel() {
  namespace orr = operations_research;
  orr::MPSolver solver("mlcut_smoke",
                       orr::MPSolver::SCIP_MIXED_INTEGER_PROGRAMMING);
  auto* x = solver.MakeIntVar(0.0, 10.0, "x");
  solver.MutableObjective()->SetCoefficient(x, 1.0);
  solver.MutableObjective()->SetMaximization();
  solver.Solve();
  return x->solution_value();
}

}  // namespace mlcut::solver

