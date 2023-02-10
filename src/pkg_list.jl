using Colors

const model_pkgs = ["ADNLPModels", "AmplNLReader", "BundleAdjustmentModels", "CUTEst", "LLSModels", "ManualNLPModels", "NLPModels", "NLPModelsJuMP", "NLPModelsModifiers", "NLPModelsTest", "NLSProblems", "OptimizationProblems", "PDENLPModels", "QuadraticModels", "QPSReader"]
const solver_pkgs = ["BenchmarkProfiles", "CaNNOLeS", "DCISolver", "JSOSolvers", "NLPModelsIpopt", "NLPModelsKnitro", "Percival", "RipQP", "SolverCore", "SolverTest", "SolverTools", "SolverBenchmark"]
const la_pkgs = ["AMD", "BasicLU", "HSL", "Krylov", "LDLFactorizations", "LimitedLDLFactorizations", "LinearOperators", "PROPACK", "MUMPS", "QRMumps", "SparseMatricesCOO", "SuiteSparseMatrixCollection"]
const jso_pkgs = model_pkgs ∪ solver_pkgs ∪ la_pkgs

const colors = [
  ("8b0000", "cb3c33"), # Red for models
  ("006400", "389826"), # Green for solvers
  ("4b0082", "9558b2"), # Purple for linear algebra
  ("fff", "000"),
]

color_of_pkg(pkg) = if pkg in model_pkgs
  colors[1]
elseif pkg in solver_pkgs
  colors[2]
elseif pkg in la_pkgs
  colors[3]
else
  colors[4]
end
