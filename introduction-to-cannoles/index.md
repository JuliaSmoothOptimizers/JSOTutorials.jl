---
author: "Abel S. Siqueira"
title: "CaNNOLeS.jl tutorial"
tags:
  - "solvers"
  - "cannoles"
---

[![CaNNOLeS 0.7.2](https://img.shields.io/badge/CaNNOLeS-0.7.2-006400?style=flat-square&labelColor=389826)](https://juliasmoothoptimizers.github.io/CaNNOLeS.jl/stable/)
[![ADNLPModels 0.5.1](https://img.shields.io/badge/ADNLPModels-0.5.1-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/ADNLPModels.jl/stable/)



CaNNOLeS is a solver for equality-constrained nonlinear least-squares problems, i.e.,
optimization problems of the form

    min ¹/₂‖F(x)‖²      s. to     c(x) = 0.

It uses other JuliaSmoothOptimizers packages for development.

In particular, [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) is used for defining the problem, and [SolverCore](https://github.com/JuliaSmoothOptimizers/SolverCore.jl) for the output.
It also uses [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl)'s `MA57` as main solver, but you can pass `linsolve=:ldlfactorizations` to use [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl).

## Fine-tune CaNNOLeS

CaNNOLeS.jl exports the function `cannoles`:
```
   cannoles(nlp :: AbstractNLPModel; kwargs...)
```

Find below a list of the main options of `cannoles`.

### Tolerances on the problem

| Parameters           | Type          | Default         | Description                                        |
| -------------------- | ------------- | --------------- | -------------------------------------------------- |
| ϵtol                 | AbstractFloat | √eps(eltype(x)) | tolerance.                                         |
| unbounded_threshold  | AbstractFloat | -1e5            | below this threshold the problem is unbounded.     |
| max_f                | Integer       | 100000          | evaluation limit, e.g. `sum_counters(nls) > max_f` |
| max_time             | AbstractFloat | 30.             | maximum number of seconds.                         |
| max_inner            | Integer       | 10000           | maximum number of iterations.                      |

### Algorithmic parameters

| Parameters                  | Type           | Default           | Description                                        |
| --------------------------- | -------------- | ----------------- | -------------------------------------------------- |
| x                           | AbstractVector | copy(nls.meta.x0) | initial guess. |
| λ                           | AbstractVector | eltype(x)[]       | initial guess for the Lagrange mutlipliers. |
| method                      | Symbol         | :Newton           | method to compute direction, `:Newton`, `:LM`, `:Newton_noFHess`, or `:Newton_vanishing`. |
| merit                       | Symbol         | :auglag           | merit function: `:norm1`, `:auglag` |
| linsolve                    | Symbol         | :ma57             | solver use to compute the factorization: `:ma57`, `:ma97`, `:ldlfactorizations` |
| check_small_residual        | Bool           | true              | |
| always_accept_extrapolation | Bool           | false             | |
| δdec                        | Real           | eltype(x)(0.1)    | |

## Examples

```julia
using CaNNOLeS, ADNLPModels

# Rosenbrock
nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
stats = cannoles(nls, ϵtol = 1e-5, x = ones(2))
```

```
"Execution stats: first-order stationary"
```



```julia
# Constrained
nls = ADNLSModel(
  x -> [x[1] - 1; 10 * (x[2] - x[1]^2)],
  [-1.2; 1.0],
  2,
  x -> [x[1] * x[2] - 1],
  [0.0],
  [0.0],
)
stats = cannoles(nls, max_time = 10., merit = :auglag)
```

```
Error: MethodError: no method matching solve!(::CaNNOLeS.CaNNOLeSSolver{Float64, Vector{Float64}}, ::ADNLPModels.ADNLSModel{Float64, Vector{Float64}, Vector{Int64}}, ::SolverCore.GenericExecutionStats
{Float64, Vector{Float64}, Vector{Float64}, Any}; max_time=10.0, merit=:auglag)
Closest candidates are:
  solve!(::CaNNOLeS.CaNNOLeSSolver, ::NLPModels.AbstractNLSModel, ::SolverCore.GenericExecutionStats; x, λ, method, linsolve, max_f, max_time, max_inner, ϵtol, verbose, check_small_residual, always_ac
cept_extrapolation, δdec) at ~/.julia/packages/CaNNOLeS/EgvjT/src/CaNNOLeS.jl:114 got unsupported keyword argument "merit"
  solve!(::SolverCore.AbstractOptimizationSolver, ::NLPModels.AbstractNLPModel, ::SolverCore.GenericExecutionStats; kwargs...) at ~/.julia/packages/SolverCore/jOu5t/src/solver.jl:43
  solve!(::SolverCore.AbstractOptimizationSolver, ::NLPModels.AbstractNLPModel; kwargs...) at ~/.julia/packages/SolverCore/jOu5t/src/solver.jl:38
  ...
```

