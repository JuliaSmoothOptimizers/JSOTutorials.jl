---
author: "Abel S. Siqueira"
title: "CaNNOLeS.jl tutorial"
tags:
  - "solvers"
  - "cannoles"
---

[![CaNNOLeS 0.7.4](https://img.shields.io/badge/CaNNOLeS-0.7.4-006400?style=flat-square&labelColor=389826)](https://juliasmoothoptimizers.github.io/CaNNOLeS.jl/stable/)
[![ADNLPModels 0.6.1](https://img.shields.io/badge/ADNLPModels-0.6.1-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/ADNLPModels.jl/stable/)



CaNNOLeS is a solver for equality-constrained nonlinear least-squares problems, i.e.,
optimization problems of the form

$$
\min \frac{1}{2}\|F(x)\|^2_2 \quad \text{s.to} \quad c(x) = 0.
$$

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

| Parameters           | Type          | Default           | Description                                        |
| -------------------- | ------------- | ----------------- | -------------------------------------------------- |
| rtol                 | AbstractFloat | `√eps(eltype(x))` | tolerance                                          |
| unbounded_threshold  | AbstractFloat | `-1e5`            | below this threshold the problem is unbounded      |
| max_f                | Integer       | `100000`          | evaluation limit, e.g. `sum_counters(nls) > max_f` |
| max_time             | AbstractFloat | `30.0`            | maximum number of seconds                          |
| max_inner            | Integer       | `10000`           | maximum number of iterations                       |

### Algorithmic parameters

| Parameters                  | Type           | Default             | Description                                        |
| --------------------------- | -------------- | ------------------- | -------------------------------------------------- |
| x                           | AbstractVector | `nls.meta.x0`       | initial guess |
| λ                           | AbstractVector | `eltype(x)[]`       | initial guess for the Lagrange mutlipliers |
| method                      | Symbol         | `:Newton`           | method to compute direction, `:Newton`, `:LM`, `:Newton_noFHess`, or `:Newton_vanishing` |
| linsolve                    | Symbol         | `:ma57`             | solver use to compute the factorization: `:ma57`, `:ma97`, `:ldlfactorizations` |
| verbose                     | Int            | `0`                 | if > 0, display iteration details every `verbose` iteration |
| check_small_residual        | Bool           | `true`              | if `true`, stop whenever $ \|F(x)\|^2_2 \leq $ ``ϵtol`` and $ \|c(x)\|_\infty \leq $ ``√ϵtol`` |
| always_accept_extrapolation | Bool           | `false`             | if `true`, run even if the extrapolation step fails |
| δdec                        | Real           | `eltype(x)(0.1)`    | reducing factor on the parameter `δ` |

## Examples

```julia
using CaNNOLeS, ADNLPModels

# Rosenbrock
nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
stats = cannoles(nls, rtol = 1e-5, x = ones(2))
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
stats = cannoles(nls, max_time = 10., linsolve = :ldlfactorizations)
```

```
"Execution stats: first-order stationary"
```

