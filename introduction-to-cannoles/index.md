---
author: "Abel S. Siqueira"
title: "CaNNOLeS.jl tutorial"
tags:
  - "solvers"
  - "cannoles"
---

[![CaNNOLeS 0.7.5](https://img.shields.io/badge/CaNNOLeS-0.7.5-006400?style=flat-square&labelColor=389826)](https://juliasmoothoptimizers.github.io/CaNNOLeS.jl/stable/)
[![ADNLPModels 0.7.0](https://img.shields.io/badge/ADNLPModels-0.7.0-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/ADNLPModels.jl/stable/)



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
| atol                 | AbstractFloat | `√eps(eltype(x))` | absolute tolerance                                        |
| rtol                 | AbstractFloat | `√eps(eltype(x))` | relative tolerance: the algorithm uses `ϵtol := atol + rtol * ‖∇F(x⁰)ᵀF(x⁰) - ∇c(x⁰)ᵀλ⁰‖` |
| Fatol                | AbstractFloat | `√eps(eltype(x))` | absolute tolerance on the residual                                       |
| Frtol                | AbstractFloat | `√eps(eltype(x))` |relative tolerance on the residual, the algorithm stops when ‖F(xᵏ)‖ ≤ Fatol + Frtol * ‖F(x⁰)‖  and ‖c(xᵏ)‖∞ ≤ √ϵtol |
| unbounded_threshold  | AbstractFloat | `-1e5`            | below this threshold the problem is unbounded      |
| max_eval             | Integer       | `100000`          | maximum number of evaluations computed by `neval_residual(nls) + neval_cons(nls)` |
| max_iter             | Integer       | `-1`              | maximum number of iterations                       |
| max_time             | AbstractFloat | `30.0`            | maximum number of seconds                          |
| max_inner            | Integer       | `10000`           | maximum number of inner iterations (return stalled if this limit is reached) |
| verbose              | Integer       | `0`               | if > 0, display iteration details every `verbose` iteration |

### Algorithmic parameters

| Parameters                  | Type           | Default             | Description                                        |
| --------------------------- | -------------- | ------------------- | -------------------------------------------------- |
| x                           | AbstractVector | `nls.meta.x0`       | initial guess |
| λ                           | AbstractVector | `nls.meta.y0`       | initial guess for the Lagrange mutlipliers |
| use_initial_multiplier      | Bool           | `false`             | if `true` use `λ` for the initial stopping tests |
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

