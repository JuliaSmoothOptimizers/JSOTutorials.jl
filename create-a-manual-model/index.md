---
author: "Abel S. Siqueira"
title: "How to create a model from the function and its derivatives"
tags:
  - "models"
  - "manual"
---

![JSON 0.21.3](https://img.shields.io/badge/JSON-0.21.3-000?style=flat-square&labelColor=999)
[![NLPModels 0.20.0](https://img.shields.io/badge/NLPModels-0.20.0-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/NLPModels.jl/stable/)
[![ManualNLPModels 0.1.4](https://img.shields.io/badge/ManualNLPModels-0.1.4-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/ManualNLPModels.jl/stable/)
[![NLPModelsJuMP 0.12.1](https://img.shields.io/badge/NLPModelsJuMP-0.12.1-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/NLPModelsJuMP.jl/stable/)
![BenchmarkTools 1.3.2](https://img.shields.io/badge/BenchmarkTools-1.3.2-000?style=flat-square&labelColor=999)
[![ADNLPModels 0.6.0](https://img.shields.io/badge/ADNLPModels-0.6.0-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/ADNLPModels.jl/stable/)
![JuMP 1.9.0](https://img.shields.io/badge/JuMP-1.9.0-000?style=flat-square&labelColor=999)
[![JSOSolvers 0.10.0](https://img.shields.io/badge/JSOSolvers-0.10.0-006400?style=flat-square&labelColor=389826)](https://juliasmoothoptimizers.github.io/JSOSolvers.jl/stable/)



When you know the derivatives of your optimization problem, it is frequently more efficient to use them directly instead of relying on automatic differentiation.
For that purpose, we have created `ManualNLPModels`. The package is very crude, due to demand being low, but let us know if you need more functionalities.

For instance, in the logistic regression problem, we have a model
$h_{\beta}(x) = \sigma(\hat{x}^T \beta) = \sigma(\beta_0 + x^T\beta_{1:p})$,
where
$\hat{x} = \begin{bmatrix} 1 \\ x \end{bmatrix}$.
The value of $\beta$ is found by finding the minimum of the negavitve of the log-likelihood function.

$$\ell(\beta) = -\frac{1}{n} \sum_{i=1}^n y_i \ln \big(h_{\beta}(x_i)\big) + (1 - y_i) \ln\big(1 - h_{\beta}(x_i)\big).$$

We'll input the gradient of this function manually. It is given by

$$\nabla \ell(\beta) = \frac{-1}{n} \sum_{i=1}^n \big(y_i - h_{\beta}(x_i)\big) \hat{x}_i = \frac{1}{n} \begin{bmatrix} e^T \\ X^T \end{bmatrix} (h_{\beta}(X) - y),$$

where $e$ is the vector with all components equal to 1.

```julia
using ManualNLPModels
using LinearAlgebra, Random
Random.seed!(0)

sigmoid(t) = 1 / (1 + exp(-t))
h(β, X) = sigmoid.(β[1] .+ X * β[2:end])

n, p = 500, 50
X = randn(n, p)
β = randn(p + 1)
y = round.(h(β, X) .+ randn(n) * 0.1)

function myfun(β, X, y)
  @views hβ = sigmoid.(β[1] .+ X * β[2:end])
  out = sum(
    yᵢ * log(ŷᵢ + 1e-8) + (1 - yᵢ) * log(1 - ŷᵢ + 1e-8)
    for (yᵢ, ŷᵢ) in zip(y, hβ)
  )
  return -out / n + 0.5e-4 * norm(β)^2
end

function mygrad(out, β, X, y)
  n = length(y)
  @views δ = (sigmoid.(β[1] .+ X * β[2:end]) - y) / n
  out[1] = sum(δ) + 1e-4β[1]
  @views out[2:end] .= X' * δ + 1e-4 * β[2:end]
  return out
end

nlp = NLPModel(
  zeros(p + 1),
  β -> myfun(β, X, y),
  grad=(out, β) -> mygrad(out, β, X, y),
)
```

```
ManualNLPModels.NLPModel{Float64, Vector{Float64}}
  Problem name: Generic
   All variables: ████████████████████ 51     All constraints: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            free: ████████████████████ 51                free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
          infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            nnzh: (100.00% sparsity)   0               linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
                                                    nonlinear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
                                                         nnzj: (------% sparsity)         

  Counters:
             obj: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 grad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
        cons_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0             cons_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              jac_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         jac_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0            jprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
       jprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0           jtprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
      jtprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
```





Notice that the `grad` function must modify the first argument so you don't waste memory creating arrays.

Only the `obj`, `grad` and `grad!` functions will be defined for this model, so you need to choose your solver carefully.
We'll use `lbfgs` from `JSOSolvers.jl`.

```julia
using JSOSolvers

output = lbfgs(nlp)
βsol = output.solution
ŷ = round.(h(βsol, X))
sum(ŷ .== y) / n
```

```
1.0
```





We can compare against other approaches.

```julia
using BenchmarkTools
using Logging

@benchmark begin
  nlp = NLPModel(
    zeros(p + 1),
    β -> myfun(β, X, y),
    grad=(out, β) -> mygrad(out, β, X, y),
  )
  output = with_logger(NullLogger()) do
    lbfgs(nlp)
  end
end
```

```
BenchmarkTools.Trial: 1982 samples with 1 evaluation.
 Range (min … max):  2.296 ms …   6.751 ms  ┊ GC (min … max): 0.00% … 60.12%
 Time  (median):     2.369 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.518 ms ± 629.929 μs  ┊ GC (mean ± σ):  4.27% ±  9.93%

  █▆▆▅▃▁                                                       
  ██████▇▆▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▃▅▅▆▃▆▆▆▅▅▅ █
  2.3 ms       Histogram: log(frequency) by time      6.08 ms <

 Memory estimate: 1.73 MiB, allocs estimate: 2655.
```



```julia
using ADNLPModels

@benchmark begin
  adnlp = ADNLPModel(β -> myfun(β, X, y), zeros(p + 1))
  output = with_logger(NullLogger()) do
    lbfgs(adnlp)
  end
end
```

```
BenchmarkTools.Trial: 55 samples with 1 evaluation.
 Range (min … max):  89.006 ms … 100.059 ms  ┊ GC (min … max): 0.00% … 4.55%
 Time  (median):     90.649 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   92.107 ms ±   2.914 ms  ┊ GC (mean ± σ):  1.48% ± 2.23%

      █ ▄ ▄                                                     
  ▄▄▄▁███▄██▄▆▄▆▄▄▄▁▄▁▄▁▁▁▄▁▁▁▁▁▁▁▁▄▁▄▄█▄▄▁▄▁▁▁▁▁▁▁▁▄▁▄▆▁▁▁▄▁▄ ▁
  89 ms           Histogram: frequency by time         98.5 ms <

 Memory estimate: 30.58 MiB, allocs estimate: 5234.
```



```julia
using JuMP
using NLPModelsJuMP

@benchmark begin
  model = Model()
  @variable(model, modelβ[1:p+1])
  @NLexpression(model,
    xᵀβ[i=1:n],
    modelβ[1] + sum(modelβ[j + 1] * X[i,j] for j = 1:p)
  )
  @NLexpression(
    model,
    hβ[i=1:n],
    1 / (1 + exp(-xᵀβ[i]))
  )
  @NLobjective(model, Min,
    -sum(y[i] * log(hβ[i] + 1e-8) + (1 - y[i] * log(hβ[i] + 1e-8)) for i = 1:n) / n + 0.5e-4 * sum(modelβ[i]^2 for i = 1:p+1)
  )
  jumpnlp = MathOptNLPModel(model)
  output = with_logger(NullLogger()) do
    lbfgs(jumpnlp)
  end
end
```

```
BenchmarkTools.Trial: 34 samples with 1 evaluation.
 Range (min … max):  134.129 ms … 167.896 ms  ┊ GC (min … max): 0.00% … 11.86%
 Time  (median):     151.700 ms               ┊ GC (median):    9.48%
 Time  (mean ± σ):   150.414 ms ±  10.865 ms  ┊ GC (mean ± σ):  6.69% ±  5.15%

  █      ▃  ▃                ▃                     █        ▃    
  █▇▇▇▁▁▁█▁▁█▁▁▁▇▁▁▇▁▁▁▁▁▁▁▇▁█▁▇▇▁▇▇▇▇▁▁▇▁▁▇▇▁▇▁▇▁▁█▇▁▁▇▁▁▁▁█▁▇ ▁
  134 ms           Histogram: frequency by time          168 ms <

 Memory estimate: 98.00 MiB, allocs estimate: 436895.
```





Or just the grad calls:

```julia
using NLPModels

@benchmark grad(nlp, β)
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  12.400 μs …  11.011 ms  ┊ GC (min … max): 0.00% … 89.41%
 Time  (median):     15.000 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   16.621 μs ± 111.105 μs  ┊ GC (mean ± σ):  5.92% ±  0.89%

    ▂▅▅▆██▇▆▅▄▅▆▅▅▅▄▄▄▂                                         
  ▂▆█████████████████████▇▆▅▄▄▃▃▂▃▂▂▂▁▂▁▂▂▂▁▁▁▁▁▁▁▁▁▂▁▂▁▁▁▁▁▁▁ ▄
  12.4 μs         Histogram: frequency by time         24.2 μs <

 Memory estimate: 18.19 KiB, allocs estimate: 8.
```



```julia
adnlp = ADNLPModel(β -> myfun(β, X, y), zeros(p + 1))
@benchmark grad(adnlp, β)
```

```
BenchmarkTools.Trial: 3578 samples with 1 evaluation.
 Range (min … max):  1.312 ms …   7.034 ms  ┊ GC (min … max): 0.00% … 74.77%
 Time  (median):     1.345 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   1.395 ms ± 304.858 μs  ┊ GC (mean ± σ):  1.14% ±  4.23%

    ▄█▇▅▂                                                      
  ▂▅█████▇▅▃▃▃▂▃▂▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▂▂▂▂▂▁▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  1.31 ms         Histogram: frequency by time        1.61 ms <

 Memory estimate: 471.91 KiB, allocs estimate: 42.
```



```julia
model = Model()
@variable(model, modelβ[1:p+1])
@NLexpression(model,
  xᵀβ[i=1:n],
  modelβ[1] + sum(modelβ[j + 1] * X[i,j] for j = 1:p)
)
@NLexpression(
  model,
  hβ[i=1:n],
  1 / (1 + exp(-xᵀβ[i]))
)
@NLobjective(model, Min,
  -sum(y[i] * log(hβ[i] + 1e-8) + (1 - y[i] * log(hβ[i] + 1e-8)) for i = 1:n) / n + 0.5e-4 * sum(modelβ[i]^2 for i = 1:p+1)
)
jumpnlp = MathOptNLPModel(model)
@benchmark grad(jumpnlp, β)
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  232.104 μs … 901.110 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     235.103 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   236.836 μs ±   9.055 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

    ▁▃▄▆▇██▇▇▆▅▄▄▃▂▁  ▁ ▂  ▁▂▂▃▂▃▃▂▂▂▁▁         ▁▁▁▁▁▁ ▁        ▂
  ▄▆█████████████████████████████████████▇▇▇▆███████████▇▇▆▆▆▆▆ █
  232 μs        Histogram: log(frequency) by time        251 μs <

 Memory estimate: 496 bytes, allocs estimate: 1.
```





Take these benchmarks with a grain of salt. They are being run on a github actions server with global variables.
If you want to make an informed option, you should consider performing your own benchmarks.

