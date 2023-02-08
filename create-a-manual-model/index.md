---
author: "Abel S. Siqueira"
title: "How to create a model from the function and its derivatives"
tags:
  - "models"
  - "manual"
---


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
BenchmarkTools.Trial: 1854 samples with 1 evaluation.
 Range (min … max):  2.301 ms …   7.886 ms  ┊ GC (min … max): 0.00% … 54.35%
 Time  (median):     2.484 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.692 ms ± 885.166 μs  ┊ GC (mean ± σ):  5.70% ± 11.18%

  ███▇▅▃    ▂▂                                                ▁
  ███████▇▆▇██▇█▆▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▅▅▆▆▆▆▇▅▇ █
  2.3 ms       Histogram: log(frequency) by time      7.32 ms <

 Memory estimate: 1.72 MiB, allocs estimate: 2594.
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
BenchmarkTools.Trial: 52 samples with 1 evaluation.
 Range (min … max):  91.143 ms … 108.271 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     95.760 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   96.220 ms ±   3.646 ms  ┊ GC (mean ± σ):  1.67% ± 2.35%

     █▁ █     ▁ ▄▁    ▁    █ ▁  ▁▁▁   ▁▁ ▁                      
  ▆▆▁██▆█▆▆▁▆▆█▁██▁▆▆▁█▁▆▆▁█▁█▁▁███▁▁▆██▁█▁▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▆ ▁
  91.1 ms         Histogram: frequency by time          106 ms <

 Memory estimate: 30.79 MiB, allocs estimate: 10075.
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
BenchmarkTools.Trial: 32 samples with 1 evaluation.
 Range (min … max):  137.333 ms … 170.311 ms  ┊ GC (min … max):  0.00% … 12.93%
 Time  (median):     160.668 ms               ┊ GC (median):    10.39%
 Time  (mean ± σ):   158.182 ms ±  10.621 ms  ┊ GC (mean ± σ):   7.41% ±  5.30%

                          ▂                              █       
  █▁▅▅▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁█▅▁▁▁▁▅▅▁▅▁▁▁▁▅▁▅▁▅▅▁▁▁▁▁▁▅▁▁███▅▅▅▅▅ ▁
  137 ms           Histogram: frequency by time          170 ms <

 Memory estimate: 98.00 MiB, allocs estimate: 436897.
```





Or just the grad calls:

```julia
using NLPModels

@benchmark grad(nlp, β)
```

```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  13.200 μs …  1.547 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     16.200 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   16.877 μs ± 19.347 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

    ▃▂▆██▃█▇█▃▇▇▂▅▃▂                                           
  ▃▇████████████████▆█▇▆▄▅▄▃▃▃▃▂▃▂▂▂▂▂▂▂▂▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂ ▄
  13.2 μs         Histogram: frequency by time        29.3 μs <

 Memory estimate: 18.19 KiB, allocs estimate: 8.
```



```julia
adnlp = ADNLPModel(β -> myfun(β, X, y), zeros(p + 1))
@benchmark grad(adnlp, β)
```

```
BenchmarkTools.Trial: 3408 samples with 1 evaluation.
 Range (min … max):  1.313 ms …  10.658 ms  ┊ GC (min … max): 0.00% … 83.99%
 Time  (median):     1.436 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   1.464 ms ± 373.879 μs  ┊ GC (mean ± σ):  1.32% ±  4.36%

    ▄█▄▁                                                       
  ▂▇████▇▅▅▅▅▅▅▄▅▅▅▆▅▅▅▆▆█▇▆▆▅▅▅▆▅▄▄▃▄▄▃▃▃▃▃▃▂▂▂▂▂▂▁▁▁▁▁▂▁▁▁▁ ▃
  1.31 ms         Histogram: frequency by time        1.72 ms <

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
 Range (min … max):  232.004 μs … 897.917 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     236.705 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   238.315 μs ±   9.441 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

          ▂▄▆█▇▄▁                                                
  ▁▁▂▃▅▆█▇███████▆▅▄▃▃▃▂▂▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▃
  232 μs           Histogram: frequency by time          256 μs <

 Memory estimate: 496 bytes, allocs estimate: 1.
```





Take these benchmarks with a grain of salt. They are being run on a github actions server with global variables.
If you want to make an informed option, you should consider performing your own benchmarks.


