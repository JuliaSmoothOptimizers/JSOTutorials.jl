---
author: "Tangi Migot"
title: "Creating an ADNLPModels backend that supports multiple precisions"
tags:
  - "models"
  - "automatic differentiation"
  - "multi-precision"
  - "tests"
---
```julia
using ADNLPModels, ForwardDiff, NLPModels, OptimizationProblems
```




One of the main strengths of Julia for scientific computing is its native usage of [arbitrary precision arithmetic](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic).
The same can be exploited for optimization models and solvers.
In the organization [JuliaSmoothOptimizers](https://juliasmoothoptimizers.github.io), the package [ADNLPModels.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) provides automatic differentiation (AD)-based model implementations that conform to the NLPModels API.
This package is modular in the sense that it implements a backend system allowing the user to use essentially any AD system available, see [ADNLPModels.jl/dev/backend/](https://juliasmoothoptimizers.github.io/ADNLPModels.jl/dev/backend/) for a tutorial.

Note that most of the solvers available in [JuliaSmoothOptimizers](https://juliasmoothoptimizers.github.io) will accept generic types.
For instance, it is possible to use the classical L-BFGS method implemented in [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/) in single precision.

```julia
using JSOSolvers
f(x) = (x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2
x32 = Float32[-1.2; 1.0]
nlp = ADNLPModel(f, x32)
stats = lbfgs(nlp)
print(stats)
```

```
Generic Execution stats
  status: first-order stationary
  objective value: 0.00035444953
  primal feasibility: 0.0
  dual feasibility: 0.029705668
  solution: [0.98121893f0  0.9626594f0]
  iterations: 33
  elapsed time: 1.099910020828247
```





To design a multi-precision algorithm, we would also need to evaluate the model at a different precision, but this fails.

```julia
x16 = Float16[-1.2; 1.0]
grad(nlp, x16)
```

```
Error: Invalid Tag object:
  Expected ForwardDiff.Tag{typeof(Main.var"##WeaveSandBox#312".f), Float16}
,
  Observed ForwardDiff.Tag{typeof(Main.var"##WeaveSandBox#312".f), Float32}
.
```





In this tutorial, we will show how to modify the AD-backend in `ADNLPModel` to overcome this issue.

## Multiprecision ADNLPModels

Let's define the famous Rosenbrock function

$$
f(x) = (x_1 - 1)^2 + 100(x_2 - x_1^2)^2
$$

with starting point $x^0 = (-1.2,1.0)$, and its associated `ADNLPModel`.

```julia
f(x) = (x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2
T = Float64
x0 = T[-1.2; 1.0]
nlp = ADNLPModel(f, x0)
```

```
ADNLPModel - Model with automatic differentiation backend ADModelBackend{
  ForwardDiffADGradient,
  ForwardDiffADHvprod,
  ForwardDiffADJprod,
  ForwardDiffADJtprod,
  SparseADJacobian{Tuple{RuntimeGeneratedFunctions.RuntimeGeneratedFunction
{(:ˍ₋arg1,), Symbolics.var"#_RGF_ModTag", Symbolics.var"#_RGF_ModTag", (0xc
004344f, 0x41fbfc17, 0x653fb755, 0xc578bdb4, 0x51f4830a)}, RuntimeGenerated
Functions.RuntimeGeneratedFunction{(:ˍ₋out, :ˍ₋arg1), Symbolics.var"#_RGF_M
odTag", Symbolics.var"#_RGF_ModTag", (0x8b830f34, 0x90fedfc4, 0xe15f67a5, 0
xe2748718, 0x2a3cb564)}}},
  ForwardDiffADHessian,
  ForwardDiffADGHjvprod,
}
  Problem name: Generic
   All variables: ████████████████████ 2      All constraints: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            free: ████████████████████ 2                 free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
          infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            nnzh: (  0.00% sparsity)   3               linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
                                                    nonlinear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
                                                         nnzj: (------% spa
rsity)         

  Counters:
             obj: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 grad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0                 cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
        cons_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0             cons_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0                 jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0              jac_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         jac_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0            jprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
       jprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0           jtprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
      jtprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0                hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0
```





The [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) are usually parametrically typed by the vector and element type of `x0` and use this type for some pre-computations.
We now see how `ADNLPModel` can still be used for other types.

### Objective Evaluation

Note that in the input of the `ADNLPModel` constructor only the function `x0` is typed, while the objective function `f` can be generic.
Therefore, the function `obj(nlp, x)` will return an element of type `eltype(x)`.

```julia
x32 = Float32[-1.2; 1.0]
obj(nlp, x32) # type Float32
```

```
24.200005f0
```





### Gradient Evaluation

An `ADNLPModel` is parametrically typed by the vector and element type of `x0` and use this type for some pre-computations.
For instance, `ADNLPModel` may compute some default backend based on the expected element type to speed up the AD process.

```julia
adbackend = get_adbackend(nlp)
adbackend.gradient_backend # returns information about the default backend for the gradient computation.
```

```
ADNLPModels.ForwardDiffADGradient(ForwardDiff.GradientConfig{ForwardDiff.Ta
g{typeof(Main.var"##WeaveSandBox#312".f), Float64}, Float64, 2, Vector{Forw
ardDiff.Dual{ForwardDiff.Tag{typeof(Main.var"##WeaveSandBox#312".f), Float6
4}, Float64, 2}}}((Partials(1.0, 0.0), Partials(0.0, 1.0)), ForwardDiff.Dua
l{ForwardDiff.Tag{typeof(Main.var"##WeaveSandBox#312".f), Float64}, Float64
, 2}[Dual{ForwardDiff.Tag{typeof(Main.var"##WeaveSandBox#312".f), Float64}}
(6.93672140646395e-310,6.93672140646395e-310,6.93672140646395e-310), Dual{F
orwardDiff.Tag{typeof(Main.var"##WeaveSandBox#312".f), Float64}}(6.93672140
646395e-310,6.93672140646395e-310,6.93672140646395e-310)]))
```





We now show how to define your gradient-backend to keep the genericity in two steps:
- Define a new structure `GenericGradientBackend <: ADNLPModels.ADBackend`;
- Implements the function `ADNLPModels.gradient!` for this new backend.

We will use [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) to compute the gradient.
Note that the same can be done using alternatives such as [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl) or [Zygote.jl](https://github.com/FluxML/Zygote.jl).

```julia
struct GenericGradientBackend <: ADNLPModels.ADBackend end
GenericGradientBackend(args...; kwargs...) = GenericGradientBackend()

function ADNLPModels.gradient!(::GenericGradientBackend, g, f, x)
  return ForwardDiff.gradient!(g, f, x)
end
```




Once the new backend is defined it is possible to use it in the `ADNLPModel` constructor:

```julia
nlp = ADNLPModel(f, x0, gradient_backend = GenericGradientBackend)
```

```
ADNLPModel - Model with automatic differentiation backend ADModelBackend{
  Main.var"##WeaveSandBox#312".GenericGradientBackend,
  ForwardDiffADHvprod,
  ForwardDiffADJprod,
  ForwardDiffADJtprod,
  SparseADJacobian{Tuple{RuntimeGeneratedFunctions.RuntimeGeneratedFunction
{(:ˍ₋arg1,), Symbolics.var"#_RGF_ModTag", Symbolics.var"#_RGF_ModTag", (0xc
004344f, 0x41fbfc17, 0x653fb755, 0xc578bdb4, 0x51f4830a)}, RuntimeGenerated
Functions.RuntimeGeneratedFunction{(:ˍ₋out, :ˍ₋arg1), Symbolics.var"#_RGF_M
odTag", Symbolics.var"#_RGF_ModTag", (0x8b830f34, 0x90fedfc4, 0xe15f67a5, 0
xe2748718, 0x2a3cb564)}}},
  ForwardDiffADHessian,
  ForwardDiffADGHjvprod,
}
  Problem name: Generic
   All variables: ████████████████████ 2      All constraints: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            free: ████████████████████ 2                 free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
          infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            nnzh: (  0.00% sparsity)   3               linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
                                                    nonlinear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
                                                         nnzj: (------% spa
rsity)         

  Counters:
             obj: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 grad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0                 cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
        cons_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0             cons_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0                 jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0              jac_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         jac_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0            jprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
       jprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0           jtprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
      jtprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0                hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0
```





It is then possible to use the NLPModel API with any precision and compute the gradient

```julia
grad(nlp, x32) # returns a vector of Float32
```

```
2-element Vector{Float32}:
 -215.60004
  -88.000015
```





or the gradient in-place

```julia
x16 = Float16[-1.2; 1.0]
g = similar(x16)
grad!(nlp, x16, g) # returns a vector of Float16
```

```
2-element Vector{Float16}:
 -215.9
  -88.06
```





The same can be done for the other backends jacobian, hessian, etc.

## Multiprecision test problems

Designing a multi-precision algorithm is very often connected with benchmarking and test problems.
The package [OptimizationProblems.jl](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl) provides a collection of optimization problems in JuMP and ADNLPModels syntax, see [introduction to OptimizationProblems.jl tutorial](https://juliasmoothoptimizers.github.io/tutorials/introduction-to-optimizationproblems/).

This package provides a `DataFrame` with all the information on the implemented problems.

```julia
OptimizationProblems.meta[!, :name] # access the names of the available  problems
```

```
277-element Vector{String}:
 "AMPGO02"
 "AMPGO03"
 "AMPGO04"
 "AMPGO05"
 "AMPGO06"
 "AMPGO07"
 "AMPGO08"
 "AMPGO09"
 "AMPGO10"
 "AMPGO11"
 ⋮
 "tquartic"
 "triangle"
 "triangle_deer"
 "triangle_pacman"
 "triangle_turtle"
 "tridia"
 "vardim"
 "woods"
 "zangwil3"
```





In the following example, we use the problem `HS68`.

```julia
using OptimizationProblems.ADNLPProblems
name = :hs68
T = Float64
# Returns an `ADNLPModel` of element type `T` with GenericGradientBackend
nlp = eval(name)(type = Val(T), gradient_backend = GenericGradientBackend)
```

```
ADNLPModel - Model with automatic differentiation backend ADModelBackend{
  Main.var"##WeaveSandBox#312".GenericGradientBackend,
  ForwardDiffADHvprod,
  ForwardDiffADJprod,
  ForwardDiffADJtprod,
  SparseADJacobian{Tuple{RuntimeGeneratedFunctions.RuntimeGeneratedFunction
{(:ˍ₋arg1,), Symbolics.var"#_RGF_ModTag", Symbolics.var"#_RGF_ModTag", (0x1
3fc82cd, 0xa1ccdbc1, 0x1a6ff6d9, 0x5b86167c, 0x413bc14a)}, RuntimeGenerated
Functions.RuntimeGeneratedFunction{(:ˍ₋out, :ˍ₋arg1), Symbolics.var"#_RGF_M
odTag", Symbolics.var"#_RGF_ModTag", (0xa98a883c, 0x7ce5503f, 0x67760d36, 0
x355372f4, 0xba393467)}}},
  ForwardDiffADHessian,
  ForwardDiffADGHjvprod,
}
  Problem name: hs68
   All variables: ████████████████████ 4      All constraints: ████████████
████████ 2     
            free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         low/upp: ████████████████████ 4              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ████████████
████████ 2     
          infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            nnzh: (  0.00% sparsity)   10              linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
                                                    nonlinear: ████████████
████████ 2     
                                                         nnzj: ( 50.00% spa
rsity)   4     

  Counters:
             obj: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 grad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0                 cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
        cons_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0             cons_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0                 jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0              jac_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         jac_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0            jprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
       jprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0           jtprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
      jtprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0                hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0
```





The keyword arguments other than specific to a problem (`n`, `type`) are all passed to the constructor of the `ADNLPModel`.
In the example above `GenericGradientBackend` is used for the gradient backend.

```julia
x16 = Float16.(get_x0(nlp))
g = similar(x16)
grad!(nlp, x16, g) # returns a vector of Float16
```

```
4-element Vector{Float16}:
 -0.3704
 -0.0
  0.368
 -0.0
```





We should pay additional attention when using multiple precisions as casting, for instance `x0`, from `Float64` into `Float16` implies that rounding errors occur.
Therefore, `x0` is different than `x16`, and the gradients evaluated for these values too.

Feel free to look at [OptimizationProblems.jl documentation](https://juliasmoothoptimizers.github.io/OptimizationProblems.jl/dev/) to learn more or the tutorials at [juliasmoothoptimizers.github.io](https://juliasmoothoptimizers.github.io).
