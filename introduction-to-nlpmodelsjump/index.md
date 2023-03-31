---
author: "Abel Soares Siqueira and Alexis Montoison"
title: "NLPModelsJuMP.jl tutorial"
tags:
  - "models"
  - "nlpmodels"
  - "manual"
---

[![NLPModels 0.20.0](https://img.shields.io/badge/NLPModels-0.20.0-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/NLPModels.jl/stable/)
[![NLPModelsJuMP 0.12.1](https://img.shields.io/badge/NLPModelsJuMP-0.12.1-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/NLPModelsJuMP.jl/stable/)
![JuMP 1.9.0](https://img.shields.io/badge/JuMP-1.9.0-000?style=flat-square&labelColor=999)
[![OptimizationProblems 0.6.0](https://img.shields.io/badge/OptimizationProblems-0.6.0-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/OptimizationProblems.jl/stable/)



NLPModelsJuMP is a combination of NLPModels and JuMP, as the name implies.
Sometimes it may be required to refer to the specific documentation, as we'll present
here only the documention specific to NLPModelsJuMP.

## MathOptNLPModel

`MathOptNLPModel` is a simple yet efficient model. It uses JuMP to define the problem,
and can be accessed through the NLPModels API.
An advantage of `MathOptNLPModel` over models such as [`ADNLPModels`](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) is that
they provide sparse derivates.

Let's define the famous Rosenbrock function
$$
f(x) = (x_1 - 1)^2 + 100(x_2 - x_1^2)^2,
$$
with starting point $x^0 = (-1.2,1.0)$.

```julia
using NLPModels, NLPModelsJuMP, JuMP

x0 = [-1.2; 1.0]
model = Model() # No solver is required
@variable(model, x[i=1:2], start=x0[i])
@NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)

nlp = MathOptNLPModel(model)
```

```
NLPModelsJuMP.MathOptNLPModel
  Problem name: Generic
   All variables: ████████████████████ 2      All constraints: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            free: ████████████████████ 2                 free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
          infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            nnzh: (  0.00% sparsity)   3               linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
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





Let's get the objective function value at $x^0$, using only `nlp`.

```julia
fx = obj(nlp, nlp.meta.x0)
println("fx = $fx")
```

```
fx = 24.199999999999996
```





Let's try the gradient and Hessian.

```julia
gx = grad(nlp, nlp.meta.x0)
Hx = hess(nlp, nlp.meta.x0)
println("gx = $gx")
println("Hx = $Hx")
```

```
gx = [-215.59999999999997, -87.99999999999999]
Hx = [1330.0 480.0; 480.0 200.0]
```





Let's do something a little more complex here, defining a function to try to
solve this problem through steepest descent method with Armijo search.
Namely, the method

1. Given $x^0$, $\varepsilon > 0$, and $\eta \in (0,1)$. Set $k = 0$;
2. If $\Vert \nabla f(x^k) \Vert < \varepsilon$ STOP with $x^* = x^k$;
3. Compute $d^k = -\nabla f(x^k)$;
4. Compute $\alpha_k \in (0,1]$ such that $f(x^k + \alpha_kd^k) < f(x^k) + \alpha_k\eta \nabla f(x^k)^Td^k$
5. Define $x^{k+1} = x^k + \alpha_kx^k$
6. Update $k = k + 1$ and go to step 2.

```julia
using LinearAlgebra

function steepest(nlp; itmax=100000, eta=1e-4, eps=1e-6, sigma=0.66)
  x = nlp.meta.x0
  fx = obj(nlp, x)
  ∇fx = grad(nlp, x)
  slope = dot(∇fx, ∇fx)
  ∇f_norm = sqrt(slope)
  iter = 0
  while ∇f_norm > eps && iter < itmax
    t = 1.0
    x_trial = x - t * ∇fx
    f_trial = obj(nlp, x_trial)
    while f_trial > fx - eta * t * slope
      t *= sigma
      x_trial = x - t * ∇fx
      f_trial = obj(nlp, x_trial)
    end
    x = x_trial
    fx = f_trial
    ∇fx = grad(nlp, x)
    slope = dot(∇fx, ∇fx)
    ∇f_norm = sqrt(slope)
    iter += 1
  end
  optimal = ∇f_norm <= eps
  return x, fx, ∇f_norm, optimal, iter
end

x, fx, ngx, optimal, iter = steepest(nlp)
println("x = $x")
println("fx = $fx")
println("ngx = $ngx")
println("optimal = $optimal")
println("iter = $iter")
```

```
x = [1.0000006499501406, 1.0000013043156974]
fx = 4.2438440239813445e-13
ngx = 9.984661274466946e-7
optimal = true
iter = 17962
```





Maybe this code is too complicated? If you're in a class you just want to show a
Newton step.

```julia
f(x) = obj(nlp, x)
g(x) = grad(nlp, x)
H(x) = hess(nlp, x)
x = nlp.meta.x0
d = -H(x) \ g(x)
```

```
2-element Vector{Float64}:
 0.02471910112359557
 0.3806741573033706
```





or a few

```julia
for i = 1:5
  global x
  x = x - H(x) \ g(x)
  println("x = $x")
end
```

```
x = [-1.1752808988764043, 1.3806741573033703]
x = [0.7631148711766087, -3.1750338547485217]
x = [0.7634296788842126, 0.5828247754973592]
x = [0.9999953110849883, 0.9440273238534098]
x = [0.9999956956536664, 0.9999913913257125]
```





### OptimizationProblems

The package
[OptimizationProblems](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl)
provides a collection of problems defined in JuMP format, which can be converted
to `MathOptNLPModel`.

```julia
using OptimizationProblems.PureJuMP  # Defines a lot of JuMP models

nlp = MathOptNLPModel(woods())
x, fx, ngx, optimal, iter = steepest(nlp)
println("fx = $fx")
println("ngx = $ngx")
println("optimal = $optimal")
println("iter = $iter")
```

```
fx = 2.200338951411302e-13
ngx = 9.97252246367067e-7
optimal = true
iter = 12688
```





Constrained problems can also be converted.

```julia
using NLPModels, NLPModelsJuMP, JuMP

model = Model()
x0 = [-1.2; 1.0]
@variable(model, x[i=1:2] >= 0.0, start=x0[i])
@NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)
@constraint(model, x[1] + x[2] == 3.0)
@NLconstraint(model, x[1] * x[2] >= 1.0)

nlp = MathOptNLPModel(model)

println("cx = $(cons(nlp, nlp.meta.x0))")
println("Jx = $(jac(nlp, nlp.meta.x0))")
```

```
cx = [-0.19999999999999996, -2.2]
Jx = sparse([1, 2, 1, 2], [1, 1, 2, 2], [1.0, 1.0, 1.0, -1.2], 2, 2)
```





## MathOptNLSModel

`MathOptNLSModel` is a model for nonlinear least squares using JuMP, The objective
function of NLS problems has the form $f(x) = \tfrac{1}{2}\|F(x)\|^2$, but specialized
methods handle $F$ directly, instead of $f$.
To use `MathOptNLSModel`, we define a JuMP model without the objective, and use `NLexpression`s to
define the residual function $F$.
For instance, the Rosenbrock function can be expressed in nonlinear least squares format by
defining
$$
F(x) = \begin{bmatrix} x_1 - 1\\ 10(x_2 - x_1^2) \end{bmatrix},
$$
and noting that $f(x) = \|F(x)\|^2$ (the constant $\frac{1}{2}$ is ignored as it
doesn't change the solution).
We implement this function as

```julia
using NLPModels, NLPModelsJuMP, JuMP

model = Model()
x0 = [-1.2; 1.0]
@variable(model, x[i=1:2], start=x0[i])
@NLexpression(model, F1, x[1] - 1)
@NLexpression(model, F2, 10 * (x[2] - x[1]^2))

nls = MathOptNLSModel(model, [F1, F2], name="rosen-nls")

residual(nls, nls.meta.x0)
```

```
2-element Vector{Float64}:
 -2.2
 -4.3999999999999995
```



```julia
jac_residual(nls, nls.meta.x0)
```

```
2×2 SparseArrays.SparseMatrixCSC{Float64, Int64} with 3 stored entries:
  1.0    ⋅ 
 24.0  10.0
```





### NLSProblems

The package
[NLSProblems](https://github.com/JuliaSmoothOptimizers/NLSProblems.jl)
provides a collection of problems already defined as `MathOptNLSModel`.