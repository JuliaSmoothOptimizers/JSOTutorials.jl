---
author: "Geoffroy Leconte"
title: "Introduction to RipQP"
tags:
  - "introduction"
  - "quadratic"
  - "regularized"
  - "solver"
---


## Input

RipQP uses the package [QuadraticModels.jl](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl) to model
convex quadratic problems.

Here is a basic example:

```julia
using QuadraticModels, LinearAlgebra, SparseMatricesCOO
Q = [6. 2. 1.
     2. 5. 2.
     1. 2. 4.]
c = [-8.; -3; -3]
A = [1. 0. 1.
     0. 2. 1.]
b = [0.; 3]
l = [0.;0;0]
u = [Inf; Inf; Inf]
QM = QuadraticModel(
  c,
  SparseMatrixCOO(tril(Q)),
  A=SparseMatrixCOO(A),
  lcon=b,
  ucon=b,
  lvar=l,
  uvar=u,
  c0=0.,
  name="QM"
)
```

```
QuadraticModels.QuadraticModel{Float64, Vector{Float64}, SparseMatricesCOO.SparseMatrixCOO{Float64, Int64}, SparseMatricesCOO.SparseMatrixCOO{Float64, Int64}}
  Problem name: QM
   All variables: ████████████████████ 3      All constraints: ████████████████████ 2     
            free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           lower: ████████████████████ 3                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ████████████████████ 2     
          infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            nnzh: (  0.00% sparsity)   6               linear: ████████████████████ 2     
                                                    nonlinear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
                                                         nnzj: ( 33.33% sparsity)   4     

  Counters:
             obj: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 grad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
        cons_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0             cons_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              jac_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         jac_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0            jprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
       jprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0           jtprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
      jtprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
```





Once your `QuadraticModel` is loaded, you can simply solve it RipQP:

```julia
using RipQP
stats = ripqp(QM)
println(stats)
```

```
Generic Execution stats
  status: first-order stationary
  objective value: 1.125
  primal feasibility: Inf
  dual feasibility: Inf
  solution: [0.0  1.5  0.0]
  multipliers: [0.0  0.0]
  multipliers_L: [0.0  4.5  0.0]
  multipliers_U: [5.0  0.0  0.0]
  iterations: 0
  elapsed time: 0.011768102645874023
  solver specific:
    presolvedQM: nothing
    psoperations: [QuadraticModels.RemoveIfix{Float64, Vector{Float64}}(1, 0.0, -8.0)  QuadraticModels.RemoveIfix{Float64, Vector{Float64}}(3, 0.0, -3.0)  QuadraticModels.EmptyRow{Float64, Vector{Floa
t64}}(1)  QuadraticModels.SingletonRow{Float64, Vector{Float64}}(2, 2, 2.0, true, true)  QuadraticModels.RemoveIfix{Float64, Vector{Float64}}(2, 1.5, -3.0)]
```





The `stats` output is a
[GenericExecutionStats](https://juliasmoothoptimizers.github.io/SolverCore.jl/dev/reference/#SolverCore.GenericExecutionStats).

It is also possible to use the package [QPSReader.jl](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) in order to
read convex quadratic problems in MPS or SIF formats: (download [QAFIRO](https://raw.githubusercontent.com/JuliaSmoothOptimizers/RipQP.jl/main/test/QAFIRO.SIF))

```julia
using QPSReader, QuadraticModels
QM = QuadraticModel(readqps("QAFIRO.SIF"))
```

```
Error: SystemError: opening file "QAFIRO.SIF": No such file or directory
```





## Logging

RipQP displays some logs at each iterate.

You can deactivate logging with

```julia
stats = ripqp(QM, display = false)
```

```
"Execution stats: first-order stationary"
```





It is also possible to get a history of several quantities such as the primal and dual residuals and the relative primal-dual gap. These quantites are available in the dictionary `solver_specific` of the `stats`.

```julia
stats = ripqp(QM, history = true)
pddH = stats.solver_specific[:pddH]
```

```
Error: KeyError: key :pddH not found
```





## Change configuration and tolerances

You can use `RipQP` without scaling with:

```julia
stats = ripqp(QM, scaling = false)
```

```
"Execution stats: first-order stationary"
```





You can also change the [`RipQP.InputTol`](https://juliasmoothoptimizers.github.io/RipQP.jl/stable/API/#RipQP.InputTol) type to change the tolerances for the stopping criteria:

```julia
stats = ripqp(QM, itol = InputTol(max_iter = 100, ϵ_rb = 1e-4), scaling = false)
```

```
"Execution stats: first-order stationary"
```





## Save the Interior-Point system

At every iteration, RipQP solves two linear systems with the default Predictor-Corrector method (the affine system and the corrector-centering system), or one linear system with the Infeasible Path-Following method.

To save these systems, you can use:

```julia
w = SystemWrite(write = true, name="test_", kfirst = 4, kgap=3)
stats1 = ripqp(QM, w = w)
```

```
"Execution stats: first-order stationary"
```





This will save one matrix and the associated two right hand sides of the PC method every three iterations starting at
iteration four.
Then, you can read the saved files with:

```julia
using DelimitedFiles, MatrixMarket
K = MatrixMarket.mmread("test_K_iter4.mtx")
rhs_aff = readdlm("test_rhs_iter4_aff.rhs", Float64)[:]
rhs_cc =  readdlm("test_rhs_iter4_cc.rhs", Float64)[:]
```

```
Error: SystemError: opening file "test_K_iter4.mtx": No such file or directory
```





## Timers

You can see the elapsed time with:

```julia
stats1.elapsed_time
```

```
6.318092346191406e-5
```





For more advance timers you can use [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl):

```julia
using TimerOutputs
TimerOutputs.enable_debug_timings(RipQP)
reset_timer!(RipQP.to)
stats = ripqp(QM)
TimerOutputs.complement!(RipQP.to) # print complement of timed sections
show(RipQP.to, sortby = :firstexec)
```

```
────────────────────────────────────────────────────────────────────
                            Time                    Allocations      
                   ───────────────────────   ────────────────────────
 Tot / % measured:      720ms /   0.0%           29.4MiB /   0.0%    

 Section   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────
 ────────────────────────────────────────────────────────────────────
```


