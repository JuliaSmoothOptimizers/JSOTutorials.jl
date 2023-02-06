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
QuadraticModels.QuadraticModel{Float64, Vector{Float64}, SparseMatricesCOO.
SparseMatrixCOO{Float64, Int64}, SparseMatricesCOO.SparseMatrixCOO{Float64,
 Int64}}
  Problem name: QM
   All variables: ████████████████████ 3      All constraints: ████████████
████████ 2     
            free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           lower: ████████████████████ 3                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ████████████
████████ 2     
          infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            nnzh: (  0.00% sparsity)   6               linear: ████████████
████████ 2     
                                                    nonlinear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
⋅⋅⋅⋅⋅⋅⋅⋅ 0     
                                                         nnzj: ( 33.33% spa
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





Once your `QuadraticModel` is loaded, you can simply solve it RipQP:

```julia
using RipQP
stats = ripqp(QM)
println(stats)
```

```
Generic Execution stats
  status: first-order stationary
  objective value: 1.1249999997850777
  primal feasibility: 6.138085685914678e-11
  dual feasibility: 3.864633058014988e-10
  solution: [5.6676609869371275e-11  1.5000000000152136  4.704246989775501e
-12]
  multipliers: [-9.305014036788608  2.2499999995728555]
  multipliers_L: [4.305014036777336  7.44283917646875e-10  7.05501403716986
]
  multipliers_U: [0.0  0.0  0.0]
  iterations: 16
  elapsed time: 8.331439971923828
  solver specific:
    nvar_slack: 3
    pdd: 6.42239516535383e-10
    absolute_iter_cnt: 4
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
5-element Vector{Float64}:
 2.0425814644358002
 0.6845391618145797
 0.0006472337157159482
 6.426950908481628e-7
 6.42239516535383e-10
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
5-element Vector{Float64}:
 -7.65907083541235e-9
  3.293857956802506e-15
  1.4101961075442407e-7
  0.0
  0.0
```





## Timers

You can see the elapsed time with:

```julia
stats1.elapsed_time
```

```
0.1480410099029541
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
──────────────────────────────────────────────────────────────────────────
──────
                                        Time                    Allocations
      
                               ───────────────────────   ──────────────────
──────
       Tot / % measured:            2.09s /  46.2%            185MiB /  58.
0%    

 Section               ncalls     time    %tot     avg     alloc    %tot   
   avg
 ──────────────────────────────────────────────────────────────────────────
──────
 ripqp                      1    965ms  100.0%   965ms    107MiB  100.0%   
107MiB
   ~ripqp~                  1    964ms   99.9%   964ms    107MiB   99.9%   
107MiB
   allocate workspace       1   78.2μs    0.0%  78.2μs   5.39KiB    0.0%  5
.39KiB
   init solver              1   43.0μs    0.0%  43.0μs   3.80KiB    0.0%  3
.80KiB
   display                  5    413μs    0.0%  82.6μs   49.8KiB    0.0%  1
0.0KiB
   update solver            4   21.6μs    0.0%  5.40μs      960B    0.0%   
  240B
   solver aff               4   2.30μs    0.0%   575ns     0.00B    0.0%   
 0.00B
   solver cc                4   1.10μs    0.0%   275ns     0.00B    0.0%   
 0.00B
 ──────────────────────────────────────────────────────────────────────────
──────
```


