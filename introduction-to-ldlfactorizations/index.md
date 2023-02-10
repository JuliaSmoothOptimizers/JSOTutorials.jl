---
author: "Geoffroy Leconte"
title: "LDLFactorizations tutorial"
tags:
  - "linear"
  - "factorization"
  - "ldlt"
---

![JSON 0.21.3](https://img.shields.io/badge/JSON-0.21.3-000?style=flat-square&labelColor=fff")
![MatrixMarket 0.3.1](https://img.shields.io/badge/MatrixMarket-0.3.1-000?style=flat-square&labelColor=fff")
[![RipQP 0.6.1](https://img.shields.io/badge/RipQP-0.6.1-006400?style=flat-square&labelColor=389826")](https://juliasmoothoptimizers.github.io/RipQP.jl/stable/)
[![SparseMatricesCOO 0.2.1](https://img.shields.io/badge/SparseMatricesCOO-0.2.1-4b0082?style=flat-square&labelColor=9558b2")](https://juliasmoothoptimizers.github.io/SparseMatricesCOO.jl/stable/)
[![QuadraticModels 0.9.3](https://img.shields.io/badge/QuadraticModels-0.9.3-8b0000?style=flat-square&labelColor=cb3c33")](https://juliasmoothoptimizers.github.io/QuadraticModels.jl/stable/)
![Plots 1.38.5](https://img.shields.io/badge/Plots-1.38.5-000?style=flat-square&labelColor=fff")
[![QPSReader 0.2.1](https://img.shields.io/badge/QPSReader-0.2.1-8b0000?style=flat-square&labelColor=cb3c33")](https://juliasmoothoptimizers.github.io/QPSReader.jl/stable/)
[![LDLFactorizations 0.10.0](https://img.shields.io/badge/LDLFactorizations-0.10.0-4b0082?style=flat-square&labelColor=9558b2")](https://juliasmoothoptimizers.github.io/LDLFactorizations.jl/stable/)
![TimerOutputs 0.5.22](https://img.shields.io/badge/TimerOutputs-0.5.22-000?style=flat-square&labelColor=fff")



LDLFactorizations.jl is a translation of Tim Davis's Concise LDLᵀ Factorization, part of [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) with several improvements.

This package is appropriate for matrices A that possess a factorization of the
form LDLᵀ without pivoting, where L is unit lower triangular and D is *diagonal* (indefinite in general), including definite and quasi-definite matrices.

## A basic example

```julia
using LinearAlgebra
n = 10
A0 = rand(n, n)
A = A0 * A0' + I # A is symmetric positive definite
b = rand(n)
```

```
10-element Vector{Float64}:
 0.44679736306833084
 0.13639980582628786
 0.9982629703253633
 0.7629195278883684
 0.48490360667055765
 0.33011916053291135
 0.45420827770306094
 0.12970399180240144
 0.43043562309322114
 0.6688148235593476
```





We solve the system $A x = b$ using LDLFactorizations.jl:

```julia
using LDLFactorizations
Au = Symmetric(triu(A), :U) # get upper triangle and apply Symmetric wrapper
LDL = ldl(Au)
x = LDL \ b
```

```
10-element Vector{Float64}:
  0.1309474733647077
 -0.29907176394538615
  0.31371173047122763
  0.27919712024167503
  0.011283767132091567
 -0.1795392351481066
  0.13842198850140885
 -0.17006052688691836
  0.15031946088362338
  0.09388146234911694
```





## A more performance-focused example

We build a problem with sparse arrays.

```julia
using SparseArrays
n = 100
# create create a SQD matrix A:
A0 = sprand(Float64, n, n, 0.1)
A1 = A0 * A0' + I
A = [A1   A0;
     A0' -A1]
b = rand(2 * n)
```

```
200-element Vector{Float64}:
 0.923673117495888
 0.8060597817052443
 0.2776980634508168
 0.22330694838001697
 0.6840044397602859
 0.7801121129807327
 0.7769252273511311
 0.0499736797787228
 0.3401065709994954
 0.20176934688168413
 ⋮
 0.6175148856647636
 0.8815797568066227
 0.9818353693955151
 0.9694605678955173
 0.46542915237794746
 0.5133701279205034
 0.6838328898640792
 0.150983075500008
 0.10207080354422882
```





Now if we want to use the factorization to solve multiple systems that have 
the same sparsity pattern as A, we only have to use `ldl_analyze` once.

```julia
Au = Symmetric(triu(A), :U) # get upper triangle and apply Symmetric wrapper
x = similar(b)

LDL = ldl_analyze(Au) # symbolic analysis
ldl_factorize!(Au, LDL) # factorization
ldiv!(x, LDL, b) # solve in-place (we could use ldiv!(LDL, b) if we want to overwrite b)

Au.data.nzval .+= 1.0 # modify Au without changing the sparsity pattern
ldl_factorize!(Au, LDL) 
ldiv!(x, LDL, b)
```

```
200-element Vector{Float64}:
 -2.226016635185447
  0.19153818495589314
  1.5224079286171288
 -0.12301117268000783
  2.6104026453678495
  0.12930237808094366
  0.9720712124269522
  1.6187703827258055
 -1.3675275327753298
 -2.949028618897882
  ⋮
  0.5917076768932824
  0.6230028619146927
  0.8541210029180326
 -0.28589878351077425
 -0.9683289577789973
  1.0326625876677429
 -0.970236005464521
 -0.11582691627290392
  2.1486203256526952
```





## Dynamic Regularization

When the matrix to factorize is (nearly) singular and the factorization encounters (nearly) zero pivots, 
if we know the signs of the pivots and if they are clustered by signs (for example, the 
`n_d` first pivots are positive and the other pivots are negative before permuting), we can use:

```julia
ϵ = sqrt(eps())
Au = Symmetric(triu(A), :U)
LDL = ldl_analyze(Au)
LDL.tol = ϵ
LDL.n_d = 10
LDL.r1 = 2 * ϵ # if any of the n_d first pivots |D[i]| < ϵ, then D[i] = sign(LDL.r1) * max(abs(D[i] + LDL.r1), abs(LDL.r1))
LDL.r2 = -ϵ # if any of the n - n_d last pivots |D[i]| < ϵ, then D[i] = sign(LDL.r2) * max(abs(D[i] + LDL.r2), abs(LDL.r2))
ldl_factorize!(Au, LDL)
```

```
LDLFactorizations.LDLFactorization{Float64, Int64, Int64, Int64}(true, true, true, 200, [26, 5, 4, 5, 11, 10, 10, 9, 10, 11  …  192, 193, 194, 195, 196, 197, 198, 199, 200, -1], [31, 58, 52, 81, 109, 
35, 61, 42, 76, 113  …  9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [197, 183, 173, 195, 195, 186, 199, 196, 196, 199  …  200, 200, 200, 200, 200, 200, 200, 200, 200, 200], [73, 63, 20, 78, 36, 6, 87, 1, 100, 57  
…  191, 192, 194, 195, 196, 197, 198, 199, 200, 149], [8, 25, 26, 27, 28, 6, 29, 30, 31, 32  …  191, 192, 13, 193, 194, 195, 196, 197, 198, 199], [1, 32, 90, 142, 223, 332, 367, 428, 470, 546  …  1724
3, 17251, 17258, 17264, 17269, 17273, 17276, 17278, 17279, 17279], [1, 1, 7, 11, 18, 23, 23, 35, 43, 50  …  783, 783, 783, 783, 783, 783, 783, 783, 783, 783], [6, 36, 57, 63, 78, 87, 36, 63, 73, 106  
…  184, 193, 184, 187, 187, 187, 187, 193, 193, 193], [26, 29, 30, 34, 35, 40, 41, 46, 50, 54  …  197, 198, 199, 200, 198, 199, 200, 199, 200, 200], [0.09203679675420152, 0.3586961770865634, 0.0183518
55821856, 0.07800007492711891, 0.0533557942849594, 0.12765448431788354, 0.1277320081983776, 0.009440488474003023, 0.0878922080011437, 0.20842621991179397  …  0.028348635099511677, 0.10245298867845785,
 -0.06986402579877125, -0.011582439846244477, -0.05676577186831453, 0.15618693811843998, 0.06273458850629737, -0.006339929923500262, -0.0355198780341459, -0.030500123784741325], [1.8972010602821314, 4
.516886376712483, 3.2312450825417405, 2.8901516378880316, 4.21864206790784, 2.6925536998536925, 4.952579314396723, 2.496941463791387, 3.09451004124551, 2.584583137973754  …  -3.382524238503639, -3.017
174364961268, -2.4150386519388425, -2.9902243826376473, -2.9280879742386126, -3.0009952526072734, -2.79155187002199, -3.6842848186700565, -2.5582175678312007, -2.8736675166218633], [0.0, 0.0, 0.0, 0.0
, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [19, 18, 22, 23, 24, 29, 30, 31, 40, 41  …  190, 191, 192, 193, 194, 195, 196, 197, 198, 199], 2.9802322387695312e
-8, -1.4901161193847656e-8, 1.4901161193847656e-8, 10)
```

