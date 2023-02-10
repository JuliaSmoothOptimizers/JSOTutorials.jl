---
author: "Geoffroy Leconte"
title: "LDLFactorizations tutorial"
tags:
  - "linear"
  - "factorization"
  - "ldlt"
---

<img class="badge" src="https://img.shields.io/badge/JSON-0.21.3-000?style=flat-square&labelColor=fff">
<img class="badge" src="https://img.shields.io/badge/MatrixMarket-0.3.1-000?style=flat-square&labelColor=fff">
<a href="https://juliasmoothoptimizers.github.io/RipQP.jl/stable/"><img class="badge" src="https://img.shields.io/badge/RipQP-0.6.1-006400?style=flat-square&labelColor=389826"></a>
<a href="https://juliasmoothoptimizers.github.io/SparseMatricesCOO.jl/stable/"><img class="badge" src="https://img.shields.io/badge/SparseMatricesCOO-0.2.1-4b0082?style=flat-square&labelColor=9558b2"></a>
<a href="https://juliasmoothoptimizers.github.io/QuadraticModels.jl/stable/"><img class="badge" src="https://img.shields.io/badge/QuadraticModels-0.9.3-8b0000?style=flat-square&labelColor=cb3c33"></a>
<img class="badge" src="https://img.shields.io/badge/Plots-1.38.5-000?style=flat-square&labelColor=fff">
<a href="https://juliasmoothoptimizers.github.io/QPSReader.jl/stable/"><img class="badge" src="https://img.shields.io/badge/QPSReader-0.2.1-8b0000?style=flat-square&labelColor=cb3c33"></a>
<a href="https://juliasmoothoptimizers.github.io/LDLFactorizations.jl/stable/"><img class="badge" src="https://img.shields.io/badge/LDLFactorizations-0.10.0-4b0082?style=flat-square&labelColor=9558b2"></a>
<img class="badge" src="https://img.shields.io/badge/TimerOutputs-0.5.22-000?style=flat-square&labelColor=fff">



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
 0.9772725258214982
 0.018980826332634093
 0.09612309556665843
 0.5694473582961732
 0.609621476855787
 0.10424223011921119
 0.7546335610105603
 0.0720474426104003
 0.5044906405199465
 0.5858182827046466
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
  0.42329959039880044
 -0.30560953767132515
 -0.18003427824764923
  0.11956761293893751
  0.2285978758413831
 -0.2291266764418185
  0.24966957705956458
 -0.22356267021366077
 -0.023811451347362466
  0.1536569938555397
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
 0.2252928383235525
 0.09607684025054142
 0.1455689959461024
 0.2906770905908683
 0.06456831603960977
 0.8972294325885234
 0.5617575562632903
 0.35060146055866837
 0.3607379958543454
 0.10547276439550501
 ⋮
 0.8375334248331843
 0.35805959117579966
 0.2890981213623115
 0.4395962819301307
 0.12660836042250734
 0.07830002266227531
 0.9558929155099075
 0.6233299721765038
 0.5781481728236497
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
 -0.42849952047233825
  0.04778546130271077
  0.5156885446770194
 -0.22581564069461865
  0.9582888356295892
 -1.3951596103780548
  0.6980982555452031
 -0.8596319060492321
 -0.9491760894934234
 -0.8936462133871886
  ⋮
  0.12822148919402887
 -0.006710789682161481
  0.11925523162005253
  0.24720772158527166
 -0.00537273409780839
  0.18251616590415326
 -0.4594097967961835
 -0.010795705763598796
 -0.30839398090177833
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
LDLFactorizations.LDLFactorization{Float64, Int64, Int64, Int64}(true, true, true, 200, [2, 5, 4, 5, 25, 7, 11, 10, 10, 11  …  192, 193, 194, 195, 196, 197, 198, 199, 200, -1], [42, 66, 67, 95, 121, 5
1, 77, 50, 57, 100  …  9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [200, 200, 200, 200, 200, 169, 188, 190, 180, 199  …  200, 200, 200, 200, 200, 200, 200, 200, 200, 200], [67, 17, 94, 25, 19, 39, 14, 64, 41, 3  …
  191, 192, 193, 195, 196, 197, 198, 199, 200, 95], [25, 26, 10, 27, 28, 29, 30, 31, 32, 33  …  191, 192, 193, 14, 194, 195, 196, 197, 198, 199], [1, 43, 109, 176, 271, 392, 443, 520, 570, 627  …  173
65, 17373, 17380, 17386, 17391, 17395, 17398, 17400, 17401, 17401], [1, 7, 15, 17, 27, 31, 35, 41, 49, 55  …  624, 625, 626, 626, 626, 626, 626, 626, 626, 626], [17, 19, 39, 41, 94, 118, 3, 14, 17, 19
  …  194, 186, 194, 194, 194, 194, 194, 194, 194, 194], [2, 27, 28, 29, 30, 32, 34, 39, 41, 43  …  197, 198, 199, 200, 198, 199, 200, 199, 200, 200], [0.02125023424611219, 0.029156991518404455, 0.4213
225452636448, 0.08105850090389313, 0.03148142426812767, 0.1826934502606528, 0.1270077303095095, 0.05479977945875698, 0.10071312722187795, 0.02698925054828756  …  0.08860980585003056, -0.01237148947825
8134, 0.046299028225396686, -0.031150657872106677, -0.05830389652118573, 0.09599315840409402, -0.00547241360334812, -0.05412258716579334, -0.0012538025800229709, 0.01138402273804919], [2.3637238555955
62, 2.0181270395463047, 4.1226943358409684, 2.67697519023688, 5.781153152370618, 2.183190302814621, 2.1882329946630765, 2.034516936838124, 2.5822038260270146, 4.290056567099096  …  -3.2672216020349674
, -2.6856437278977747, -4.154135850789975, -4.1360126232822845, -3.244255821415339, -4.433457680955283, -4.14769760599786, -2.203727807016382, -3.994831808538892, 3.580197759518608], [0.0, 0.0, 0.0, 0
.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [14, 15, 16, 17, 29, 30, 31, 32, 33, 34  …  190, 191, 192, 193, 194, 195, 196, 197, 198, 199], 2.980232238769531
2e-8, -1.4901161193847656e-8, 1.4901161193847656e-8, 10)
```

