---
author: "Geoffroy Leconte"
title: "LDLFactorizations tutorial"
tags:
  - "linear"
  - "factorization"
  - "ldlt"
---


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
 0.5660694086865138
 0.5273551349536265
 0.6735903537477889
 0.1561574382691362
 0.17893628011342344
 0.5721260059416873
 0.15083159688355785
 0.3691127937862808
 0.6541589342736945
 0.3272233457971805
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
  0.10078524257098356
  0.11999160722507905
  0.10857294298364037
 -0.13446746085716196
 -0.08089179380941809
  0.14540789809236696
 -0.09942094396271997
 -0.06556478935034776
  0.0601389175020337
 -0.07438101435897126
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
 0.4312721811985666
 0.2493767122714866
 0.4395250557934327
 0.7815864121162811
 0.5225808617219967
 0.6291199978277422
 0.4273042789613629
 0.27381910540993326
 0.8813304514303829
 0.24825300633214709
 ⋮
 0.8608114352057055
 0.6247109093272826
 0.3870381167247071
 0.14261506556506542
 0.2712942181949832
 0.1906716440941686
 0.28934141458084384
 0.6503469290773818
 0.6310160663069263
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
  0.10897297529400288
  0.5518306481759249
  0.03587708330331003
  0.1566644106503779
 -0.08919471622858571
 -0.543885234977905
 -0.6080490524262658
 -0.4492992384252682
  0.3431938978502821
 -0.07006295526282472
  ⋮
 -0.3771204892578631
  0.30944191870102
 -0.26199463780011034
  0.0770332277239776
  0.45581870171416894
 -0.04785752982798173
  0.017216290117848684
  0.42576352474763224
  0.028516338655068182
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
LDLFactorizations.LDLFactorization{Float64, Int64, Int64, Int64}(true, true, true, 200, [3, 3, 30, 31, 30, 9, 8, 9, 30, 11  …  192, 193, 194, 195, 196, 197, 198, 199, 200, -1], [48, 50, 100, 46, 41, 5
8, 56, 81, 121, 16  …  9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [199, 174, 199, 173, 198, 199, 196, 200, 200, 196  …  200, 200, 200, 200, 200, 200, 200, 200, 200, 200], [181, 51, 14, 46, 81, 153, 151, 114, 116,
 11  …  192, 193, 194, 195, 196, 197, 198, 199, 200, 162], [30, 31, 32, 33, 34, 35, 36, 27, 37, 38  …  16, 191, 192, 193, 194, 195, 196, 197, 198, 199], [1, 49, 99, 199, 245, 286, 344, 400, 481, 602  
…  16762, 16770, 16777, 16783, 16788, 16792, 16795, 16797, 16798, 16798], [1, 10, 18, 30, 41, 54, 67, 76, 81, 91  …  725, 725, 725, 725, 725, 725, 725, 725, 725, 725], [13, 14, 40, 51, 55, 68, 81, 82,
 151, 8  …  181, 182, 181, 182, 191, 191, 191, 191, 191, 191], [3, 32, 34, 35, 42, 60, 71, 82, 101, 102  …  197, 198, 199, 200, 198, 199, 200, 199, 200, 200], [-0.05006145563815363, -0.515181503476433
5, -0.07071764936680519, -0.4827072920352396, -0.2397320529072592, -0.02768743016770725, -0.2766407256013685, -0.31621610248774357, -0.11509956519874202, -0.08358199965040854  …  -0.03552443596852614,
 -0.07773129626422923, -0.039165118105957096, 0.02199454802822853, -0.012148155123951529, 0.13660357419563443, -0.04635070336595878, 0.01056788920974842, 0.07008089289077785, -0.06545137761842088], [-
1.3188964056788095, 2.7389725200714063, 3.1073603727460566, 2.298590628587391, 1.3188964056788095, -3.117735153792235, -2.7389725200714063, -3.1040550213886307, -4.30174992997638, 2.5674554309881055  
…  -3.3535841802651563, -4.298199215577154, -4.106688992747375, -2.3293851446089637, -2.7894406550010054, -4.213574579195694, -3.087166286068581, -3.7840197532571485, -4.8274459697760825, -3.803435842
0665164], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [12, 9, 16, 32, 33, 34, 35, 36, 37, 38  …  190, 191, 192, 193, 194, 195, 196, 197, 19
8, 199], 2.9802322387695312e-8, -1.4901161193847656e-8, 1.4901161193847656e-8, 10)
```


