---
author: "Geoffroy Leconte"
title: "LDLFactorizations tutorial"
tags:
  - "linear"
  - "factorization"
  - "ldlt"
---

![JSON 0.21.3](https://img.shields.io/badge/JSON-0.21.3-000?style=flat-square&labelColor=fff)
![MatrixMarket 0.3.1](https://img.shields.io/badge/MatrixMarket-0.3.1-000?style=flat-square&labelColor=fff)
[![RipQP 0.6.1](https://img.shields.io/badge/RipQP-0.6.1-006400?style=flat-square&labelColor=389826)](https://juliasmoothoptimizers.github.io/RipQP.jl/stable/)
[![SparseMatricesCOO 0.2.1](https://img.shields.io/badge/SparseMatricesCOO-0.2.1-4b0082?style=flat-square&labelColor=9558b2)](https://juliasmoothoptimizers.github.io/SparseMatricesCOO.jl/stable/)
[![QuadraticModels 0.9.3](https://img.shields.io/badge/QuadraticModels-0.9.3-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/QuadraticModels.jl/stable/)
![Plots 1.38.5](https://img.shields.io/badge/Plots-1.38.5-000?style=flat-square&labelColor=fff)
[![QPSReader 0.2.1](https://img.shields.io/badge/QPSReader-0.2.1-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/QPSReader.jl/stable/)
[![LDLFactorizations 0.10.0](https://img.shields.io/badge/LDLFactorizations-0.10.0-4b0082?style=flat-square&labelColor=9558b2)](https://juliasmoothoptimizers.github.io/LDLFactorizations.jl/stable/)
![TimerOutputs 0.5.22](https://img.shields.io/badge/TimerOutputs-0.5.22-000?style=flat-square&labelColor=fff)



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
 0.9977179249913947
 0.9777089489715103
 0.29673332818545894
 0.7398480261635807
 0.9224854034439071
 0.781598308703541
 0.9215454087642969
 0.16384364261792284
 0.3443193752091759
 0.39961376669860815
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
  0.12601466606726366
  0.3199856084291996
 -0.21442399579246707
  0.1275407418371889
  0.17168820992576747
 -0.04373707627143434
  0.24977388422525504
 -0.1617337966453858
 -0.013665445672811917
 -0.15717518503310443
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
 0.41185465039576663
 0.3966000946659183
 0.5934265904404439
 0.17274886364073683
 0.33046931873150065
 0.6360484974466485
 0.980079858363043
 0.41196314207243667
 0.9618398927425295
 0.17738329409013853
 ⋮
 0.9554353441675286
 0.29103046541566
 0.4480425458426587
 0.2493742885119008
 0.9568588707109307
 0.08147330966173927
 0.5199248657829982
 0.10008437064168596
 0.09509797532802988
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
 -1.0371156090339337
  0.11006312518498189
  0.07482528797684385
 -0.1333309527864173
 -1.06845251528816
  0.9848800018631857
 -0.0502595887741095
  0.3303442784864204
 -1.2797987758961937
 -0.6307221972032542
  ⋮
  0.176956218542843
 -0.1457875992995222
 -0.5202787125993733
 -0.2788836467863046
 -0.35455434280580533
  0.03259144532828587
 -0.4953814792454132
  0.2874207458521518
 -0.6398148615601961
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
LDLFactorizations.LDLFactorization{Float64, Int64, Int64, Int64}(true, true, true, 200, [6, 6, 5, 5, 6, 27, 8, 12, 11, 11  …  192, 193, 194, 195, 196, 197, 198, 199, 200, -1], [36, 51, 30, 53, 80, 109
, 46, 86, 49, 69  …  9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [200, 199, 188, 200, 200, 200, 200, 200, 200, 200  …  200, 200, 200, 200, 200, 200, 200, 200, 200, 200], [33, 80, 29, 27, 97, 89, 60, 63, 70, 31  … 
 191, 192, 193, 194, 195, 196, 198, 199, 200, 74], [27, 28, 29, 30, 31, 32, 33, 34, 35, 36  …  191, 192, 193, 194, 195, 196, 20, 197, 198, 199], [1, 37, 88, 118, 171, 251, 360, 406, 492, 541  …  16962
, 16970, 16977, 16983, 16988, 16992, 16995, 16997, 16998, 16998], [1, 8, 14, 21, 28, 34, 40, 50, 57, 67  …  807, 808, 808, 808, 809, 809, 809, 809, 809, 809], [27, 33, 63, 71, 80, 89, 97, 27, 63, 80  
…  180, 197, 186, 197, 186, 197, 186, 197, 197, 197], [6, 27, 30, 32, 33, 41, 48, 50, 53, 56  …  197, 198, 199, 200, 198, 199, 200, 199, 200, 200], [0.22969136662593811, 0.007895872589531712, 0.248610
7060010613, 0.5024096918967353, 0.05312829329568176, 0.1613948980181861, 0.02050507892097995, 0.014968385893559158, 0.11177515922945357, 0.14141993566042643  …  0.012843908017086713, -0.04551813278128
736, -0.06163296842454068, -0.00911468161740913, -0.0355401980073358, 0.0646221004730968, -0.021038839754146898, 0.05253192887977991, 0.0242546124250311, -0.05580769787992197], [1.655711057044528, 4.6
29631237320515, 2.290755478999105, 3.0872408782322354, 3.104880153110535, 3.9814576855530515, 2.3921336082568265, 6.071781525153877, 2.7801997255435205, 4.509756381325145  …  -3.4982331841078618, -1.6
729155472803339, -3.9177279445449606, -3.285264161062077, -2.47007177398019, -3.817528842728698, -3.061914092478808, -3.9750291259143298, -3.764690499816542, 2.9813839708080643], [0.0, 0.0, 0.0, 0.0, 
0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [26, 8, 12, 30, 31, 32, 33, 34, 35, 36  …  190, 191, 192, 193, 194, 195, 196, 197, 198, 199], 2.9802322387695312e-8,
 -1.4901161193847656e-8, 1.4901161193847656e-8, 10)
```





## Choosing the precision of the factorization

It is possible to factorize a matrix in a different type than the type of its elements:

```julia
# with eltype(Au) == Float64
LDL64 = ldl(Au) # factorization in eltype(Au) = Float64
LDL32 = ldl(Au, Float32) # factorization in Float32
```

```
LDLFactorizations.LDLFactorization{Float32, Int64, Int64, Int64}(true, true, true, 200, [6, 6, 5, 5, 6, 27, 8, 12, 11, 11  …  192, 193, 194, 195, 196, 197, 198, 199, 200, -1], [36, 51, 30, 53, 80, 109
, 46, 86, 49, 69  …  9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [200, 199, 188, 200, 200, 200, 200, 200, 200, 200  …  200, 200, 200, 200, 200, 200, 200, 200, 200, 200], [33, 80, 29, 27, 97, 89, 60, 63, 70, 31  … 
 191, 192, 193, 194, 195, 196, 198, 199, 200, 74], [27, 28, 29, 30, 31, 32, 33, 34, 35, 36  …  191, 192, 193, 194, 195, 196, 20, 197, 198, 199], [1, 37, 88, 118, 171, 251, 360, 406, 492, 541  …  16962
, 16970, 16977, 16983, 16988, 16992, 16995, 16997, 16998, 16998], [1, 8, 14, 21, 28, 34, 40, 50, 57, 67  …  807, 808, 808, 808, 809, 809, 809, 809, 809, 809], [27, 33, 63, 71, 80, 89, 97, 27, 63, 80  
…  180, 197, 186, 197, 186, 197, 186, 197, 197, 197], [6, 27, 30, 32, 33, 41, 48, 50, 53, 56  …  197, 198, 199, 200, 198, 199, 200, 199, 200, 200], Float32[0.22969137, 0.007895872, 0.2486107, 0.502409
7, 0.05312829, 0.16139491, 0.020505078, 0.014968386, 0.11177516, 0.14141993  …  0.012843925, -0.045518123, -0.061632916, -0.009114685, -0.03554021, 0.0646221, -0.02103883, 0.052532002, 0.024254637, -0
.055807747], Float32[1.655711, 4.629631, 2.2907555, 3.087241, 3.1048803, 3.9814572, 2.3921337, 6.0717816, 2.7801998, 4.5097566  …  -3.498236, -1.6729151, -3.9177287, -3.2852643, -2.4700718, -3.8175278
, -3.0619137, -3.9750273, -3.7646878, 2.9813836], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [26, 8, 12, 30, 31, 32, 33, 34, 35, 36
  …  190, 191, 192, 193, 194, 195, 196, 197, 198, 199], 0.0f0, 0.0f0, 0.0f0, 200)
```



```julia
# with eltype(Au) == Float64
LDL64 = ldl_analyze(Au) # symbolic analysis in eltype(Au) = Float64
LDL32 = ldl_analyze(Au, Float32) # symbolic analysis in Float32
ldl_factorize!(Au, LDL64)
ldl_factorize!(Au, LDL32)
```

```
LDLFactorizations.LDLFactorization{Float32, Int64, Int64, Int64}(true, true, true, 200, [6, 6, 5, 5, 6, 27, 8, 12, 11, 11  …  192, 193, 194, 195, 196, 197, 198, 199, 200, -1], [36, 51, 30, 53, 80, 109
, 46, 86, 49, 69  …  9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [200, 199, 188, 200, 200, 200, 200, 200, 200, 200  …  200, 200, 200, 200, 200, 200, 200, 200, 200, 200], [33, 80, 29, 27, 97, 89, 60, 63, 70, 31  … 
 191, 192, 193, 194, 195, 196, 198, 199, 200, 74], [27, 28, 29, 30, 31, 32, 33, 34, 35, 36  …  191, 192, 193, 194, 195, 196, 20, 197, 198, 199], [1, 37, 88, 118, 171, 251, 360, 406, 492, 541  …  16962
, 16970, 16977, 16983, 16988, 16992, 16995, 16997, 16998, 16998], [1, 8, 14, 21, 28, 34, 40, 50, 57, 67  …  807, 808, 808, 808, 809, 809, 809, 809, 809, 809], [27, 33, 63, 71, 80, 89, 97, 27, 63, 80  
…  180, 197, 186, 197, 186, 197, 186, 197, 197, 197], [6, 27, 30, 32, 33, 41, 48, 50, 53, 56  …  197, 198, 199, 200, 198, 199, 200, 199, 200, 200], Float32[0.22969137, 0.007895872, 0.2486107, 0.502409
7, 0.05312829, 0.16139491, 0.020505078, 0.014968386, 0.11177516, 0.14141993  …  0.012843925, -0.045518123, -0.061632916, -0.009114685, -0.03554021, 0.0646221, -0.02103883, 0.052532002, 0.024254637, -0
.055807747], Float32[1.655711, 4.629631, 2.2907555, 3.087241, 3.1048803, 3.9814572, 2.3921337, 6.0717816, 2.7801998, 4.5097566  …  -3.498236, -1.6729151, -3.9177287, -3.2852643, -2.4700718, -3.8175278
, -3.0619137, -3.9750273, -3.7646878, 2.9813836], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [26, 8, 12, 30, 31, 32, 33, 34, 35, 36
  …  190, 191, 192, 193, 194, 195, 196, 197, 198, 199], 0.0f0, 0.0f0, 0.0f0, 200)
```

