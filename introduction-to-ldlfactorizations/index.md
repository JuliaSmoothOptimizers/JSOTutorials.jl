---
author: "Geoffroy Leconte"
title: "LDLFactorizations tutorial"
tags:
  - "linear"
  - "factorization"
  - "ldlt"
---

![JSON 0.21.4](https://img.shields.io/badge/JSON-0.21.4-000?style=flat-square&labelColor=999)
![MatrixMarket 0.3.1](https://img.shields.io/badge/MatrixMarket-0.3.1-000?style=flat-square&labelColor=999)
[![RipQP 0.6.2](https://img.shields.io/badge/RipQP-0.6.2-006400?style=flat-square&labelColor=389826)](https://juliasmoothoptimizers.github.io/RipQP.jl/stable/)
![DelimitedFiles 1.9.1](https://img.shields.io/badge/DelimitedFiles-1.9.1-000?style=flat-square&labelColor=999)
[![QuadraticModels 0.9.4](https://img.shields.io/badge/QuadraticModels-0.9.4-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/QuadraticModels.jl/stable/)
[![SparseMatricesCOO 0.2.1](https://img.shields.io/badge/SparseMatricesCOO-0.2.1-4b0082?style=flat-square&labelColor=9558b2)](https://juliasmoothoptimizers.github.io/SparseMatricesCOO.jl/stable/)
![Plots 1.38.15](https://img.shields.io/badge/Plots-1.38.15-000?style=flat-square&labelColor=999)
[![QPSReader 0.2.1](https://img.shields.io/badge/QPSReader-0.2.1-8b0000?style=flat-square&labelColor=cb3c33)](https://juliasmoothoptimizers.github.io/QPSReader.jl/stable/)
[![LDLFactorizations 0.10.0](https://img.shields.io/badge/LDLFactorizations-0.10.0-4b0082?style=flat-square&labelColor=9558b2)](https://juliasmoothoptimizers.github.io/LDLFactorizations.jl/stable/)
![TimerOutputs 0.5.23](https://img.shields.io/badge/TimerOutputs-0.5.23-000?style=flat-square&labelColor=999)



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
 0.7410151519637049
 0.5803173738730819
 0.9162605977012206
 0.440697725052859
 0.3058647693045615
 0.7251509182841179
 0.5651214645786362
 0.06887028186606126
 0.9383303831008762
 0.12791375405281702
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
  0.031062548178427067
 -0.0288313533994344
  0.22662835336315346
 -0.0017206635148817786
  0.0144508351872706
  0.22136204577502708
  0.07366088962299415
 -0.2141058032302209
  0.2670262563983708
 -0.32594504749573433
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
 0.97451827066654
 0.35219051701592796
 0.7670241003715759
 0.27755948756397686
 0.2261977151423713
 0.2959430002823329
 0.4394158032275185
 0.23858040106006617
 0.44584988272366166
 0.5041635487980847
 ⋮
 0.3648725514994158
 0.37891581752117165
 0.46776858594277926
 0.19910973871863602
 0.46931157489658437
 0.4814397508887842
 0.24938941253172198
 0.8618046844393147
 0.5867752737974512
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
 -0.10395268401273323
 -0.11636578663130666
  0.6631541009139968
  0.3364560473187304
  0.33921441392818064
 -0.47714612409460705
  0.5254245159646647
 -0.07538396377919897
  0.17604677754672513
  0.19625575149835262
  ⋮
  0.27810765676176585
  0.3087799091206352
 -0.6153940010007604
  0.13293439330792706
 -0.018244751249800094
 -0.4366926448582921
 -0.25239986923733015
 -0.5482430025231264
  0.6270460987305891
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
LDLFactorizations.LDLFactorization{Float64, Int64, Int64, Int64}(true, true, true, 200, [34, 41, 11, 5, 10, 9, 8, 9, 10, 11  …  192, 193, 194, 195, 196, 197, 198, 199, 200, -1], [56, 53, 28, 54, 88, 6
4, 43, 75, 115, 135  …  9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [199, 199, 198, 197, 200, 200, 198, 199, 200, 200  …  200, 200, 200, 200, 200, 200, 200, 200, 200, 200], [104, 127, 107, 120, 179, 199, 186, 106,
 165, 182  …  191, 192, 193, 194, 195, 196, 197, 198, 200, 110], [31, 32, 33, 24, 34, 21, 17, 19, 35, 36  …  191, 192, 193, 194, 195, 196, 197, 198, 6, 199], [1, 57, 110, 138, 192, 280, 344, 387, 462,
 577  …  16629, 16637, 16644, 16650, 16655, 16659, 16662, 16664, 16665, 16665], [1, 5, 12, 21, 21, 29, 30, 30, 32, 36  …  792, 792, 792, 793, 794, 795, 795, 796, 796, 796], [17, 20, 72, 73, 8, 20, 28,
 41, 73, 86  …  199, 199, 186, 199, 199, 199, 199, 199, 199, 199], [34, 36, 51, 52, 58, 79, 82, 85, 96, 102  …  197, 198, 199, 200, 198, 199, 200, 199, 200, 200], [-0.17858290289643114, -0.08919237538
790274, -0.06453943119029718, -0.08074981241974985, -0.148716028643219, -0.1079151694540959, -0.031753457351889086, -0.08839876579302026, -0.10615098162705923, -0.20683849899215664  …  -0.011119804798
925149, 0.0027349141211280075, -0.03167591458741246, -0.03535569897207626, 0.10258868109496486, 0.03494334303370605, 0.03364826904390819, 0.0371845782747629, 0.010039337773387096, 0.047092827801259105
], [-4.631265251501746, -2.7932593127464767, -2.4899267375417455, -3.0035989510240757, -4.279332941421245, -2.872999130748176, -4.135679337426587, -2.3311589027373705, -3.01665361103017, -7.3900232214
21596  …  -2.660895638536931, -3.3086748009980136, -2.1696692859288507, -2.30076515243789, -1.9886819918922096, -2.9767018510499943, -2.4067454598282496, -2.3651297187687135, -2.5100259106820464, -2.0
944782986205523], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [6, 9, 11, 31, 32, 33, 34, 35, 36, 37  …  190, 191, 192, 193, 194, 195, 196, 
197, 198, 199], 2.9802322387695312e-8, -1.4901161193847656e-8, 1.4901161193847656e-8, 10)
```





## Choosing the precision of the factorization

It is possible to factorize a matrix in a different type than the type of its elements:

```julia
# with eltype(Au) == Float64
LDL64 = ldl(Au) # factorization in eltype(Au) = Float64
LDL32 = ldl(Au, Float32) # factorization in Float32
```

```
LDLFactorizations.LDLFactorization{Float32, Int64, Int64, Int64}(true, true, true, 200, [34, 41, 11, 5, 10, 9, 8, 9, 10, 11  …  192, 193, 194, 195, 196, 197, 198, 199, 200, -1], [56, 53, 28, 54, 88, 6
4, 43, 75, 115, 135  …  9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [199, 199, 198, 197, 200, 200, 198, 199, 200, 200  …  200, 200, 200, 200, 200, 200, 200, 200, 200, 200], [104, 127, 107, 120, 179, 199, 186, 106,
 165, 182  …  191, 192, 193, 194, 195, 196, 197, 198, 200, 110], [31, 32, 33, 24, 34, 21, 17, 19, 35, 36  …  191, 192, 193, 194, 195, 196, 197, 198, 6, 199], [1, 57, 110, 138, 192, 280, 344, 387, 462,
 577  …  16629, 16637, 16644, 16650, 16655, 16659, 16662, 16664, 16665, 16665], [1, 5, 12, 21, 21, 29, 30, 30, 32, 36  …  792, 792, 792, 793, 794, 795, 795, 796, 796, 796], [17, 20, 72, 73, 8, 20, 28,
 41, 73, 86  …  199, 199, 186, 199, 199, 199, 199, 199, 199, 199], [34, 36, 51, 52, 58, 79, 82, 85, 96, 102  …  197, 198, 199, 200, 198, 199, 200, 199, 200, 200], Float32[-0.1785829, -0.089192376, -0.
06453943, -0.08074981, -0.14871603, -0.10791517, -0.031753458, -0.08839877, -0.106150985, -0.2068385  …  -0.011119786, 0.0027349156, -0.03167594, -0.035355713, 0.102588706, 0.034943312, 0.03364818, 0.
037184507, 0.010039314, 0.047092833], Float32[-4.631265, -2.7932594, -2.4899268, -3.003599, -4.2793326, -2.8729992, -4.1356792, -2.331159, -3.0166535, -7.3900237  …  -2.6608965, -3.3086727, -2.1696684
, -2.3007638, -1.9886813, -2.9767003, -2.406745, -2.3651314, -2.510026, -2.0944774], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [6,
 9, 11, 31, 32, 33, 34, 35, 36, 37  …  190, 191, 192, 193, 194, 195, 196, 197, 198, 199], 0.0f0, 0.0f0, 0.0f0, 200)
```



```julia
# with eltype(Au) == Float64
LDL64 = ldl_analyze(Au) # symbolic analysis in eltype(Au) = Float64
LDL32 = ldl_analyze(Au, Float32) # symbolic analysis in Float32
ldl_factorize!(Au, LDL64)
ldl_factorize!(Au, LDL32)
```

```
LDLFactorizations.LDLFactorization{Float32, Int64, Int64, Int64}(true, true, true, 200, [34, 41, 11, 5, 10, 9, 8, 9, 10, 11  …  192, 193, 194, 195, 196, 197, 198, 199, 200, -1], [56, 53, 28, 54, 88, 6
4, 43, 75, 115, 135  …  9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [199, 199, 198, 197, 200, 200, 198, 199, 200, 200  …  200, 200, 200, 200, 200, 200, 200, 200, 200, 200], [104, 127, 107, 120, 179, 199, 186, 106,
 165, 182  …  191, 192, 193, 194, 195, 196, 197, 198, 200, 110], [31, 32, 33, 24, 34, 21, 17, 19, 35, 36  …  191, 192, 193, 194, 195, 196, 197, 198, 6, 199], [1, 57, 110, 138, 192, 280, 344, 387, 462,
 577  …  16629, 16637, 16644, 16650, 16655, 16659, 16662, 16664, 16665, 16665], [1, 5, 12, 21, 21, 29, 30, 30, 32, 36  …  792, 792, 792, 793, 794, 795, 795, 796, 796, 796], [17, 20, 72, 73, 8, 20, 28,
 41, 73, 86  …  199, 199, 186, 199, 199, 199, 199, 199, 199, 199], [34, 36, 51, 52, 58, 79, 82, 85, 96, 102  …  197, 198, 199, 200, 198, 199, 200, 199, 200, 200], Float32[-0.1785829, -0.089192376, -0.
06453943, -0.08074981, -0.14871603, -0.10791517, -0.031753458, -0.08839877, -0.106150985, -0.2068385  …  -0.011119786, 0.0027349156, -0.03167594, -0.035355713, 0.102588706, 0.034943312, 0.03364818, 0.
037184507, 0.010039314, 0.047092833], Float32[-4.631265, -2.7932594, -2.4899268, -3.003599, -4.2793326, -2.8729992, -4.1356792, -2.331159, -3.0166535, -7.3900237  …  -2.6608965, -3.3086727, -2.1696684
, -2.3007638, -1.9886813, -2.9767003, -2.406745, -2.3651314, -2.510026, -2.0944774], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [6,
 9, 11, 31, 32, 33, 34, 35, 36, 37  …  190, 191, 192, 193, 194, 195, 196, 197, 198, 199], 0.0f0, 0.0f0, 0.0f0, 200)
```

