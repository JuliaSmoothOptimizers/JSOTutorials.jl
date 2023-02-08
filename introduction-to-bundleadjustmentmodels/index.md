---
author: "Antonin Kenens and Tangi Migot"
title: "Introduction to BundleAdjustmentModels"
tags:
  - "introduction"
  - "model"
  - "least-squares"
  - "nlsmodels"
  - "test set"
  - "bundle adjustment"
---


A Julia repository of [bundle adjustment](https://en.wikipedia.org/wiki/Bundle_adjustment) problems from the [Bundle Adjustment in the Large](http://grail.cs.washington.edu/projects/bal/) repository.

```julia
using BundleAdjustmentModels
```




## Get the list of problems

The function `problems_df()` returns a `Dataframe` of all the bundle adjustment problems.

There are 74 different models organized in the `Dataframe` with the following elements:

- the name of the model;
- the group of the model;
- the size of the jacobian matrix (`nequ`, `nvar`);
- the number of non-zeros elements in the jacobian matrix (`nnzj`).

```julia
df = problems_df()
```

```
JLD2.ReconstructedTypes.var"##DataFrames.DataFrame#313"(AbstractVector[["problem-16-22106-pre", "problem-88-64298-pre", "problem-135-90642-pre", "problem-142-93602-pre", "problem-150-95821-pre", "prob
lem-161-103832-pre", "problem-173-111908-pre", "problem-182-116770-pre", "problem-202-132796-pre", "problem-237-154414-pre"  …  "problem-744-543562-pre", "problem-951-708276-pre", "problem-1102-780462
-pre", "problem-1158-802917-pre", "problem-1184-816583-pre", "problem-1238-843534-pre", "problem-1288-866452-pre", "problem-1350-894716-pre", "problem-1408-912229-pre", "problem-1778-993923-pre"], ["d
ubrovnik", "dubrovnik", "dubrovnik", "dubrovnik", "dubrovnik", "dubrovnik", "dubrovnik", "dubrovnik", "dubrovnik", "dubrovnik"  …  "venice", "venice", "venice", "venice", "venice", "venice", "venice",
 "venice", "venice", "venice"], [167436, 767874, 1106672, 1131216, 1136238, 1184038, 1269140, 1337410, 1503304, 1716662  …  6117726, 7497784, 8104680, 8261006, 8358094, 8581006, 8766012, 9034252, 9269
060, 10003892], [66462, 193686, 273141, 282084, 288813, 312945, 337281, 351948, 400206, 465375  …  1637382, 2133387, 2351304, 2419173, 2460405, 2541744, 2610948, 2696298, 2749359, 2997771], [2009232, 
9214488, 13280064, 13574592, 13634856, 14208456, 15229680, 16048920, 18039648, 20599944  …  73412712, 89973408, 97256160, 99132072, 100297128, 102972072, 105192144, 108411024, 111228720, 120046704]], 
DataFrames.Index(Dict(:nnzj => 5, :group => 2, :nequ => 3, :name => 1, :nvar => 4), [:name, :group, :nequ, :nvar, :nnzj]))
```





For instance, it is possible to select the problems where the Jacobian matrix of the residual has at least 50000 lines and less than 34000 columns.

```julia
filter_df = df[ ( df.nequ .≥ 50000 ) .& ( df.nvar .≤ 34000 ), :]
```

```
Error: type ##DataFrames.DataFrame#313 has no field nequ
```





The `Dataframe` is listing the matrices that you can have access to, but they still need to be downloaded.

Following the example above, we filtered two problems. 
What we want to do now is to select the first one in the listing.

```julia
name = filter_df[1, :name] # select the name of the first problem
```

```
Error: UndefVarError: filter_df not defined
```





Now that the name is selected, we need to access the problem itself, and there are 2 solutions:

- You can download the problem's archive file;
- You can automatically create a nonlinear least squares problem using [`NLPModels`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) from [JuliaSmoothOptimizers](https://juliasmoothoptimizers.github.io/).

## Get the problem archive file

This package uses Julia Artifacts to handle the problems archives so that

1. The models are downloaded only once;
2. They are identified with a unique hash;
3. They can be deleted with a single command line.

The method [`fetch_ba_name`](https://juliasmoothoptimizers.github.io/BundleAdjustmentModels.jl/dev/reference/#BundleAdjustmentModels.fetch_ba_name-Tuple{AbstractString}) will automatically download the problem (if needed) and return its path.

```julia
path = fetch_ba_name(name)
```

```
Error: UndefVarError: name not defined
```





It is also possible to directly download and get access to an entire group of problems using [`fetch_ba_group`](https://juliasmoothoptimizers.github.io/BundleAdjustmentModels.jl/dev/reference/#BundleAdjustmentModels.fetch_ba_group-Tuple{AbstractString}).

```julia
paths = fetch_ba_group("ladybug")
```

```
30-element Vector{String}:
 "/home/runner/.julia/artifacts/dd2da5f94014b5f9086a2b38a87f8c1bc171b9c2"
 "/home/runner/.julia/artifacts/3d0853a3ca8e585814697fea9cd4d6956692e103"
 "/home/runner/.julia/artifacts/5c5c938c998d6c083f549bc584cfeb07bd296d89"
 "/home/runner/.julia/artifacts/e8ebdcec7cd5dc3ec4825dcb6642064e541fa625"
 "/home/runner/.julia/artifacts/111ebd1dcdf5670eac8a12f4e2ce5a3b79b4f7f6"
 "/home/runner/.julia/artifacts/3e8acd1ef4cb13d93a7140be1c3702b5eb165978"
 "/home/runner/.julia/artifacts/69595726ce3990604ef56b80c273fe7f95506f2e"
 "/home/runner/.julia/artifacts/1eab751f4850b1e22302a7f97894d4b769c7350e"
 "/home/runner/.julia/artifacts/90ce1588d82c76a81c2b07d9e6abcc984e0e4586"
 "/home/runner/.julia/artifacts/05e4248f24d897a97e9be1f386e18d1de7e511e5"
 ⋮
 "/home/runner/.julia/artifacts/72cdf832cd3eedb4cc4c589127e7ab5b91a60f00"
 "/home/runner/.julia/artifacts/1e01d6927d4d4e0acd3086d9c75d2ee783ab1220"
 "/home/runner/.julia/artifacts/b92a8dfd520e7440769c4a72685e817632f9ea43"
 "/home/runner/.julia/artifacts/366a8368c091027a23a190b1ed498158eef947a5"
 "/home/runner/.julia/artifacts/ee08f4f6898cc26471d3a63502fe908cfa9a5dcc"
 "/home/runner/.julia/artifacts/1a0ebf23b35d784ad3d39f4987bc7d785f5976b6"
 "/home/runner/.julia/artifacts/00be55410c27068ec73261e122a39258100a1a11"
 "/home/runner/.julia/artifacts/0303e7ae8256c494c9da052d977277f21265899b"
 "/home/runner/.julia/artifacts/389ecea5c2f2e2b637a2b4439af0bd4ca98e6d84"
```





## Generate a nonlinear least squares model

Now, it is possible to load the model using [`BundleAdjustmentModel`](https://juliasmoothoptimizers.github.io/BundleAdjustmentModels.jl/dev/reference/#BundleAdjustmentModels.BundleAdjustmentModel-Tuple{AbstractString})

```julia
df = problems_df()
filter_df = df[ ( df.nequ .≥ 50000 ) .& ( df.nvar .≤ 34000 ), :]
name = filter_df[1, :name]
model = BundleAdjustmentModel(name);
```

```
Error: type ##DataFrames.DataFrame#314 has no field nequ
```





or

```julia
model = BundleAdjustmentModel("problem-49-7776-pre");
```




The function `BundleAdjustmentModel` will instantiate the model and automatically download it if needed.
The resulting structure is an instance of `AbstractNLPModel`.
So, it is possible to access its API as any other [`NLPModel`](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/).

```julia
using NLPModels
```




Using [`residual`](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/api/#NLPModels.residual), it is possible to compute the residual of the model

```julia
model = BundleAdjustmentModel("problem-49-7776-pre.txt.bz2")
x = get_x0(model) # or `model.meta.x0`
Fx = residual(model, x)
```

```
63686-element Vector{Float64}:
 -9.020226301243213
 11.263958304987227
 -1.8332297149469525
  5.304698960898122
 -4.332321480806684
  7.117305031392988
 -0.5632751791501462
 -1.062178017695885
 -3.9692059546843836
 -2.2850712830954194
  ⋮
 -0.4377574792022081
 -0.024461522651677114
 -0.1961281328666189
  0.008375315306523134
  0.23044496991071384
  0.04927878647407624
  0.47289578243411867
 -0.01443314653496941
 -0.4486499211288866
```





or use the in-place method [`residual!`](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/api/#NLPModels.residual!)

```julia
model = BundleAdjustmentModel("problem-49-7776-pre.txt.bz2")
x = get_x0(model) # or `model.meta.x0`
nequ = get_nequ(model) # or `model.nls_meta.nequ`
Fx = zeros(nequ)
residual!(model, x, Fx);
```




You can also have access to the [`LinearOperator`](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) of the Jacobian matrix of the residual of the model which is calculated by hand (in contradiction to automatic differentiation).

You need to call [`jac_structure_residual!`](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/api/#NLPModels.jac_structure_residual!) at least once before calling [`jac_op_residual!`](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/api/#NLPModels.jac_op_residual!).

```julia
model = BundleAdjustmentModel("problem-49-7776")
meta_nls = nls_meta(model)
nnzj = meta_nls.nnzj # number of nonzeros in the jacobian
rows = Vector{Int}(undef, nnzj)
cols = Vector{Int}(undef, nnzj)
jac_structure_residual!(model, rows, cols);
```




You need to call [`jac_coord_residual!`](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/api/#NLPModels.jac_coord_residual!) to update it to the current point.

```julia
model = BundleAdjustmentModel("problem-49-7776")
x = get_x0(model)
vals = similar(x, nnzj)
jac_coord_residual!(model, x, vals)
```

```
764232-element Vector{Float64}:
   545.1179297695714
    -5.058282392703829
  -478.0666614182794
  -283.5120110272222
 -1296.3388697208218
  -320.60334752077165
   551.1773498438256
     0.00020469082949125088
  -471.09490058346296
  -409.36200783910084
     ⋮
  1246.0491342213195
   -96.64898276178089
   622.4628407807937
     2.852854717671426e-7
   305.00859581263086
    19.561762292226746
     6.5984133899730155
     1.680958440896614
     0.06413511779846102
```





Finally you can use [`jac_op_residual!`](https://juliasmoothoptimizers.github.io/NLPModels.jl/dev/api/#NLPModels.jac_op_residual!):

```julia
model = BundleAdjustmentModel("problem-49-7776")
meta_nls = nls_meta(model)
nnzj = meta_nls.nnzj

rows = Vector{Int}(undef, nnzj)
cols = Vector{Int}(undef, nnzj)
jac_structure_residual!(model, rows, cols)

vals = similar(model.meta.x0, nnzj)
jac_coord_residual!(model, model.meta.x0, vals)

Jv = similar(model.meta.x0, meta_nls.nequ)
Jtv = similar(model.meta.x0, meta_nls.nvar)
Jx = jac_op_residual!(model, rows, cols, vals, Jv, Jtv)
```

```
Linear operator
  nrow: 63686
  ncol: 23769
  eltype: Float64
  symmetric: false
  hermitian: false
  nprod:   0
  ntprod:  0
  nctprod: 0
```





There is no second-order information available for the problems in this package.

## Delete the problems

Once you have finished working with a specific model you can delete it the same way you downloaded it.

```julia
delete_ba_artifact!("problem-49-7776")
```




If you want to clean your workspace, you can also delete all the problems at once.

```julia
delete_all_ba_artifacts!()
```

