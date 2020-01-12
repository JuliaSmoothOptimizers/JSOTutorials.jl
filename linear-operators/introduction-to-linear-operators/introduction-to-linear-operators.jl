
using Pkg
pkg"activate ."
pkg"add LinearOperators"
pkg"add FFTW"
pkg"add Plots"
pkg"add PyPlot"
pkg"add Weave"
pkg"instantiate"
pkg"status"


pkgs = ["LinearOperators"]

using Pkg
ctx=Pkg.Types.Context()
display("text/html", "<img src=\"https://img.shields.io/badge/julia-$VERSION-3a5fcc.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAMAAAAolt3jAAAB+FBMVEUAAAA3lyU3liQ4mCY4mCY3lyU4lyY1liM3mCUskhlSpkIvkx0zlSEeigo5mSc8mio0liKPxYQ/nC5NozxQpUBHoDY3lyQ5mCc3lyY6mSg3lyVPpD9frVBgrVFZqUpEnjNgrVE3lyU8mio8mipWqEZhrVJgrVFfrE9JoTkAVAA3lyXJOjPMPjNZhCowmiNOoz1erE9grVFYqUhCnjFmk2KFYpqUV7KTWLDKOjK8CADORj7GJx3SJyVAmCtKojpOoz1DnzFVeVWVSLj///+UV7GVWbK8GBjPTEPMQTjPTUXQUkrQSEGZUycXmg+WXbKfZ7qgarqbYraSVLCUV7HLPDTKNy7QUEjUYVrVY1zTXFXPRz2UVLmha7upeMCqecGlcb6aYLWfaLrLPjXLPjXSWFDVZF3VY1zVYlvRTkSaWKqlcr6qesGqecGpd8CdZbjo2+7LPTTKOS/QUUnVYlvVY1zUXVbPSD6TV7OibLuqecGqecGmc76aYbaibLvKOC/SWlPMQjrQUEjRVEzPS0PLPDL7WROZX7WgarqibLucY7eTVrCVWLLLOzLGLCLQT0bIMynKOC7FJx3MPjV/Vc+odsCRUa+SVLCDPaWVWLKWWrLJOzPHOTLKPDPLPDPLOzLLPDOUV6+UV7CVWLKVWLKUV7GUWLGPUqv///8iGqajAAAAp3RSTlMAAAAAAAAAAAAAABAZBAAAAABOx9uVFQAAAB/Y////eQAAADv0////pgEAAAAAGtD///9uAAAAAAAAAAcOQbPLfxgNAAAAAAA5sMyGGg1Ht8p6CwAAFMf///94H9j///xiAAAw7////65K+f///5gAABjQ////gibg////bAAAAEfD3JwaAFfK2o0RAAAAAA4aBQAAABEZAwAAAAAAAAAAAAAAAAAAAIvMfRYAAAA6SURBVAjXtcexEUBAFAXAfTM/IDH6uAbUqkItyAQYR26zDeS0UxieBvPVbArjXd9GS295raa/Gmu/A7zfBRgv03cCAAAAAElFTkSuQmCC\">")
for p in pkgs
  uuid=ctx.env.project.deps[p]
  v=ctx.env.manifest[uuid].version
  c=string(hash(p) % 0x1000000, base=16)
  display("text/html", "<img src=\"https://img.shields.io/badge/$p-$v-brightgreen?color=$c\">")
end
pkg"status"


using LinearOperators

prod(v) = [3v[1] - v[2]; 2v[1] + 2v[2]]
#                     type, nrows, ncols, symm?, herm?, prod[; tprod, ctprod]
A = LinearOperator(Float64,     2,     2, false, false, prod)

@show A
A * ones(2)


prod(v) = [3v[1] - v[2]; 2v[1] + 2v[2]]
tprod(v) = [3v[1] + 2v[2]; -v[1] + 2v[2]]

A = LinearOperator(Float64, 2, 2, false, false, prod, tprod)

transpose(A) * ones(2)


A' * ones(2)


using FFTW, LinearAlgebra

A = LinearOperator(16, 16, false, false, fft, nothing, ifft)

v = rand(16) + im * rand(16)
norm(A * v - fft(v)), norm(A' * v - ifft(v)), norm(Matrix(A' * A) - I)


n = 4000
A = rand(n, n)
B = rand(n, n)
opA = LinearOperator(A)
opB = LinearOperator(B)

@show opA * opB

# Run twice
@time A * B
@time opA * opB

v = rand(n)
@time A * (B * v)
@time (opA * opB) * v;


m, n = 50, 30
A = rand(50, 30)
op1 = PreallocatedLinearOperator(A)
op2 = LinearOperator(A)
v = rand(n)

op1 * v
al = @allocated for i = 1:100
  op1 * v
end
println("Allocation of op1: $al")
op2 * v
al = @allocated for i = 1:100
  op2 * v
end
println("Allocation of op2: $al")


A = rand(5,5)
A = A' * A
op = opCholesky(A)  # Use, e.g., as a preconditioner
v = rand(5)

norm(A \ v - op * v) / norm(v)


B = LBFGSOperator(5, scaling=false) # B₀ = I
H = InverseLBFGSOperator(5, scaling=false) # H₀ = I

# Example of s₀ and y₀
s = rand(5)
y = rand(5)
push!(B, s, y)
push!(H, s, y)

q = rand(5)
Bq = q + dot(y, q) / dot(y, s) * y - dot(s, q) / dot(s, s) * s
Hauxq = q - dot(s, q) / dot(y, s) * y
Hq = Hauxq - dot(y, Hauxq) / dot(y, s) * s + dot(s, q) / dot(y, s) * s

norm(B * q - Bq), norm(H * q - Hq), norm(B * H * q - q)


function HeatEquationOp(L, m, δ, α)
  h = L / (m - 1)
  γ = α * δ / h^2
  κ = 1 - 4γ

  Tprod(v) = [κ * v[1] + 2γ * v[2];
             [γ * v[i-1] + κ * v[i] + γ * v[i+1] for i = 2:m-1];
              κ * v[m] + 2γ * v[m-1]]

  T = LinearOperator(Float64, m, m, false, false, Tprod)
  D = opEye(m) * γ

  function prod(v)
    Hv = zeros(m^2)
    Hv[1:m] .= [T  2D] * v[1:2m]
    for i = 2:m-1
      Hv[(i-1)*m+1:i*m] .= [D T D] * v[(i-2)*m+1:(i+1)*m]
    end
    Hv[end-m+1:end] .= [2D T] * v[end-2m+1:end]
    return Hv
  end

  return LinearOperator(Float64, m^2, m^2, false, false, prod)
end


A = HeatEquationOp(1.0, 3, 0.1, 0.1)
Matrix(A)


using SparseArrays, Plots
pyplot(size=(600,600))

A = HeatEquationOp(1.0, 20, 0.1, 0.1)
spy(sparse(Matrix(A)))


function HeatEquation(u0, L, m, δ, α)
  h = L / (m - 1)
  U = zeros(m^2)
  for i = 1:m
    x = (i - 1) * h
    for j = 1:m
      y = (j - 1) * h
      U[(i - 1)*m + j] = u0(x, y)
    end
  end
  A = HeatEquationOp(L, m, δ, α)

  return U, A
end


const dark_purple = Colors.RGB(0.584, 0.345, 0.698)
const dark_red    = Colors.RGB(0.796, 0.235, 0.200)
const dark_green  = Colors.RGB(0.220, 0.596, 0.149)
const black       = Colors.RGB(0.0, 0.0, 0.0)
colors = [black, dark_green, dark_green, dark_red, dark_red, dark_purple, dark_purple]

L = 5
u0(x, y) = begin
  d = [(x,y) -> ((x - sin(2π/3*i))^2 + (y - cos(2π/3*i))^2)^2 for i = 1:3]
  xx, yy = x - L / 2, y - L / 2
  return 3*exp(-sqrt(2)*d[1](xx,yy)) + 2*exp(-d[2](xx,yy)) + exp(-4*d[3](xx,yy))
end

grid = range(0, L, length=100)
maxu = maximum([u0(xi,yi) for xi in grid, yi in grid])
p = plot(; leg=false, size=(1000,500), layout=@layout [a b])
surface!(p[1], grid, grid, u0, c=ColorGradient(colors), camera=(210,30))
contour!(p[2], grid, grid, u0, levels=range(0.1, maxu, length=50), c=ColorGradient(colors))


L = 5
m = 30
δ = 0.01
α = 0.5

U0, A = HeatEquation(u0, L, m, δ, α)
U = copy(U0)

plot_rows = 4
plot_cols = 4
plots = []

Δt = 10

for i = 1:plot_rows
  for j = 1:plot_cols
    global U

    p = surface(reshape(U, m, m), leg=false, c=ColorGradient(colors), camera=(210,30))
    zlims!(0, maximum(U0))
    xticks!(Float64[])
    yticks!(Float64[])
    k = (i - 1) * plot_cols + j
    title!("t = $(round(k * Δt * δ, digits=3))")
    push!(plots, p)

    for t = 1:Δt
      U = A * U
    end
  end
end
plot(plots..., layout=(plot_rows, plot_cols), size=(1000, plot_rows * 250))


U0, A = HeatEquation(u0, L, m, δ, α)
U = copy(U0)

Δt = 5

anim = Animation()
for i = 1:60
  global U

  rU = reshape(U, m, m)
  p = plot(; leg=false, size=(1000,500), layout=@layout [a b])
  surface!(p[1], rU, c=ColorGradient(colors), camera=(210,30))
  contour!(p[2], rU, levels=range(0.1, maximum(U0), length=50), c=ColorGradient(colors))
  zlims!(0, maximum(U0))
  frame(anim)

  for t = 1:Δt
    U = A * U
  end
end
gif(anim, "heat-equation.gif", fps=12)
nothing

