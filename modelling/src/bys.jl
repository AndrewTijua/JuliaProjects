using ParameterizedFunctions
using DiffEqOperators
using OrdinaryDiffEq
using DifferentialEquations
using Turing
using Random
using Distributions

using Plots
plotlyjs()
#using PlotlyJS
# using ORCA
#
# using StatsPlots
#=
In which we numerically solve pdes using discrete approximations to partial derivatives and inserting the resulting (system of)
odes into julias ode solvers. note that pdes generally give stiff systems of odes, and often need energy preservation (esp if oscillatory),
so the choice of solver is critical. KenCarp4 works well for single derived systems and over (relatively) short timespans,
but trapezoid works better for oscillatory pdes resulting from coupled systems (ie 2nd order pdes). check eigenvalues, if complex are present
use trapezoid or similar
=#

#heat equation
#u₀(x) = sin(2πx)
#dirichlet 0 boundaries
u_analytic(x, t) = sin(2 * π * x) * exp(-t * (2 * π)^2)

nknots = 100
h = 1.0 / (nknots + 1)
knots = collect(range(h, step = h, length = nknots))
ord_deriv = 2
ord_approx = 2

Δ = CenteredDifference(ord_deriv, ord_approx, h, nknots)
bc = Dirichlet0BC(Float64)

t0 = 0.0
t1 = 0.07
u0 = u_analytic.(knots, t0)

step_he(u, p, t) = Δ * bc * u
prob = ODEProblem(step_he, u0, (t0, t1))
alg = KenCarp4()
sol = solve(prob, alg)

plot_t = collect(t0:((t1-t0)/200):t1)

p_array = Array(sol(plot_t))
sp_t = p_array[:, :]

xs = collect(knots)
ts = plot_t
zs = sp_t

plot(
    xs,
    ts,
    zs',
    st = :surface,
    xlabel = "X",
    ylabel = "t",
    zlabel = "Δ",
    colorbar = false,
    camera = [75, 30],
    title = "Heat Eqⁿ w/ sinusoid init)",
    size = (1000, 1000),
)

using Test
@test u_analytic.(knots, t1) ≈ sol[end] atol = 1e-3
####
#poisson equation

f = 1.0
a = -1.0
b = 2.0

u_analytic_poisson(x) = f / 2 * x^2 + (b - a - f / 2) * x + a

nknots = 10
h = 1.0 / (nknots + 1)
ord_deriv = 2
ord_approx = 2

Δ = CenteredDifference(ord_deriv, ord_approx, h, nknots)
bc = DirichletBC(a, b)

# display(Array(Δ))
# display(bc * zeros(nknots))

u = (Δ * bc) \ fill(f, nknots)
knots = collect(range(h, step = h, length = nknots))

using Test
@test u ≈ u_analytic_poisson.(knots)
####
#wave equation

t0 = 0.0
t1 = 100.0

nknots = 100
h = 1.0 / (nknots + 1)
knots = collect(range(h, step = h, length = nknots))
ord_deriv = 2
ord_approx = 4

v = 1.0

u0 = 0 .* knots
up0 = -abs.(-2.0 .* knots .+ 1.0) .+ 1.0

Δ = CenteredDifference(ord_deriv, ord_approx, h, nknots)
bc = Dirichlet0BC(Float64)

function step_w(du, u, p, t)
    a = u[:, 1]
    da = u[:, 2]
    du[:, 1] = da
    du[:, 2] = (1.0^2) * Δ * bc * a
end


prob = ODEProblem(step_w, hcat(u0, up0), (t0, t1))
alg = Trapezoid()
sol = solve(prob, alg)
plot(sol)

tsol = collect(t0:((t1-t0)/200):t1)

S_array = Array(sol(tsol))
P_t = S_array[50, 1, :]



function fcn(n::Int64)
    return (8.0 / (n * π)^2 * sin(n * π / 2))
end

function f_decon(x::Float64, t::Float64, n::Int64 = 10)
    sm::Float64 = 0.0
    for i = 1:n
        cn::Float64 = fcn(i)
        sm += (i * π)^(-1) * cn * sin(i * π * x) * sin(i * π * t)
    end
    return sm
end

#f_decon(knots[1], 5., 10)
plot(tsol, P_t)
plot!(tsol, f_decon.(knots[50], tsol, 15))

plot_t = collect(0:0.05:5)

p_array = Array(sol(plot_t))
sp_t = p_array[:, 1, :]



xs = collect(knots)
ts = plot_t
zs = sp_t

plot(xs, ts, zs', st = :surface, xlabel = "X", ylabel = "t", zlabel = "Δ", colorbar = false, camera = [75, 30])
#Plots.savefig("wave_eqn_uisoln.png")
#it works now. yay. (read bottom to top its like twitter)
#okay deffo incorrect
#i dont think this is correct but it sure looks pretty
####
#klein gordon pde

#1/c² * ∂ₜₜu - ∂ₓₓu + μ²u = 0
# ∂ₜₜu = c²(∂ₓₓu - μ²u)

t0 = 0.0
t1 = 10.0

l = 5.0

nknots = 250
h = 2l / (nknots + 1)
knots = collect(range(-l, step = h, length = nknots))
ord_deriv = 2
ord_approx = 4

#c = 3e8
#μ = (9.11e-31 * c) / (1.0545718e-34)
# c = 10.0
# μ = 100.0

mₚ = 2.176435e-8
m = 1.67262192369e-27 / mₚ
Δ = CenteredDifference(ord_deriv, ord_approx, h, nknots)
bc = Dirichlet0BC(Float64)

up0 = sinc.(knots)
u0 = 0 .* knots

function step_kg(du, u, p, t)
    a = u[:, 1]
    da = u[:, 2]
    du[:, 1] = da
    du[:, 2] = (Δ * bc * a - m^2 * a)
end


prob = ODEProblem(step_kg, hcat(u0, up0), (t0, t1))
alg = Trapezoid()
sol = solve(prob, alg)
#plot(sol)

plot_t = collect(t0:((t1-t0)/200):t1)

p_array = Array(sol(plot_t))
sp_t = p_array[:, 1, :]

xs = collect(knots)
ts = plot_t
zs = sp_t

plot(
    xs,
    ts,
    zs',
    st = :surface,
    xlabel = "X",
    ylabel = "t",
    zlabel = "Δ",
    colorbar = false,
    camera = [75, 30],
    title = "Klein-Gordon (relativistic wave eqⁿ)",
    size = (1000, 1000),
)
####
