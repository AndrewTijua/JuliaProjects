using ParameterizedFunctions
using DiffEqOperators
using OrdinaryDiffEq
using DifferentialEquations
using Turing
using Random
using Distributions

using Plots
plotlyjs()

using StatsPlots
#=
In which we numerically solve pdes using discrete approximations to derivatives and inserting the resulting (system of) odes into
julias ode solvers. note that pdes generally give stiff systems of odes, and often need energy preservation (esp if oscillatory),
so the choice of solver is critical
=#

#heat equation

u_analytic(x, t) = sin(2 * π * x) * exp(-t * (2 * π)^2)

nknots = 100
h = 1.0 / (nknots + 1)
knots = collect(range(h, step = h, length = nknots))
ord_deriv = 2
ord_approx = 2

Δ = CenteredDifference(ord_deriv, ord_approx, h, nknots)
bc = Dirichlet0BC(Float64)

t0 = 0.0
t1 = 0.03
u0 = u_analytic.(knots, t0)

step_he(u, p, t) = Δ * bc * u
prob = ODEProblem(step_he, u0, (t0, t1))
alg = KenCarp4()
sol = solve(prob, alg)

plot(sol)

using Test
@test u_analytic.(knots, t1) ≈ sol[end] rtol = 1e-3
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

display(Array(Δ))
display(bc * zeros(nknots))

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
up0 = -abs.(-2.0 .* knots .+ 1.) .+ 1.

Δ = CenteredDifference(ord_deriv, ord_approx, h, nknots)
bc = Dirichlet0BC(Float64)

function step_w(du, u, p, t)
    a = u[:,1]
    da = u[:,2]
    du[:,1] = da
    du[:,2] = (1.0^2) * Δ * bc * a
end


prob = ODEProblem(step_w, hcat(u0, up0), (t0, t1))
alg = Trapezoid()
sol = solve(prob, alg)
plot(sol)

tsol = collect(t0:((t1-t0)/2000):t1)

S_array = Array(sol(tsol))
P_t = S_array[50,1,:]



function fcn(n::Int64)
    return (8.0 / (n*π)^2 * sin(n * π / 2))
end

function f_decon(x::Float64, t::Float64, n::Int64 = 10)
    sm::Float64 = 0.
    for i in 1:n
        cn::Float64 = fcn(i)
        sm += (i * π)^(-1) * cn * sin(i * π * x) * sin(i * π * t)
    end
    return sm
end

f_decon(knots[1], 5., 10)
plot(tsol, P_t)
plot!(tsol, f_decon.(knots[50], tsol, 15))

#it works now. yay.
#okay deffo incorrect
#i dont think this is correct but it sure looks pretty
####
