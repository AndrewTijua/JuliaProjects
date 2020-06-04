using ModelingToolkit
using OrdinaryDiffEq
using Plots
plotlyjs();

@parameters t σ ρ β
@variables x(t) y(t) z(t)
@derivatives D'~t

#2nd order lorenz
eqs = [D(D(x)) ~ σ * (y - x), D(y) ~ x * (ρ - z) - y, D(z) ~ x * y - β * z]

sys = ODESystem(eqs)
sys = ode_order_lowering(sys)

u₀ = [D(x) => 2.0, x => 1.0, y => 0.0, z => 0.0]

p = [σ => 28.0, ρ => 10.0, β => 8 / 3]

tspan = (0.0, 100.0)

prob = ODEProblem(sys, u₀, tspan, p, jac = true)
soln = solve(prob, Tsit5())

plot(soln, vars = (x, y, z))

###
@parameters t σ ρ β
@variables x(t) y(t) z(t)
@derivatives D'~t

eqs = [D(x) ~ σ * (y - x), D(y) ~ x * (ρ - z) - y, D(z) ~ x * y - β * z]

lorenz1 = ODESystem(eqs, name = :lorenz1)
lorenz2 = ODESystem(eqs, name = :lorenz2)

@variables a
@parameters γ
#connected lorenz
connections = [0 ~ lorenz1.x + lorenz2.y + a * γ]
connected = ODESystem(connections, t, [a], [γ], systems = [lorenz1, lorenz2])

u0 = [lorenz1.x => 1.0, lorenz1.y => 0.0, lorenz1.z => 0.0, lorenz2.x => 0.0, lorenz2.y => 1.0, lorenz2.z => 0.0, a => 2.0]

p = [lorenz1.σ => 10.0, lorenz1.ρ => 28.0, lorenz1.β => 8 / 3, lorenz2.σ => 10.0, lorenz2.ρ => 28.0, lorenz2.β => 8 / 3, γ => 2.0]

tspan = (0.0, 100.0)
prob = ODEProblem(connected, u0, tspan, p)
sol = solve(prob, Rodas5())

plot(sol, vars = (a, lorenz1.z, lorenz2.z))
###
using DifferentialEquations
using DiffEqSensitivity
using Random
using Distributions
using Turing
using DataFrames
using StatsPlots

function sir_ode_sys!(du, u, p, t)
      (Su, In, Re, Cu) = u
      (β, c, γ) = p
      N = Su + In + Re
      infection = β * c * In / N * Su
      recovery = γ * In
      @inbounds begin
            du[1] = -infection
            du[2] = infection - recovery
            du[3] = recovery
            du[4] = infection
      end
      nothing
end

t₀ = 0.0
t₁ = 40.0

tspan = (t₀, t₁)

Δobs = 1.0
t_obs = (t₀+Δobs):Δobs:t₁

pop = 1e3
inf = 1e1
u₀ = [pop - inf, inf, 0.0, 0.0]
p = [0.05, 10.0, 0.25]

prob_ode = ODEProblem(sir_ode_sys!, u₀, tspan, p)

soln_ode = solve(prob_ode, Tsit5(), saveat = Δobs)

C = Array(soln_ode)[4, :]
X = C[2:end] - C[1:(end-1)]

#Random.seed!(1234)
Y = rand.(Poisson.(X))

bar(t_obs, Y, legend = false)
plot!(t_obs, X, legend = false)

@model bayes_sir(y) = begin
      # Calculate number of timepoints
      l = length(y)
      i₀ ~ Uniform(1 / pop, 1.0 - (1 / pop))
      β ~ Uniform(0.0, 1.0)
      I = i₀ * pop
      u0 = [pop - I, I, 0.0, 0.0]
      p = [β, 10.0, 0.25]
      tspan = (0.0, float(l))
      prob = ODEProblem(sir_ode_sys!, u0, tspan, p)
      sol = solve(prob, Tsit5(), saveat = 1.0)
      sol_C = Array(sol)[4, :] # Cumulative cases
      sol_X = sol_C[2:end] - sol_C[1:(end-1)]
      l = length(y)
      for i = 1:l
            y[i] ~ Poisson(sol_X[i])
      end
end;

ode_mynuts = sample(bayes_sir(Y), NUTS(0.65), 10000)

describe(ode_mynuts)
plot(ode_mynuts)

post_mynuts = DataFrame(ode_mynuts)

histogram2d(post_mynuts[!, :β], post_mynuts[!, :i₀], bins = 100, xlabel = "β", ylabel = "i₀")

function predict_sir(y, chain)
      # Length of data
      l = length(y)
      # Length of chain
      m = length(chain)
      # Choose random
      idx = sample(1:m)
      i₀ = chain[:i₀].value[idx]
      β = chain[:β].value[idx]
      I = i₀ * pop
      u0 = [pop - I, I, 0.0, 0.0]
      p = [β, 10.0, 0.25]
      tspan = (0.0, float(l))
      prob = ODEProblem(sir_ode_sys!, u0, tspan, p)
      sol = solve(prob, Tsit5(), saveat = 1.0)
      out = Array(sol)
      sol_X = [0.0; out[4, 2:end] - out[4, 1:(end-1)]]
      hcat(sol.t, out', sol_X)
end;

Xp = []
for i = 1:10
      pred = predict_sir(Y, ode_mynuts)
      push!(Xp, pred[2:end, 6])
end

scatter(t_obs, Y, legend = false)
plot!(t_obs, Xp, legend = false)
