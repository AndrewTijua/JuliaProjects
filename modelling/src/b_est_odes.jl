using DifferentialEquations
using DiffEqSensitivity
using Random
using Distributions
using Turing
using DataFrames
using StatsPlots

using Plots
plotlyjs()

function sird_ode_sys!(du, u, p, t)
      (Su, In, Re, De) = u
      (β, c, γ, δ, hccap) = p
      N = Su + In + Re
      recovery = γ * In

      c_mod = maximum([c * (1-10In/N), 0.0])

      infection = β * c_mod * In / N * Su
      # if (In <= hccap)
            death = δ * In
      # else
      #       death = δ * In * (In/hccap)
      # end
      @inbounds begin
            du[1] = -infection
            du[2] = infection - recovery - death
            du[3] = recovery
            du[4] = death
      end
      nothing
end

t₀ = 0.0
t₁ = 1000.0

tspan = (t₀, t₁)

pop = 66e6
inf = 1e1
u₀ = [pop - inf, inf, 0.0, 0.0]
p = [0.05, 10.0, 0.1, 0.0001, pop/1e3]

prob_ode = ODEProblem(sird_ode_sys!, u₀, tspan, p)

soln_ode = solve(prob_ode)

plot(soln_ode)

@model bayes_sirm(y, times, Δt) = begin
      # Calculate number of timepoints
      l = length(y)
      i₀ ~ Uniform(1 / pop, 1.0 - (1 / pop))
      β ~ Uniform(0.0, 0.5)
      γ ~ Uniform(0.0, 1.0)
      δ ~ Uniform(0.0, 0.5)
      I = i₀ * pop
      u0 = [pop - I, I, 0.0, 0.0]
      p = [β, 10.0, γ, δ]
      tspan = (minimum(times), maximum(times))
      prob = ODEProblem(sirm_ode_sys!, u0, tspan, p)
      sol = solve(prob)
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
      prob = ODEProblem(sirm_ode_sys!, u0, tspan, p)
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
