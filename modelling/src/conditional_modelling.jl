using DifferentialEquations
using OrdinaryDiffEq
using DiffEqCallbacks
using ModelingToolkit

using Random
using Distributions
using Turing
using DataFrames

using Plots
plotlyjs()

@parameters t δ
@variables x(t)
@derivatives D'~t

eqs = [D(x) ~ -δ * x]
de = ODESystem(eqs, t, [x], [δ])
ode_f = ODEFunction(de)

u0 = [1.0]
tspan = (0.0, 10.0)
p = [0.5]

condition(u, t, integrator) = u[1] - 0.25
effect!(integrator) = integrator.u[1] += 1.0
cb = ContinuousCallback(condition, effect!)

prob = ODEProblem(ode_f, u0, tspan, p, callback = cb)
sol = solve(prob)

plot(sol)
####
@parameters t g d
@variables x(t) y(t)
@derivatives D'~t
x_t = Variable(D(x))(t)
y_t = Variable(D(y))(t)
sgX = x_t * abs(x_t)
sgY = y_t * abs(y_t)

eqs = [D(D(x)) ~ -(sgX * d), D(D(y)) ~ -g - (sgY * d)]

de = ODESystem(eqs)
de = ode_order_lowering(de)

ode_f = ODEFunction(de)

u0 = [D(x) => 150.0, D(y) => 10.0, x => 1.0, y => 5.0]
tspan = (0.0, 100.0)
p = [g => 9.81, d => 0.1]

function condition(out, u, t, integrator)
    out[1] = u[4] #y condition
    out[2] = u[3] * (u[3] - 10.0) #x condition
    out[3] = u[4] + 0.005 * abs(u[1])
end

cor = 0.9

function effect!(integrator, idx)
    if idx == 2
        integrator.u[1] = -cor * integrator.u[1]
    elseif idx == 1
        integrator.u[2] = -cor * integrator.u[2]
    elseif idx == 3
        terminate!(integrator)
    end
end

cb = VectorContinuousCallback(condition, effect!, 3)

prob = ODEProblem(de, u0, tspan, p, callback = cb)
sol = solve(prob, Vern6())

plot(sol, vars = (3, 4))
####
