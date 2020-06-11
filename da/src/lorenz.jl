using DifferentialEquations, ParameterizedFunctions, Plots
plotlyjs()
lorenz = @ode_def begin
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z
end σ ρ β

u0 = [1.0, 1.0, 1.0]
# tspan = (0.0, 10000.0)
tspan = (0.0, 100.0)
p = [10.0, 28.0, 8 / 3]

condition(u, t, integrator) = u[1]
effect!(integrator) = nothing

cb = ContinuousCallback(condition, effect!, save_positions = (true, false))

prob = ODEProblem(lorenz, u0, tspan, p)
sol = solve(prob, Tsit5())
plot(sol, vars = (1, 2, 3), legend = :none)
# sol_pc = solve(prob, Vern9(), callback = cb, save_everystep = false)
# scatter(sol_pc, vars = (2, 3), legend = :none)
