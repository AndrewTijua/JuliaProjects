using DifferentialEquations, ParameterizedFunctions, Plots

lorenz = @ode_def begin
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z
end σ ρ β

u0 = [1.0, 1.0, 1.0]
tspan = (0.0, 100.0)
p = [10.0, 28.0, 8 / 3]
prob = ODEProblem(lorenz, u0, tspan, p)
sol = solve(prob)
plot(sol, vars = (1, 2, 3), legend = :none)
plot(sol, vars = (1, 2), legend = :none)
