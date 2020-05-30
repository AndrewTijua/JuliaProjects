using DifferentialEquations, ParameterizedFunctions, Plots

lorenz = @ode_def begin                  # define the system
 dx = σ * (y - x)
 dy = x * (ρ - z) - y
 dz = x * y - β*z
end σ ρ β

u0 = [1.0,1.0,1.0]                       # initial conditions
tspan = (0.0,100.0)                      # timespan
p = [10.0,28.0,8/3]                      # parameters
prob = ODEProblem(lorenz, u0, tspan, p)  # define the problem
sol = solve(prob)                        # solve it
plot(sol, vars = (1, 2, 3))              # plot solution in phase space
