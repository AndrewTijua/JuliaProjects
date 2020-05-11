using DifferentialEquations
using Plots;
plotlyjs();

const g = 9.81
const L = 1.0
const m = 0.5
const b = 0.1

u0 = [0, pi / 4]
tspan = (0.0, 100.0)

function pendulum(du, u, p, t)
    theta = u[1]
    dtheta = u[2]
    du[1] = dtheta
    du[2] = -(g / L) * sin(theta) - dtheta * m * b
end

penprob = ODEProblem(pendulum, u0, tspan)
soln = solve(penprob)

plot(soln)

function dampedharmonic(du, u, p, t)
    return (-(g / L) * sin(u) - du * m * b)
end

prob = SecondOrderODEProblem(dampedharmonic, pi / 4, 0.0, tspan)
soln_2 = solve(prob, McAte5(), dt = 0.1)

plot(soln_2)
